"""State-guided cloth manipulation on exported current House geometry.

Contact-only cloth diagnostic with prescribed robot motion. Cloth is never
attached to the robot. This is NOT a fully dynamic robot simulator and is
explicitly excluded from production until native robot/contact coupling exists.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import newton
import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation, Slerp
from scipy.optimize import brentq

from tablecloth_robot_kinematics import Robot
from tablecloth_robot_collisions import add_robot_collision, convex_contact_mesh
from tablecloth_scene_binding import add_newton_collisions, load_scene_collisions, source_record, station_xyyaw


@wp.kernel
def pad_surface_distance(mesh: wp.uint64, points: wp.array(dtype=wp.vec3),
                         distance: wp.array(dtype=float), closest: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    query = wp.mesh_query_point_sign_parity(mesh, points[i], 1.)
    if query.result:
        q = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        closest[i] = q
        distance[i] = query.sign * wp.length(points[i] - q)
    else:
        closest[i] = points[i]
        distance[i] = 1.


@wp.kernel
def drive(qd: wp.array(dtype=wp.vec3),
          body: wp.array(dtype=wp.transform), body_b: wp.array(dtype=wp.transform),
          body_vb: wp.array(dtype=wp.spatial_vector),
          start: wp.array(dtype=wp.transform), stop: wp.array(dtype=wp.transform),
          body_com: wp.array(dtype=wp.vec3), substep_dt: float,
          fraction: float, alpha_offset: wp.array(dtype=wp.float32), damping: float, n_body: int):
    # Diagnostic prescribed rigid boundary motion only. No particle position,
    # owner, anchor, mass or target velocity is available to this kernel.
    i = wp.tid()
    alpha = fraction + alpha_offset[0]
    if i < n_body:
        a, b = start[i], stop[i]
        p = wp.lerp(wp.transform_get_translation(a), wp.transform_get_translation(b), alpha)
        r = wp.quat_slerp(wp.transform_get_rotation(a), wp.transform_get_rotation(b), alpha)
        x = wp.transform(p, r)
        previous = body[i]
        body_b[i] = x
        linear = (wp.transform_point(x, body_com[i]) - wp.transform_point(previous, body_com[i])) / substep_dt
        dq = r * wp.quat_inverse(wp.transform_get_rotation(previous))
        if dq[3] < 0.0:
            dq = -dq
        angular = wp.vec3(dq[0], dq[1], dq[2]) * (2.0 / substep_dt)
        body_vb[i] = wp.spatial_vector(linear, angular)
    if i < qd.shape[0]:
        qd[i] = qd[i] * damping


def smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (10 + t * (-15 + 6 * t))


@wp.kernel
def strain_delta(q: wp.array(dtype=wp.vec3), inv_mass: wp.array(dtype=wp.float32),
                 edges: wp.array2d(dtype=wp.int32), rest: wp.array(dtype=wp.float32),
                 ratio: float, delta: wp.array(dtype=wp.vec3), count: wp.array(dtype=wp.float32)):
    i = wp.tid()
    a, b = edges[i, 0], edges[i, 1]
    weight = inv_mass[a] + inv_mass[b]
    d = q[b] - q[a]
    length = wp.length(d)
    if weight > 0.0 and length > rest[i] * ratio:
        correction = d * ((length - rest[i] * ratio) / (length * weight))
        wp.atomic_add(delta, a, correction * inv_mass[a])
        wp.atomic_add(delta, b, -correction * inv_mass[b])
        wp.atomic_add(count, a, 1.0)
        wp.atomic_add(count, b, 1.0)


@wp.kernel
def strain_apply(q: wp.array(dtype=wp.vec3), delta: wp.array(dtype=wp.vec3), count: wp.array(dtype=wp.float32)):
    i = wp.tid()
    if count[i] > 0.0:
        q[i] = q[i] + delta[i] / count[i]
    delta[i] = wp.vec3(0.0)
    count[i] = 0.0


@wp.kernel
def strain_velocity(q:wp.array(dtype=wp.vec3),previous:wp.array(dtype=wp.vec3),
                    qd:wp.array(dtype=wp.vec3),inv_mass:wp.array(dtype=wp.float32),inv_dt:float):
    i=wp.tid()
    if inv_mass[i]>0.0:
        qd[i]=qd[i]+(q[i]-previous[i])*inv_dt


@wp.kernel
def proposed_displacement(q:wp.array(dtype=wp.vec3),previous:wp.array(dtype=wp.vec3),
                          displacement:wp.array(dtype=wp.vec3)):
    i=wp.tid()
    displacement[i]=q[i]-previous[i]


def folded_initial(uv, size, basket_center, floor, folds=2):
    """Continuous accordion folds inside the measured basket footprint.

    No initial corner is tied to a historical gripper coordinate. Small fold
    inclinations separate layers without discontinuities at crease vertices.
    """
    def zig(s, length):
        phase = s / (length / folds)
        index = np.minimum(np.floor(phase).astype(int), folds-1)
        local = phase - index
        return np.where(index % 2, 1 - local, local) * (length / folds)
    u, v = uv.T
    p = np.zeros((len(uv), 3))
    p[:, 0] = basket_center[0] + zig(v, size[1]) - size[1] / (2*folds)
    # Fold the already layered long strip around a cylinder. Reflecting its
    # layer offset on the second leg preserves layer ordering; adding two
    # independent zigzag heights causes sheets to intersect at crossed folds.
    radius = .016
    layer = .022 * (v-size[1]/2)
    radii = radius+layer
    # Preserve arc length along every u strip. Using one common angle with
    # different layer radii pre-stretched the original folded mesh by ~40%.
    straight = (size[0]-np.pi*radii)/2
    theta = np.clip((u-straight)/radii,0,np.pi)
    y = np.where(u < straight,u-straight,np.where(u > straight+np.pi*radii,
        size[0]-u-straight,radii*np.sin(theta)))
    # All bend circles share a center, so different-radius layers stay nested.
    mean_straight = (size[0]-np.pi*radius)/2
    p[:, 1] = basket_center[1]+y+(mean_straight-radius)/2
    p[:, 2] = floor+.005+radius+.022*size[1]/2+radii*np.cos(theta)
    return p


class Hands:
    def __init__(self, robot, calibration, detailed_contacts=None):
        # NpzFile otherwise decompresses every CAD array on each IK iteration.
        calibration = {key: calibration[key] for key in calibration}
        self.robot, self.calibration = robot, calibration
        self.detailed_contacts = detailed_contacts
        self.values = json.loads(str(calibration["seed_json"]))
        self.links = calibration["links"].tolist()
        self.grip = {"L": 0., "R": 0.}
        self.apply_grips()
        self.centres_local = {}
        # Use the actual opposing distal collision faces, rather than link
        # origins, to locate the cloth inside the jaws.
        for side in ("L", "R"):
            self.centres_local[side] = self.mouth_local(side)[0]

    def apply_grips(self):
        for key in self.calibration:
            if key.startswith("joint_"):
                name = key[6:]
                side = "L" if "_l_" in name else "R"
                self.values[name] = float(np.interp(self.grip[side], self.calibration["grip"], self.calibration[key]))

    def mouth_local(self, side, poses=None):
        poses = self.robot.fk(self.values) if poses is None else poses
        base = poses[f"gripper_{side.lower()}_base_link"]
        points = []
        origins = []
        for half in ("inner", "outer"):
            link = f"gripper_{side.lower()}_{half}_link4"
            xf = np.linalg.inv(base) @ poses[link]
            origins.append(xf[:3, 3])
            points.append(self.calibration[f"points_{link}"] @ xf[:3, :3].T + xf[:3, 3])
        axis = origins[0] - origins[1]
        axis /= np.linalg.norm(axis)
        low, high = (points[1] @ axis).max(), (points[0] @ axis).min()
        first = np.cross(axis, [0, 0, 1.])
        if np.linalg.norm(first) < .1:
            first = np.cross(axis, [1., 0, 0])
        first /= np.linalg.norm(first)
        second = np.cross(axis, first)
        bounds = []
        for direction in (first, second):
            bounds.append([max((p @ direction).min() for p in points), min((p @ direction).max() for p in points)])
        centre = axis * (low + high) / 2 + first * np.mean(bounds[0]) + second * np.mean(bounds[1])
        return centre, axis, first, second, low, high, np.array(bounds)

    def centres(self):
        poses = self.robot.fk(self.values)
        return {s: poses[f"gripper_{s.lower()}_base_link"][:3, :3] @ self.mouth_local(s)[0] + poses[f"gripper_{s.lower()}_base_link"][:3, 3] for s in ("L", "R")}

    def shallow_grasp_offset(self, side, depth):
        """Mouth-center offset for a cloth corner near the distal jaw tips.

        The controller still commands the mouth center; this changes the bite
        location, not the cloth or the geometry-gated attachment condition.
        """
        centre, axis, first, second, low, high, bounds = self.mouth_local(side)
        # In this CAD frame second points toward the palm. Bounds describe the
        # overlap of the actual two distal faces, including jaw linkage motion.
        if np.dot(second, centre) >= 0:
            raise ValueError("Unexpected distal jaw frame")
        if not 0 <= depth <= np.ptp(bounds[1]):
            raise ValueError("Grasp depth must lie inside the opposing pad footprint")
        local_offset = second * (bounds[1, 0] + depth - bounds[1].mean())
        rotation = self.robot.fk(self.values)[f"gripper_{side.lower()}_base_link"][:3, :3]
        return -rotation @ local_offset

    def command(self, targets, grips, orientations=None):
        self.grip.update(grips)
        self.apply_grips()
        errors = {}
        for side in ("L", "R"):
            self.values, errors[side] = self.robot.solve(self.values, side, np.asarray(targets[side]),
                self.mouth_local(side)[0], None if orientations is None else orientations.get(side))
        return errors

    def transforms(self):
        poses = self.robot.fk(self.values)
        return np.asarray([np.r_[poses[n][:3, 3], Rotation.from_matrix(poses[n][:3, :3]).as_quat()] for n in self.links], dtype=np.float32)

    def enclosed(self, points, side, padding=.003, poses=None):
        poses = self.robot.fk(self.values) if poses is None else poses
        pose = poses[f"gripper_{side.lower()}_base_link"]
        local = (points - pose[:3, 3]) @ pose[:3, :3]
        c, axis, first, second, low, high, bounds = self.mouth_local(side,poses)
        # At closure the two penalty surfaces can overlap by the shell margin.
        # A grasp is local to their common transverse footprint and mid-slab.
        a, b, d = local @ axis, local @ first, local @ second
        mask = (a >= min(low, high) - padding) & (a <= max(low, high) + padding)
        mask &= (b >= bounds[0, 0] - padding) & (b <= bounds[0, 1] + padding)
        mask &= (d >= bounds[1, 0] - padding) & (d <= bounds[1, 1] + padding)
        return mask, local

    def pad_surfaces(self, points, side, poses=None):
        """Closest points on the same closed distal surfaces used by Newton.

        Whole-mesh projected bounds include empty space beyond rounded tips;
        they are insufficient to establish a physical two-pad contact.
        """
        poses = self.robot.fk(self.values) if poses is None else poses
        if not hasattr(self, '_pad_meshes'):
            self._pad_meshes = {}
        distances, vectors = [], []
        for half in ('inner', 'outer'):
            link = f'gripper_{side.lower()}_{half}_link4'
            if link not in self._pad_meshes:
                from tablecloth_gripper_contacts import contact_parts
                self._pad_meshes[link] = [wp.Mesh(
                    points=wp.array(vertices, dtype=wp.vec3, device='cpu'),
                    indices=wp.array(faces.reshape(-1), dtype=wp.int32, device='cpu'))
                    for vertices, faces in contact_parts(self.calibration, link, self.detailed_contacts)]
            pose = poses[link]
            local = (np.asarray(points) - pose[:3, 3]) @ pose[:3, :3]
            d = wp.empty(len(local), dtype=float, device='cpu')
            closest = wp.empty(len(local), dtype=wp.vec3, device='cpu')
            signed, nearest = [], []
            for mesh in self._pad_meshes[link]:
                wp.launch(pad_surface_distance, dim=len(local), inputs=[
                    mesh.id, wp.array(local, dtype=wp.vec3, device='cpu'), d, closest], device='cpu')
                signed.append(d.numpy().copy()); nearest.append(closest.numpy().copy())
            signed, nearest = np.stack(signed), np.stack(nearest)
            selected = signed.argmin(axis=0)
            distances.append(signed[selected, np.arange(len(local))])
            vectors.append((local - nearest[selected, np.arange(len(local))]) @ pose[:3, :3].T)
        return np.stack(distances, axis=1), np.stack(vectors, axis=1)


class Simulation:
    def __init__(self, args):
        if not getattr(args, 'diagnostic_kinematic_robot', False):
            raise ValueError('Native dynamic robot/contact coupling is not implemented here. '
                             'Use --diagnostic-kinematic-robot only for excluded diagnostic trials.')
        if args.start_held or args.linen_support_top:
            raise ValueError('Attachment-based held initialization and artificial linen support are retired')
        self.args, self.output = args, Path(args.output)
        self.policy = json.loads(Path(args.policy).read_text()) if args.policy else {}
        self.output.mkdir(parents=True, exist_ok=False)
        self.source = source_record()
        self.geometry = np.load(args.geometry, allow_pickle=False)
        self.meshes = load_scene_collisions(self.geometry)
        self.robot = Robot(json.loads(Path(args.robot).read_text()), station_xyyaw(args.task))
        contact_path = getattr(args, 'gripper_contact_meshes', None)
        detailed = dict(np.load(contact_path)) if contact_path else None
        self.hands = Hands(self.robot, np.load(Path(args.robot).with_suffix(".npz"), allow_pickle=False), detailed)
        initial_plan = self.policy.get("cloth_initialization", {})
        self.size = np.asarray(initial_plan.get("size_m", [.42, .62]), dtype=float)
        self.cloth_origin_xy = np.asarray(initial_plan.get("origin_xy_m", [-2.43, -.61]), dtype=float)
        if self.size.shape != (2,) or not np.isfinite(self.size).all() or (self.size <= 0).any():
            raise ValueError("Cloth dimensions must be two finite positive lengths")
        if self.cloth_origin_xy.shape != (2,) or not np.isfinite(self.cloth_origin_xy).all():
            raise ValueError("Cloth origin must be a finite XY pair")
        self.hands.values["idx02_body_joint2"] = args.waist_pitch
        self.hands.apply_grips()
        if args.start_held:
            if args.mode != "spread" or args.task != "8.10" or args.initial or args.pickup_corner:
                raise ValueError("--start-held is an explicit independent 8.10 initialization")
            short = args.grasp_edge == "short"
            self.held_init = dict(axis=0 if short else 1,span=.28 if short else .40,
                                  x=-2.43 if short else -2.38,height=1.17 if short else 1.00)
            hi = self.held_init
            errors = self.hands.command({"R":[hi["x"],-.32-hi["span"]/2,hi["height"]],
                                        "L":[hi["x"],-.32+hi["span"]/2,hi["height"]]}, {"R":1.,"L":1.})
            if max(errors.values()) > .003:
                raise ValueError(f"Unreachable held-cloth initializer: {errors}")
        nx, ny = args.nx, args.ny
        self.primary_n = (nx + 1) * (ny + 1)
        self.additional_cloth = self.policy.get('additional_cloth')
        self.n = self.primary_n * (2 if self.additional_cloth else 1)
        self.cloth_ids = np.repeat(np.arange(2 if self.additional_cloth else 1), self.primary_n)
        self.dt = 1 / args.fps
        builder = newton.ModelBuilder()
        # This is a cloth solver with prescribed robot motion. Rigid/rigid
        # impulses are not consumed; retain all cloth/shape contacts only.
        builder.default_shape_cfg.has_shape_collision = False
        builder.default_shape_cfg.ke = args.contact_stiffness
        builder.default_shape_cfg.kd = 100.
        for _ in range(2 if self.additional_cloth else 1):
            builder.add_cloth_grid(pos=wp.vec3(0., 0., 1.), rot=wp.quat_identity(), vel=wp.vec3(0.),
                dim_x=nx, dim_y=ny, cell_x=self.size[0]/nx, cell_y=self.size[1]/ny,
                mass=args.density*np.prod(self.size)/self.primary_n, tri_ke=args.stretch, tri_ka=args.stretch,
                tri_kd=args.stretch_damping, edge_ke=args.bend, edge_kd=args.bend_damping, particle_radius=args.radius)
        for group in dict.fromkeys(e["group"] for e, _, _ in self.meshes):
            add_newton_collisions(builder, self.meshes, group)
        builder.add_ground_plane()
        scene_shape_count = builder.shape_count
        self.linen_support=None
        if args.mode=="spread" and args.linen_support_top>0:
            basket=next(v for e,v,f in self.meshes if e["group"]=="basket" and "riser" not in e["prim_path"])
            centre=(basket.min(0)+basket.max(0))/2
            bottom=float(basket[:,2].min())+.026
            half=np.array([.16,.105,(args.linen_support_top-bottom)/2])
            centre[2]=bottom+half[2]
            if half[2]<=0:raise ValueError("Linen support top must be above the basket floor")
            cfg=builder.default_shape_cfg.copy();cfg.mu=.4
            builder.add_shape_box(body=-1,xform=wp.transform(centre,wp.quat_identity()),hx=half[0],hy=half[1],hz=half[2],cfg=cfg)
            self.linen_support=dict(centre=centre.tolist(),half_extents=half.tolist(),model="static packed neighboring linens; compliance not calibrated")
        self.body_ids = {}
        self.gripper_contact_meshes = []
        from tablecloth_gripper_contacts import contact_parts
        for link, xf in zip(self.hands.links, self.hands.transforms()):
            body = builder.add_body(xform=wp.transform(xf[:3], xf[3:]), is_kinematic=True, label=link)
            cfg = builder.default_shape_cfg.copy()
            cfg.mu, cfg.ke, cfg.kd, cfg.margin = 1.2, args.contact_stiffness, 50., .0005
            parts = contact_parts(self.hands.calibration, link, detailed)
            for points, triangles in parts:
                mesh = newton.Mesh(points, triangles.reshape(-1))
                if detailed is not None and bool(detailed.get(f'concave_{link}', False)):
                    builder.add_shape_mesh(body=body, mesh=mesh, cfg=cfg)
                else:
                    builder.add_shape_convex_hull(body=body, mesh=mesh, cfg=cfg)
            self.gripper_contact_meshes.append(dict(link=link, parts=len(parts),
                vertices=sum(len(v) for v, f in parts), triangles=sum(len(f) for v, f in parts),
                approximation='detailed closed CAD surface union' if detailed is not None and f'part_count_{link}' in detailed
                              else 'closed outward convex hull of current CAD vertices'))
            self.body_ids[link] = body
        self.robot_body_shapes = []
        if args.robot_body_collisions:
            self.robot_body_shapes = self.robot.metadata.get('nonhand_collision_shapes', [])
            if not self.robot_body_shapes:
                raise ValueError('Re-export current robot CAD with nonhand_collision_shapes, '
                                 'or explicitly request --no-robot-body-collisions for an ablation')
            poses = self.robot.fk(self.hands.values)
            for shape in self.robot_body_shapes:
                link = shape['link']
                if link not in self.body_ids:
                    pose = poses[link]
                    self.body_ids[link] = builder.add_body(
                        xform=wp.transform(pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_quat()),
                        is_kinematic=True, label=link)
                cfg = builder.default_shape_cfg.copy()
                cfg.mu, cfg.ke, cfg.kd, cfg.margin = args.friction, args.contact_stiffness, 50., .0005
                add_robot_collision(builder, self.body_ids[link], shape, cfg)
        # Stable body ordering is retained for saved-state rendering.
        self.body_names = list(self.body_ids)
        builder.color(include_bending=True)
        self.model = builder.finalize()
        self.model.soft_contact_ke = args.contact_stiffness
        self.model.soft_contact_kd = 50.
        self.model.soft_contact_mu = args.friction
        mu = self.model.shape_material_mu.numpy()
        mu[:scene_shape_count] = args.friction
        self.model.shape_material_mu.assign(mu)
        self.solver = newton.solvers.SolverVBD(self.model, iterations=args.iterations,
            integrate_with_external_rigid_solver=True, particle_enable_self_contact=args.self_contact,
            particle_self_contact_radius=args.self_radius,
            particle_self_contact_margin=args.self_margin or args.self_radius*3,
            particle_collision_detection_interval=args.collision_refresh,
            particle_topological_contact_filter_threshold=args.contact_filter_rings,
            friction_epsilon=getattr(args, 'friction_epsilon', 1e-5), rigid_contact_k_start=args.contact_stiffness,
            deterministic=wp.DeterministicMode.NOT_GUARANTEED)
        self.collision = newton.CollisionPipeline(self.model, include_static_kinematic_pairs=False,
            rigid_contact_max=1, soft_contact_max=self.n*24, soft_contact_margin=.006)
        self.contacts = self.collision.contacts()
        self.a, self.b, self.control = self.model.state(), self.model.state(), self.model.control()
        rest = self.a.particle_q.numpy()
        self.uv = rest[:, :2].copy()
        self.uv -= self.uv.min(0)
        self.faces = self.model.tri_indices.numpy()
        self.edges = np.unique(np.sort(np.vstack([self.faces[:, [0, 1]], self.faces[:, [1, 2]], self.faces[:, [2, 0]]]), axis=1), axis=0)
        self.rest_lengths = np.linalg.norm(rest[self.edges[:, 0]] - rest[self.edges[:, 1]], axis=1)
        self.edges_wp=wp.array(self.edges,dtype=wp.int32)
        self.rest_wp=wp.array(self.rest_lengths,dtype=wp.float32)
        self.delta=wp.zeros(self.n,dtype=wp.vec3)
        self.count=wp.zeros(self.n,dtype=wp.float32)
        self.before_strain=wp.zeros(self.n,dtype=wp.vec3)
        # Current collision bounds; the main table mesh includes its top.
        table = next(v for e, v, f in self.meshes if e["group"] == "table")
        basket = next(v for e, v, f in self.meshes if e["group"] == "basket" and "riser" not in e["prim_path"])
        self.table_top = float(table[:, 2].max())
        self.table_bounds = np.array([table.min(0), table.max(0)])
        self.basket_bounds = np.array([basket.min(0), basket.max(0)])
        self.basket_center = self.basket_bounds.mean(0)
        self.basket_floor = float(basket[:, 2].min()) + .024
        self.corner_ids = [int(np.argmin(np.linalg.norm(self.uv - x, axis=1))) for x in ([0, 0], [self.size[0], 0], [0, self.size[1]], self.size)]
        if args.start_held:
            # A folded curtain is an explicit independent task initial state.
            # Arc length is the rest cloth width; the two held endpoints have
            # the explicitly configured separation. No cloth shape is reset after simulation starts.
            hi = self.held_init
            length = self.size[hi["axis"]]
            angle = brentq(lambda a:2*length/a*np.sin(a/2)-hi["span"],.01,6.27)
            radius = length/angle
            theta = (self.uv[:,hi["axis"]]/length-.5)*angle
            initial = np.column_stack((hi["x"]+radius*(np.cos(theta)-np.cos(angle/2)),
                -.32+radius*np.sin(theta),hi["height"]-self.uv[:,1-hi["axis"]]))
        elif args.mode == "spread":
            initial = folded_initial(self.uv, self.size, self.basket_center, max(self.basket_floor,args.linen_support_top),args.folds)
        else:
            initial = np.column_stack((self.cloth_origin_xy[0] + self.uv[:self.primary_n, 0], self.cloth_origin_xy[1] + self.uv[:self.primary_n, 1], np.full(self.primary_n, self.table_top + args.radius + .004)))
            if initial_plan.get('require_on_table') and not args.initial and not args.replay_action:
                margin = float(initial_plan.get('table_margin_m', .005))
                if (np.any(initial[:, :2] < self.table_bounds[0, :2] + margin)
                        or np.any(initial[:, :2] > self.table_bounds[1, :2] - margin)):
                    raise ValueError('Old cloth must start fully supported inside the table boundary')
        if self.additional_cloth:
            if args.mode != 'remove' or args.start_held or args.linen_support_top:
                raise ValueError('Two-cloth replacement requires old cloth on table and free clean cloth in basket')
            from tablecloth_canvas_initial import basket_single_fold
            clean = basket_single_fold(self.uv[self.primary_n:], self.size, self.basket_bounds)
            initial = np.concatenate((initial, clean))
        if args.initial:
            initial = np.load(args.initial)["cloth_q"][-1]
            if initial.shape != (self.n, 3):
                raise ValueError("Initial cloth grid differs from configured grid")
        if args.replay_action:
            with np.load(Path(args.replay_action)/"full_state.npz") as reference:
                initial=reference["initial_cloth_q"].copy()
                if (initial.shape!=(self.n,3) or not np.array_equal(reference["faces"],self.faces)
                        or not np.array_equal(reference["uv"], self.uv)):
                    raise ValueError("Same-action replay requires the same cloth topology and rest dimensions")
        self.a.particle_q.assign(initial)
        self.b.particle_q.assign(initial)
        self.mass = self.model.particle_mass.numpy().copy()
        self.inv_mass = self.model.particle_inv_mass.numpy().copy()
        self.owner = np.full(self.n, -1, dtype=np.int32)
        self.local = np.zeros((self.n, 3), dtype=np.float32)
        self.owner_wp = wp.array(self.owner, dtype=wp.int32)
        self.local_wp = wp.array(self.local, dtype=wp.vec3)
        self.start = wp.array(self.body_transforms(), dtype=wp.transform)
        self.stop = wp.array(self.body_transforms(), dtype=wp.transform)
        self.alpha_offset = wp.zeros(1,dtype=wp.float32)
        self.previous_body = self.body_transforms()
        self.previous_grip = self.hands.grip.copy()
        # Compatibility name for action planners: observations, never constraints.
        self.clamps, self.events, self.frames = {}, [], []
        self.contact_targets = {}
        self.stack_clamps, self.stack_grasp_sides = {}, set()
        self.orientations = {}
        self.pending_grasps = {}
        self.t = 0.
        self.graph = None
        self.initial = initial.copy()
        if args.start_held:
            if not self.clamp("R",self.corner_ids[0]) or not self.clamp("L",self.corner_ids[1 if args.grasp_edge == "short" else 2]):
                raise ValueError("Held initialization must enclose both local cloth corners")

    def body_transforms(self):
        poses = self.robot.fk(self.hands.values)
        return np.asarray([np.r_[poses[name][:3, 3], Rotation.from_matrix(poses[name][:3, :3]).as_quat()]
                           for name in self.body_names], dtype=np.float32)

    def physics_frame(self):
        # Reuse a two-substep graph. Capturing a whole frame duplicates Warp's
        # deterministic scatter scratch buffers and can exhaust a 24 GB GPU.
        for sub in range(2):
            self.a.clear_forces()
            wp.launch(drive, dim=max(self.n, self.model.body_count), inputs=[self.a.particle_qd,
                self.a.body_q, self.b.body_q, self.b.body_qd,
                self.start, self.stop, self.model.body_com, self.dt/self.args.substeps,
                (sub+1)/self.args.substeps, self.alpha_offset,
                np.exp(-self.args.damping*self.dt/self.args.substeps), self.model.body_count])
            self.collision.collide(self.a, self.contacts)
            self.solver.step(self.a, self.b, self.control, self.contacts, self.dt/self.args.substeps)
            # Material strain limiting, independent of the task and goal.
            # Account for the correction in velocity; applying it only before
            # VBD lets its predictor erase the associated constraint impulse.
            wp.copy(self.before_strain,self.b.particle_q)
            guarded=self.args.strain_contact_guard and self.args.self_contact and self.args.strain_iters>0
            batch = self.args.strain_contact_interval or self.args.strain_iters or 1
            for begin in range(0,self.args.strain_iters,batch):
                if guarded:
                    # Refresh between short batches so a large unconstrained
                    # correction is not computed and then almost all discarded.
                    self.solver._collision_detection_penetration_free(self.b)
                for _ in range(min(batch,self.args.strain_iters-begin)):
                    wp.launch(strain_delta,dim=len(self.edges),inputs=[self.b.particle_q,self.model.particle_inv_mass,
                        self.edges_wp,self.rest_wp,self.args.max_stretch,self.delta,self.count])
                    wp.launch(strain_apply,dim=self.n,inputs=[self.b.particle_q,self.delta,self.count])
                if guarded:
                    wp.launch(proposed_displacement,dim=self.n,inputs=[self.b.particle_q,
                        self.solver.pos_prev_collision_detection,self.solver.particle_displacements])
                    self.solver._penetration_free_truncation(self.b.particle_q)
            wp.launch(strain_velocity,dim=self.n,inputs=[self.b.particle_q,self.before_strain,self.b.particle_qd,
                self.model.particle_inv_mass,self.args.substeps/self.dt])
            self.a, self.b = self.b, self.a

    def setup_graph(self):
        if self.args.substeps % 2:
            raise ValueError("Even substeps are required for repeatable CUDA graph state swapping")
        if self.policy.get('bvh_rebuild_interval_frames'):
            # A refit of the flat rest-mesh tree can interleave separate cloths
            # or distant folded layers. Rebuild on actual positions before
            # capturing the new tree handles; no physical state is changed.
            self.solver.rebuild_bvh(self.a)
        with wp.ScopedCapture() as capture:
            self.physics_frame()
        self.graph = capture.graph

    def actual_hand_poses(self, side):
        """Current physical transforms, including the distal closing pads."""
        simulated_body = self.a.body_q.numpy()
        poses = {}
        for name in self.hands.links:
            if not name.startswith(f"gripper_{side.lower()}_"):
                continue
            xf=simulated_body[self.body_ids[name]]
            pose=np.eye(4);pose[:3,:3]=Rotation.from_quat(xf[3:]).as_matrix();pose[:3,3]=xf[:3]
            poses[name]=pose
        return poses

    def clamp(self, side, target_id, *, grip=None, event_time=None, substep=0, record_miss=True):
        """Compatibility API: observe present geometry, never impose a grasp.

        Recomputed even for an already observed pinch; losing contact or slipping
        out must clear the observation. Enclosure alone is not a successful lift.
        """
        self.contact_targets[side] = int(target_id)
        previous = self.clamps.get(side)
        found = self._observe_pinch(side, target_id, grip=grip, event_time=event_time,
                                    substep=substep, record_miss=record_miss)
        if not found:
            self.clamps.pop(side, None)
            self.stack_clamps.pop(side, None)
            if previous is not None:
                self.events.append(dict(time=self.t if event_time is None else event_time,
                    frame_index=len(self.frames), substep_index=substep,
                    event='pad_contact_lost', side=side, particles=previous.tolist()))
        return found

    def record_pad_contact(self, side, ids, *, event_time, substep, **evidence):
        ids = np.asarray(ids, dtype=int)
        previous = self.clamps.get(side)
        self.clamps[side] = ids
        if previous is None or not np.array_equal(previous, ids):
            self.events.append(dict(time=self.t if event_time is None else event_time,
                frame_index=len(self.frames), substep_index=substep,
                event='pad_contact_observed', side=side, particles=ids.tolist(),
                observation_only=True, holding_force_validated=False, **evidence))

    def _observe_pinch(self, side, target_id, *, grip=None, event_time=None, substep=0, record_miss=True):
        grip = self.hands.grip[side] if grip is None else grip
        if grip >= .9 and side in self.stack_grasp_sides:
            from tablecloth_stack_grasp import try_stack_clamp
            return try_stack_clamp(self, side, target_id, grip=grip, event_time=event_time, substep=substep)
        if grip < .9:
            return False
        # Both enclosure and local anchor use the same actual simulated pose.
        poses=self.actual_hand_poses(side)
        centre,axis,first,second,low,high,bounds=self.hands.mouth_local(side,poses)
        effective_radius=self.args.radius+.0005  # actual hand collision margin
        # The rigid pad meshes must not overlap. A late catch inside their
        # interpenetrating volume is not a physically enclosed cloth pinch.
        if high-low < 0. or high-low > 2*effective_radius:
            return False
        points = self.a.particle_q.numpy()
        # A small contiguous cloth patch near the requested hem only. Do not
        # accidentally weld together every folded layer in the jaw volume.
        material_distance = np.linalg.norm(self.uv - self.uv[target_id], axis=1)
        candidates = np.flatnonzero((material_distance <= self.args.clamp_radius) & (self.cloth_ids == self.cloth_ids[target_id]))
        distances, contact_vectors = self.hands.pad_surfaces(points[candidates], side, poses=poses)
        contact_tolerance = self.policy.get('deep_grasp', {}).get('contact_distance_tolerance_m', .001)
        contact = (distances >= -.0005).all(axis=1) & (distances <= effective_radius + contact_tolerance).all(axis=1)
        pose = poses[f'gripper_{side.lower()}_base_link']
        local = (points - pose[:3, 3]) @ pose[:3, :3]
        depths = local @ second - bounds[1, 0]
        minimum_depth = self.policy.get('deep_grasp', {}).get('minimum_depth_m', 0.)
        if minimum_depth:
            # The whole-pad projected gap is the narrowest tip gap. A deep
            # concave contact can be wider: use its two actual opposing faces,
            # together with the width/depth footprint, rather than that slab.
            across = local[candidates] @ first
            world_axis = pose[:3, :3] @ axis
            sides = contact_vectors @ world_axis
            contact &= ((across >= bounds[0, 0]) & (across <= bounds[0, 1])
                        & (depths[candidates] >= minimum_depth)
                        & (depths[candidates] <= np.ptp(bounds[1]))
                        & (sides[:, 0] <= 0.) & (sides[:, 1] >= 0.))
        ids = candidates[contact]
        if not len(ids):
            if record_miss:
                self.events.append(dict(time=self.t, event="missed_pad_contact", side=side,
                    target_distance_m=float(np.linalg.norm(points[target_id] - self.hands.centres()[side]))))
            return False
        ids = ids[np.argsort(material_distance[ids])[:self.args.clamp_vertices]]
        self.record_pad_contact(side, ids, event_time=event_time, substep=substep,
            actual_grip_fraction=float(grip), opposing_pad_gap_m=float(high-low),
            insertion_depths_m=depths[ids].tolist(), minimum_required_insertion_m=minimum_depth,
            contact_shell_m=effective_radius, extra_contact_tolerance_m=contact_tolerance,
            opposing_surface_distances_m=[distances[np.flatnonzero(candidates == i)[0]].tolist() for i in ids],
            model='opposing_pad_geometry_observation')
        return True

    def release(self, side, *, event_time=None, substep=0, reason="explicit_rule_release", pad_gap=None):
        # A planner notification only. Opening the actual fingers must release
        # cloth through contact dynamics; neither mass nor velocity is changed.
        ids = self.clamps.pop(side, np.array([], dtype=int))
        self.stack_clamps.pop(side, None)
        self.contact_targets.pop(side, None)
        self.pending_grasps.pop(side, None)
        if len(ids):
            self.events.append(dict(time=self.t if event_time is None else event_time,
                frame_index=len(self.frames), substep_index=substep,
                event='contact_observation_cleared', side=side, particles=ids.tolist(),
                reason=reason, opposing_pad_gap_m=pad_gap, observation_only=True))

    def update_clamps(self):
        """Reject legacy callers attempting to attach particles; never repair silently."""
        self.assert_contact_only_state()

    def assert_contact_only_state(self):
        if np.any(self.owner != -1):
            raise RuntimeError('Particle attachment is forbidden')
        mass = self.model.particle_mass.numpy()
        inverse = self.model.particle_inv_mass.numpy()
        if (not np.array_equal(mass, self.mass) or not np.array_equal(inverse, self.inv_mass)
                or not np.all(mass > 0) or not np.all(inverse > 0)):
            raise RuntimeError('Cloth mass/inverse mass changed or a fixed particle was introduced')

    def frame(self, phase, targets, grips):
        old_values, old_grips = self.hands.values.copy(), self.hands.grip.copy()
        errors = self.hands.command(targets, grips, self.orientations)
        final_values, final_grips = self.hands.values.copy(), self.hands.grip.copy()
        arm_names = self.robot.arm_names("L") + self.robot.arm_names("R")
        delta = max(abs(final_values[n]-old_values[n]) for n in arm_names)
        limit = self.args.max_joint_speed*self.dt
        steps = 1 if delta <= limit else int(np.ceil(1.875*delta/limit))
        # Insert actual physics steps, not interpolated pictures. Joint-space
        # retiming prevents IK branch changes from becoming single-frame jumps.
        for k in range(steps):
            u = smooth((k+1)/steps)
            self.hands.values = {n:old_values.get(n,0.)*(1-u)+v*u for n,v in final_values.items()}
            self.hands.grip = {side:old_grips[side]*(1-u)+final_grips[side]*u for side in ("L","R")}
            self.hands.apply_grips()
            commanded = targets if k == steps-1 else self.hands.centres()
            self.advance(phase, commanded, errors if k == steps-1 else {"L":0.,"R":0.}, targets, steps>1)

    def update_grasp_contacts(self, fraction, grip_start):
        """Refresh contact observations without modifying any physical state."""
        substep = round(fraction*self.args.substeps)
        event_time = self.t + fraction*self.dt
        targets = {**self.contact_targets, **self.pending_grasps}
        for side, target in targets.items():
            grip = grip_start[side]*(1-fraction) + self.hands.grip[side]*fraction
            self.clamp(side, target, grip=grip, event_time=event_time,
                       substep=substep, record_miss=False)

    def advance(self, phase, targets, errors, planner_targets, retimed):
        if self.args.max_frames and len(self.frames) >= self.args.max_frames:
            raise StopIteration('Explicit diagnostic frame limit reached; incomplete episode')
        self.assert_contact_only_state()
        body = self.body_transforms()
        self.start.assign(self.previous_body)
        self.stop.assign(body)
        self.previous_body = body.copy()
        rebuild_interval = int(self.policy.get('bvh_rebuild_interval_frames', 0))
        if self.graph is None or (rebuild_interval and len(self.frames) % rebuild_interval == 0):
            self.setup_graph()
        grip_start=self.previous_grip.copy()
        self.update_grasp_contacts(0.,grip_start)
        for sub in range(0,self.args.substeps,2):
            self.alpha_offset.assign(np.array([sub/self.args.substeps],dtype=np.float32))
            wp.capture_launch(self.graph)
            self.update_grasp_contacts((sub+2)/self.args.substeps,grip_start)
        self.previous_grip=self.hands.grip.copy()
        points = self.a.particle_q.numpy()
        if not np.isfinite(points).all():
            raise RuntimeError("Non-finite cloth state")
        self.frames.append(dict(phase=phase, q=points, qd=self.a.particle_qd.numpy(),
            joint=np.array([self.hands.values.get(n, 0.) for n in self.robot.names]), body=body.copy(),
            grip=[self.hands.grip[s] for s in ("L", "R")], eef=np.array([self.hands.centres()[s] for s in ("L", "R")]),
            target=np.array([targets[s] for s in ("L", "R")]), planner_target=np.array([planner_targets[s] for s in ("L", "R")]), retimed=retimed, ik=np.array([errors[s] for s in ("L", "R")]), owner=self.owner.copy()))
        self.t += self.dt
        if len(self.frames) % 20 == 0:
            ratio = np.linalg.norm(points[self.edges[:, 0]]-points[self.edges[:, 1]],axis=1)/self.rest_lengths
            print(json.dumps(dict(frame=len(self.frames), time=round(self.t,2), phase=phase,
                min_z=float(points[:,2].min()), max_z=float(points[:,2].max()), strain_p99=float(np.quantile(ratio,.99)),
                strain_max=float(ratio.max()), particle_speed_max=float(np.linalg.norm(self.frames[-1]["qd"],axis=1).max()), ik=errors,
                clamps={s:len(i) for s,i in self.clamps.items()})), flush=True)
        if max(errors.values()) > self.args.max_ik_error:
            raise RuntimeError(f"Unreachable EEF target: {errors}")

    def move(self, phase, seconds, destinations=None, grips=None, follow=None, orientations=None):
        override = self.policy.get(phase,{})
        seconds = override.get("seconds",seconds)
        destinations = {**(destinations or {}),**override.get("destinations",{})}
        grips = {**(grips or {}),**override.get("grips",{})}
        # These controls change robot commands only. They do not prescribe a
        # cloth shape or change contact/material parameters during the rollout.
        for side in override.get("clear_orientations",[]):
            self.orientations.pop(side,None)
        orientations = {**(orientations or {}), **{
            side:Rotation.from_quat(quaternion).as_matrix()
            for side,quaternion in override.get("orientations_xyzw",{}).items()}}
        follow = {side:(index,np.asarray(override.get("follow_offsets",{}).get(side,offset)))
                  for side,(index,offset) in (follow or {}).items()}
        start = self.hands.centres()
        interpolators = {}
        for side, rotation in (orientations or {}).items():
            pose = self.robot.fk(self.hands.values)[f"gripper_{side.lower()}_base_link"]
            interpolators[side] = Slerp([0, 1], Rotation.from_matrix([pose[:3,:3], rotation]))
        destination = {s: np.asarray((destinations or {}).get(s, start[s])) for s in ("L", "R")}
        grip0 = self.hands.grip.copy()
        grip1 = {s: (grips or {}).get(s, grip0[s]) for s in ("L", "R")}
        steps = max(1, round(seconds*self.args.fps))
        print(f"[rule] {phase}: {steps} frames", flush=True)
        for k in range(steps):
            u = smooth((k+1)/steps)
            for side, interpolate in interpolators.items():
                self.orientations[side] = interpolate(u).as_matrix()
            goal = {s: start[s]*(1-u)+destination[s]*u for s in ("L", "R")}
            if "arc" in override:
                arc=override["arc"]
                theta=u*np.deg2rad(arc.get("angle_end_deg",90.))
                for side in ("L","R"):
                    goal[side][0]=arc["anchor_x"]+arc.get("direction",-1)*arc["radius"]*np.sin(theta)
                    goal[side][2]=arc["surface_z"]+arc["radius"]*np.cos(theta)
            if follow:
                q = self.a.particle_q.numpy()
                for s, (idx, offset) in follow.items():
                    bite_offset = self.hands.shallow_grasp_offset(s,self.args.grasp_depth) if self.args.grasp_depth>=0 else 0.
                    goal[s] = start[s]*(1-u)+(q[idx]+offset+bite_offset)*u
            for side,wave in override.get("oscillation",{}).items():
                # Smoothly switch a small rule-based shake on and off. Joint
                # retiming below remains responsible for actual speed limits.
                fraction=(k+1)/steps
                envelope=np.sin(np.pi*fraction)**2
                goal[side] += np.asarray(wave["amplitude_m"])*envelope*np.sin(
                    2*np.pi*wave.get("cycles",2)*fraction)
            self.frame(phase, goal, {s: grip0[s]*(1-u)+grip1[s]*u for s in ("L", "R")})
            if self.args.max_frames and len(self.frames) >= self.args.max_frames:
                raise StopIteration("Configured trial frame limit")

    def grasp_corner(self, side, corner, seconds=4.):
        if self.args.mode=="remove":
            hem=self.a.particle_q.numpy()[corner]
            outside=min(float(hem[0])-.14,float(self.table_bounds[0,0])-.14)
            orientation = None
            if side == "L":
                orientation = {side: Rotation.from_quat([0.,.866025404,0.,.5]).as_matrix()}
            if side == "R":
                # Engineering wrist waypoint: put the palm behind the hem,
                # outside the tabletop, rather than over its solid lip.
                orientation = {side: Rotation.from_quat([.819427993,-.000612775,.573181652,-.000428627]).as_matrix()}
            self.move(f"{side}_approach",seconds,{side:[outside,hem[1],max(.61,float(hem[2])+.15)]}, orientations=orientation)
            self.move(f"{side}_descend_outside",2.,{side:[outside,hem[1],hem[2]]})
            self.move(f"{side}_enter_below_edge",2.,follow={side:(corner,np.zeros(3))})
        elif side == "L":
            # Stay outside the hanging sheet before approaching its lower hem.
            # A direct descending chord crosses the free cloth and folds it
            # through the kinematic fingers before the intended pinch.
            start = self.hands.centres()[side]
            self.move("L_outside",2.,{side:[-2.55,start[1],start[2]]})
            self.move("L_align_hem",3.,follow={side:(corner,np.array([-.20,0,0]))})
            self.move("L_enter_hem",2.,follow={side:(corner,np.zeros(3))})
        else:
            self.move(f"{side}_approach", seconds, follow={side:(corner,np.array([0,0,.075]))})
            self.move(f"{side}_descend", 1.2, follow={side:(corner,np.zeros(3))})
        self.pending_grasps[side] = corner
        self.move(f"{side}_close", .7, grips={side:1.}, follow={side:(corner,np.zeros(3))})
        self.pending_grasps.pop(side)
        if not self.clamp(side,corner):
            raise RuntimeError(f"{side} gripper closed without cloth in its jaw footprint")

    def grasp_remove_pair(self, a, c):
        corners = {"R":a,"L":c}
        q = self.a.particle_q.numpy()
        rotations = {
            "R":Rotation.from_quat([.819427993,-.000612775,.573181652,-.000428627]).as_matrix(),
            "L":Rotation.from_quat([0.,.866025404,0.,.5]).as_matrix()}
        self.move("pair_approach_outside",4.,{side:[-2.50,q[idx,1],.61] for side,idx in corners.items()},orientations=rotations)
        self.move("pair_descend_outside",2.,{side:[-2.50,q[idx,1],q[idx,2]] for side,idx in corners.items()})
        follow = {side:(idx,np.zeros(3)) for side,idx in corners.items()}
        self.move("pair_enter_below_edge",2.,follow=follow)
        self.pending_grasps.update(corners)
        self.move("pair_close",.7,grips={"R":1.,"L":1.},follow=follow)
        self.pending_grasps.clear()
        if not all(side in self.clamps for side in corners):
            raise RuntimeError("Two-hand hem grasp did not enclose both corners")

    def run_policy(self):
        self.move("settle", self.args.settle)
        a,b,c,d = self.corner_ids
        if self.args.mode == "spread":
            short = self.args.grasp_edge == "short"
            if not self.args.start_held:
                # Choosing a different physical corner is an action change;
                # the folded mesh, material and basket initialization stay fixed.
                first = self.args.pickup_corner
                partner = first ^ (1 if short else 2)
                self.grasp_corner("R",self.corner_ids[first])
                self.move("single_corner_lift",8.,{"R":[-2.28,-.80,1.23]})
                self.move("present_free_corner",3.,{"R":[-2.28,-.40,1.20]})
                self.grasp_corner("L",self.corner_ids[partner])
            span = .40 if short else .595
            yr, yl = -.32-span/2, -.32+span/2
            height = 1.17 if short else 1.00
            self.move("open_held_edge",5.,{"R":[-2.43 if short else -2.38,yr,height],"L":[-2.43 if short else -2.38,yl,height]})
            self.move("unfold_settle",3.)
            # Lay forwards: the free hem starts near the table edge, then
            # the held hem extends away from the robot while descending.
            # Reversing that stroke piles both hems on the near edge.
            self.move("carry_above_table",3.,{"R":[-2.20,yr,.92 if short else .82],"L":[-2.20,yl,.92 if short else .82]})
            self.move("extend_over_table",3.5,{"R":[-1.98,yr,.64],"L":[-1.98,yl,.64]})
            self.move("lower_over_table",1.2,{"R":[-1.98,yr,.60],"L":[-1.98,yl,.60]})
            self.move("release_clearance",.5)
            self.move("release",.8,grips={"R":0.,"L":0.})
            self.release("R"); self.release("L")
        else:
            self.grasp_remove_pair(a,c)
            centres=self.hands.centres()
            self.move("withdraw_below_edge",2.,{s:[-2.47,centres[s][1],centres[s][2]] for s in ("L","R")})
            self.move("raise_outside_table",3.,{"R":[-2.47,-.61,.72],"L":[-2.47,.005,.72]})
            self.orientations.clear()
            self.move("lift_near_edge",5.,{"R":[-2.25,-.61,1.00],"L":[-2.25,.005,1.00]},
                orientations={"L":Rotation.from_quat([.356961550,.692332093,.031251051,.626321081]).as_matrix(),
                              "R":Rotation.from_quat([.349754686,.383748943,.694816936,-.497632228]).as_matrix()})
            self.orientations.clear()
            self.move("gather",3.,{"R":[-2.25,-.46,1.02],"L":[-2.25,-.25,1.02]})
            self.move("carry_to_basket",4.,{"R":[-2.24,-.97,1.03],"L":[-2.24,-.78,1.03]})
            self.move("lower_into_basket",4.,{"R":[-2.24,-.97,.49],"L":[-2.24,-.78,.49]})
            self.move("release",.8,grips={"R":0.,"L":0.})
            self.release("R"); self.release("L")
        centres = self.hands.centres()
        self.move("clear_released_cloth",2.,{side:[xyz[0],xyz[1],xyz[2]+.20] for side,xyz in centres.items()})
        self.move("retreat",3.,{"R":[-2.43,-.85,.77],"L":[-2.43,-.15,.77]})
        self.move("final_settle",4.)

    def run_replay(self):
        """Replay saved commands only; historical attachments are not replayed."""
        root=Path(self.args.replay_action)
        action=dict(np.load(root/"action.npz"))
        with np.load(root/"full_state.npz") as archive:
            state = {key: archive[key] for key in ('joint_names', 'phase', 'active_task_id')
                     if key in archive.files}
        meta=json.loads((root/"trial.json").read_text())
        if meta["fps"]!=self.args.fps or meta["mode"]!=self.args.mode:
            raise ValueError("Replay mode and clock must match the reference")
        names = action.get('joint_names', state['joint_names']).tolist()
        if set(names) != set(self.robot.names) or len(names) != len(self.robot.names):
            raise ValueError('Replay joint names must match the current robot')
        count = len(action['joint_target'])
        if (len(state['phase']) != count or len(action['time_s']) != count
                or not np.allclose(action['time_s'], np.arange(count)/self.args.fps, rtol=0, atol=1e-8)):
            raise ValueError('Replay requires a complete uniform command clock')
        for k,joints in enumerate(action["joint_target"]):
            if hasattr(self, "active_task") and "active_task_id" in state:
                self.active_task = str(state["active_task_id"][k])
            self.hands.values=dict(zip(names,joints))
            self.hands.grip={"L":float(action["action_lgrip"][k]),"R":float(action["action_rgrip"][k])}
            # No historical clamp events or saved cloth/robot states are applied.
            targets={s:action["eef_target"][k,i] for i,s in enumerate(("L","R"))}
            planner={s:action["planner_eef_goal"][k,i] for i,s in enumerate(("L","R"))}
            actual=self.hands.centres()
            errors={s:float(np.linalg.norm(actual[s]-targets[s])) for s in ("L","R")}
            self.advance(str(state["phase"][k]),targets,errors,planner,bool(action["retimed"][k]))

    def save(self, failure=None):
        if not self.frames:
            return
        self.assert_contact_only_state()
        f = self.frames
        q = np.stack([x["q"] for x in f])
        ratio = np.linalg.norm(q[:,self.edges[:,0]]-q[:,self.edges[:,1]],axis=2)/self.rest_lengths
        final=q[-1]
        meta=dict(schema="real2sim.tablecloth_rule_trial.v1",benchmark_task_id=self.args.task,
            mode=self.args.mode,initial_state_model="independent folded curtain held in two hands" if self.args.start_held else "folded basket cloth" if self.args.mode=="spread" else "explicit saved diagnostic cloth" if self.args.initial else "flat cloth fully supported within table" if self.policy.get('cloth_initialization', {}).get('require_on_table') else "flat cloth with near-edge overhang",source=self.source,config=vars(self.args),policy=self.policy,frames=len(f),fps=self.args.fps,
            failure=failure,grasp_model="contact_and_friction_only_no_attachments",
            diagnostic_only=True, eligible_for_production=False, fully_physical_robot=False,
            attachment_count=0, particle_mass_invariant=True,
            fixed_material=True,linen_support=self.linen_support,
            events=self.events,initial_bounds=[self.initial.min(0).tolist(),self.initial.max(0).tolist()],
            final_bounds=[final.min(0).tolist(),final.max(0).tolist()],
            stretch_p99_max=float(np.quantile(ratio,.99,axis=1).max()),stretch_max=float(ratio.max()),
            ik_error_max_m=float(np.max([x["ik"] for x in f])),
            final_table_contact_fraction=float(np.mean((abs(final[:,2]-self.table_top)<.015)&np.all(final[:,:2]>=self.table_bounds[0,:2],axis=1)&np.all(final[:,:2]<=self.table_bounds[1,:2],axis=1))),
            final_basket_xy_fraction=float(np.mean(np.all(final[:,:2]>=self.basket_bounds[0,:2],axis=1)&np.all(final[:,:2]<=self.basket_bounds[1,:2],axis=1))),
            final_below_basket_floor_fraction=float(np.mean(final[:,2] < self.basket_floor-.006)),
            numerical_determinism="not_guaranteed; repeatability requires tolerance checks",
            grasp_event_timing="geometry observations at two-substep boundaries; no state changes",
            strain_contact_guard="Newton 1.4 conservative self-contact truncation" if self.args.strain_contact_guard else "disabled",
            rigid_body_control="prescribed kinematics; rigid/rigid dynamics not validated",
            robot_cloth_collision_model="current USD closed gripper contact surfaces plus nonhand CAD primitives with hull fallback" if self.robot_body_shapes else "closed gripper contact surfaces only",
            gripper_contact_meshes=self.gripper_contact_meshes,
            nonhand_collision_shape_count=len(self.robot_body_shapes),
            nonhand_native_primitive_count=sum(bool(s.get('primitive')) for s in self.robot_body_shapes),
            retimed_frames=sum(x["retimed"] for x in f),motion_retiming="joint-space minimum-jerk interpolation with additional physical simulation steps; intermediate EEF commands are FK of commanded joints",
            owner_review="pending",automatic_benchmark_success=None)
        joint = np.stack([x["joint"] for x in f])
        np.savez_compressed(self.output/"full_state.npz",cloth_q=q,cloth_qd=np.stack([x["qd"] for x in f]),
            joint_position=joint,joint_names=np.array(self.robot.names),body_q=np.stack([x["body"] for x in f]),
            body_names=np.array(self.body_names),faces=self.faces,uv=self.uv,initial_cloth_q=self.initial,
            particle_mass=self.mass,particle_inv_mass=self.inv_mass,
            cloth_id=self.cloth_ids,clamp_owner=np.stack([x["owner"] for x in f]),phase=np.array([x["phase"] for x in f]),
            time_s=np.arange(len(f))/self.args.fps,benchmark_task_id=self.args.task,source_record_json=json.dumps(self.source))
        arm_indices=[self.robot.names.index(n) for s in ("L","R") for n in self.robot.arm_names(s)]
        np.savez_compressed(self.output/"action.npz",action_arm=joint[:,arm_indices],action_lgrip=np.array([x["grip"][0] for x in f]),
            action_rgrip=np.array([x["grip"][1] for x in f]),joint_target=joint,joint_names=np.array(self.robot.names),
            eef_target=np.stack([x["target"] for x in f]),eef_actual=np.stack([x["eef"] for x in f]),
            ik_error=np.stack([x["ik"] for x in f]),planner_eef_goal=np.stack([x["planner_target"] for x in f]),retimed=np.array([x["retimed"] for x in f]),time_s=np.arange(len(f))/self.args.fps,
            benchmark_task_id=self.args.task,control_mode="kinematic_joint_targets",fps=self.args.fps)
        (self.output/"trial.json").write_text(json.dumps(meta,indent=2)+"\n")
        print(json.dumps({k:v for k,v in meta.items() if k not in ("config","events")},indent=2),flush=True)


def main(simulation_class=Simulation):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("geometry","robot","output","task"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--mode",choices=["spread","remove"],required=True)
    p.add_argument("--diagnostic-kinematic-robot", action="store_true",
                   help="Explicitly exclude this prescribed-robot contact diagnostic from production")
    p.add_argument("--initial")
    p.add_argument("--gripper-contact-meshes", help="Detailed CAD contact parts exported by tablecloth_gripper_contacts.py")
    p.add_argument("--replay-action",help="Prior raw rule trial: replay identical joint/grip commands and initial cloth")
    p.add_argument("--start-held",action="store_true",help="Independent 8.10 setup: folded curtain already held in both hands")
    p.add_argument("--grasp-edge",choices=["short","long"],default="long")
    p.add_argument("--pickup-corner",type=int,choices=range(4),default=0,
                   help="Basket pickup material corner: (0,0), (width,0), (0,length), (width,length)")
    p.add_argument("--folds",type=int,choices=[2,3],default=2)
    p.add_argument("--clamp-vertices",type=int,choices=[1,2,4],default=1)
    p.add_argument("--policy",help="Phase duration/EEF waypoint overrides; task semantics still come from the catalog")
    p.add_argument("--physics-profile", help="Versioned material/solver defaults; explicit CLI values take precedence")
    p.add_argument("--self-contact",action=argparse.BooleanOptionalAction,default=True)
    p.add_argument("--robot-body-collisions",action=argparse.BooleanOptionalAction,default=True,
                   help="Include exported current nonhand robot CAD in cloth dynamics")
    p.add_argument("--strain-contact-guard",action=argparse.BooleanOptionalAction,default=True)
    p.add_argument("--contact-filter-rings",type=int,choices=[1,2,3],default=1,
        help="Self-contact exclusion rings; 1 retains nearby nonincident fold contacts")
    for name,default in dict(nx=36,ny=54,fps=20,substeps=32,iterations=24,max_frames=0,strain_iters=48,collision_refresh=4,strain_contact_interval=8).items():
        p.add_argument("--"+name.replace("_","-"),type=int,default=default)
    for name,default in dict(density=.35,stretch=500000.,stretch_damping=1.,bend=.003,bend_damping=.0001,
        radius=.0015,self_radius=.001,self_margin=0.,friction=.35,friction_epsilon=1e-5,contact_stiffness=1000000.,damping=.8,clamp_radius=.018,grasp_depth=.004,settle=1.5,max_ik_error=.01,waist_pitch=.65,max_joint_speed=1.5,max_stretch=1.025,linen_support_top=0.).items():
        p.add_argument("--"+name.replace("_","-"),type=float,default=default)
    preliminary=p.parse_args()
    if preliminary.physics_profile:
        profile=json.loads(Path(preliminary.physics_profile).read_text())
        allowed={"substeps","iterations","strain_iters","collision_refresh","strain_contact_interval",
                 "density","stretch","stretch_damping","bend","bend_damping","radius",
                 "self_radius","self_margin","friction","friction_epsilon","contact_stiffness",
                 "damping","max_stretch"}
        if not isinstance(profile,dict) or set(profile)-allowed:
            p.error("Physics profile must contain only material and solver parameters")
        actions={action.dest:action for action in p._actions}
        for name,value in profile.items():
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value):
                p.error(f"Invalid numeric physics profile value: {name}")
            if actions[name].type is int and value != int(value):
                p.error(f"Physics profile value must be an integer: {name}")
            profile[name]=actions[name].type(value)
        p.set_defaults(**profile)
    args=p.parse_args()
    wp.init()
    sim=simulation_class(args)
    error=None
    try:
        if args.replay_action:
            sim.run_replay()
        else:
            sim.run_policy()
    except StopIteration as exc:
        error=str(exc)
    except BaseException as exc:
        error=f"{type(exc).__name__}: {exc}"
        raise
    finally:
        sim.save(error)


if __name__ == "__main__":
    main()
