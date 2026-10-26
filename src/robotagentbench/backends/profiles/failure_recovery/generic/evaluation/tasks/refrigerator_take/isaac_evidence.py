"""Read-only native PhysX facts for refrigerator take; no simulation setters."""
import itertools
import math
import numpy as np
from .initial_state import FRIDGE, ROBOT, TARGET
from .observation_geometry import rotation_wxyz

DOORS = (FRIDGE + '/left_door', FRIDGE + '/right_door')
SOURCE_PARTS = ('col_shelf4_floor', 'col_shelf4_lip', 'col_shelf4_panel')
SOURCE_SUPPORTS = (DOORS[1] + '/col_shelf4_floor', DOORS[1] + '/col_shelf4_lip')


def outside_bounds(points, lower, upper):
    points = np.asarray(points, float)
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    if (points.ndim != 2 or points.shape[1] != 3 or not len(points)
            or lo.shape != (3,) or hi.shape != (3,)
            or not all(np.isfinite(x).all() for x in (points, lo, hi))
            or np.any(lo >= hi)):
        raise ValueError('Missing complete object/compartment geometry')
    return bool(np.any(points.min(0) > hi) or np.any(points.max(0) < lo))


def opposed_branches(forces):
    """Pad OR finger link contacts on opposing branches, never a command test.

    Zero is a numerical force test, not a tuned grasp-force threshold. This
    requires both load-bearing branches of the same hand on the same target.
    """
    values = np.asarray(forces, float)
    if values.shape != (2, 2, 3) or not np.isfinite(values).all():
        raise ValueError('Missing current inner/outer branch forces')
    return bool(any(np.linalg.norm(a) > 1e-9 and np.linalg.norm(b) > 1e-9
                    and np.dot(a, b) < 0 for a, b in values))


def _under(path, root):
    return path == root or path.startswith(root + '/')


def allowed_pair(a, b, source_parts=SOURCE_PARTS, target=TARGET):
    # Robot self-contact is not re-scored here; robot's native self-collision
    # policy remains untouched. Gripping target and hand-to-door are intended.
    if _under(a, ROBOT) and _under(b, ROBOT):
        return True
    for hand, other in ((a, b), (b, a)):
        if _under(hand, ROBOT) and '/gripper_' in hand:
            if _under(other, target) or any(_under(other, d) for d in DOORS):
                return True
    # Initial/remaining support in the source compartment is not a strike.
    for box, support in ((a, b), (b, a)):
        if _under(box, target) and support in tuple(DOORS[1] + '/' + name for name in source_parts):
            return True
    return False


class IsaacEvidenceReader:
    target = TARGET  # Preserve the original beef reader/test contract.
    target_root = TARGET

    def __init__(self, stage, random_plan=None, *, target_kind='beef'):
        from pxr import Usd, UsdGeom, UsdPhysics, Gf
        self.stage = stage
        from .random_reset import PARTS
        from .variants import VARIANTS
        selected = next((v for v in VARIANTS.values() if v.kind == target_kind), None)
        if selected is None or (target_kind != 'beef' and random_plan is None):
            raise ValueError('Missing selected-target initial layout')
        self.target, self.target_root = selected.body, selected.root
        self.random_plan = random_plan
        self.source_parts = (PARTS[random_plan['boxes'][target_kind]['source_compartment']]
                             if random_plan else SOURCE_PARTS)
        self.source_supports = tuple(DOORS[1]+'/'+n for n in self.source_parts[:2])
        cache = UsdGeom.XformCache()
        self.closed_rotations = []
        for path in DOORS:
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise ValueError('Missing refrigerator door: ' + path)
            # Rotation only; body transforms may contain a uniform asset scale.
            m = np.asarray(cache.GetLocalToWorldTransform(prim), float)[:3, :3].T
            self.closed_rotations.append(m / np.linalg.norm(m, axis=0))
        self.target_points = []
        target = stage.GetPrimAtPath(self.target)
        if not target or not target.IsActive():
            raise ValueError('Missing selected refrigerator box: '+self.target)
        inv = cache.GetLocalToWorldTransform(target).GetInverse()
        for prim in Usd.PrimRange(target, Usd.TraverseInstanceProxies()):
            if prim.IsA(UsdGeom.Mesh):
                m = cache.GetLocalToWorldTransform(prim) * inv
                self.target_points.extend(list(m.Transform(p)) for p in UsdGeom.Mesh(prim).GetPointsAttr().Get())
        self.target_points = np.asarray(self.target_points, float)
        if self.target_points.ndim != 2 or not len(self.target_points):
            raise ValueError('Missing full target surface')
        # Cache a conservative envelope of the original source bin, door-local.
        inv = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(DOORS[1])).GetInverse()
        points = []
        for name in self.source_parts:
            prim = stage.GetPrimAtPath(DOORS[1] + '/' + name)
            if not prim or not prim.IsA(UsdGeom.Cube):
                raise ValueError('Missing original source bin part: ' + name)
            h = float(UsdGeom.Cube(prim).GetSizeAttr().Get()) / 2
            m = cache.GetLocalToWorldTransform(prim) * inv
            points.extend(list(m.Transform(Gf.Vec3d(*p)))
                          for p in itertools.product((-h, h), repeat=3))
        points = np.asarray(points, float)
        self.lower, self.upper = points.min(0), points.max(0)
        self.extra_geometry = {}
        if random_plan:
            for kind, box in random_plan['boxes'].items():
                if kind == target_kind:
                    continue
                body = stage.GetPrimAtPath(box['body'])
                if not body or not body.HasAPI(UsdPhysics.RigidBodyAPI):
                    raise ValueError('Missing background box body: '+box['body'])
                body_inv = cache.GetLocalToWorldTransform(body).GetInverse()
                surface = []
                for prim in Usd.PrimRange(body):
                    if prim.IsA(UsdGeom.Mesh):
                        tf = cache.GetLocalToWorldTransform(prim)*body_inv
                        surface.extend(list(tf.Transform(p)) for p in UsdGeom.Mesh(prim).GetPointsAttr().Get())
                # Tensor body transforms omit authored scale; incorporate it
                # into the local points, as opposed to changing the asset.
                body_tf = np.asarray(cache.GetLocalToWorldTransform(body),float)[:3,:3].T
                scale = np.linalg.norm(body_tf,axis=0)
                self.extra_geometry[kind] = dict(body=box['body'], root=box['root'],
                    points=np.asarray(surface,float)*scale,
                    supports=tuple(DOORS[1]+'/'+n for n in PARTS[box['source_compartment']][:2]))
        self.views = {}
        self.contact_views = []
        self.last = None
        self.paths = {}

    def initialize(self, world):
        from omni.physics.tensors import create_simulation_view
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools
        self.world = world
        self.native = create_simulation_view('numpy')
        self.interface = get_physx_simulation_interface()
        self.decode = PhysicsSchemaTools.intToSdfPath
        for path in (self.target, *DOORS, *(g['body'] for g in self.extra_geometry.values())):
            view = self.native.create_rigid_body_view(path)
            if view.count != 1 or list(view.prim_paths) != [path]:
                raise ValueError('Wrong body view: ' + path)
            self.views[path] = view
        # Public worker already instruments the robot articulation. Native
        # per-link tensor contact views require additional per-link APIs and
        # cannot be used for link2 without modifying instrumentation. Instead
        # reduce the existing report (also used by the accepted diagnostic).
        from pxr import PhysxSchema
        if not self.stage.GetPrimAtPath(ROBOT).HasAPI(PhysxSchema.PhysxContactReportAPI):
            raise ValueError('Existing robot contact reporter is unavailable')
        self.branches = {f'{ROBOT}/gripper_{hand}_{side}_link{link}': (h, s)
                         for h, hand in enumerate(('l', 'r'))
                         for s, side in enumerate(('inner', 'outer')) for link in (2, 4)}

    def pose(self, path):
        p = np.asarray(self.views[path].get_transforms(), float)
        if p.shape != (1, 7) or not np.isfinite(p).all():
            raise ValueError('Missing current rigid pose')
        return p[0, :3], rotation_wxyz([p[0, 6], *p[0, 3:6]])

    def sample(self):
        now = float(self.world.current_time)
        if self.last is not None and now <= self.last:
            raise ValueError('Stale physical sample')
        dt = float(self.world.get_physics_dt())
        branches = np.zeros((2, 2, 3), float)
        poses = [self.pose(path) for path in DOORS]
        angles = [math.degrees(math.acos(float(np.clip((np.trace(ref.T @ rotation)-1)/2, -1, 1))))
                  for ref, (_, rotation) in zip(self.closed_rotations, poses)]
        p, r = self.pose(self.target)
        points_world = self.target_points @ r.T + p
        door_p, door_r = poses[1]
        points_local = (points_world - door_p) @ door_r
        out = outside_bounds(points_local, self.lower, self.upper)
        # Only reduce relevant positive-impulse contacts; no point logs or old
        # force reuse. Existing public/L1 reporters are retained, not modified.
        headers, data = self.interface.get_contact_report()[:2]
        collision = False
        support = False
        external_support = False
        extra_support = {kind: False for kind in self.extra_geometry}
        for h in headers:
            pair = []
            for token in (h.collider0, h.collider1):
                key = int(token)
                if key not in self.paths:
                    self.paths[key] = str(self.decode(token))
                pair.append(self.paths[key])
            relevant = any(_under(p, ROBOT) or _under(p, self.target_root) for p in pair)
            extra_relevant = any(_under(p,g['root']) for p in pair for g in self.extra_geometry.values())
            if not relevant and not extra_relevant:
                continue
            start, count = int(h.contact_data_offset), int(h.num_contact_data)
            if count == 0:
                continue  # CONTACT_LOST carries no current contact points.
            if start < 0 or count < 0 or start + count > len(data):
                raise ValueError('Invalid contact range')
            impulses = [np.asarray(tuple(data[i].impulse), float) for i in range(start, start+count)]
            if not all(np.isfinite(v).all() for v in impulses):
                raise ValueError('Nonfinite contact impulse')
            loaded = any(np.linalg.norm(v) > 1e-9 for v in impulses)
            if not loaded:
                continue
            for kind,g in self.extra_geometry.items():
                for i in (0,1):
                    if _under(pair[i],g['root']) and pair[1-i] in g['supports']:
                        extra_support[kind] |= float(sum(impulses,np.zeros(3))[2])*(1 if i==0 else -1)>1e-9
            if any(_under(p, self.target_root) for p in pair):
                box_index = next(i for i, path in enumerate(pair) if _under(path, self.target_root))
                other_index = 1 - box_index
                other = pair[other_index]
                on_box = sum(impulses, np.zeros(3)) * (1 if box_index == 0 else -1)
                support |= other in self.source_supports and on_box[2] > 1e-9
                if not _under(other, ROBOT) and on_box[2] > 1e-9:
                    external_support = True
                for branch, (h, s) in self.branches.items():
                    if _under(other, branch):
                        # Match existing pad normal-impulse force reduction.
                        sign = 1 if other_index == 0 else -1
                        for i, impulse in zip(range(start, start+count), impulses):
                            normal = np.asarray(tuple(data[i].normal), float)
                            norm = np.linalg.norm(normal)
                            if not np.isfinite(normal).all() or norm < 1e-9:
                                raise ValueError('Invalid contact normal')
                            normal /= norm
                            branches[h, s] += sign * normal * abs(np.dot(impulse, normal)) / dt
                        break
            if relevant:
                collision |= not allowed_pair(*pair, self.source_parts, target=self.target_root)
        self.last = now
        # Grasp can begin while still in the source bin; once extracted, a box
        # supported by a table/floor is not counted as suspended robot holding.
        held = opposed_branches(branches) and not (out and external_support)
        return dict(time_s=now, door_angles_deg=angles, target_held=held,
                    target_fully_out=out, non_operational_collision=bool(collision),
                    source_floor_contact=bool(support), external_support=bool(external_support),
                    branch_forces_N=branches.tolist(),
                    background_box_support=extra_support,
                    background_box_positions={kind: self.pose(g['body'])[0].tolist()
                                              for kind,g in self.extra_geometry.items()},
                    target_position_m=p.tolist(), target_rotation=r.tolist())

    def close(self):
        self.views.clear()
        self.contact_views.clear()
        self.native = self.interface = self.world = self.stage = None
