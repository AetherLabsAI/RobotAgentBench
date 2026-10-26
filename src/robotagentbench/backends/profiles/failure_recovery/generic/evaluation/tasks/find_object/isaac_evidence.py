"""Live physical readout of the find_object family: poses + PhysX contact reports -> Facts (facts.py). Only session-layer contact-report
instrumentation is authored; no pose, drive, material, collision or solver setting is ever written. The reader never scores.

Conventions (DEFINITION_TABLE 实现绑定):
  contact sample   a contact point with non-zero impulse (impulse^2 > 1e-18)
  supported by X   a non-robot collider under X pushes the target upward: contact normal on the target has world z > 0 and the contact
                   point is not above the target's collision centre (world z of the body origin + half extent is the centre proxy)
  grasped          a confirmed grasp (both pads of one gripper touch the target with opposing normals and the target kept its pose relative to that hand for 0.1 s), then held while that hand keeps bilateral contact (gaps < 0.1 s bridged; relative jitter allowed, user 2026-10-06)
                   for 0.10 s within 3 mm / 5 deg (public GraspDetector)
  ground           /World/Background/Shell/Floor (the only floor collider of the house baseline)
  above_rim        drawer tasks: lowest world z of the target's collision vertices > highest world z of the drawer's wall collision vertices
  in_destination   4.17: every target collision vertex projects inside the horizontal convex outline of the TV-table top colliders
"""
from collections import deque
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from arena_api.contact_identity import ContactIdentityCache
from arena_api.policy_execution import PHYSICS_DT, SUBSTEPS
from arena_api.bimanual import CONTROL_DT
from evaluation.tasks.shelf_pick.physical_detection import GraspDetector, DetectionConfig
from evaluation.tasks.shelf_pick.geometry import rotation
from .specs import ROBOT, TARGET, DISTRACTORS, DRAWER, TV, FLOOR, SHELF_SUPPORTS, STATION, spec as key_spec
from .grasp_latch import GraspLatch
from .facts import Facts, TargetFacts, DistractorFacts, SUPPORT, DRAWER_FLOOR, DRAWER_WALL, DESTINATION, GROUND, FURNITURE as ON_FURNITURE, OBJECT as ON_OBJECT

READER_VERSION = 'arena.find_object.reader.v1'
PADS = {h: tuple(ROBOT + f'/gripper_{h}_{side}_link4' for side in ('inner', 'outer')) for h in ('l', 'r')}
HANDS = {h: ROBOT + f'/gripper_{h}_base_link' for h in PADS}
IMPULSE_SQUARED_MIN = 1e-18
INITIAL_WINDOW_S = 0.5


def under(path, root):
    return path == root or path.startswith(root.rstrip('/') + '/')


def serializable(v):
    if isinstance(v, dict): return {k: serializable(x) for k, x in v.items()}
    if isinstance(v, (tuple, list, set, frozenset)): return [serializable(x) for x in v]
    if isinstance(v, np.generic): return v.item()
    if isinstance(v, np.ndarray): return v.tolist()
    return v


def collider_points(prim, to_root):
    from pxr import UsdGeom, Gf
    import itertools
    if prim.IsA(UsdGeom.Mesh):
        pts = prim.GetAttribute('points').Get()
    elif prim.IsA(UsdGeom.Cube):
        s = UsdGeom.Cube(prim).GetSizeAttr().Get() / 2
        pts = list(itertools.product((-s, s), repeat=3))
    elif prim.IsA(UsdGeom.Cylinder):
        c = UsdGeom.Cylinder(prim)
        r, h, ax = c.GetRadiusAttr().Get(), c.GetHeightAttr().Get() / 2, 'XYZ'.index(c.GetAxisAttr().Get())
        pts = []
        for s in (-h, h):
            for a in np.linspace(0, 2 * np.pi, 48, endpoint=False):
                v = [r * np.cos(a), r * np.sin(a)]; v.insert(ax, s); pts.append(v)
    else:
        raise ValueError('Unsupported collider type: ' + str(prim.GetPath()))
    return np.asarray([list(to_root.Transform(Gf.Vec3d(*[float(x) for x in p]))) for p in pts], dtype=float)


def body_colliders(stage, root, scaled_ok=False):
    """Collision vertices of one rigid body in the body frame (the frame PhysX reports poses in: body world transform without scale)."""
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    prim = stage.GetPrimAtPath(root)
    cache = UsdGeom.XformCache()
    world = cache.GetLocalToWorldTransform(prim)
    # PhysX body frame = rotation + translation of the prim's world transform (scale factored out)
    rot = Gf.Matrix4d(world); rot.Orthonormalize()
    frame = Gf.Matrix4d(rot.ExtractRotationMatrix(), world.ExtractTranslation())
    inverse = frame.GetInverse()
    out = {}
    for p in Usd.PrimRange(prim):
        if p.HasAPI(UsdPhysics.CollisionAPI) and p.GetAttribute('physics:collisionEnabled').Get() is not False and p.IsActive():
            out[str(p.GetPath())[len(root):]] = collider_points(p, cache.GetLocalToWorldTransform(p) * inverse)
    if not out:
        raise ValueError('No collision geometry: ' + root)
    return out


def hull2d(points):
    from scipy.spatial import ConvexHull
    pts = np.asarray(points, dtype=float)[:, :2]
    return pts[ConvexHull(pts).vertices]


def inside_polygon(points, poly):
    """All points inside (or on) the convex polygon (counter-clockwise vertices)."""
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        edge = b - a
        if np.any((points[:, 0] - a[0]) * edge[1] - (points[:, 1] - a[1]) * edge[0] > 1e-6):
            return False
    return True


class Contacts:
    def __init__(self, resolve, sleeping, event_types, roots):
        self.sleeping = sleeping
        self.found, self.persist, self.lost = event_types
        self.roots = tuple(roots)
        self.identity = ContactIdentityCache(resolve, self.relevant)
        self.active = {}

    def relevant(self, paths):
        return any(under(p, r) for p in paths for r in self.roots)

    def ingest(self, report):
        if not isinstance(report, tuple) or len(report) < 2 or report[0] is None or report[1] is None:
            raise RuntimeError('Missing current PhysX contact report')
        headers, data = report[:2]
        seen = set()
        for h in headers:
            paths, key, relevant = self.identity.get(h.stage_id, h.collider0, h.collider1)
            if not relevant:
                continue
            seen.add(key)
            if h.type == self.lost:
                self.active.pop(key, None)
                continue
            if h.type not in (self.found, self.persist):
                raise RuntimeError('Unknown contact event')
            count, offset = int(h.num_contact_data), int(h.contact_data_offset)
            if count <= 0 or offset < 0 or offset + count > len(data):
                raise RuntimeError('Incomplete physical contact')
            samples = []
            for i in range(offset, offset + count):
                v = data[i]
                impulse = (v.impulse.x, v.impulse.y, v.impulse.z)
                normal = (v.normal.x, v.normal.y, v.normal.z)
                point = (v.position.x, v.position.y, v.position.z)
                if not all(math.isfinite(x) for x in impulse + normal + point):
                    raise RuntimeError('Non-finite physical contact')
                squared = sum(x * x for x in impulse)
                if squared > IMPULSE_SQUARED_MIN:
                    samples.append((normal, point, math.sqrt(squared)))
            self.active[key] = (paths, int(h.stage_id), (int(h.actor0), int(h.actor1)), tuple(samples))
        for key, (_, stage_id, actors, _) in self.active.items():
            if key not in seen and not all(self.sleeping(stage_id, a) is True for a in actors):
                raise RuntimeError('Missing awake contact pair: ' + str(key))

    def pairs(self):
        for paths, _, _, samples in self.active.values():
            if samples:
                yield paths, samples


class FindObjectReader:
    def __init__(self, stage, output, episode_id, key, plan, config=DetectionConfig()):
        from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, PhysicsSchemaTools
        from omni.physx import get_physx_simulation_interface
        from omni.physx.bindings._physx import ContactEventType
        self.stage, self.spec, self.key, self.episode_id, self.config, self.plan = stage, key_spec(key), key, episode_id, config, plan
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        if UsdGeom.GetStageUpAxis(stage) != 'Z' or UsdGeom.GetStageMetersPerUnit(stage) != 1:
            raise ValueError('Expected metres/Z-up')

        def colliding(path):
            p = stage.GetPrimAtPath(path)
            return bool(p and p.IsActive() and p.HasAPI(UsdPhysics.CollisionAPI) and p.GetAttribute('physics:collisionEnabled').Get() is not False)
        if not colliding(FLOOR):
            raise ValueError('Missing ground collider: evidence would be invalid')
        self.distractors = tuple(DISTRACTORS + '/' + n for n in plan['distractors'])
        self.support_root = self.spec.support_root
        self.destination = self.spec.destination_root
        self.target_hull = np.vstack(list(body_colliders(stage, TARGET).values()))
        from scipy.spatial import ConvexHull
        self.target_hull = self.target_hull[ConvexHull(self.target_hull).vertices]
        self.drawer_walls = None
        if self.spec.kind == 'drawer':
            cols = body_colliders(stage, DRAWER)
            walls = [v for name, v in cols.items() if 'Bottom' not in name]
            if not walls:
                raise ValueError('Drawer wall colliders not found')
            self.drawer_walls = np.vstack(walls)
        self.destination_outline = None
        if self.spec.kind == 'place':
            pts = np.vstack(list(body_colliders(stage, self.destination).values()))     # tv table colliders in the table frame
            cache = UsdGeom.XformCache(); world = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(self.destination))
            from pxr import Gf
            rot = Gf.Matrix4d(world); rot.Orthonormalize()
            frame = Gf.Matrix4d(rot.ExtractRotationMatrix(), world.ExtractTranslation())
            w = np.asarray([list(frame.Transform(Gf.Vec3d(*p))) for p in pts])
            top = w[w[:, 2] >= w[:, 2].max() - 0.01]
            self.destination_outline = hull2d(top)
            self.destination_top_z = float(w[:, 2].max())
        self.interface = get_physx_simulation_interface()
        bodies_paths = [TARGET, *self.distractors] + ([DRAWER] if self.spec.kind == 'drawer' else [])
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            bodies = [stage.GetPrimAtPath(p) for p in bodies_paths]
            bodies += [p for p in Usd.PrimRange(stage.GetPrimAtPath(ROBOT)) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
            self.arm_links = [str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(ROBOT)) if p.HasAPI(UsdPhysics.RigidBodyAPI)
                              and p.GetName().startswith(('arm_r_link', 'arm_l_link', 'arm_r_end', 'arm_l_end', 'gripper_r_', 'gripper_l_'))]
            for p in bodies:
                if not p or not p.HasAPI(UsdPhysics.RigidBodyAPI):
                    raise ValueError('Missing contact-report body')
                PhysxSchema.PhysxContactReportAPI.Apply(p).CreateThresholdAttr(0.)

        def sleeping(stage_id, actor):
            p = stage.GetPrimAtPath(PhysicsSchemaTools.intToSdfPath(actor))
            if not p:
                raise RuntimeError('Unknown contact actor')
            if not p.HasAPI(UsdPhysics.RigidBodyAPI) or p.GetAttribute('physics:kinematicEnabled').Get() is True:
                return True
            return bool(self.interface.is_sleeping(stage_id, actor))
        roots = (TARGET, *self.distractors) + ((TV,) if self.spec.kind == 'place' else ())
        self.contacts = Contacts(PhysicsSchemaTools.intToSdfPath, sleeping,
                                 (ContactEventType.CONTACT_FOUND, ContactEventType.CONTACT_PERSIST, ContactEventType.CONTACT_LOST), roots=roots)
        self.detector = GraspDetector((TARGET,), config)
        self.latch = GraspLatch(tuple(PADS), PHYSICS_DT)          # user 2026-10-06: jitter after a confirmed grasp keeps it held
        self.hashes = {l.realPath: hashlib.sha256(Path(l.realPath).read_bytes()).hexdigest() for l in stage.GetUsedLayers() if l.realPath and Path(l.realPath).is_file()}
        self.step = -1
        self.latest = None
        self.start_step = None
        self.initial_position = None
        self.closed = False
        self.history = deque(maxlen=int(math.ceil(INITIAL_WINDOW_S / PHYSICS_DT - 1e-9)) + 2)
        self.touched = {d: False for d in self.distractors}       # sticky: robot touched the distractor at any time
        self.reset_accumulators()
        self.trace = (self.output / 'physical_facts.jsonl').open('x')
        self.frames = (self.output / 'control_facts.jsonl').open('x')
        (self.output / 'detection-profile.json').write_text(json.dumps(serializable(dict(
            reader_version=READER_VERSION, key=key, kind=self.spec.kind, target=TARGET, support_root=self.support_root, destination=self.destination, drawer=DRAWER if self.spec.kind == 'drawer' else None,
            distractors=list(self.distractors), ground=[FLOOR], pads=PADS, hands=HANDS, impulse_squared_min=IMPULSE_SQUARED_MIN, grasp=asdict(config),
            target_hull_vertices=len(self.target_hull), destination_outline=self.destination_outline, initial_window_s=INITIAL_WINDOW_S, physics_modified=False,
            instrumentation='session-layer contact reports')), indent=2, allow_nan=False) + '\n')

    def initialize(self, world):
        from evaluation.physics_readers import ReadOnlyRigidBody
        self.world = world
        self.last_time = float(world.current_time)
        self.views = {}
        # finger pad links are recorded too (evidence only: the pad gap around the target can then be measured offline)
        # arm links recorded too (evidence only): a stalled arm can then be traced to the link that collides
        arm_links = self.arm_links
        for path in dict.fromkeys((TARGET, *self.distractors, *HANDS.values(), *[pad for pads in PADS.values() for pad in pads], *arm_links) + ((DRAWER,) if self.spec.kind == 'drawer' else ())):
            view = ReadOnlyRigidBody(prim_path=path)
            view.initialize()
            self.views[path] = view

    def reset_accumulators(self):
        self.window = dict(target_ground=False, distractor_ground={d: False for d in self.distractors}, tv=False)

    def classify(self, other):
        """Surface tag of a non-robot collider path."""
        if under(other, FLOOR): return GROUND
        if self.spec.kind == 'drawer' and under(other, DRAWER): return DRAWER_FLOOR     # any drawer collider that pushes the target upward is its floor (the public box mesh is one shell)
        if self.spec.kind == 'place' and under(other, self.destination): return DESTINATION
        if self.support_root and under(other, self.support_root): return SUPPORT
        if self.support_root == '/World/Assets/ShelvingUnit' and any(under(other, s) for s in SHELF_SUPPORTS): return SUPPORT   # public w120 shelf: tier boards live under the overlay prims
        if any(under(other, d) for d in self.distractors): return ON_OBJECT
        return ON_FURNITURE

    def summarize(self, target_pose):
        target = dict(robot=False, supports=set(), ground=False)
        grips = {h: [[0., 0., 0.], [0., 0., 0.]] for h in PADS}
        touched = {h: [False, False] for h in PADS}
        distract = {d: dict(robot=False, ground=False) for d in self.distractors}
        tv = False
        centre_z = float(target_pose[0][2]) + 0.5 * float(self.target_hull[:, 2].max() - self.target_hull[:, 2].min())
        for paths, samples in self.contacts.pairs():
            index = 0 if under(paths[0], TARGET) else (1 if under(paths[1], TARGET) else None)
            if index is not None:
                other = paths[1 - index]; sign = 1 if index == 0 else -1
                if under(other, ROBOT):
                    target['robot'] = True
                    for hand, pads in PADS.items():
                        for k, pad in enumerate(pads):
                            if under(other, pad):
                                touched[hand][k] = True
                                for n, _, _ in samples:
                                    for a in range(3):
                                        grips[hand][k][a] += sign * n[a]
                else:
                    tag = self.classify(other)
                    if tag == GROUND: target['ground'] = True
                    if any(sign * n[2] > 0 and pt[2] <= centre_z for n, pt, _ in samples):
                        target['supports'].add(tag)
                continue
            for d in self.distractors:
                index = 0 if under(paths[0], d) else (1 if under(paths[1], d) else None)
                if index is None: continue
                other = paths[1 - index]
                if under(other, ROBOT): distract[d]['robot'] = True
                elif under(other, FLOOR): distract[d]['ground'] = True
                break
            if self.spec.kind == 'place' and any(under(p, TV) for p in paths) and any(under(p, ROBOT) for p in paths):
                tv = True
        bilateral = {h: bool(all(touched[h]) and sum(a * b for a, b in zip(*grips[h])) < 0) for h in PADS}
        return target, bilateral, distract, tv

    def after_step(self):
        if self.closed:
            raise RuntimeError('Reader closed')
        now = float(self.world.current_time)
        if abs(now - self.last_time - PHYSICS_DT) > 1e-6:
            raise RuntimeError('Missing physical substep')
        poses = {}
        for path, view in self.views.items():
            p, q = view.get_world_pose()
            if np.asarray(p).shape != (3,) or not np.isfinite(p).all():
                raise ValueError('Invalid physical position')
            rotation(q)
            poses[path] = dict(position_m=[float(v) for v in p], quaternion_wxyz=[float(v) for v in q])
        self.contacts.ingest(self.interface.get_contact_report())
        tp = (poses[TARGET]['position_m'], poses[TARGET]['quaternion_wxyz'])
        target, bilateral, distract, tv = self.summarize(tp)
        grasped, grasp_details = self.detector.observe(now, poses, {TARGET: bilateral})
        confirmed = {h: bool(grasp_details.get(f'{TARGET}:{h}', {}).get('confirmed')) for h in PADS}
        held = self.latch.update(confirmed, bilateral)
        hull = rotation(tp[1]).apply(self.target_hull) + np.asarray(tp[0])
        above_rim, rim_z = True, 0.
        if self.spec.kind == 'drawer':
            dp = poses[DRAWER]
            walls = rotation(dp['quaternion_wxyz']).apply(self.drawer_walls) + np.asarray(dp['position_m'])
            rim_z = float(walls[:, 2].max())
            above_rim = bool(hull[:, 2].min() > rim_z)
        in_destination = bool(self.destination_outline is not None and inside_polygon(hull[:, :2], self.destination_outline))
        for d in self.distractors:
            self.touched[d] |= distract[d]['robot']
            self.window['distractor_ground'][d] |= distract[d]['ground']
        self.window['target_ground'] |= target['ground']
        self.window['tv'] |= tv
        self.step += 1
        self.latest = dict(time_s=now, reader_step=self.step, poses=poses,
                           target=dict(robot=target['robot'], supports=sorted(target['supports']), ground=target['ground'], bilateral=bilateral,
                                       grasped=bool(any(held.values())), grasp_confirmed=bool(grasped.get(TARGET, False)), held=held, hull_min_z=float(hull[:, 2].min()), above_rim=above_rim, in_destination=in_destination),
                           drawer=dict(rim_z=rim_z, pose=poses.get(DRAWER)) if self.spec.kind == 'drawer' else None,
                           distractors={d: dict(robot=distract[d]['robot'], ground=distract[d]['ground'], touched=self.touched[d]) for d in self.distractors},
                           tv_contact=tv, grasp_details=grasp_details)
        self.history.append(self.latest)
        self.trace.write(json.dumps(serializable(self.latest), allow_nan=False) + '\n')
        self.last_time = now

    def frame(self):
        if self.start_step is None:
            raise RuntimeError('Initial state not accepted')
        delta = self.step - self.start_step
        if delta % SUBSTEPS:
            raise RuntimeError('Control frame is not aligned with the physics substeps')
        row, window = self.latest, self.window
        t, pose = row['target'], row['poses'][TARGET]
        target = TargetFacts(position=tuple(pose['position_m']), quaternion=tuple(pose['quaternion_wxyz']),
                             pads_bilateral=bool(any(t['bilateral'].values()) and t['robot']), grasped=bool(t['grasped'] and t['robot']), robot_contact=bool(t['robot']),
                             supports=frozenset(t['supports']), ground_contact=bool(window['target_ground'] or t['ground']), above_rim=bool(t['above_rim']),
                             in_destination=bool(t['in_destination']), hull_min_z=float(t['hull_min_z']))
        distractors = {d.rsplit('/', 1)[-1]: DistractorFacts(robot_touched=bool(self.touched[d]), ground_contact=bool(window['distractor_ground'][d] or row['distractors'][d]['ground']))
                       for d in self.distractors}
        index = delta // SUBSTEPS
        facts = Facts(step=index, time_s=index * CONTROL_DT, target=target, distractors=distractors, tv_contact=bool(window['tv'] or row['tv_contact']),
                      drawer_rim_z=float(row['drawer']['rim_z']) if row['drawer'] else 0.)
        self.reset_accumulators()
        self.frames.write(json.dumps(serializable(dict(facts.as_dict(), simulation_time_s=row['time_s'], reader_step=self.step)), allow_nan=False) + '\n')
        self.frames.flush()
        self.trace.flush()
        return facts

    def begin_episode(self):
        if self.start_step is not None or self.latest is None:
            raise RuntimeError('Invalid initial acceptance')
        self.start_step = self.step
        self.initial_position = np.asarray(self.latest['poses'][TARGET]['position_m'], dtype=float)
        self.touched = {d: False for d in self.distractors}       # the settle is not part of the episode
        self.reset_accumulators()

    def close(self):
        if not self.closed:
            self.trace.close()
            self.frames.close()
            self.closed = True
