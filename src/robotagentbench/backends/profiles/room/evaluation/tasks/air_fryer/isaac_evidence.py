"""Live physical readout of the air fryer family (2.1 / 2.2 / 2.3 / 2.5): poses, contacts, drawer opening -> Facts.

Authored into the session layer, before physics starts: contact-report instrumentation only. Never authored: poses,
drives, materials, collision or solver settings. The one runtime write is the data producer's rail mechanism of 2.3
(DEFINITION_TABLE 2.3 实现绑定, vendor_producer/rail_release145.py): once the basket stroke reaches 0.145 m the drawer
joint is disabled in the session layer, exactly as the producer's worker did; no pose or velocity is touched. The reader
never scores; process_rules never reads the simulator.

Conventions (DEFINITION_TABLE 实现绑定; values marked "supervisor" are implementation choices listed in the README):
  contact sample    a contact point with non-zero impulse (public readers: impulse^2 > 1e-18)
  pushes upward     contact normal on the object has z > 0.5 (public readers; supervisor), counter support of the
                    basket: z > 0.7 (DEFINITION_TABLE 2.3)
  grasped           both finger pads of one gripper touch the object with opposing normals and the object has kept its
                    pose relative to that hand for 0.1 s within 3 mm / 5 deg (public GraspDetector defaults)
  opening           displacement of the basket along the drawer joint axis in the body frame (joint anchors), closed = 0
  inside basket     every vertex of the cake's collision hull lies inside the inner basket wall contour, between the
                    basket floor and the wall top (same test as the v1 reader)
  outside basket    the horizontal convex hull of the cake's collision shape does not intersect the horizontal convex
                    hull of the whole basket (all basket colliders, handle included)
  ground            /World/Background/Shell/Floor (the only floor collider of the house baseline)
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
from .specs import ROBOT, BODY, TRAY, JOINT, BASKET_FLOOR, TARGET, OTHER, PLATE, PLATE_SUPPORT, COUNTER, FLOOR, STATION, spec as key_spec
from .facts import (Facts, CakeFacts, TrayFacts, BASKET_FLOOR as ON_FLOOR, BASKET_WALL as ON_WALL, PLATE as ON_PLATE, COUNTER as ON_COUNTER,
                    GROUND as ON_GROUND, FRYER_BODY as ON_BODY, OTHER as ON_OTHER)

READER_VERSION = 'arena.air_fryer.reader.v2'
PADS = {h: tuple(ROBOT + f'/gripper_{h}_{side}_link4' for side in ('inner', 'outer')) for h in ('l', 'r')}
HANDS = {h: ROBOT + f'/gripper_{h}_base_link' for h in PADS}
IMPULSE_SQUARED_MIN = 1e-18
UP_NORMAL_Z = 0.5
COUNTER_SUPPORT_NORMAL_Z = 0.7
RAIL_RELEASE_STROKE_M = 0.145
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
    """Sample points of one collider in the frame of its rigid body."""
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


def body_colliders(stage, root):
    from pxr import Usd, UsdGeom, UsdPhysics
    prim = stage.GetPrimAtPath(root)
    cache = UsdGeom.XformCache()
    world = cache.GetLocalToWorldTransform(prim)
    if max(abs(world.GetRow3(i).GetLength() - 1) for i in range(3)) > 1e-6:
        raise ValueError('Scaled rigid-body frame: ' + root)
    inverse = world.GetInverse()
    out = {}
    for p in Usd.PrimRange(prim):
        if p.HasAPI(UsdPhysics.CollisionAPI) and p.GetAttribute('physics:collisionEnabled').Get() is not False:
            out[str(p.GetPath())[len(root):]] = collider_points(p, cache.GetLocalToWorldTransform(p) * inverse)
    if not out:
        raise ValueError('No collision geometry: ' + root)
    return out


def hull2d(points):
    from scipy.spatial import ConvexHull
    pts = np.asarray(points, dtype=float)[:, :2]
    return pts[ConvexHull(pts).vertices]


def polygons_separate(a, b):
    """Separating-axis test of two convex polygons (counter-clockwise vertex arrays): True when they do not overlap."""
    for poly in (a, b):
        n = len(poly)
        for i in range(n):
            edge = poly[(i + 1) % n] - poly[i]
            axis = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(axis)
            if norm < 1e-12:
                continue
            axis /= norm
            pa, pb = a @ axis, b @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return True
    return False


class Geometry:
    """Basket contour, drawer axis, whole-basket outline and cake hulls from the composed scene (before physics)."""
    def __init__(self, stage):
        from scipy.spatial import ConvexHull
        from pxr import Gf, UsdPhysics
        tray = body_colliders(stage, TRAY)
        floor = tray.get('/Collision/BasketFloor')
        if floor is None:
            raise ValueError('Missing active basket floor')
        centre = (floor.min(axis=0) + floor.max(axis=0))[:2] / 2
        inner, tops = [], []
        walls = [v for name, v in tray.items() if name.startswith('/Collision/MatchedBasket/')]
        if not walls:
            raise ValueError('Missing approved inner basket contour')
        for v in walls:
            xy = np.unique(v[:, :2], axis=0)
            if len(xy) != 4:
                raise ValueError('Unexpected basket wall topology')
            inner.extend(xy[np.argsort(((xy - centre) ** 2).sum(axis=1))[:2]])
            tops.append(v[:, 2].max())
        if len(inner) < 6:
            raise ValueError('Incomplete basket contour')
        self.planes = ConvexHull(np.asarray(inner)).equations
        self.bottom, self.top = float(floor[:, 2].min()), float(min(tops))
        every = np.vstack(list(tray.values()))
        self.tray_hull = every[ConvexHull(every).vertices]          # 3D hull of the whole basket (handle included), tray frame
        self.cake_hull = {}
        for root in (TARGET, OTHER):
            v = np.vstack(list(body_colliders(stage, root).values()))
            if not np.isfinite(v).all():
                raise ValueError('Missing cake geometry')
            self.cake_hull[root] = v[ConvexHull(v).vertices]
        joint = stage.GetPrimAtPath(JOINT)
        if (not joint or not joint.IsA(UsdPhysics.Joint) or joint.GetAttribute('limit:transX:physics:low').Get() != 0
                or [str(p) for p in joint.GetRelationship('physics:body0').GetTargets()] != [BODY]
                or [str(p) for p in joint.GetRelationship('physics:body1').GetTargets()] != [TRAY]):
            raise ValueError('Unexpected physical drawer joint')
        self.joint_limit_high = float(joint.GetAttribute('limit:transX:physics:high').Get())
        self.joint_enabled = joint.GetAttribute('physics:jointEnabled').Get() is not False
        self.p0 = np.asarray(joint.GetAttribute('physics:localPos0').Get(), dtype=float)
        self.p1 = np.asarray(joint.GetAttribute('physics:localPos1').Get(), dtype=float)
        q = joint.GetAttribute('physics:localRot0').Get()
        self.axis = np.asarray(q.Transform(Gf.Vec3f(1, 0, 0)), dtype=float)

    def opening(self, body, tray):
        bp, bq = body
        tp, tq = tray
        br, tr = rotation(bq), rotation(tq)
        return float(np.dot(np.asarray(tp) + tr.apply(self.p1) - np.asarray(bp) - br.apply(self.p0), br.apply(self.axis)))

    def inside(self, cake, tray_pose, cake_pose):
        tp, tq = tray_pose
        cp, cq = cake_pose
        local = rotation(tq).inv().apply(rotation(cq).apply(self.cake_hull[cake]) + np.asarray(cp) - np.asarray(tp))
        return bool(np.all(local[:, :2] @ self.planes[:, :2].T + self.planes[:, 2] <= 1e-7)
                    and local[:, 2].min() >= self.bottom - 1e-7 and local[:, 2].max() <= self.top + 1e-7)

    def outside(self, cake, tray_pose, cake_pose):
        tp, tq = tray_pose
        cp, cq = cake_pose
        tray_xy = hull2d(rotation(tq).apply(self.tray_hull) + np.asarray(tp))
        cake_xy = hull2d(rotation(cq).apply(self.cake_hull[cake]) + np.asarray(cp))
        return polygons_separate(tray_xy, cake_xy)

    def description(self):
        return dict(source='collision geometry of the composed scene', basket_floor_z_m=self.bottom, basket_wall_top_z_m=self.top,
                    inner_contour_planes=len(self.planes), basket_hull_vertices=len(self.tray_hull),
                    cake_hull_vertices={k.rsplit('/', 1)[-1]: len(v) for k, v in self.cake_hull.items()},
                    joint_limit_high_m=None if math.isinf(self.joint_limit_high) else self.joint_limit_high, joint_enabled_at_start=self.joint_enabled,
                    drawer_axis_body_frame=self.axis.tolist())


class Contacts:
    """Reduce PhysX pairs to object identities; keep only measured (non-zero impulse) samples."""
    def __init__(self, resolve, sleeping, event_types, roots, ignored=()):
        self.sleeping = sleeping
        self.found, self.persist, self.lost = event_types
        self.roots, self.ignored = tuple(roots), tuple(ignored)
        self.identity = ContactIdentityCache(resolve, self.relevant)
        self.active = {}

    def relevant(self, paths):
        return (any(under(p, r) for p in paths for r in self.roots)
                and not any(under(p, r) for p in paths for r in self.ignored))

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
                if not all(math.isfinite(x) for x in impulse + normal):
                    raise RuntimeError('Non-finite physical contact')
                squared = sum(x * x for x in impulse)
                if squared > IMPULSE_SQUARED_MIN:
                    samples.append((normal, math.sqrt(squared)))
            self.active[key] = (paths, int(h.stage_id), (int(h.actor0), int(h.actor1)), tuple(samples))
        for key, (_, stage_id, actors, _) in self.active.items():
            if key not in seen and not all(self.sleeping(stage_id, a) is True for a in actors):
                raise RuntimeError('Missing awake contact pair: ' + str(key))

    def pairs(self):
        for paths, _, _, samples in self.active.values():
            if samples:
                yield paths, samples


class RailRelease:
    """The data producer's basket rail of 2.3 (vendor_producer/rail_release145.py): the drawer joint is disabled once the
    stroke reaches 0.145 m; one way until reset; no pose, velocity, collision, mass or friction is written."""
    def __init__(self, stage, threshold=RAIL_RELEASE_STROKE_M):
        from pxr import Usd
        self.stage, self.threshold = stage, threshold
        joint = stage.GetPrimAtPath(JOINT)
        self.attr = joint.GetAttribute('physics:jointEnabled')
        if not self.attr or self.attr.Get() is False:
            raise ValueError('Rail release needs an enabled drawer joint at the start')
        fryer = stage.GetPrimAtPath(TRAY).GetParent().GetParent()
        declared = fryer.GetAttribute('arena:railReleaseStrokeM').Get()
        if declared is None or abs(float(declared) - threshold) > 1e-8:
            raise ValueError('Fryer asset does not declare the 0.145 m rail release')
        self.released = False
        self.events = []

    def step(self, opening, physics_step, time_s):
        if self.released or opening < self.threshold:
            return False
        from pxr import Usd
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            self.attr.Set(False)
        self.released = True
        self.events.append(dict(physics_step=int(physics_step), time_s=float(time_s), measured_stroke_m=float(opening), threshold_m=self.threshold,
                                mechanism='stroke-triggered joint release (producer rail_release145); pose and velocity untouched'))
        return True


class AirFryerReader:
    def __init__(self, stage, output, episode_id, key, config=DetectionConfig()):
        from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, PhysicsSchemaTools
        from omni.physx import get_physx_simulation_interface
        from omni.physx.bindings._physx import ContactEventType
        self.stage, self.spec, self.key, self.episode_id, self.config = stage, key_spec(key), key, episode_id, config
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        if UsdGeom.GetStageUpAxis(stage) != 'Z' or UsdGeom.GetStageMetersPerUnit(stage) != 1:
            raise ValueError('Expected metres/Z-up')

        def colliding(path):
            p = stage.GetPrimAtPath(path)
            return bool(p and p.IsActive() and p.HasAPI(UsdPhysics.CollisionAPI) and p.GetAttribute('physics:collisionEnabled').Get() is not False)
        if not colliding(FLOOR) or not colliding(COUNTER) or not colliding(BASKET_FLOOR) or not colliding(PLATE_SUPPORT):
            raise ValueError('Missing ground, counter, basket floor or plate support collider: evidence would be invalid')
        self.cakes = (TARGET, OTHER)
        self.geometry = Geometry(stage)
        if self.spec.basket_free and self.geometry.joint_enabled:
            raise ValueError('2.5 expects the drawer joint disabled (free basket)')
        if not self.spec.basket_free and not self.geometry.joint_enabled:
            raise ValueError('The drawer joint must be enabled at the start')
        self.rail = RailRelease(stage) if self.spec.rail_release else None
        self.interface = get_physx_simulation_interface()
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            bodies = [stage.GetPrimAtPath(p) for p in (TARGET, OTHER, TRAY, BODY)]
            bodies += [p for p in Usd.PrimRange(stage.GetPrimAtPath(ROBOT)) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
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
        import os
        self.diag_robot = os.environ.get('AIRFRYER_DIAG_ROBOT_CONTACTS') == '1'
        self.contacts = Contacts(PhysicsSchemaTools.intToSdfPath, sleeping,
                                 (ContactEventType.CONTACT_FOUND, ContactEventType.CONTACT_PERSIST, ContactEventType.CONTACT_LOST),
                                 roots=(TARGET, OTHER, TRAY) + ((ROBOT,) if self.diag_robot else ()))
        self.diag = (self.output / 'robot_contacts.jsonl').open('x') if self.diag_robot else None
        self.detector = GraspDetector((TARGET,), config)
        self.hashes = {l.realPath: hashlib.sha256(Path(l.realPath).read_bytes()).hexdigest()
                       for l in stage.GetUsedLayers() if l.realPath and Path(l.realPath).is_file()}
        self.step = -1
        self.latest = None
        self.start_step = None
        self.initial_position = None
        self.closed = False
        self.history = deque(maxlen=int(math.ceil(INITIAL_WINDOW_S / PHYSICS_DT - 1e-9)) + 2)
        self.reset_accumulators()
        self.trace = (self.output / 'physical_facts.jsonl').open('x')
        self.frames = (self.output / 'control_facts.jsonl').open('x')
        (self.output / 'detection-profile.json').write_text(json.dumps(dict(
            reader_version=READER_VERSION, key=key, target=TARGET, other=OTHER, tray=TRAY, body=BODY, counter=COUNTER, ground=[FLOOR],
            plate=[PLATE, PLATE_SUPPORT], pads=PADS, hands=HANDS, geometry=self.geometry.description(),
            impulse_squared_min=IMPULSE_SQUARED_MIN, up_normal_z=UP_NORMAL_Z, counter_support_normal_z=COUNTER_SUPPORT_NORMAL_Z,
            grasp=asdict(config), rail_release=(dict(threshold_m=RAIL_RELEASE_STROKE_M, code='vendor_producer/rail_release145.py') if self.rail else None),
            initial_window_s=INITIAL_WINDOW_S, physics_modified=False,
            instrumentation='session-layer contact reports' + ('; producer rail release (joint disabled at 0.145 m stroke)' if self.rail else '')),
            indent=2, allow_nan=False) + '\n')

    # ---- runtime ----
    def initialize(self, world):
        from evaluation.physics_readers import ReadOnlyRigidBody
        self.world = world
        self.last_time = float(world.current_time)
        self.views = {}
        for path in (TARGET, OTHER, TRAY, BODY, *HANDS.values()):
            view = ReadOnlyRigidBody(prim_path=path)
            view.initialize()
            self.views[path] = view

    def reset_accumulators(self):
        self.window = dict(ground={o: False for o in (TARGET, OTHER, TRAY)}, basket={o: False for o in self.cakes}, wall={o: False for o in self.cakes}, tray_body=False)

    def summarize(self):
        result = {o: dict(robot=False, supports=set(), ground=False, basket=False, wall=False) for o in self.cakes}
        tray = dict(robot=False, body=False, counter=False, ground=False)
        grips = {o: {h: [[0., 0., 0.], [0., 0., 0.]] for h in PADS} for o in (*self.cakes, TRAY)}
        touched = {o: {h: [False, False] for h in PADS} for o in (*self.cakes, TRAY)}
        for paths, samples in self.contacts.pairs():
            for obj in self.cakes:
                index = 0 if under(paths[0], obj) else (1 if under(paths[1], obj) else None)
                if index is None:
                    continue
                other = paths[1 - index]
                sign = 1 if index == 0 else -1
                up = any(sign * n[2] > UP_NORMAL_Z for n, _ in samples)
                flags = result[obj]
                if under(other, ROBOT):
                    flags['robot'] = True
                    for hand, pads in PADS.items():
                        for k, pad in enumerate(pads):
                            if under(other, pad):
                                touched[obj][hand][k] = True
                                for n, _ in samples:
                                    for a in range(3):
                                        grips[obj][hand][k][a] += sign * n[a]
                elif under(other, TRAY):
                    flags['basket'] = True
                    flags['wall'] |= not under(other, BASKET_FLOOR)
                    if up:
                        flags['supports'].add(ON_FLOOR if under(other, BASKET_FLOOR) else ON_WALL)
                elif under(other, BODY):
                    if up:
                        flags['supports'].add(ON_BODY)
                elif under(other, FLOOR):
                    flags['ground'] = True
                    if up:
                        flags['supports'].add(ON_GROUND)
                elif under(other, PLATE) or under(other, PLATE_SUPPORT):
                    if up:
                        flags['supports'].add(ON_PLATE)
                elif up:
                    flags['supports'].add(ON_COUNTER if under(other, COUNTER) else ON_OTHER)
            index = 0 if under(paths[0], TRAY) else (1 if under(paths[1], TRAY) else None)
            if index is not None and not any(under(paths[1 - index], c) for c in self.cakes):
                other = paths[1 - index]
                sign = 1 if index == 0 else -1
                if under(other, ROBOT):
                    tray['robot'] = True
                    for hand, pads in PADS.items():
                        for k, pad in enumerate(pads):
                            if under(other, pad):
                                touched[TRAY][hand][k] = True
                                for n, _ in samples:
                                    for a in range(3):
                                        grips[TRAY][hand][k][a] += sign * n[a]
                elif under(other, BODY):
                    tray['body'] = True
                elif under(other, COUNTER):
                    tray['counter'] |= any(sign * n[2] > COUNTER_SUPPORT_NORMAL_Z for n, _ in samples)
                elif under(other, FLOOR):
                    tray['ground'] = True
        bilateral = {o: {h: bool(all(touched[o][h]) and sum(a * b for a, b in zip(*grips[o][h])) < 0) for h in PADS} for o in touched}
        return result, tray, bilateral

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
        if self.diag is not None:
            rows = [dict(a=paths[0], b=paths[1], impulse=max(s for _, s in samples), n=len(samples))
                    for paths, samples in self.contacts.pairs() if any(under(p, ROBOT) for p in paths)]
            if rows:
                self.diag.write(json.dumps(dict(step=self.step + 1, time_s=now, pairs=rows)) + '\n')
        contact, tray, bilateral = self.summarize()
        grasped, grasp_details = self.detector.observe(now, poses, {TARGET: bilateral[TARGET]})
        body_pose = (poses[BODY]['position_m'], poses[BODY]['quaternion_wxyz'])
        tray_pose = (poses[TRAY]['position_m'], poses[TRAY]['quaternion_wxyz'])
        opening = self.geometry.opening(body_pose, tray_pose)
        released_now = self.rail.step(opening, self.step + 1, now) if self.rail is not None else False
        for obj in self.cakes:
            self.window['ground'][obj] |= contact[obj]['ground']
            self.window['basket'][obj] |= contact[obj]['basket']
            self.window['wall'][obj] |= contact[obj]['wall']
        self.window['ground'][TRAY] |= tray['ground']
        self.window['tray_body'] |= tray['body']
        self.step += 1
        self.latest = dict(
            time_s=now, reader_step=self.step, poses=poses,
            cakes={o: dict(robot=contact[o]['robot'], supports=sorted(contact[o]['supports']), ground=contact[o]['ground'], basket=contact[o]['basket'], wall=contact[o]['wall'],
                           bilateral=bilateral[o], grasped=bool(grasped.get(o, False)), tilt_deg=self.tilt(poses[o]['quaternion_wxyz']),
                           inside=self.geometry.inside(o, tray_pose, (poses[o]['position_m'], poses[o]['quaternion_wxyz'])),
                           outside=self.geometry.outside(o, tray_pose, (poses[o]['position_m'], poses[o]['quaternion_wxyz']))) for o in self.cakes},
            tray=dict(opening_m=opening, robot=tray['robot'], body=tray['body'], counter=tray['counter'], ground=tray['ground'], bilateral=bilateral[TRAY],
                      tilt_deg=self.tilt(poses[TRAY]['quaternion_wxyz']), joint_released=bool(self.rail.released) if self.rail else (not self.geometry.joint_enabled),
                      released_now=released_now),
            grasp_details=grasp_details)
        self.history.append(self.latest)
        self.trace.write(json.dumps(serializable(self.latest), allow_nan=False) + '\n')
        self.last_time = now

    @staticmethod
    def tilt(quaternion):
        up = rotation(quaternion).apply([0., 0., 1.])
        return float(np.degrees(np.arccos(np.clip(up[2], -1., 1.))))

    def frame(self):
        """Facts of the control frame that ends at the latest substep."""
        if self.start_step is None:
            raise RuntimeError('Initial state not accepted')
        delta = self.step - self.start_step
        if delta % SUBSTEPS:
            raise RuntimeError('Control frame is not aligned with the physics substeps')
        row, window = self.latest, self.window
        cakes = {}
        for obj in self.cakes:
            c, pose = row['cakes'][obj], row['poses'][obj]
            cakes[obj] = CakeFacts(
                position=tuple(pose['position_m']), quaternion=tuple(pose['quaternion_wxyz']), tilt_deg=c['tilt_deg'],
                pads_bilateral=bool(any(c['bilateral'].values()) and c['robot']), grasped=bool(c['grasped'] and c['robot']),
                robot_contact=bool(c['robot']), supports=frozenset(c['supports']), ground_contact=bool(window['ground'][obj] or c['ground']),
                basket_contact=bool(window['basket'][obj] or c['basket']), inside_basket=bool(c['inside']), outside_basket=bool(c['outside']),
                displacement_m=float(np.linalg.norm(np.asarray(pose['position_m']) - self.initial_position[obj])))
        t, pose = row['tray'], row['poses'][TRAY]
        tray = TrayFacts(position=tuple(pose['position_m']), quaternion=tuple(pose['quaternion_wxyz']), tilt_deg=t['tilt_deg'], opening_m=float(t['opening_m']),
                         body_contact=bool(t['body']), counter_supported=bool(t['counter']), robot_contact=bool(t['robot']),
                         pads_bilateral=bool(any(t['bilateral'].values()) and t['robot']), ground_contact=bool(window['ground'][TRAY] or t['ground']),
                         joint_released=bool(t['joint_released']),
                         displacement_m=float(np.linalg.norm(np.asarray(pose['position_m']) - self.initial_position[TRAY])))
        index = delta // SUBSTEPS
        facts = Facts(step=index, time_s=index * CONTROL_DT, target=cakes[TARGET], other=cakes[OTHER], tray=tray)
        self.reset_accumulators()
        self.frames.write(json.dumps(serializable(dict(facts.as_dict(), simulation_time_s=row['time_s'], reader_step=self.step)), allow_nan=False) + '\n')
        self.frames.flush()
        self.trace.flush()
        return facts

    def begin_episode(self):
        if self.start_step is not None or self.latest is None:
            raise RuntimeError('Invalid initial acceptance')
        self.start_step = self.step
        self.initial_position = {o: np.asarray(self.latest['poses'][o]['position_m'], dtype=float) for o in (*self.cakes, TRAY)}
        self.reset_accumulators()

    def close(self):
        if not self.closed:
            self.trace.close()
            self.frames.close()
            if self.diag is not None:
                self.diag.close()
            if self.rail is not None:
                (self.output / 'rail_release.json').write_text(json.dumps(dict(released=self.rail.released, events=self.rail.events), indent=2) + '\n')
            self.closed = True
