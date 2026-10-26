"""Coffee CPU animation revision (2026-10-04). Physical rigid-body robot/object evidence;
visual-only coffee surfaces and stream. No PBD liquid is instantiated.
Historical particle-based sources and videos are retained separately."""
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
from .specs import (ROBOT, MACHINE, MACHINE_BODY, BUTTON, CUP, BAG, BEANS, LIQUID, CUP_SUPPORT, TRAY_COLLIDER, COUNTER,
                    FLOOR, spec as api_spec)
from .facts import (Facts, ObjectFacts, CupPlacement, LiquidFacts, ButtonFacts, COUNTER as ON_COUNTER, TRAY as ON_TRAY,
                    GROUND as ON_GROUND, MACHINE_BODY as ON_MACHINE, OTHER as ON_OTHER)
from . import vendor_producer

READER_VERSION = 'arena.coffee_machine.reader.v1'
PADS = {h: tuple(ROBOT + f'/gripper_{h}_{side}_link4' for side in ('inner', 'outer')) for h in ('l', 'r')}
HANDS = {h: ROBOT + f'/gripper_{h}_base_link' for h in PADS}
IMPULSE_SQUARED_MIN = 1e-18
UP_NORMAL_Z = 0.5
BUTTON_PRESS_IMPULSE = 1e-5
BAY_EXIT_LOCAL_Z = 0.1054
CAVITY_RADIAL_MARGIN_M, CAVITY_FLOOR_MARGIN_M, CAVITY_RIM_MARGIN_M = 0.0003, 0.0005, 0.0009
SPILL_SPEED_M_S, SPILL_SECONDS = 0.05, 0.25
BAG_MOUTH_MARGIN_M, BAG_FLOOR_MARGIN_M = 0.005, 0.001          # producer run_pour.py:237-240
INITIAL_WINDOW_S, INITIAL_SPAN_M = 0.25, 0.002                 # public DetectionConfig defaults


def under(path, root):
    return path == root or path.startswith(root.rstrip('/') + '/')


def serializable(v):
    if isinstance(v, dict): return {k: serializable(x) for k, x in v.items()}
    if isinstance(v, (tuple, list, set, frozenset)): return [serializable(x) for x in v]
    if isinstance(v, np.generic): return v.item()
    if isinstance(v, np.ndarray): return v.tolist()
    return v


def collider_points(prim, to_root):
    """Sample points of one collider in the frame of its rigid body; spheres are returned as (centre, radius)."""
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
    elif prim.IsA(UsdGeom.Sphere):
        centre = np.asarray(to_root.Transform(Gf.Vec3d(0, 0, 0)), dtype=float)
        radius = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get()) * to_root.TransformDir(Gf.Vec3d(1, 0, 0)).GetLength()
        return None, (centre, radius)
    else:
        raise ValueError('Unsupported collider type: ' + str(prim.GetPath()))
    return np.asarray([list(to_root.Transform(Gf.Vec3d(*[float(x) for x in p]))) for p in pts], dtype=float), None


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


class CupGeometry:
    """Cavity and outline of the bound cup style, in the cup frame (Y up), from its collision geometry."""
    def __init__(self, stage, path=CUP):
        from scipy.spatial import ConvexHull
        colliders = body_colliders(stage, path)
        walls = np.vstack([pts for name, (pts, _) in colliders.items() if name.startswith('/Collision/Wall')])
        bottom = np.vstack([pts for name, (pts, _) in colliders.items() if name.startswith('/Collision/Bottom')])
        self.spheres = [ball for name, (_, ball) in colliders.items() if ball is not None]
        if len(walls) < 16 or len(bottom) < 8 or not all(n.startswith('/Collision/Handle') for n, (p, b) in colliders.items() if b is not None):
            raise ValueError('Unexpected cup collision layout')
        radial = np.hypot(walls[:, 0], walls[:, 2])
        levels = np.unique(np.round(walls[:, 1], 5))
        self.heights = levels
        self.inner = np.asarray([radial[np.abs(walls[:, 1] - y) < 1e-5].min() for y in levels])
        self.floor_y, self.rim_y = float(bottom[:, 1].max()), float(walls[:, 1].max())
        if abs(levels[0] - self.floor_y) > 1e-4 or not np.all(np.diff(self.inner) > 0):
            raise ValueError('Unexpected cup cavity profile')
        self.opening_radius = float(self.inner[-1])
        solid = np.vstack([pts for pts, _ in colliders.values() if pts is not None])
        self.outline = solid[ConvexHull(solid).vertices]

    def inside(self, local):
        """Particle centres (n, 3) in the cup frame -> inside the cavity."""
        h = local[:, 1]
        r = np.hypot(local[:, 0], local[:, 2])
        return ((r < np.interp(h, self.heights, self.inner) + CAVITY_RADIAL_MARGIN_M)
                & (h > self.floor_y - CAVITY_FLOOR_MARGIN_M) & (h < self.rim_y + CAVITY_RIM_MARGIN_M))

    def description(self):
        return dict(source='collision geometry of the bound public cup', heights_m=self.heights.tolist(), inner_radius_m=self.inner.tolist(),
                    floor_y_m=self.floor_y, rim_y_m=self.rim_y, opening_radius_m=self.opening_radius, handle_spheres=len(self.spheres),
                    outline_points=len(self.outline),
                    margins_m=dict(radial=CAVITY_RADIAL_MARGIN_M, floor=CAVITY_FLOOR_MARGIN_M, rim=CAVITY_RIM_MARGIN_M))


class BagGeometry:
    """Inside of the bag for the initial bean count: horizontal outline of its colliders, floor to mouth + 5 mm."""
    def __init__(self, stage, path=BAG):
        from scipy.spatial import ConvexHull
        solid = np.vstack([pts for pts, _ in body_colliders(stage, path).values() if pts is not None])
        self.planes = ConvexHull(solid[:, [0, 2]]).equations
        self.height = float(solid[:, 1].max())

    def inside(self, local):
        return (np.all(local[:, [0, 2]] @ self.planes[:, :2].T + self.planes[:, 2] <= 1e-9, axis=1)
                & (local[:, 1] > BAG_FLOOR_MARGIN_M) & (local[:, 1] < self.height + BAG_MOUTH_MARGIN_M))


class LiquidTracker:
    """in cup / airborne / spilled per particle index; one observation per control frame."""
    def __init__(self, cavity, dt=CONTROL_DT, speed=SPILL_SPEED_M_S, seconds=SPILL_SECONDS):
        self.cavity, self.dt, self.speed, self.seconds = cavity, dt, speed, seconds
        self.previous = np.zeros((0, 3))
        self.still = np.zeros(0)
        self.spilled = np.zeros(0, dtype=bool)

    def classify(self, points, cup_position, cup_quaternion):
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        if not np.isfinite(points).all():
            raise ValueError('Non-finite liquid particle')
        local = rotation(cup_quaternion).inv().apply(points - np.asarray(cup_position)) if len(points) else points
        return points, (self.cavity.inside(local) if len(points) else np.zeros(0, dtype=bool))

    def observe(self, points, cup_position, cup_quaternion):
        points, inside = self.classify(points, cup_position, cup_quaternion)
        known = len(self.previous)
        if len(points) < known:
            raise RuntimeError('Liquid particles disappeared; particle indices are no longer comparable')
        grown = len(points) - known
        self.still = np.concatenate([self.still, np.zeros(grown)])
        self.spilled = np.concatenate([self.spilled, np.zeros(grown, dtype=bool)])
        slow = np.zeros(len(points), dtype=bool)
        slow[:known] = np.linalg.norm(points[:known] - self.previous, axis=1) <= self.speed * self.dt
        resting = slow & ~inside
        self.still = np.where(resting, self.still + self.dt, 0.)
        self.spilled |= self.still >= self.seconds - 1e-9
        self.previous = points.copy()
        return dict(particles=len(points), in_cup=int(inside.sum()), airborne=int((~inside & ~self.spilled).sum()),
                    spilled=int(self.spilled.sum()))


class HeadCamera:
    """The data producer's layout gate (button_task/build_random.py:19-27) on LIVE poses.

    All 8 corners of the cup's world bounding box must project inside the head camera frame (|x|, |y| <= 0.96 of the
    half frame, between the clipping planes). The producer evaluated it on its authored stage; here the camera pose
    is the live pose of the head link, so it is the camera the model actually sees through.
    """
    SAFE_FRAME = 0.96

    def __init__(self, stage, camera=ROBOT + '/head_link2/Head_Camera', cup=CUP):
        from pxr import Usd, UsdGeom, UsdPhysics, Gf
        prim = stage.GetPrimAtPath(camera)
        if not prim or not prim.IsA(UsdGeom.Camera):
            raise ValueError('Missing head camera')
        link = prim.GetParent()
        while link and not link.HasAPI(UsdPhysics.RigidBodyAPI):
            link = link.GetParent()
        if not link:
            raise ValueError('Head camera is not mounted on a robot link')
        self.link = str(link.GetPath())
        cache = UsdGeom.XformCache()
        self.mount = np.asarray(cache.GetLocalToWorldTransform(prim) * cache.GetLocalToWorldTransform(link).GetInverse(), dtype=float)
        cam = UsdGeom.Camera(prim)
        self.focal, self.width, self.height = cam.GetFocalLengthAttr().Get(), cam.GetHorizontalApertureAttr().Get(), cam.GetVerticalApertureAttr().Get()
        self.near, self.far = cam.GetClippingRangeAttr().Get()
        # the producer's box: UsdGeom.BBoxCache world bound of the cup prim = its local-space box moved with the cup
        import itertools
        box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]).ComputeUntransformedBound(stage.GetPrimAtPath(cup))
        low, high, matrix = box.GetRange().GetMin(), box.GetRange().GetMax(), box.GetMatrix()
        self.points = np.asarray([list(matrix.Transform(Gf.Vec3d(*corner))) for corner in itertools.product(*[(low[i], high[i]) for i in range(3)])], dtype=float)
        if self.points.shape != (8, 3) or not np.isfinite(self.points).all():
            raise ValueError('Missing cup bound for the camera frame check')

    def check(self, link_position, link_quaternion, cup_position, cup_quaternion):
        import itertools
        world = rotation(cup_quaternion).apply(self.points) + np.asarray(cup_position)
        lo, hi = world.min(axis=0), world.max(axis=0)
        # row-vector convention of Gf: camera_world = mount * link_world
        rows = []
        link_rotation = rotation(link_quaternion)
        for corner in itertools.product(*[(lo[i], hi[i]) for i in range(3)]):
            in_link = link_rotation.inv().apply(np.asarray(corner) - np.asarray(link_position))
            q = (in_link - self.mount[3, :3]) @ self.mount[:3, :3].T
            depth = -q[2]
            rows.append(dict(x=float(2 * self.focal * q[0] / max(depth, 1e-12) / self.width),
                             y=float(2 * self.focal * q[1] / max(depth, 1e-12) / self.height), depth=float(depth)))
        visible = all(self.near < r['depth'] < self.far and abs(r['x']) <= self.SAFE_FRAME and abs(r['y']) <= self.SAFE_FRAME for r in rows)
        return dict(visible=bool(visible), safe_frame=self.SAFE_FRAME, corners=rows)


class Contacts:
    """Reduce PhysX pairs to object identities; keep only measured (non-zero impulse) samples."""
    def __init__(self, resolve, sleeping, event_types, roots, ignored):
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


class Bodies:
    """Read-only poses of many live rigid bodies through one PhysX view (the beans)."""
    def __init__(self, paths):
        from isaacsim.core.simulation_manager import SimulationManager
        simulation = SimulationManager.get_physics_sim_view()
        if simulation is None:
            raise RuntimeError('Live physics must be initialized before binding a reader')
        self.view = simulation.create_rigid_body_view(list(paths))
        if self.view.count != len(paths):
            raise RuntimeError(f'Expected {len(paths)} live rigid bodies, got {self.view.count}')
        reported = [str(p) for p in self.view.prim_paths]
        if sorted(reported) != sorted(paths):
            raise RuntimeError('Rigid-body view bound other prims')
        self.order = np.asarray([reported.index(p) for p in paths])

    def positions(self):
        values = self.view.get_transforms()
        if hasattr(values, 'detach'): values = values.detach().cpu().numpy()
        elif hasattr(values, 'numpy'): values = values.numpy()
        values = np.asarray(values, dtype=float).reshape(-1, 7)[self.order]
        if not np.isfinite(values).all():
            raise RuntimeError('Invalid live bean pose')
        return values[:, :3].copy()


class CoffeeReader:
    def __init__(self, stage, output, episode_id, api, config=DetectionConfig()):
        from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, PhysicsSchemaTools
        from omni.physx import get_physx_simulation_interface
        from omni.physx.bindings._physx import ContactEventType
        self.stage, self.spec, self.api, self.episode_id, self.config = stage, api_spec(api), api, episode_id, config
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        if UsdGeom.GetStageUpAxis(stage) != 'Z' or UsdGeom.GetStageMetersPerUnit(stage) != 1:
            raise ValueError('Expected metres/Z-up')
        def colliding(path):
            p = stage.GetPrimAtPath(path)
            return bool(p and p.IsActive() and p.HasAPI(UsdPhysics.CollisionAPI) and p.GetAttribute('physics:collisionEnabled').Get() is not False)
        if not colliding(FLOOR) or not colliding(COUNTER) or not colliding(CUP_SUPPORT) or not colliding(TRAY_COLLIDER):
            raise ValueError('Missing ground, counter or tray collider: evidence would be invalid')
        self.objects = (CUP, BAG)
        self.machine_matrix = UsdGeom.Xformable(stage.GetPrimAtPath(MACHINE)).ComputeLocalToWorldTransform(0)
        m = np.asarray(self.machine_matrix, dtype=float)
        if np.abs(m[:3, :3] @ m[:3, :3].T - np.eye(3)).max() > 1e-6:
            raise ValueError('Scaled machine frame')
        self.machine_rotation, self.machine_origin = m[:3, :3], m[3, :3]
        self.contract = vendor_producer.load('machine_contract').read_contract(stage, MACHINE)
        self.anchor = self.machine_point(self.contract.cup)
        self.outlet = self.machine_point(self.contract.outlet)
        self.cup_geometry = CupGeometry(stage)
        self.head_camera = HeadCamera(stage)
        self.bag_geometry = BagGeometry(stage)
        self.controller = vendor_producer.load('coffee_logic').CoffeeController(self.contract.hopper) if self.spec.has_beans else None
        self.bean_paths = [f'{BEANS}/Bean_{i:03d}' for i in range(100)] if self.spec.has_beans else []
        from .visual_coffee import VisualCoffee
        from .initial_state import FILL_PARTICLES
        self.animation = (VisualCoffee(stage, LIQUID, self.cup_geometry, self.contract.outlet,
                          prefilled=FILL_PARTICLES if api == '1-4' else 0) if self.spec.has_liquid else None)
        self.fluid = self.animation if self.spec.has_device else None
        self.prefill = self.animation if api == '1-4' else None
        self.liquid_points = self.tracker = None
        self.interface = get_physx_simulation_interface()
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            bodies = [stage.GetPrimAtPath(p) for p in (CUP, BAG, MACHINE_BODY, BUTTON)]
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
                return True     # static or kinematic: never an awake dynamic body
            return bool(self.interface.is_sleeping(stage_id, actor))
        import os
        self.diag_robot = os.environ.get('COFFEE_DIAG_ROBOT_CONTACTS') == '1'     # diagnostic only: log every contact the robot makes
        self.contacts = Contacts(PhysicsSchemaTools.intToSdfPath, sleeping,
                                 (ContactEventType.CONTACT_FOUND, ContactEventType.CONTACT_PERSIST, ContactEventType.CONTACT_LOST),
                                 roots=(CUP, BAG, MACHINE) + ((ROBOT,) if self.diag_robot else ()), ignored=(BEANS, LIQUID))
        self.diag = (self.output / 'robot_contacts.jsonl').open('x') if self.diag_robot else None
        self.detector = GraspDetector(self.objects, config)
        self.hashes = {l.realPath: hashlib.sha256(Path(l.realPath).read_bytes()).hexdigest()
                       for l in stage.GetUsedLayers() if l.realPath and Path(l.realPath).is_file()}
        self.step = -1
        self.latest = None
        self.start_step = None
        self.initial_position = None
        self.initial_in_cup = 0
        self.closed = False
        self.history = deque(maxlen=int(math.ceil(INITIAL_WINDOW_S / PHYSICS_DT - 1e-9)) + 2)
        self.press_by_robot = False
        self.reset_accumulators()
        self.trace = (self.output / 'physical_facts.jsonl').open('x')
        self.frames = (self.output / 'control_facts.jsonl').open('x')
        (self.output / 'detection-profile.json').write_text(json.dumps(dict(
            reader_version=READER_VERSION, api=api, cup=CUP, bag=BAG, machine=MACHINE, counter=COUNTER, ground=[FLOOR],
            tray_colliders=[CUP_SUPPORT, TRAY_COLLIDER], pads=PADS, hands=HANDS,
            machine_world_matrix=m.tolist(), cup_placement_anchor_world=self.anchor.tolist(), outlet_world=self.outlet.tolist(),
            machine_contract=serializable(self.contract.to_dict()), cup_geometry=self.cup_geometry.description(),
            bay_exit_local_z_m=BAY_EXIT_LOCAL_Z, impulse_squared_min=IMPULSE_SQUARED_MIN, up_normal_z=UP_NORMAL_Z,
            button_press_impulse=BUTTON_PRESS_IMPULSE, grasp=asdict(config),
            liquid=dict(mode='visual_cpu_v1', physical_liquid=False, accounting='volume units, not particles'),
            liquid_prefill=(dict(code='visual_coffee.py', volume_units=self.prefill.count) if self.prefill is not None else None),
            physics_modified=False, instrumentation='session-layer contact reports' + ('; visual-only stream and cup surface' if self.fluid else '') + ('; visual-only prefilled cup surface' if self.prefill else '')),
            indent=2, allow_nan=False) + '\n')

    # ---- geometry ----
    def machine_point(self, local):
        return np.asarray(local, dtype=float) @ self.machine_rotation + self.machine_origin

    def machine_local(self, world):
        return (np.asarray(world, dtype=float) - self.machine_origin) @ self.machine_rotation.T

    def placement(self, position, quaternion):
        r = rotation(quaternion)
        outlet_in_cup = r.inv().apply(self.outlet - position)
        outline = self.machine_local(r.apply(self.cup_geometry.outline) + position)[:, 2]
        spheres = [self.machine_local(r.apply(c) + position)[2] - radius for c, radius in self.cup_geometry.spheres]
        return CupPlacement(xy_error_m=float(np.linalg.norm((position - self.anchor)[:2])),
                            outlet_over_opening=bool(np.hypot(outlet_in_cup[0], outlet_in_cup[2]) < self.cup_geometry.opening_radius),
                            outside_bay=bool(min(outline.min(), *spheres) > BAY_EXIT_LOCAL_Z))

    # ---- runtime ----
    def initialize(self, world):
        from evaluation.physics_readers import ReadOnlyRigidBody
        from pxr import UsdGeom
        self.world = world
        self.last_time = float(world.current_time)
        self.views = {}
        for path in (*self.objects, BUTTON, *HANDS.values(), self.head_camera.link):
            view = ReadOnlyRigidBody(prim_path=path)
            view.initialize()
            self.views[path] = view
        self.beans = Bodies(self.bean_paths) if self.bean_paths else None
        if self.beans is not None:
            cache = UsdGeom.XformCache()
            authored = np.asarray([list(cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(p)).ExtractTranslation()) for p in self.bean_paths])
            if np.abs(self.beans.positions() - authored).max() > 1e-3:
                raise RuntimeError('Bean view order does not match the authored beans')

    def reset_accumulators(self):
        self.window = dict(ground={o: False for o in self.objects}, machine={o: False for o in self.objects},
                           robot_machine=False, robot_on_button=False, events=[], event_by_robot=[])

    def summarize(self):
        tray = (CUP_SUPPORT, TRAY_COLLIDER)
        result = {o: dict(robot=False, supports=set(), ground=False, machine=False) for o in self.objects}
        grips = {o: {h: [[0., 0., 0.], [0., 0., 0.]] for h in PADS} for o in self.objects}
        touched = {o: {h: [False, False] for h in PADS} for o in self.objects}
        robot_machine = robot_on_button = False
        button_impulse = 0.
        for paths, samples in self.contacts.pairs():
            for obj in self.objects:
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
                elif under(other, MACHINE):
                    if up and any(under(other, t) for t in tray):
                        flags['supports'].add(ON_TRAY)
                    else:
                        flags['machine'] = True
                        if up:
                            flags['supports'].add(ON_MACHINE)
                elif under(other, FLOOR):
                    flags['ground'] = True
                    if up:
                        flags['supports'].add(ON_GROUND)
                elif up:
                    flags['supports'].add(ON_COUNTER if under(other, COUNTER) else ON_OTHER)
            for i in (0, 1):
                if under(paths[i], ROBOT) and under(paths[1 - i], MACHINE):
                    if under(paths[1 - i], BUTTON):
                        strongest = max(s for _, s in samples)
                        if '/gripper_' in paths[i] and strongest > BUTTON_PRESS_IMPULSE:
                            robot_on_button = True
                            button_impulse = max(button_impulse, strongest)
                        if not self.spec.has_device:
                            robot_machine = True
                    else:
                        robot_machine = True
        bilateral = {o: {h: bool(all(touched[o][h]) and sum(a * b for a, b in zip(*grips[o][h])) < 0) for h in PADS} for o in self.objects}
        return result, bilateral, robot_machine, robot_on_button, button_impulse

    def after_step(self):
        if self.closed:
            raise RuntimeError('Reader closed')
        now = float(self.world.current_time)
        if abs(now - self.last_time - PHYSICS_DT) > 1e-6:
            raise RuntimeError('Missing physical substep')
        from pxr import UsdGeom
        if UsdGeom.Xformable(self.stage.GetPrimAtPath(MACHINE)).ComputeLocalToWorldTransform(0) != self.machine_matrix:
            raise RuntimeError('The coffee machine moved; anchors would be stale')
        poses = {}
        for path, view in self.views.items():
            p, q = view.get_world_pose()
            if np.asarray(p).shape != (3,) or not np.isfinite(p).all():
                raise ValueError('Invalid physical position')
            rotation(q)
            poses[path] = dict(position_m=[float(v) for v in p], quaternion_wxyz=[float(v) for v in q])
        self.contacts.ingest(self.interface.get_contact_report())
        if self.diag is not None:
            rows = []
            for paths, samples in self.contacts.pairs():
                if any(under(p, ROBOT) for p in paths):
                    rows.append(dict(a=paths[0], b=paths[1], impulse=max(s for _, s in samples), n=len(samples)))
            if rows:
                self.diag.write(json.dumps(dict(step=self.step + 1, time_s=now, pairs=rows)) + '\n')
        contact, bilateral, robot_machine, robot_on_button, button_impulse = self.summarize()
        grasped, grasp_details = self.detector.observe(now, poses, bilateral)
        displacement = float(self.machine_local(poses[BUTTON]['position_m'])[1])
        device = None
        if self.animation is not None:
            self.animation.sync_cup(poses[CUP]['position_m'], poses[CUP]['quaternion_wxyz'])
        if self.controller is not None:
            beans = self.machine_local(self.beans.positions())
            pressed_before = self.controller.pressed
            device = self.controller.update(PHYSICS_DT, {i: tuple(v) for i, v in enumerate(beans)}, displacement)
            # a press stroke belongs to the robot when a gripper link pushed the button at any substep of the stroke
            if displacement < -0.001 and robot_on_button:
                self.press_by_robot = True
            if device['event'] is not None:
                self.window['events'].append(device['event'])
                self.window['event_by_robot'].append(bool(self.press_by_robot))
            if pressed_before and not self.controller.pressed:
                self.press_by_robot = False
            if displacement >= -0.001 and not robot_on_button:
                self.press_by_robot = False
            emitted = self.fluid.update(PHYSICS_DT, device['dispensing'], self.machine_matrix) if self.fluid is not None else 0
            device = dict(bean_count=device['bean_count'], raw_bean_count=device['raw_bean_count'], dispensing=device['dispensing'],
                          event=device['event'], pressed=device['button_pressed'], emitted_now=emitted,
                          emitted_total=self.fluid.count if self.fluid is not None else 0)
        for obj in self.objects:
            self.window['ground'][obj] |= contact[obj]['ground']
            self.window['machine'][obj] |= contact[obj]['machine']
        self.window['robot_machine'] |= robot_machine
        self.window['robot_on_button'] |= robot_on_button
        self.step += 1
        self.latest = dict(
            time_s=now, reader_step=self.step, poses=poses,
            contacts={o: dict(robot=contact[o]['robot'], supports=sorted(contact[o]['supports']), ground=contact[o]['ground'],
                              machine=contact[o]['machine'], bilateral=bilateral[o], grasped=bool(grasped[o])) for o in self.objects},
            tilt_deg={o: self.tilt(poses[o]['quaternion_wxyz']) for o in self.objects},
            robot_machine=robot_machine, robot_on_button=robot_on_button, button_impulse=button_impulse,
            button_displacement_m=displacement, device=device, grasp_details=grasp_details)
        self.history.append(self.latest)
        self.trace.write(json.dumps(serializable(self.latest), allow_nan=False) + '\n')
        self.last_time = now

    @staticmethod
    def tilt(quaternion):
        up = rotation(quaternion).apply([0., 1., 0.])
        return float(np.degrees(np.arccos(np.clip(up[2], -1., 1.))))

    def read_liquid(self):
        value = self.liquid_points.Get()
        return np.zeros((0, 3)) if value is None else np.asarray(value, dtype=float).reshape(-1, 3)

    def liquid_now(self):
        """Counts without advancing the spill tracker (initial acceptance)."""
        return self.animation.counts()

    def cup_in_head_frame(self):
        link, cup = self.latest['poses'][self.head_camera.link], self.latest['poses'][CUP]
        return self.head_camera.check(link['position_m'], link['quaternion_wxyz'], cup['position_m'], cup['quaternion_wxyz'])

    def beans_in_bag(self):
        bag = self.latest['poses'][BAG]
        local = rotation(bag['quaternion_wxyz']).inv().apply(self.beans.positions() - np.asarray(bag['position_m']))
        return int(self.bag_geometry.inside(local).sum())

    def frame(self):
        """Facts of the control frame that ends at the latest substep."""
        if self.start_step is None:
            raise RuntimeError('Initial state not accepted')
        delta = self.step - self.start_step
        if delta % SUBSTEPS:
            raise RuntimeError('Control frame is not aligned with the physics substeps')
        row, window = self.latest, self.window
        objects = {}
        for obj in self.objects:
            c, pose = row['contacts'][obj], row['poses'][obj]
            ground = bool(window['ground'][obj] or c['ground'])
            machine = bool(window['machine'][obj] or c['machine'])
            objects[obj] = ObjectFacts(
                position=tuple(pose['position_m']), tilt_deg=row['tilt_deg'][obj], pads_bilateral=bool(c['grasped'] and c['robot']),
                robot_contact=bool(c['robot']), supports=frozenset(c['supports']), ground_contact=ground, machine_body_contact=machine,
                displacement_m=float(np.linalg.norm(np.asarray(pose['position_m']) - self.initial_position[obj])))
        cup = row['poses'][CUP]
        liquid = button = None
        if self.animation is not None:
            counts = self.animation.counts()
            liquid = LiquidFacts(emitted=counts['particles'], in_cup=counts['in_cup'],
                                 airborne=counts['airborne'], spilled=counts['spilled'],
                                 initial_in_cup=int(self.initial_in_cup), mode='visual_cpu_v1')
        if self.spec.has_device:
            button = ButtonFacts(displacement_m=row['button_displacement_m'], robot_on_button=bool(window['robot_on_button'] or row['robot_on_button']),
                                 dispensing=bool(row['device']['dispensing']), events=tuple(window['events']),
                                 event_by_robot=tuple(window['event_by_robot']))
        index = delta // SUBSTEPS
        facts = Facts(step=index, time_s=index * CONTROL_DT, cup=objects[CUP], bag=objects[BAG],
                      placement=self.placement(np.asarray(cup['position_m']), cup['quaternion_wxyz']),
                      robot_machine_contact=bool(window['robot_machine'] or row['robot_machine']),
                      beans_in_hopper=int(row['device']['bean_count']) if self.controller is not None else 0,
                      beans_total=len(self.bean_paths), liquid=liquid, button=button)
        self.reset_accumulators()
        self.frames.write(json.dumps(serializable(dict(facts.as_dict(), simulation_time_s=row['time_s'], reader_step=self.step)), allow_nan=False) + '\n')
        self.frames.flush()
        self.trace.flush()
        return facts

    def begin_episode(self):
        if self.start_step is not None or self.latest is None:
            raise RuntimeError('Invalid initial acceptance')
        self.start_step = self.step
        self.initial_position = {o: np.asarray(self.latest['poses'][o]['position_m'], dtype=float) for o in self.objects}
        if self.animation is not None and self.fluid is None:
            self.initial_in_cup = self.liquid_now()['in_cup']
        self.reset_accumulators()
        self.press_by_robot = False

    def close(self):
        if not self.closed:
            self.trace.close()
            self.frames.close()
            if self.diag is not None:
                self.diag.close()
            self.closed = True
