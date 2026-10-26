"""Live physical readout for the tidy_sofa family. Only session contact-report instrumentation is authored.

Per physics substep it records target/scenery/drawer/hand poses, reduces the PhysX contact report to
object identities (finger pads, support/destination, floor, furniture) and emits one PhysicalFrame.
No physics parameters, drives or poses are written.
"""
from collections import deque
from dataclasses import asdict
import hashlib
import json
import logging
from pathlib import Path
import numpy as np
from arena_api.contact_identity import ContactIdentityCache
from arena_api.policy_execution import PHYSICS_DT
from evaluation.tasks.shelf_pick.physical_detection import GraspDetector, DetectionConfig
from evaluation.tasks.shelf_pick.geometry import rotation
from .specs import ROBOT, DRAWER, COFFEE_TABLE, DRAWER_INITIAL_LIP_M
from .process_rules import PhysicalFrame, RULE_VERSION

FLOOR = '/World/Background/Shell/Floor'
GENERATED_FLOORS = '/World/HouseTopologyCandidate/Shell/Floors'
PADS = {h: tuple(ROBOT + f'/gripper_{h}_{side}_link4' for side in ('inner', 'outer')) for h in ('l', 'r')}
HANDS = {h: ROBOT + f'/gripper_{h}_base_link' for h in PADS}
BASE_LINK = ROBOT + '/base_link'
# Drawer interior (L1 destination_aabb, canonical layout, drawer at its closed pose) expressed relative to the
# drawer body origin so it travels with the drawer; z from the drawer floor to the wall lip.
DRAWER_INTERIOR_LOCAL = ((-1.8405820086964546 + 1.705296933972675, -1.5300252365197225 + 1.705296933972675),
                         (-0.6297119681218109 + 0.5176450936458411, -0.40557776113885174 + 0.5176450936458411),
                         (0.19999322958861535 - 0.20000892877578746, 0.3373112690122057 - 0.20000892877578746))


def under(path, root):
    return path == root or path.startswith(root.rstrip('/') + '/')


def serializable(v):
    if isinstance(v, dict): return {k: serializable(x) for k, x in v.items()}
    if isinstance(v, (tuple, list, set, frozenset)): return [serializable(x) for x in v]
    if isinstance(v, np.generic): return v.item()
    return v


class Contacts:
    """Reduce PhysX pairs to object identities; keep only measured (nonzero-impulse) samples."""
    def __init__(self, resolve, sleeping, event_types, roots):
        self.sleeping = sleeping; self.found, self.persist, self.lost = event_types
        self.roots = tuple(roots)
        self.identity = ContactIdentityCache(resolve, lambda paths: any(under(p, r) for p in paths for r in self.roots))
        self.active = {}
        self.missing_awake_pair_count = 0
        self._warned_missing_awake_pairs = set()

    def ingest(self, report):
        if not isinstance(report, tuple) or len(report) < 2 or report[0] is None or report[1] is None:
            raise RuntimeError('Missing current PhysX contact report')
        headers, data = report[:2]; seen = set()
        for h in headers:
            paths, key, relevant = self.identity.get(h.stage_id, h.collider0, h.collider1)
            if not relevant: continue
            seen.add(key)
            if h.type == self.lost:
                self.active.pop(key, None); continue
            if h.type not in (self.found, self.persist): raise RuntimeError('Unknown contact event')
            count, offset = int(h.num_contact_data), int(h.contact_data_offset)
            if count <= 0 or offset < 0 or offset + count > len(data): raise RuntimeError('Incomplete physical contact')
            samples = []
            for i in range(offset, offset + count):
                v = data[i]
                imp = (v.impulse.x, v.impulse.y, v.impulse.z); normal = (v.normal.x, v.normal.y, v.normal.z); point = (v.position.x, v.position.y, v.position.z)
                if not all(np.isfinite(x) for x in imp + normal + point): raise RuntimeError('Nonfinite physical contact')
                if sum(x * x for x in imp) > 1e-18: samples.append((normal, point, imp))
            self.active[key] = (paths, int(h.stage_id), (int(h.actor0), int(h.actor1)), tuple(samples))
        for key, (_, stage, actors, _) in tuple(self.active.items()):
            if key not in seen and not all(self.sleeping(stage, a) is True for a in actors):
                # No fresh evidence: never retain stale touch/grasp samples.
                self.active.pop(key, None)
                self.missing_awake_pair_count += 1
                if key not in self._warned_missing_awake_pairs:
                    self._warned_missing_awake_pairs.add(key)
                    logging.getLogger(__name__).warning(
                        'Missing awake contact pair: %s; stale contact discarded, continuing '
                        '(further warnings for this pair suppressed; count=%d)',
                        key, self.missing_awake_pair_count)

    def pairs(self):
        for paths, _, _, samples in self.active.values():
            if samples: yield paths, samples


class LivingroomReader:
    def __init__(self, stage, output, episode_id, spec, plan, config=DetectionConfig()):
        from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, PhysicsSchemaTools, Gf
        from omni.physx import get_physx_simulation_interface
        from omni.physx.bindings._physx import ContactEventType
        self.stage = stage; self.output = Path(output); self.output.mkdir(parents=True, exist_ok=False)
        self.spec = spec; self.plan = plan; self.episode_id = episode_id; self.config = config
        self.target = spec.target; self.scenery = tuple(spec.scenery); self.objects = (self.target, *self.scenery)
        self.support_root = spec.support if spec.kind == 'pick' else spec.destination
        floor = stage.GetPrimAtPath(FLOOR); generated = stage.GetPrimAtPath(GENERATED_FLOORS)
        def colliding(p): return bool(p and p.IsActive() and p.HasAPI(UsdPhysics.CollisionAPI) and p.GetAttribute('physics:collisionEnabled').Get() is not False)
        self.floor_paths = [FLOOR] if colliding(floor) else ([str(c.GetPath()) for c in generated.GetChildren() if colliding(c)] if generated else [])
        if not self.floor_paths: raise ValueError('No active physical floor')
        self.frame_delta = np.asarray(plan['source_to_layout_matrix'], dtype=float)   # row-major Gf.Matrix4d: world' = world * M
        self.zone = self.zone_geometry()
        self.interface = get_physx_simulation_interface()
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            bodies = [stage.GetPrimAtPath(p) for p in self.objects] + ([stage.GetPrimAtPath(DRAWER)] if spec.kind == 'drawer' else [])
            bodies += [p for p in Usd.PrimRange(stage.GetPrimAtPath(ROBOT)) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
            for p in bodies:
                if not p or not p.HasAPI(UsdPhysics.RigidBodyAPI): raise ValueError('Missing contact-report body')
                PhysxSchema.PhysxContactReportAPI.Apply(p).CreateThresholdAttr(0.)
        def sleeping(stage_id, actor):
            p = stage.GetPrimAtPath(PhysicsSchemaTools.intToSdfPath(actor))
            if not p: raise RuntimeError('Unknown contact actor')
            return bool(not p.HasAPI(UsdPhysics.RigidBodyAPI) or self.interface.is_sleeping(stage_id, actor))
        self.contacts = Contacts(PhysicsSchemaTools.intToSdfPath, sleeping,
                                 (ContactEventType.CONTACT_FOUND, ContactEventType.CONTACT_PERSIST, ContactEventType.CONTACT_LOST),
                                 (ROBOT, *self.objects))
        self.detector = GraspDetector(self.objects, config)
        self.hashes = {l.realPath: hashlib.sha256(Path(l.realPath).read_bytes()).hexdigest() for l in stage.GetUsedLayers() if l.realPath and Path(l.realPath).is_file()}
        self.fixed = {p: UsdGeom.Xformable(stage.GetPrimAtPath(p)).ComputeLocalToWorldTransform(0) for p in (*self.floor_paths, *spec.furniture) if stage.GetPrimAtPath(p)}
        self.step = -1; self.latest = None; self.start_step = None; self.initial_z = None; self.drawer_origin = None
        self.pending = []; self.closed = False; self.history = deque(maxlen=60)
        self.trace = (self.output / 'physical_facts.jsonl').open('x')
        (self.output / 'detection-profile.json').write_text(json.dumps(dict(rule_version=RULE_VERSION, api_id=spec.api_id, kind=spec.kind, target=self.target,
            scenery=list(self.scenery), support_root=self.support_root, furniture=list(spec.furniture), zone=serializable(self.zone), config=asdict(config),
            floors=self.floor_paths, physics_modified=False, instrumentation='session-layer contact reports'), indent=2) + '\n')

    # ---- geometry ----
    def to_layout(self, xyz):
        v = np.asarray([*xyz, 1.0], dtype=float) @ self.frame_delta
        return v[:3]

    def zone_geometry(self):
        sp = self.spec
        if sp.kind == 'pick': return None
        if sp.kind == 'drawer':
            return dict(kind='drawer_interior', local=DRAWER_INTERIOR_LOCAL)
        cx, cy, hx, hy = sp.zone_xy
        rot = self.frame_delta[:3, :3]
        center = self.to_layout((cx, cy, sp.zone_top_z))
        return dict(kind=sp.kind, center=center.tolist(), half=(hx, hy), rotation=rot.tolist(), top_z=float(center[2]), margin=sp.edge_margin_m)

    def in_zone(self, poses):
        z = self.zone; p = np.asarray(poses[self.target]['position_m'])
        if z is None: return True
        if z['kind'] == 'drawer_interior':
            d = poses[DRAWER]
            # interior box is world-aligned in the canonical layout; undo the layout rotation (row-vector M: canonical = layout @ M^-1)
            rel = self.frame_delta[:3, :3] @ (p - np.asarray(d['position_m']))
            (x0, x1), (y0, y1), (z0, z1) = z['local']
            # inside = centre below the wall rim (an object bridging the rims sits at z1 and is NOT inside)
            return bool(x0 <= rel[0] <= x1 and y0 <= rel[1] <= y1 and z0 - 0.005 <= rel[2] <= z1 - 0.01)
        rel = np.asarray(z['rotation']) @ (p - np.asarray(z['center']))   # layout -> canonical axes
        hx, hy = z['half']; m = z['margin']
        inside = abs(rel[0]) <= hx - m and abs(rel[1]) <= hy - m
        if z['kind'] == 'basket': inside = inside and p[2] < z['top_z'] + 0.10
        return bool(inside)

    # ---- runtime ----
    def initialize(self, world):
        from evaluation.physics_readers import ReadOnlyRigidBody
        self.world = world; self.last_time = float(world.current_time); self.views = {}
        paths = (*self.objects, *HANDS.values(), BASE_LINK) + ((DRAWER,) if self.spec.kind == 'drawer' else ())
        for i, path in enumerate(paths):
            view = ReadOnlyRigidBody(prim_path=path)
            view.initialize(); self.views[path] = view
        if self.spec.kind == 'drawer':
            p, q = self.views[DRAWER].get_world_pose()
            r = rotation(q)
            self.drawer_axis_local = r.inv().apply(self.to_layout((1, 0, 0)) - self.to_layout((0, 0, 0)))
            self.drawer_origin = np.asarray(p, dtype=float)

    def after_step(self):
        if self.closed: raise RuntimeError('Reader closed')
        now = float(self.world.current_time)
        if abs(now - self.last_time - PHYSICS_DT) > 1e-6: raise RuntimeError('Missing physical substep')
        from pxr import UsdGeom
        for path, tf in self.fixed.items():
            if UsdGeom.Xformable(self.stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(0) != tf: raise RuntimeError('Static furniture moved: ' + path)
        poses = {}
        for path, view in self.views.items():
            p, q = view.get_world_pose()
            if np.asarray(p).shape != (3,) or not np.isfinite(p).all(): raise ValueError('Invalid physical position')
            rotation(q); poses[path] = dict(position_m=[float(v) for v in p], quaternion_wxyz=[float(v) for v in q])
        self.contacts.ingest(self.interface.get_contact_report())
        c = self.summarize()
        grasped, details = self.detector.observe(now, poses, c['bilateral'])
        travel = 0.0
        if self.spec.kind == 'drawer':
            d = np.asarray(poses[DRAWER]['position_m']); axis = rotation(poses[DRAWER]['quaternion_wxyz']).apply(self.drawer_axis_local)
            travel = float(np.dot(d - self.drawer_origin, axis)) + DRAWER_INITIAL_LIP_M
        self.step += 1
        self.latest = dict(time_s=now, reader_step=self.step, poses=poses, contacts=c, grasped=grasped, grasp_details=details,
                           in_zone=self.in_zone(poses), drawer_travel_m=travel)
        self.history.append(self.latest)
        if self.start_step is not None: self.pending.append(self.frame())
        self.trace.write(json.dumps(serializable(self.latest), allow_nan=False) + '\n'); self.trace.flush(); self.last_time = now

    def summarize(self):
        sp = self.spec
        grips = {obj: {h: [[0., 0., 0.], [0., 0., 0.]] for h in PADS} for obj in self.objects}
        pad_contact = {obj: False for obj in self.objects}
        supported = {obj: False for obj in self.objects}; on_floor = {obj: False for obj in self.objects}
        support_contact = {obj: False for obj in self.objects}     # any contact with the support/destination body (drawer rim, wall, floor)
        foreign_support = {obj: False for obj in self.objects}     # held up by something that is neither the support/destination nor the robot
        furniture = set()
        for paths, samples in self.contacts.pairs():
            for obj in self.objects:
                index = 0 if under(paths[0], obj) else (1 if under(paths[1], obj) else None)
                if index is None: continue
                other = paths[1 - index]; sign = 1 if index == 0 else -1
                for hand, pads in PADS.items():
                    for k, pad in enumerate(pads):
                        if under(other, pad):
                            pad_contact[obj] = True
                            for n, _, _ in samples:
                                for a in range(3): grips[obj][hand][k][a] += sign * n[a]
                up = any(sign * n[2] > .5 for n, _, _ in samples)   # the other body pushes the object up
                if under(other, self.support_root):
                    support_contact[obj] = True; supported[obj] |= up
                elif not under(other, ROBOT) and up: foreign_support[obj] = True
                if any(under(other, f) for f in self.floor_paths): on_floor[obj] = True
            # robot vs furniture (deductions); drawer contact is part of the drawer task, seat/support contact is not furniture
            for i in (0, 1):
                subject, other = paths[i], paths[1 - i]
                if not under(subject, ROBOT): continue
                if sp.kind == 'drawer' and under(other, DRAWER): continue
                for root in sp.furniture:
                    if under(other, root): furniture.add(root)
        bilateral = {obj: {h: (sum(a * b for a, b in zip(*pads)) < 0) for h, pads in hands.items()} for obj, hands in grips.items()}
        return dict(bilateral=bilateral, pad_contact=pad_contact, supported=supported, support_contact=support_contact, foreign_support=foreign_support,
                    on_floor=on_floor, furniture_contacts=frozenset(furniture))

    def frame(self):
        if self.start_step is None: raise RuntimeError('Initial state not accepted')
        row = self.latest; c = row['contacts']; t = self.target
        wrong = frozenset(s for s in self.scenery if row['grasped'][s] and row['poses'][s]['position_m'][2] - self.initial_z[s] >= self.spec.lift_m)
        return PhysicalFrame(self.episode_id, self.step - self.start_step, row['time_s'],
                             bool(row['grasped'][t]), bool(c['pad_contact'][t]), bool(c['supported'][t]), bool(row['in_zone']),
                             row['poses'][t]['position_m'][2] - self.initial_z[t], bool(c['on_floor'][t]),
                             wrong_grasped=wrong, furniture_contacts=c['furniture_contacts'],
                             scenery_on_floor=frozenset(s for s in self.scenery if c['on_floor'][s]), drawer_travel_m=float(row['drawer_travel_m']),
                             target_support_contact=bool(c['support_contact'][t]), target_foreign_support=bool(c['foreign_support'][t]))

    def begin_episode(self):
        if self.start_step is not None or self.latest is None: raise RuntimeError('Invalid initial acceptance')
        self.start_step = self.step
        self.initial_z = {o: self.latest['poses'][o]['position_m'][2] for o in self.objects}
        self.pending = [self.frame()]

    def take_batch(self):
        if not self.pending: raise RuntimeError('Missing physical frame batch')
        result = tuple(self.pending); self.pending = []
        return result

    def close(self):
        if not self.closed: self.trace.close(); self.closed = True
