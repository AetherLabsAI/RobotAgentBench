"""Scene-owned persistent coffee device and 20 Hz evidence bridge.

No scene reset, robot commands, physics setters or global registration. The
scene owner supplies a native substep source and owns this service for the
whole episode, including all four scoring phases. CPU fixtures are labelled.
"""
from dataclasses import dataclass
import hashlib
import math
import numpy as np
from pxr import Gf, Usd, UsdGeom

from arena_api.policy_execution import PHYSICS_DT, SUBSTEPS
from evaluation.tasks.coffee_machine import isaac_evidence as old, vendor_producer
from evaluation.tasks.coffee_machine.visual_coffee import VisualCoffee, MODE
from evaluation.tasks.coffee_machine.facts import Facts, ObjectFacts, CupPlacement, LiquidFacts, ButtonFacts
from .scene_binding import ReadOnlySceneBinding
from .composite import Frame, CoffeeComposite, PHASES


@dataclass(frozen=True)
class Substep:
    time_s: float
    poses: dict
    bean_positions: np.ndarray
    contact_pairs: tuple


class SceneCoffeeService:
    """Construct ONCE by the scene owner before physics, never per phase."""
    def __init__(self, stage, manifest, episode_id, instance_id):
        self.binding = ReadOnlySceneBinding(stage, manifest)
        self.identity = self.binding.task_binding(episode_id, instance_id)
        self.roles = self.binding.roles
        self.counter_paths = self.binding.counter_paths
        self.stage = stage
        animation_root = '/World/OpenendedCoffeeVisual/M' + hashlib.sha256(self.roles['machine'].encode()).hexdigest()[:16]
        self.animation_root = animation_root
        if MODE != 'visual_cpu_v1':
            raise ValueError('Only the approved visual coffee mode is supported')
        if stage.GetPrimAtPath(animation_root):
            raise ValueError('Existing device/animation: do not reconstruct or clear it')
        if not animation_root.startswith('/World/') or any(
                old.under(animation_root, path) or old.under(path, animation_root)
                for path in self.roles.values()):
            raise ValueError('Animation root must be separate from physical objects')
        self.objects = (self.roles['cup'], self.roles['bag'])
        self.machine = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(self.roles['machine']))
        self.m = np.asarray(self.machine)
        if np.max(np.abs(self.m[:3, :3] @ self.m[:3, :3].T - np.eye(3))) > 1e-6:
            raise ValueError('Scaled machine frame')
        self.controller = vendor_producer.load('coffee_logic').CoffeeController(self.binding.contract.hopper)
        # Visual-only session prims; never save or overwrite a source scene layer.
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            self.animation = VisualCoffee(stage, animation_root, self.binding.cup, self.binding.contract.outlet, prefilled=0)
        self.detector = old.GraspDetector(self.objects)
        self.last_time = None
        self.sequence = -1
        self.latest = None
        self.press_by_robot = False
        self.consumer = None
        self.closed = False
        self._clear_window()

    def _clear_window(self):
        self.window = dict(ground={p: False for p in self.objects}, machine={p: False for p in self.objects},
                           robot_machine=False, robot_button=False, robot_on_button=False, events=[], by_robot=[])

    def machine_local(self, points):
        return (np.asarray(points) - self.m[3, :3]) @ self.m[:3, :3].T

    def _contacts(self, pairs, floor):
        from evaluation.tasks.coffee_machine.facts import COUNTER, TRAY, GROUND, MACHINE_BODY, OTHER
        result = {p: dict(robot=False, supports=set(), ground=False, machine=False) for p in self.objects}
        normals = {p: {h: np.zeros((2, 3)) for h in old.PADS} for p in self.objects}
        touched = {p: {h: [False, False] for h in old.PADS} for p in self.objects}
        machine_contact = button_contact = button_press = False
        tray = (self.roles['cup_support'], self.roles['machine'] + '/Body/Collision/Tray')
        for paths, samples in pairs:
            if len(paths) != 2 or not samples:
                raise ValueError('Incomplete contact pair')
            for normal, impulse in samples:
                if len(normal) != 3 or not all(math.isfinite(v) for v in (*normal, impulse)) or impulse <= 0:
                    raise ValueError('Expected measured positive-impulse contacts')
            for obj in self.objects:
                idx = 0 if old.under(paths[0], obj) else (1 if old.under(paths[1], obj) else None)
                if idx is None:
                    continue
                other, sign = paths[1 - idx], 1 if idx == 0 else -1
                up = any(sign * n[2] > old.UP_NORMAL_Z for n, _ in samples)
                flags = result[obj]
                if old.under(other, old.ROBOT):
                    flags['robot'] = True
                    for h, pads in old.PADS.items():
                        for k, pad in enumerate(pads):
                            if old.under(other, pad):
                                touched[obj][h][k] = True
                                normals[obj][h][k] += sum((sign * np.asarray(n) for n, _ in samples), np.zeros(3))
                elif old.under(other, self.roles['machine']):
                    if up and any(old.under(other, p) for p in tray):
                        flags['supports'].add(TRAY)
                    else:
                        flags['machine'] = True
                        if up: flags['supports'].add(MACHINE_BODY)
                elif old.under(other, floor):
                    flags['ground'] = True
                    if up: flags['supports'].add(GROUND)
                elif up:
                    flags['supports'].add(COUNTER if any(old.under(other,p) for p in self.counter_paths) else OTHER)
            for i in (0, 1):
                if old.under(paths[i], old.ROBOT) and old.under(paths[1-i], self.roles['machine']):
                    if old.under(paths[1-i], self.roles['button']):
                        button_contact = True
                        button_press |= '/gripper_' in paths[i] and max(v for _, v in samples) > old.BUTTON_PRESS_IMPULSE
                    else:
                        machine_contact = True
        bilateral = {p: {h: bool(all(touched[p][h]) and np.dot(*normals[p][h]) < 0) for h in old.PADS}
                     for p in self.objects}
        return result, bilateral, machine_contact, button_contact, bool(button_press)

    def advance(self, sample, *, floor, physics_dt=PHYSICS_DT):
        """Called exactly once after each native physics substep; no physics step here."""
        if self.closed or not isinstance(sample, Substep):
            raise RuntimeError('Closed service or invalid substep')
        if physics_dt not in (PHYSICS_DT, 1/60):
            raise RuntimeError('Unsupported scene clock')
        if self.consumer is not None and not self.consumer.process.ended and physics_dt != PHYSICS_DT:
            raise RuntimeError('Active coffee scoring must retain 240 Hz; leaving cannot pause/reset it')
        if not math.isfinite(sample.time_s) or (self.last_time is not None and
                abs(sample.time_s - self.last_time - physics_dt) > 1e-6):
            raise RuntimeError('Repeated, skipped or reset physics clock')
        for path in (*self.objects, self.roles['button'], *old.HANDS.values()):
            pose = sample.poses[path]
            pos, quat = np.asarray(pose['position_m']), np.asarray(pose['quaternion_wxyz'])
            if pos.shape != (3,) or quat.shape != (4,) or not np.isfinite(pos).all() or not np.isfinite(quat).all():
                raise ValueError('Invalid live pose: ' + path)
            old.rotation(quat)
        beans = np.asarray(sample.bean_positions)
        if beans.shape != (len(self.binding.resolved['bean_bodies']), 3) or not np.isfinite(beans).all():
            raise ValueError('Missing or invalid bean inventory')
        if not isinstance(sample.contact_pairs, tuple):
            raise ValueError('Explicit current contact pairs required; missing is not no contact')
        c, bilateral, machine_contact, button_contact, button_press = self._contacts(sample.contact_pairs, floor)
        grasped, details = self.detector.observe(sample.time_s, sample.poses, bilateral)
        button_depth = float(self.machine_local(sample.poses[self.roles['button']]['position_m'])[1])
        pressed_before = self.controller.pressed
        d = self.controller.update(physics_dt, dict(enumerate(map(tuple, self.machine_local(beans)))), button_depth)
        if button_depth < -.001 and button_press: self.press_by_robot = True
        if d['event'] is not None:
            self.window['events'].append(d['event'])
            self.window['by_robot'].append(bool(self.press_by_robot))
        if (pressed_before and not self.controller.pressed) or (button_depth >= -.001 and not button_press):
            self.press_by_robot = False
        cup = sample.poses[self.roles['cup']]
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            self.animation.sync_cup(cup['position_m'], cup['quaternion_wxyz'])
            self.animation.update(physics_dt, d['dispensing'], self.machine)
        for obj in self.objects:
            self.window['ground'][obj] |= c[obj]['ground']
            self.window['machine'][obj] |= c[obj]['machine']
        self.window['robot_machine'] |= machine_contact
        self.window['robot_button'] |= button_contact
        self.window['robot_on_button'] |= button_press
        self.sequence += 1
        self.last_time = sample.time_s
        self.last_physics_dt = physics_dt
        self.latest = dict(sample=sample, contacts=c, grasped=grasped, grasp_details=details,
                           device=d, button_depth=button_depth)

    def facts(self, index, phase):
        if self.latest is None or phase not in PHASES:
            raise RuntimeError('No current physical sample or unknown phase')
        row, w = self.latest, self.window
        objs = {}
        for role in ('cup', 'bag'):
            path = self.roles[role]
            pose, c = row['sample'].poses[path], row['contacts'][path]
            objs[role] = ObjectFacts(tuple(pose['position_m']), old.CoffeeReader.tilt(pose['quaternion_wxyz']),
                                    bool(row['grasped'][path] and c['robot']), bool(c['robot']),
                                    frozenset(c['supports']), bool(w['ground'][path]), bool(w['machine'][path]))
        cup = row['sample'].poses[self.roles['cup']]
        pos, rot = np.asarray(cup['position_m']), old.rotation(cup['quaternion_wxyz'])
        geometry = self.binding.cup
        outlet = self.machine.Transform(Gf.Vec3d(*self.binding.contract.outlet))
        anchor = self.machine.Transform(Gf.Vec3d(*self.binding.contract.cup))
        local = rot.inv().apply(np.asarray(outlet) - pos)
        outline = list(self.machine_local(rot.apply(geometry.outline) + pos)[:, 2])
        outline += [float(self.machine_local(rot.apply(c) + pos)[2] - radius) for c, radius in geometry.spheres]
        counts = self.animation.counts()
        return Facts(index, index * old.CONTROL_DT, objs['cup'], objs['bag'],
                     CupPlacement(float(np.linalg.norm((pos - anchor)[:2])),
                                  bool(np.hypot(local[0], local[2]) < geometry.opening_radius),
                                  bool(min(outline) > old.BAY_EXIT_LOCAL_Z)),
                     bool(w['robot_machine'] or (phase != '1-3' and w['robot_button'])),
                     int(row['device']['bean_count']), len(self.binding.resolved['bean_bodies']),
                     LiquidFacts(counts['particles'], counts['in_cup'], counts['airborne'], counts['spilled'], mode=MODE),
                     ButtonFacts(row['button_depth'], bool(w['robot_on_button']), bool(row['device']['dispensing']),
                                 tuple(w['events']), tuple(w['by_robot'])))


class CoffeeEvidenceBridge:
    """One scorer attached to an existing service. Does not reset its device."""
    def __init__(self, service, *, accepted_binding, evidence_source):
        if accepted_binding != service.identity or service.latest is None:
            raise ValueError('Scene owner must first accept this exact scene binding')
        if service.last_physics_dt != PHYSICS_DT:
            raise ValueError('Switch to coffee physics and accept its native initial state before scoring')
        if service.consumer is not None:
            raise ValueError('Service already has a scorer; leave/re-entry is not a reset')
        if evidence_source not in ('native_physx', 'cpu_synthetic_fixture'):
            raise ValueError('Explicit evidence provenance required')
        self.service = service
        self.process = CoffeeComposite(service.identity)
        self.process.source = evidence_source
        self.start = service.sequence
        self.index = -1
        # Initial logical checks must succeed before consuming the service.
        self.process._validate_global(Frame(service.identity, service.facts(0, PHASES[0])))
        service.consumer = self

    def collect_and_score(self):
        delta = self.service.sequence - self.start
        if delta % SUBSTEPS or delta // SUBSTEPS != self.index + 1:
            raise RuntimeError('Repeated, missing or misaligned 20 Hz frame')
        index = delta // SUBSTEPS
        result = self.process.observe(Frame(self.service.identity, self.service.facts(index, self.process.phase)))
        self.index = index
        self.service._clear_window()
        return result
