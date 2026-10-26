"""Native multi-object readout for the existing 20 Hz OpenEnd host.

Only contact instrumentation is authored. No poses, physics parameters, helper
events, synthetic grasps or task-specific clocks are used to award points.
"""
from collections import deque
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import json
import math
import numpy as np
from .registration import ClothClearRegistration
from .scoring import Binding, Facts
from .group import ClothClearGroup
from openend.tasks.all_objects_integration.touch_wake import TouchWake
from openend.tasks.all_objects_integration.selective_read import contact_report


class ClothClearPort(ClothClearRegistration):
    def __init__(self, stage, manifest, output, episode_id, instance_id, parameters):
        self.validate_parameters(parameters)
        from pxr import Usd, UsdPhysics, PhysxSchema, PhysicsSchemaTools
        from omni.physx import get_physx_simulation_interface
        from omni.physx.bindings._physx import ContactEventType
        from evaluation.tasks.shelf_pick.geometry import HANDS
        from evaluation.tasks.shelf_pick.contact_stream import ROBOT
        from openend.tasks.holding_follow_v1.detection import DetectionConfig, GraspDetector
        from .native_geometry import SceneGeometry
        from .contacts import ClothContacts
        self.geometry = g = SceneGeometry(stage, manifest)
        self.stage, self.hands = stage, HANDS
        self.bindings = tuple(Binding(episode_id, instance_id+'_'+str(i), p, g.cloth, g.table)
                              for i, p in enumerate(g.targets))
        self.binding = SimpleNamespace(objects={**{b.instance_id: b.target_path for b in self.bindings},
                                                'cloth': g.cloth, 'table': g.table})
        self.config = DetectionConfig()
        self.detector = GraspDetector(g.targets, self.config)
        self.interface = get_physx_simulation_interface()
        if not stage.GetSessionLayer().anonymous:
            raise ValueError('Anonymous session instrumentation required')
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            bodies = [stage.GetPrimAtPath(p) for p in g.bodies.values()]
            bodies += [p for p in Usd.PrimRange(stage.GetPrimAtPath(ROBOT)) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
            for p in bodies:
                attr = p.GetAttribute('physxContactReport:threshold')
                if attr and attr.HasAuthoredValueOpinion() and attr.Get() != 0:
                    raise ValueError('Conflicting contact instrumentation')
                PhysxSchema.PhysxContactReportAPI.Apply(p).CreateThresholdAttr(0.)
        def sleeping(sid, actor):
            p = stage.GetPrimAtPath(PhysicsSchemaTools.intToSdfPath(actor))
            if not p:
                raise RuntimeError('Unknown contact actor')
            return bool(not p.HasAPI(UsdPhysics.RigidBodyAPI) or self.interface.is_sleeping(sid, actor))
        self.contacts = ClothContacts(PhysicsSchemaTools.intToSdfPath, sleeping,
            (ContactEventType.CONTACT_FOUND, ContactEventType.CONTACT_PERSIST, ContactEventType.CONTACT_LOST),
            (), g.top['z'], self.config.support_face_tolerance_m, floor_paths=g.floors,
            targets=g.targets, support=g.table)
        self.output = Path(output)
        self.output.mkdir(exist_ok=False)
        self.trace = (self.output/'facts.jsonl').open('x')
        self.history = deque(maxlen=math.ceil(self.config.initial_history_s*240)+2)
        self.latest = self.composite = self.score = None
        self.pending, self.index, self.closed = [], -1, False
        self.contact_selection = TouchWake()
        self.selection_time = None
        (self.output/'geometry.json').write_text(json.dumps(dict(targets=g.targets, bodies=g.bodies,
            table=g.top, cloth=g.cloth_polygon, exclusions=g.exclusions, detector=asdict(self.config),
            scene_pose_writes=0, physics_parameters_changed=False,
            stability_semantics='continuous_loaded_upward_table_support_released_1s_no_new_velocity_threshold'), indent=2)+'\n')

    @property
    def ended(self):
        return self.composite is not None and self.composite.ended

    def initialize(self, clock, view):
        self.clock, self.last_time = clock, clock.current_time
        self.views = {}
        for key, path in [*self.geometry.bodies.items(), *((p, p) for p in self.hands.values())]:
            v = view.create_rigid_body_view(path)
            if v.count != 1 or list(v.prim_paths) != [path]:
                raise ValueError('Native body binding mismatch ' + path)
            self.views[key] = v

    def _contact_step(self, helper):
        if any(helper.get(k) for k in ('events', 'docked_object_count', 'tray_dock_enabled')):
            raise RuntimeError('Helper-modified evidence')
        now, dt = self.clock.current_time, 1/self.clock.hz
        if self.clock.hz not in (60, 240) or abs(now-self.last_time-dt) > 1e-6:
            raise RuntimeError('Missing native substep')
        self.contacts.ingest(contact_report(self, helper))
        contacts = self.contacts.facts(self.geometry)
        self.contact_selection.observe(time_s=now, finger_contact=any(c['finger_contact'] for c in contacts.values()),
            supports=frozenset().union(*(c['supports'] for c in contacts.values())),
            external_contact=any(c['external_contact'] for c in contacts.values()),
            on_floor=any(c['on_floor'] for c in contacts.values()))
        self.selection_time = now
        return now, dt, contacts

    def after_physics(self, helper):
        now, dt, contacts = self._contact_step(helper)
        self.geometry.validate_static(self.stage)
        poses = {}
        for path, view in self.views.items():
            pose = np.asarray(view.get_transforms()).copy()[0]
            if pose.shape != (7,) or not np.isfinite(pose).all():
                raise RuntimeError('Invalid native body pose')
            poses[path] = dict(position_m=pose[:3].tolist(), quaternion_wxyz=pose[[6,3,4,5]].tolist())
        held, _ = self.detector.observe(now, poses, {p:c['bilateral'] for p,c in contacts.items()})
        row = dict(time_s=now, physics_dt_s=dt, contacts=contacts, poses=poses, held=held,
                   footprints={p:self.geometry.footprint(p, poses[p]) for p in self.geometry.targets})
        self.latest = row
        self.history.append(row)
        self.last_time = now
        if self.composite is not None:
            self.pending.append(row)

    def after_inactive(self, helper):
        if self.pending:
            raise RuntimeError('Unconsumed selected batch')
        now, dt, _ = self._contact_step(helper)
        for h in self.detector.history.values():
            h.clear()
        self.composite.suspend_substep(now, dt)
        self.last_time = now
        self.inactive_substeps = getattr(self, 'inactive_substeps', 0)+1

    def facts(self, row, index):
        g = self.geometry
        result = []
        for b in self.bindings:
            c = row['contacts'][b.target_path]
            result.append(Facts(b, index, row['time_s'], row['physics_dt_s'], bool(row['held'][b.target_path]),
                c['robot_contact'], c['supports'], c['on_floor'], c['external_contact'],
                c['supports'] == frozenset((g.table,)), row['footprints'][b.target_path],
                g.top['polygon'], g.cloth_polygon, g.exclusions, support_contact=c['support_contact']))
        return tuple(result)

    def accept_initial(self):
        from openend.tasks.meal_tray.nonmotion_closeout_v1.initial_windows import initial_window
        if self.composite is not None or self.latest is None:
            raise RuntimeError('Accept initial once')
        rows, complete = initial_window(self.history, self.config.initial_history_s, self.clock.get_physics_dt())
        checks = dict(history_complete=complete and all(r['physics_dt_s'] == self.clock.get_physics_dt() for r in rows))
        for p in self.geometry.targets:
            checks[p] = bool(rows) and all(r['contacts'][p]['supports'] == frozenset((self.geometry.cloth,))
                and not r['held'][p] and not any(r['contacts'][p][k] for k in ('robot_contact','on_floor','external_contact')) for r in rows)
            checks[p+':stable'] = bool(rows) and bool(np.ptp([r['poses'][p]['position_m'] for r in rows], axis=0).max()
                                                    <= self.config.initial_position_span_m)
        receipt = dict(accepted=all(checks.values()), checks=checks, evidence_source='native_physx',
                       scene_pose_writes=0, complete_manipulation_verified=False)
        # Preserve the exact acceptance predicate; expose its individual facts
        # once at admission rather than logging contacts every physics step.
        receipt['diagnostics'] = {
            'expected_support': self.geometry.cloth,
            'window_s': self.config.initial_history_s,
            'position_span_limit_m': self.config.initial_position_span_m,
            'rows': [dict(time_s=r['time_s'], physics_dt_s=r['physics_dt_s'],
                targets={p:dict(contacts=r['contacts'][p], held=bool(r['held'][p]),
                    pose=r['poses'][p], footprint=r['footprints'][p])
                    for p in self.geometry.targets}) for r in rows],
            'last_active_contacts': [
                dict(paths=paths, samples=[dict(normal=n, point=p, impulse=imp)
                    for n, p, imp in samples])
                for paths, _, _, samples in self.contacts.active.values()
                if any(path == t or path.startswith(t+'/')
                    for path in paths for t in self.geometry.targets)],
        }
        (self.output/'initial_acceptance.json').write_text(json.dumps(receipt, indent=2,
            default=lambda x: x.tolist() if hasattr(x, 'tolist') else list(x))+'\n')
        if not receipt['accepted']:
            raise RuntimeError('Cloth group initial rejected ' + str(checks))
        composite = ClothClearGroup(self.bindings)
        self.score = composite.observe(self.facts(self.latest, 0))
        self.composite, self.index, self.pending = composite, 0, []
        return receipt

    def selection_request(self):
        if self.selection_time is None or abs(self.selection_time-self.clock.current_time) > 1e-6:
            raise RuntimeError('Stale touch evidence')
        return self.contact_selection.request(ended=self.ended, time_s=self.clock.current_time)

    def consume_control(self, active):
        if active is not True or not self.pending:
            raise RuntimeError('Missing active physical evidence batch')
        rows, self.pending = self.pending, []
        for i, row in enumerate(rows):
            self.index += 1
            batch = self.facts(row, self.index)
            self.score = self.composite.observe(batch, final_sample=i == len(rows)-1)
            self.trace.write(json.dumps(dict(facts=[asdict(f) for f in batch], score=self.score), default=list)+'\n')
        self.score = dict(self.score, evidence_status='native_physx', native_full_trajectory_verified=False)
        self.trace.flush()
        return self.score

    def pause(self):
        self.composite.pause()

    def finalize(self, reason):
        self.pending = []
        self.score = dict(self.composite.finalize('success' if self.composite.reason == 'success' else reason),
                          evidence_status='native_physx', native_full_trajectory_verified=False)
        return self.score

    def close(self):
        if not self.closed:
            self.trace.close()
            self.closed = True
