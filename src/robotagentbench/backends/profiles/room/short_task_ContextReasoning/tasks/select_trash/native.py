"""Native contact-only port. No manipulation helper or state teleportation."""
from collections import deque
from copy import deepcopy
from pathlib import Path
import json
import math
from .task import NAME, NativeEvidence
from short_tasks.common.episode import SingleTaskEpisode, NO_PROGRESS_TIMEOUT_S
from short_tasks.runtime import ShortTaskSession

class SelectionEpisode(SingleTaskEpisode):
    def observe(self, snapshot, *, simulation_time_s, finalizer=None):
        if snapshot.get('failure_reason') != 'wrong_candidate_touch':
            return super().observe(snapshot, simulation_time_s=simulation_time_s, finalizer=finalizer)
        if self.terminal is not None:
            raise RuntimeError('Finalized episode cannot accept another action')
        self._time(simulation_time_s)
        if simulation_time_s <= self.last_time:
            raise ValueError('Stale failure evidence')
        self._snapshot(snapshot)
        if simulation_time_s > self.progress_time + NO_PROGRESS_TIMEOUT_S + 1e-9:
            return super().observe(snapshot, simulation_time_s=simulation_time_s, finalizer=finalizer)
        previous = self.last_time, self.latest
        self.last_time, self.latest = float(simulation_time_s), deepcopy(snapshot)
        try:
            return self.finish('wrong_candidate_touch', finalizer=finalizer)
        except BaseException:
            self.last_time, self.latest = previous
            raise

class SelectionSession(ShortTaskSession):
    def accept_initial(self):
        receipt = super().accept_initial()
        previous = self.ledger
        # Preserve the freshly accepted clock/snapshot. This is a task-owned
        # ledger specialization, not a reset or mutation of shared task rules.
        ledger = SelectionEpisode(task_name=previous.task_name, episode_id=previous.episode_id,
            actual_layout_id=previous.layout, rule_version=previous.rule_version,
            start_time_s=previous.start_time, target_roles=previous.target_roles,
            maximum_score=previous.maximum, score_arithmetic=previous.score_arithmetic)
        ledger.latest = deepcopy(previous.latest)
        self.ledger = ledger
        return receipt

class SelectionPort:
    def __init__(self, stage, selection, output, episode_id, *, evidence_factory=NativeEvidence,
                 support_root='/World/Assets/CoffeeTable', support_roots=None):
        from pxr import Usd, UsdPhysics, PhysxSchema, PhysicsSchemaTools
        from omni.physx import get_physx_simulation_interface
        from omni.physx.bindings._physx import ContactEventType
        from evaluation.tasks.tidy_sofa.isaac_reader import Contacts, ROBOT
        from openend.tasks.holding_follow_v1.detection import DetectionConfig
        self.config = DetectionConfig()
        self.stage, self.selection = stage, selection
        self.support_root = support_root
        self.support_roots = tuple(support_roots or (support_root,))
        self.interface = get_physx_simulation_interface()
        self.bodies = {}
        for row in selection['candidates']:
            bodies = [p for p in Usd.PrimRange(stage.GetPrimAtPath(row['path'])) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
            if len(bodies) != 1:
                raise ValueError('Each candidate must bind one native rigid body')
            body = bodies[0]
            if UsdPhysics.RigidBodyAPI(body).GetKinematicEnabledAttr().Get():
                raise ValueError('Candidate must remain dynamic')
            self.bodies[row['instance_id']] = str(body.GetPath())
        if not stage.GetSessionLayer().anonymous:
            raise ValueError('Anonymous instrumentation layer required')
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            bodies = [stage.GetPrimAtPath(p) for p in self.bodies.values()]
            bodies += [p for p in Usd.PrimRange(stage.GetPrimAtPath(ROBOT)) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
            for body in bodies:
                a = body.GetAttribute('physxContactReport:threshold')
                if a and a.HasAuthoredValueOpinion() and a.Get() != 0:
                    raise ValueError('Conflicting contact instrumentation')
                PhysxSchema.PhysxContactReportAPI.Apply(body).CreateThresholdAttr(0.)
        def sleeping(sid, actor):
            prim = stage.GetPrimAtPath(PhysicsSchemaTools.intToSdfPath(actor))
            if not prim:
                raise RuntimeError('Unknown contact actor')
            return bool(not prim.HasAPI(UsdPhysics.RigidBodyAPI) or self.interface.is_sleeping(sid, actor))
        self.contacts = Contacts(PhysicsSchemaTools.intToSdfPath, sleeping,
            (ContactEventType.CONTACT_FOUND, ContactEventType.CONTACT_PERSIST, ContactEventType.CONTACT_LOST),
            tuple(a['path'] for a in selection['candidates']))
        self.evidence = evidence_factory(stage, selection, episode_id)
        self.output = Path(output)
        self.output.mkdir(exist_ok=False)
        self.trace = (self.output/'facts.jsonl').open('x')
        self.history = deque(maxlen=math.ceil(self.config.initial_history_s*240)+2)
        self.score = deepcopy(self.evidence.evaluator.result)
        self.ended = self.closed = self.paused = False

    def initialize(self, clock, view):
        self.clock = clock
        self.views = {k: view.create_rigid_body_view(p) for k, p in self.bodies.items()}
        if any(v.count != 1 or list(v.prim_paths) != [self.bodies[k]] for k,v in self.views.items()):
            raise ValueError('Ambiguous native body view')

    def after_physics(self, helper_report):
        from openend.tasks.glasses_transfer.contacts import reduce_pairs
        report = helper_report.get('_contact_report')
        if report is None:
            report = self.interface.get_contact_report()
        now = float(self.clock.current_time)
        if not self.evidence.evaluator.armed:
            self.contacts.ingest(report)
            pairs = tuple(self.contacts.pairs())
            values = {}
            for a in self.selection['candidates']:
                c = reduce_pairs(pairs, target=a['path'], source=self.support_root,
                    bag=None, cabinet=None, floors=(), robot='/World/Robot', pads={})
                values[a['instance_id']] = dict(position=self.views[a['instance_id']].get_transforms()[0,:3].tolist(),
                    supported=any(s==root or s.startswith(root+'/') for s in c['supports']
                                  for root in self.support_roots), robot_contact=c['robot_contact'])
            self.history.append(dict(time_s=now, physics_dt_s=1/self.clock.hz, values=values))
            self.evidence.read_touches(report)
        elif not self.paused:
            before = self.score['terminated']
            self.score = self.evidence.after_physics(report, now)
            if self.score['terminated'] and not before:
                self.trace.write(json.dumps(dict(time_s=now, evidence='native_physx_finger_contact', **self.score))+'\n')
                self.trace.flush()

    def accept_initial(self):
        import numpy as np
        from openend.tasks.meal_tray.nonmotion_closeout_v1.initial_windows import initial_window
        rows, complete = initial_window(self.history, self.config.initial_history_s, 1/self.clock.hz)
        checks = {}
        for key in self.bodies:
            values = [r['values'][key] for r in rows]
            checks[key] = dict(supported=bool(values) and all(v['supported'] for v in values),
                no_robot_contact=bool(values) and not any(v['robot_contact'] for v in values),
                stable=bool(values) and bool(np.ptp([v['position'] for v in values], axis=0).max() <= self.config.initial_position_span_m))
        receipt = dict(accepted=complete and all(all(v.values()) for v in checks.values()),
            history_complete=complete, checks=checks, evidence_source='native_physx',
            candidate_count=len(checks), physics_parameters_changed=False)
        (self.output/'initial_acceptance.json').write_text(json.dumps(receipt, indent=2)+'\n')
        if not receipt['accepted']:
            raise RuntimeError('Selection initial state rejected: '+str(checks))
        from .task import touched_instances
        touches = touched_instances(self.evidence.native.contacts.pairs(), self.evidence.native.fingers, self.evidence.bindings)
        self.evidence.evaluator.arm(float(self.clock.current_time), initial_touches=touches)
        return receipt

    def consume_control(self, active):
        return deepcopy(self.score)

    def begin_selected_control(self, active):
        self.paused = False

    def finalize(self, reason):
        self.ended = True
        self.score = self.evidence.evaluator.finish(reason)
        return deepcopy(self.score)

    def pause(self):
        self.paused = True

    def after_inactive(self, report):
        self.paused = True
        self.evidence.read_touches(report.get('_contact_report') or self.interface.get_contact_report())

    def close(self):
        if not self.closed:
            self.trace.close()
            self.closed = True

def build_selection_session(plan, stage, manifest, output, *, episode_id, workspace_selector, sensor):
    from short_tasks import registry
    from short_tasks.scene import verify_scene_sources, verify_manifest
    verify_scene_sources()
    verify_manifest(manifest)
    plan['source_receipt'] = registry.verify_sources(plan)
    if workspace_selector is None:
        if sensor is None:
            raise ValueError('Existing calibrated workspace sensor required')
        from openend.tasks.all_objects_integration.workspace_physics import from_scene
        workspace_selector = from_scene(stage, manifest, sensor)
    output.mkdir(parents=True, exist_ok=False)
    port = SelectionPort(stage, plan['selection'], output/NAME, episode_id)
    return SelectionSession(plan, manifest, {NAME:port}, episode_id=episode_id,
                            workspace_selector=workspace_selector)

def build_session(request, stage, manifest, output, *, episode_id, workspace_selector=None,
                  sensor=None, runtime_settings=None):
    from short_tasks import registry
    plan = registry.bind(registry.resolve(request), manifest)
    return build_selection_session(plan, stage, manifest, Path(output), episode_id=episode_id,
                                   workspace_selector=workspace_selector, sensor=sensor)
