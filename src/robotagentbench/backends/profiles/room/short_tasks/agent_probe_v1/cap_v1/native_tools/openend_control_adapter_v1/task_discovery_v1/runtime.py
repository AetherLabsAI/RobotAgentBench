"""Discovery-only scoring over the full native OpenEnd controls/contact stream."""
from copy import deepcopy
import json
from openend.room import MultiTaskRoom
from openend.dispatch import MultiTaskDispatcher
from .policy import CoverageLedger, RULE_VERSION


class DiscoveryDispatcher(MultiTaskDispatcher):
    def _timer(self):
        result = super()._timer()
        result['timed_out'] = self.ended and self.termination_reason == 'no_score_increase_timeout'
        return result

    def record_discovery(self, scores):
        # Discovery evidence has no operation-phase/awake-task prerequisite.
        # Use the original native clock/deadline and atomic progress accounting.
        if self.ended or not self.pending or self.scored:
            raise RuntimeError('One on-time discovery batch per physical frame')
        if set(scores) != set(self._scores):
            raise ValueError('All frozen object owners required')
        if any(type(r['score']) is not int or r['score'] not in (0, 1) for r in scores.values()):
            raise ValueError('One deduplicated point per physical target')
        before = self.total_score
        self._scores = deepcopy(scores)
        self.scored = True
        after = self.total_score
        if after > before:
            self.last_progress_time = self.last_time
            self.progress.append(dict(instance_ids=sorted(scores), time_s=self.last_time,
                previous_total_score=before, total_score=after, delta=after-before))


class DiscoveryRoom(MultiTaskRoom):
    def __init__(self, stage, manifest, output, episode_id, *, request, sensor, workspace_selector):
        super().__init__(stage, manifest, output, episode_id, request=request,
                         sensor=sensor, workspace_selector=workspace_selector)
        try:
            # Preserve the existing OpenEnd bag support semantics, not a new
            # asset/physics override. Discovery never calls operation scorers.
            from ..bag_support_v2.policy import enable_for_ports
            registration = json.loads((self.output/'registration.json').read_text())
            kinds = {t['instance_id']: t['type'] for t in registration['request']['tasks']}
            allowed = {'wallet_transfer','powerbank_transfer','water_bottle_transfer',
                       'glasses_transfer','book_transfer','remote_transfer'}
            if set(kinds.values()) - allowed:
                raise ValueError('This pilot admits only the verified outing portable-object mapping')
            support = enable_for_ports(kinds, self.ports)
            targets = {key: (self.ports[key].geometry.target,) for key in kinds}
            if len({v[0] for v in targets.values()}) != len(targets):
                raise ValueError('Duplicate physical target ownership')
            target_paths = {v[0] for v in targets.values()}
            background = {row['path']: (row['path'],) for row in manifest['objects']
                          if row['path'] not in target_paths}
            # Include visible furniture/destinations omitted from object rows.
            # Group collider descendants into their actual authored object root.
            from pxr import UsdPhysics
            for prim in stage.Traverse():
                path = str(prim.GetPath())
                if not prim.HasAPI(UsdPhysics.CollisionAPI) or path.startswith('/World/Robot/'):
                    continue
                if any(path == root or path.startswith(root+'/')
                       for root in [*target_paths, *background]):
                    continue
                background[path] = (path,)
            self.discovery = CoverageLedger(episode_id=episode_id, targets=targets,
                non_targets=background, fingers=self.contact_routes.native.fingers)
            old = self.dispatch
            self.dispatch = DiscoveryDispatcher(old.registrations, sector_radius=old.radius,
                sector_half_angle=old.half, no_progress_timeout_s=old.timeout, selection_order=old.order)
            self.event_trace = (self.output/'discovery_events.jsonl').open('x')
            (self.output/'discovery_rule.json').write_text(json.dumps(dict(rule_version=RULE_VERSION,
                targets=targets, non_targets=background, denominator=len(targets),
                operation_points_enabled=False, wrong_touch_penalty=0,
                bag_support_policy=support, native_controls_and_workspace_rates_unchanged=True),indent=2)+'\n')
        except BaseException:
            super().close()
            raise

    def accept_initial(self):
        receipt = super().accept_initial()
        self.discovery.arm(self.clock.current_time)
        return receipt

    def _after_physics(self, helper_report):
        if not self.started:
            return super()._after_physics(helper_report)
        if self.closed or self.faulted or self.pending is None:
            raise RuntimeError('Substep outside a live native action')
        if self.fridge_helper is not None:
            self.fridge_helper.after_physics(self.clock.hz)
        report = helper_report.get('_contact_report')
        if report is None:
            report = self.contact_routes.interface.get_contact_report()
        self.contact_routes.after_physics(report, self.clock.current_time)
        now = self.clock.current_time
        # A physical interval arriving after the original deadline is not banked.
        on_time = (self.pending[0] + .05 <=
                   self.dispatch.last_progress_time + self.dispatch.timeout + 1e-9)
        events = self.discovery.observe(self.contact_routes.native.contacts.pairs(),
            episode_id=self.discovery.episode_id, time_s=now, accepting_contacts=on_time)
        for event in events:
            self.event_trace.write(json.dumps(event)+'\n')
        if events:
            self.event_trace.flush()

    def discovery_scores(self):
        return {key: dict(score=int(key in self.discovery.target_events), maximum_score=1,
                          started=key in self.discovery.target_events,
                          success=key in self.discovery.target_events)
                for key in self.discovery.targets}

    def finish_control(self, *, completed_substeps):
        if self.pending is None or self.faulted:
            raise RuntimeError('No pending native frame')
        start, plan = self.pending
        if (completed_substeps != plan['substeps'] or self.clock.hz != plan['physics_hz']
                or abs(self.clock.current_time-start-.05) > 1e-6):
            self.faulted = True
            raise RuntimeError('Missing substeps or broken native 20Hz interval')
        try:
            timing = self.dispatch.complete_physics_frame(simulation_time_s=self.clock.current_time)
            if not timing['timed_out']:
                self.dispatch.record_discovery(self.discovery_scores())
                timing = self.dispatch.finish_frame()
                if self.discovery.report()['all_targets_contacted']:
                    self.dispatch.ended = True
                    self.dispatch.current = None
                    self.dispatch.termination_reason = 'all_targets_contacted'
            if self.dispatch.ended:
                timing = self._settle_terminal()
            result = dict(plan=plan, outcome=timing, discovery=self.discovery.report())
            self.trace.write(json.dumps(result, default=list)+'\n'); self.trace.flush()
        except BaseException:
            self.faulted = True
            raise
        self.pending = None; self.frame += 1
        return result

    def _settle_terminal(self):
        if hasattr(self, 'terminal_result'):
            return self.terminal_result
        final = self.discovery.finalize(evaluation_valid=True,
            termination_reason=self.dispatch.termination_reason)
        self.terminal_result = dict(self.dispatch.settle_terminal_scores(self.discovery_scores()),
                                   discovery=final, success=final['success'])
        (self.output/'final_result.json').write_text(json.dumps(self.terminal_result,indent=2)+'\n')
        return self.terminal_result

    def close(self):
        try:
            super().close()
        finally:
            stream = getattr(self, 'event_trace', None)
            if stream is not None:
                stream.close()


from ..session_adapter import OpenEndControlSession


class DiscoverySession(OpenEndControlSession):
    def result(self):
        result = super().result()
        coverage = self.room.discovery.report()
        # Native transport/cleanup validity still gates the persisted evaluation.
        result.update(discovery=coverage, success=bool(result['evaluation_valid'] and
            coverage.get('success')), scores=dict(total=self.room.dispatch.total_score,
            maximum=len(self.room.discovery.targets)), rule_version=RULE_VERSION)
        return result
