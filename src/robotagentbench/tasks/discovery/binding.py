"""Task-owned budget binding for an original, not-yet-accepted Discovery room.

Original initialization, contacts, physical intervals, clocks, coverage-to-port
mapping and terminal settlement are retained. No native modules are imported.
"""
from copy import deepcopy
import json
from pathlib import Path

from .policy import CoverageLedger, RULE_VERSION


def describe_plan(plan):
    if plan.get('task_name') != 'task_discovery':
        raise ValueError('Only Task Discovery receives the object touch budget')
    result = deepcopy(plan)
    result['rule_version'] = RULE_VERSION
    result['score_policy']['touch_budget'] = 'N+2 distinct physical objects'
    return result


def install(room):
    """Replace task-local rule state before the original accepted-state arm."""
    previous = room.discovery
    if (room.started or room.closed or room.faulted or room.pending is not None
            or previous.origin is not None or previous.final_result is not None
            or previous.target_events or previous.wrong_events):
        raise ValueError('Attach the approved budget only before initial acceptance')
    ledger = CoverageLedger(episode_id=previous.episode_id, targets=previous.targets,
        non_targets=previous.non_targets, fingers=previous.fingers)
    # Original four-scenario catalogue already groups collider children into
    # physical objects and deduplicates shared subtask targets.
    path = Path(room.output) / 'discovery_touch_budget_rule.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(dict(rule_version=RULE_VERSION, target_count=len(ledger.targets),
            touch_limit=ledger.touch_limit, non_target_score=0,
            repeated_object_contacts='count_once',
            simultaneous_contact_order='original_stable_object_id_order',
            contact_that_exhausts_budget_is_scored=True,
            unmapped_contact='invalid_missing_physical_object_binding',
            original_clock_timeout_physics_and_contacts_unchanged=True), stream, indent=2)
        stream.write('\n')
    room.discovery = ledger
    return room


class DiscoveryControlSession:
    """Compose the original session; do not patch the original room's methods."""
    def __init__(self, original):
        if not isinstance(original.room.discovery, CoverageLedger):
            raise ValueError('Task-owned budget must be bound before session creation')
        self._original = original
        self._error = None

    def __getattr__(self, name):
        return getattr(self._original, name)

    @property
    def faulted(self):
        return self._error is not None or self._original.faulted

    def result(self):
        result = self._original.result()
        result['rule_version'] = RULE_VERSION
        result['discovery'] = self.room.discovery.report()
        if self._error is not None:
            result.update(status='error', evaluation_valid=False, success=False,
                error=deepcopy(self._error))
        return result

    def finish_control(self, *, completed_substeps):
        if self.faulted:
            raise RuntimeError('No frame completion after a budget session fault')
        try:
            # This banks the last admitted physical interval and its final
            # target contact before budget termination. It preserves original
            # all-target/timeout priority and the source's unique-object map.
            self._original.finish_control(completed_substeps=completed_substeps)
            if self.room.discovery.budget_exhausted and not self.room.dispatch.ended:
                self.room.dispatch.ended = True
                self.room.dispatch.current = None
                self.room.dispatch.termination_reason = 'touch_budget_exhausted'
                self.room._settle_terminal()
            return self.result()
        except BaseException as error:
            self._error = dict(type=type(error).__name__, message=str(error))
            raise

    def begin_control(self, *args, **kwargs):
        if self.faulted:
            raise RuntimeError('No control after a budget session fault')
        return self._original.begin_control(*args, **kwargs)

    def after_physics(self, *args, **kwargs):
        if self.faulted:
            raise RuntimeError('No physics evidence after a budget session fault')
        return self._original.after_physics(*args, **kwargs)

    def close(self):
        return self._original.close()
