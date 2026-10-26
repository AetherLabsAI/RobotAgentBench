"""Terminal completion evidence from an already measured native FULL sample.

No physics step, native scorer observe, score write, or timestamp substitution.
The original port finalizer still owns its pending buffer, deductions and result.
"""
from copy import copy, deepcopy
import math

KINDS = frozenset(('book_transfer', 'wallet_transfer', 'cap_transfer',
                   'glasses_transfer', 'trash_transfer', 'coffee', 'fryer'))


def _fresh(sample_time, now):
    if any(type(t) not in (int, float) or not math.isfinite(t) for t in (sample_time, now)) or sample_time != now:
        raise ValueError('Terminal native FULL sample is missing or stale')


def _row(met, now, source):
    if type(met) is not bool:
        raise ValueError('Typed original completion predicate required')
    return dict(available=True, met=met, simulation_time_s=now, source=source,
                scoring_changed=False, terminal_rule_changed=False)


def capture(port, kind):
    """Run once before original finalize; current native facts remain private."""
    now = port.clock.current_time
    previous = getattr(port, '_long_terminal_current', None)
    if isinstance(previous, dict) and previous.get('simulation_time_s') == now:
        return deepcopy(previous)
    component = port.composite
    try:
        if kind == 'coffee':
            if not component.component.success:
                return _row(False, now, 'required_coffee_process_incomplete')
            _fresh(port.owner.service.latest['sample'].time_s, now)
            if getattr(port, '_long_current_observed_time', None) == now:
                current = deepcopy(component.current_result)
            else:
                facts = port.owner.service.facts(port.index + 1, component.facts_phase)
                current = component._current(facts)
            result = _row(current['valid'], now, 'original_prepare_coffee_terminal_current')
            result['original_predicate'] = current
            return result
        if kind == 'fryer':
            if not component.success:
                return _row(False, now, 'required_fryer_process_incomplete')
            _fresh(port.source.latest['time_s'], now)
            current = getattr(port, '_long_current_result', None)
            if isinstance(current, dict) and current.get('simulation_time_s') == now:
                return deepcopy(current)
            from .current_devices import fryer_current_result
            facts = port.source.selected_facts(port.index + 1)
            return fryer_current_result(history_success=True, facts=facts,
                sample_time_s=port.source.latest['time_s'], current_time_s=now)
        if kind not in KINDS:
            raise ValueError('Unregistered terminal completion adapter')
        # Do not let a shadow placement acquire a new process milestone.
        if port.score.get('operation_complete') is False:
            return _row(False, now, 'required_operation_incomplete')
        if port.score.get('operation_complete') is not True:
            raise ValueError('Original typed operation completion required')
        latest = port.latest
        _fresh(latest['time_s'], now)
        if getattr(component.last, 'time_s', None) == now:
            return _row(port.score['success'], now, 'original_current_component_success')
        facts = port.facts(latest, True, port.index + 1)
        if hasattr(component, 'destination_state'):
            # Exact WalletTransfer/GlassesTransfer source final channel. Its
            # existing policy permits held objects after placement completed.
            destination = deepcopy(component.destination_state)
            supported = (bool(facts.supports) if component.allow_packed_support
                else facts.supports == frozenset((component.binding.bag,)))
            destination.observe(process_complete=bool(component.pick.ever_complete),
                located=bool(supported and facts.inside_cavity and not facts.rim_only and not facts.on_floor),
                released=bool(not facts.robot_contact and not facts.held))
            return _row(destination.met, now, 'original_destination_state_terminal_current')
        # Deepcopy ONLY placement: LegacyPickup owns a Python module; copying
        # it would fail and could alter the original finalize isinstance branch.
        placement = component.place
        penalties = getattr(placement, 'penalties', None)
        frozen_spec = getattr(penalties, 'spec', None)
        shadow = copy(component)
        shadow.place = deepcopy(placement, {} if frozen_spec is None else {id(frozen_spec): frozen_spec})
        shadow.last = facts
        shadow.current_success = shadow.place.observe(facts, component.b,
            enabled=bool(component.pickup_complete), final_sample=True)
        # Reuse the exact subclass summary: Short book/wallet require release,
        # while the original cap/trash predicates have their own semantics.
        return _row(shadow.summary()['success'], now, 'original_placement_terminal_current')
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        return dict(available=False, met=None, simulation_time_s=now,
            reason='terminal_native_current_unavailable', error_type=type(error).__name__, error=str(error))


def terminal_port(original, kind):
    if kind not in KINDS:
        raise ValueError('Unsupported terminal port kind')

    class TerminalCurrentPort(original):
        def finalize(self, reason):
            # Cache before original finalization clears pending native evidence.
            # Repeated result() never invokes a reader or consumes contact windows.
            self._long_terminal_current = capture(self, kind)
            return super().finalize(reason)

    return TerminalCurrentPort


def wrap_factories(compiled, overrides):
    from openend.room import FACTORIES
    result = dict(overrides)
    for task in compiled['runtime_request']['tasks']:
        key, kind = task['instance_id'], task['type']
        if kind in KINDS:
            result[key] = terminal_port(result.get(key, FACTORIES[kind]), kind)
    return result
