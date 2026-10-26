"""Remove numerical deductions at OpenEnd's live score submission boundary.

Native evidence, earned milestones, prerequisites, maxima, success/failure and
termination flags remain unchanged. This module never controls physics and never
mutates a shared component scorer. Raw totals are rebuilt BEFORE zero clamping.
"""
from copy import deepcopy
import math

RULE_VERSION = 'openend.positive_only.additive_target_contact_1pt.v1'
COMPONENTS = ('pickup', 'placement', 'pick', 'place', 'operations', 'refrigerator', 'microwave')
PROVENANCE = frozenset(('events', 'contact_credit', 'handoff', 'fridge_handoff', 'transitions'))


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def penalty_key(key):
    return 'deduct' in key or 'penalt' in key


def own_deduction(node):
    # Prefer an explicit aggregate over its itemization; never count both a
    # shelf collision total and its neighbor/shelf diagnostic subtotals.
    for key in ('deduction_total', 'total_deductions', 'deducted_points'):
        if number(node.get(key)):
            return node[key]
    if number(node.get('deductions')):
        return node['deductions']
    if isinstance(node.get('deductions'), list):
        return sum(item.get('points', item.get('amount', 0)) for item in node['deductions'])
    if isinstance(node.get('deductions'), dict):
        return sum(value for value in node['deductions'].values() if number(value))
    return sum(value for key, value in node.items() if penalty_key(key) and number(value)
               and key not in ('neighbor_collision_deduction', 'shelf_collision_deduction'))


def _walk(node, fridge_points):
    if isinstance(node, list):
        return [_walk(value, fridge_points) for value in node]
    if not isinstance(node, dict):
        return node
    old = node
    new = {key: deepcopy(value) if key in PROVENANCE else _walk(value, fridge_points)
           for key, value in old.items()}
    own = own_deduction(old)
    if own < 0:
        raise ValueError('Deductions must be nonnegative')
    component_keys = [key for key in COMPONENTS if isinstance(old.get(key), dict)]
    raw_change = sum(new[key].get('raw_score', new[key].get('score', 0))
                     - old[key].get('raw_score', old[key].get('score', 0)) for key in component_keys)
    net_change = sum(new[key].get('score', 0) - old[key].get('score', 0) for key in component_keys)

    if number(old.get('raw_score')):
        if component_keys:
            raw = old['raw_score'] + raw_change + own
        elif number(old.get('earned_points')):
            raw = old['earned_points']
        elif number(old.get('earned')):
            raw = old['earned']
        else:
            raw = old['raw_score'] + own
        new['raw_score'] = raw
        if 'score' in old:
            new['score'] = max(0, raw)
    elif old.get('schema') == 'openend.meal_preparation.v1':
        # Original preparation sums phase RAW scores and only then clamps.
        new['score'] = max(0, sum(phase['raw_score'] for phase in new['phase_results']))
    elif 'illumination_completed' in old and 'extraction_score' in old:
        # Keep the original completed-illumination prerequisite. Removing a
        # penalty is not permission to award unearned or gated operations.
        operation = sum(old['awards'].values()) if old['illumination_completed'] else 0
        illumination = old['illumination_score'] if old['illumination_completed'] else 0
        new['extraction_score'] = operation
        new['operation_score'] = operation + illumination
        new['score'] = operation + illumination + new['final_state']['score'] if old['illumination_completed'] else 0
    elif 'hold_failure_deduction' in old and isinstance(old.get('stages'), (tuple, list)):
        # The reused refrigerator Result has no raw or earned field. Adding a
        # deduction back to its CLAMPED score would invent points at reset.
        if not fridge_points or set(old['stages']) - set(fridge_points):
            raise ValueError('Original refrigerator stage-point contract required')
        new['score'] = sum(fridge_points[stage] for stage in old['stages'])
    elif number(old.get('score')):
        if own:
            # Other currently supported leaf summaries must expose their gross
            # score. Refuse to guess an unknown clamped arithmetic contract.
            if number(old.get('earned_points')):
                new['score'] = old['earned_points']
            elif number(old.get('earned')):
                new['score'] = old['earned']
            else:
                raise ValueError('Unrecognized penalty-bearing score summary')
        else:
            new['score'] = old['score'] + net_change

    if 'operation_score' in old and not ('illumination_completed' in old and 'extraction_score' in old):
        for key in ('operations', 'pickup'):
            if key in component_keys:
                new['operation_score'] = old['operation_score'] + new[key].get('score', 0) - old[key].get('score', 0)
                break
    for key, value in old.items():
        if penalty_key(key):
            if number(value):
                new[key] = 0
            elif key == 'deductions' and isinstance(value, list):
                new[key] = []
            elif key == 'deductions' and isinstance(value, dict):
                if not all(type(points) is bool or number(points) and points >= 0 for points in value.values()):
                    raise ValueError('Unrecognized deduction itemization')
                # Wipe uses boolean condition flags, not numerical charges.
                # Preserve those physical diagnostics; its deducted_points=0.
                new[key] = {name: points if type(points) is bool else 0 for name, points in value.items()}
    if number(new.get('score')):
        if new['score'] < -1e-8 or (number(new.get('maximum_score')) and new['score'] > new['maximum_score'] + 1e-8):
            raise ValueError('Positive score exceeds original score budget')
    for key in ('success', 'failure', 'ended', 'finalized', 'operation_complete', 'termination', 'reason'):
        if key in old and new[key] != old[key]:
            raise AssertionError('Physical/lifecycle predicate changed: ' + key)
    return new


def positive_score(native, *, fridge_points):
    if not isinstance(native, dict) or 'contact_credit' in native:
        raise ValueError('Unaugmented original native task result required')
    result = _walk(native, fridge_points)
    result['score_policy'] = dict(rule_version=RULE_VERSION, numerical_deductions_enabled=False,
                                original_earned_points_and_physical_predicates_unchanged=True)
    return result


class PositiveLedger:
    """Explicit composition, scoped to the OpenEnd factory, not a global patch."""
    def __init__(self, native_ledger, *, fridge_points):
        self.native_ledger = native_ledger
        self.fridge_points = dict(fridge_points)

    def __getattr__(self, name):
        return getattr(self.native_ledger, name)

    def augment(self, key, native):
        result = self.native_ledger.augment(key, positive_score(native, fridge_points=self.fridge_points))
        result['contact_credit']['rule_version'] = RULE_VERSION
        result['contact_credit']['completion_and_native_penalties_unchanged'] = False
        result['contact_credit']['physical_completion_unchanged'] = True
        result['contact_credit']['numerical_deductions_enabled'] = False
        return result

    def finalize(self):
        result = self.native_ledger.finalize()
        result['rule_version'] = RULE_VERSION
        result['numerical_deductions_enabled'] = False
        return result
