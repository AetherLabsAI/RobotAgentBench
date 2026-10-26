"""Positive-only public scoring after canonical component validation.

Native component details remain unmodified diagnostic evidence. Only canonical
scores feed the benchmark result, final settlement and progress deadline.
"""
from copy import deepcopy
import math

VERSION = 'short_tasks.no_deductions.v1'
SUPPORTED = frozenset((
    'prepare_coffee', 'bake_bread', 'stock_wooden_tray', 'retrieve_from_drawer',
    'hang_cap', 'clear_table_trash', 'fridge_to_heated_box', 'clear_tablecloth',
    'pack_object', 'prepare_cutlery', 'carry_meal_tray', 'sofa_to_coffee_table',
    'table_to_storage_basket', 'sofa_to_drawer',
))


def descriptor():
    return dict(version=VERSION, enabled=False,
                final_state_revocation_preserved=True, success_unchanged=True,
                component_details='legacy_diagnostic_only')


def attach(plan):
    if plan.get('task_name') in SUPPORTED and plan.get('available'):
        plan['deduction_policy'] = descriptor()


def apply(plan, snapshot):
    return project(snapshot) if plan.get('task_name') in SUPPORTED else snapshot


def positive_scores(scores):
    keys = {'process', 'current_final', 'deductions', 'total', 'maximum'}
    if not isinstance(scores, dict) or set(scores) != keys:
        raise ValueError('Canonical validated score channels required')
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
           for v in scores.values()):
        raise ValueError('Finite nonnegative score channels required')
    gross = scores['process'] + scores['current_final']
    if gross > scores['maximum'] + 1e-9 or scores['total'] > scores['maximum'] + 1e-9:
        raise ValueError('Awards exceed the existing maximum')
    out = deepcopy(scores)
    # Use gross awards, never net + deductions: legacy clipping may have
    # discarded negative intermediate totals independently per component.
    out.update(deductions=0., total=gross)
    return out


def project(snapshot):
    if type(snapshot.get('success')) is not bool:
        raise ValueError('Success must come from the existing task predicate')
    out = deepcopy(snapshot)
    out['scores'] = positive_scores(snapshot['scores'])
    out['deduction_policy'] = descriptor()
    return out
