"""Common supplementary policy, isolated from component scoring versions."""
from copy import deepcopy
from .scoring import VERSION, APPROACH_TIERS, APPROACH_POINTS, TOUCH_POINTS

SIMPLE = frozenset(('hang_cap', 'pack_object', 'carry_meal_tray',
    'sofa_to_coffee_table', 'table_to_storage_basket'))
MULTI = frozenset(('stock_wooden_tray', 'clear_table_trash',
    'clear_tablecloth', 'prepare_cutlery'))
SPECIAL_COUNTS = {'prepare_coffee': 3, 'bake_bread': 2,
    'fridge_to_heated_box': 3, 'retrieve_from_drawer': 2, 'sofa_to_drawer': 2}
SUPPORTED = SIMPLE | MULTI | SPECIAL_COUNTS.keys()


def attach(plan):
    if not plan['available'] or plan['task_name'] not in SUPPORTED:
        return plan
    if 'interaction_policy' in plan:
        raise ValueError('Interaction policy attached twice')
    name=plan['task_name']
    count=(len(plan['target_categories']) if name in MULTI else
           SPECIAL_COUNTS.get(name,1))
    if not count:raise ValueError('Empty interaction target set')
    plan['base_score_policy']=deepcopy(plan['score_policy'])
    plan['interaction_policy']=dict(version=VERSION, target_count=count,
        approach_points=APPROACH_POINTS, touch_points=TOUCH_POINTS,
        approach_tiers=[dict(distance_m=d, points=p) for d,p in APPROACH_TIERS],
        initially_satisfied_tiers='skipped_without_credit',
        maximum=1.5*count, once_per_episode_target=True,
        success_unchanged=True, initial_credit=False)
    plan['score_policy']['maximum']+=1.5*count
    plan['score_policy']['process_maximum']+=1.5*count
    return plan


def component_plan(plan):
    out=deepcopy(plan)
    if 'interaction_policy' in out:
        out['score_policy']=deepcopy(out['base_score_policy'])
    return out


def active_roles(name, port):
    """Read the existing state machine; never advance or unlock a stage."""
    if name=='prepare_coffee':
        return {'1-1':{'cup'},'1-2':{'bag'},'1-3':{'button'},'1-4':{'cup'},'complete':set()}[port.composite.phase]
    if name=='bake_bread':
        # The insert phase requires opening the basket and loading the cake.
        phase=port.composite.phase
        return {'2.1':{'basket','target'},'2.2':{'basket','target'},
                '2.3':{'basket'},'2.5':{'target'},'complete':set()}[phase]
    if name=='fridge_to_heated_box':
        c=port.composite
        if c.phase=='microwave':return {'microwave_handle','target'}
        if c.phase!='refrigerator':raise ValueError('Unknown refrigerator phase')
        stages=set(c.fridge._stages)
        return {'fridge_handle'} | ({'target'} if 'open' in stages else set())
    if name=='retrieve_from_drawer':
        return {'target'} if port.short_drawer.opened else {'drawer'}
    if name=='sofa_to_drawer':
        return {'target','drawer'} if getattr(port.composite,'pickup_complete',False) else {'target'}
    return None
