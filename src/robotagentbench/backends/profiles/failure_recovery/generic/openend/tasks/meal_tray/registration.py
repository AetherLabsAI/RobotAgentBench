"""Meal task dependency/identity preflight. Not a native port registration.

Input tasks use the existing Openend instance_id/type/parameters contract.
Sequential preparation and serving must retain one scene-owned target.
"""
from copy import deepcopy

DEFINITIONS = {
    'coffee_to_meal_tray': dict(name='咖啡摆盘', prerequisite='coffee',
        target_asset='coffee_cup_blue_compact', source_asset='coffee_machine',
        pickup_component='1-4', pickup_points=5, placement_points=3,
        prerequisite_maximum=16),
    'bread_to_meal_tray': dict(name='面包摆盘', prerequisite='fryer',
        source_asset='air_fryer', pickup_component='2.5', pickup_points=4,
        placement_points=3, prerequisite_maximum=8),
    'microwave_box_to_meal_tray': dict(name='微波炉取盒摆盘', prerequisite='fridge_to_microwave',
        source_asset='microwave', placement_points=3, prerequisite_maximum=18),
}


def dependencies(tasks):
    """Require the user-selected preparation; never synthesize a scored task."""
    if not isinstance(tasks, list):
        raise ValueError('Explicit task list required')
    ids = [t.get('instance_id') for t in tasks]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Distinct task instance IDs required')
    result = []
    for t in tasks:
        kind = t.get('type')
        if kind not in DEFINITIONS:
            continue
        if set(t) != {'instance_id', 'type', 'parameters'}:
            raise ValueError('Meal task uses scene identities, no parameter overrides')
        if kind=='microwave_box_to_meal_tray':
            if (set(t['parameters'])!={'payload_variant'} or
                    t['parameters']['payload_variant'] not in ('beef_box','leftover_box')):
                raise ValueError('Explicit microwave payload_variant required')
        elif t['parameters'] != {}:
            raise ValueError('Meal task uses scene identities, no parameter overrides')
        d = DEFINITIONS[kind]
        peers = [p for p in tasks if p.get('type') == kind]
        upstream = [p for p in tasks if p.get('type') == d['prerequisite']]
        if len(peers) != 1 or len(upstream) != 1:
            raise ValueError(d['name']+'必须同时且唯一注册前置任务：'+d['prerequisite'])
        p = upstream[0]
        if kind == 'microwave_box_to_meal_tray':
            if p.get('parameters') != t['parameters']:
                raise ValueError('冰箱交接与微波炉取盒必须选择同一种盒子')
            asset=t['parameters']['payload_variant']
        elif kind == 'bread_to_meal_tray':
            if (set(p.get('parameters', {})) != {'bread_variant'}
                    or p['parameters']['bread_variant'] not in ('no_sugar', 'sugar')):
                raise ValueError('Use explicit existing fryer bread_variant')
            asset = 'cake_plain_yellow' if p['parameters']['bread_variant']=='no_sugar' else 'cake_frosted_white'
        else:
            if p.get('parameters') != {}:
                raise ValueError('No coffee preparation overrides')
            asset = d['target_asset']
        result.append(dict(instance_id=t['instance_id'], type=kind,
            prerequisite_instance_id=p['instance_id'], prerequisite_type=d['prerequisite'],
            target_asset=asset, source_asset=d['source_asset'], destination_asset='meal_tray_large',
            score_owner=dict(preparation=p['instance_id'], pickup_and_placement=t['instance_id']),
            scene_initialization='existing_preparation_start_not_prefilled',
            reset_at_handoff=False))
    return result


def bind(tasks, manifest):
    edges = dependencies(tasks)
    if not edges:
        return dict(bindings=[], runtime_registered=False, native_ready=False)
    def one(asset):
        rows = [o for o in manifest['objects'] if o.get('id') == asset]
        if len(rows) != 1:
            raise ValueError('One exact scene-owned instance required: '+asset)
        row = rows[0]
        from .scoring import path
        if not path(row.get('path')):
            raise ValueError('Canonical scene path required: '+asset)
        return row['path']
    claimed = set()
    out = []
    for edge in edges:
        row = deepcopy(edge)
        row.update(target_wrapper=one(edge['target_asset']), source_station_wrapper=one(edge['source_asset']),
                   destination_wrapper=one(edge['destination_asset']),
                   pickup_source_role={'coffee_to_meal_tray':'machine','bread_to_meal_tray':'basket',
                       'microwave_box_to_meal_tray':'microwave'}[edge['type']])
        if row['target_wrapper'] in claimed:
            raise ValueError('Two meal tasks cannot score the same target')
        if len({row[k] for k in ('target_wrapper','source_station_wrapper','destination_wrapper')}) != 3:
            raise ValueError('Overlapping meal task roles')
        claimed.add(row['target_wrapper']); out.append(row)
    return dict(bindings=out, shared_meal_tray=one('meal_tray_large'),
        runtime_registered=False, native_ready=False,
        pending=['native_body_and_pickup_source_role_resolution',
                 'shared_native_device_owner_and_meal_ports', 'scene_generation_and_web_wiring',
                 'native_settling_and_full_action_validation'])
