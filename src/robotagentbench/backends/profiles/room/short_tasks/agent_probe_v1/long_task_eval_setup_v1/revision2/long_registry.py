"""Trusted document-15..22 recipes; original OpenEnd room owns scores/clock.

Installed at short_tasks/agent_probe_v1/long_task_eval_setup_v1/revision2.
No arbitrary factory/module/physics/rule overrides are accepted from a request.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

RULE_VERSION = 'long_task_eval_setup_v1.original_openend.v2'
NAMES = {'15': 'Stock the tray (Multi)', '16': 'Tidy the sofa area',
         '17': 'Tidy the table', '18': 'Replace the tablecloth',
         '19': 'Find and pack an item', '20': 'Prepare a meal',
         '21': 'Prepare coffee and bread', '22': 'Pack bags and baskets'}
PACK = {'wallet': 'wallet_transfer', 'glasses_case': 'glasses_transfer'}
SOURCES = ('sofa', 'small_table', 'coffee_table.objects')
CONTAINERS = ('fixed_cpu_bag', 'storage_basket')


def source_root():
    return Path(__file__).resolve().parents[4]


def _pins(root, paths):
    result = {}
    for name in sorted(set(paths)):
        path = root / name
        result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _packing(rows, *, multiple):
    if not isinstance(rows, list) or not rows or (multiple and len(rows) < 2):
        raise ValueError('Explicit distinct requested target instances required')
    seen = set(); scene = []; runtime = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {'asset', 'source', 'destination'}:
            raise ValueError('Each target requires asset/source/destination')
        asset, source, destination = row['asset'], row['source'], row['destination']
        if asset not in PACK or source not in SOURCES or destination not in CONTAINERS:
            raise ValueError('Select an existing wallet/glasses source and bag/basket destination')
        if asset in seen:
            raise ValueError('Native scene has one instance of each selected portable asset')
        seen.add(asset); identity = 'pack_' + asset
        scene.append(dict(id=identity, type=PACK[asset], source=source, destination=destination))
        runtime.append(dict(instance_id=identity, type=PACK[asset], parameters=dict(destination=destination)))
    return scene, runtime


def compile_task(document_id, parameters=None, *, source=None):
    """Resolve a trusted recipe. ``source`` only supports CPU source-snapshot tests.

    Available means a native recipe is defined, never physics acceptance/pass.
    Runtime task paths for cutlery are deliberately bound only from the manifest.
    """
    if type(document_id) is not str or document_id not in NAMES:
        raise ValueError('Unknown appendix document identity')
    p = deepcopy(parameters) if parameters is not None else {}
    if not isinstance(p, dict):
        raise ValueError('Explicit parameter object required')
    allowed = {'seed', 'layout_id'} | {
        '15': set(), '16': set(), '17': {'waste', 'belongings'}, '18': set(),
        '19': {'asset', 'source'}, '20': {'payload_variant', 'required_cutlery'},
        '21': {'bread_variant', 'heating_required', 'finish_on_small_plate'}, '22': {'targets'}}[document_id]
    if set(p) - allowed:
        raise ValueError('Unknown or forbidden Long parameter: ' + ', '.join(sorted(set(p)-allowed)))
    if type(p.get('seed')) is not int or type(p.get('layout_id')) is not int:
        raise ValueError('Explicit existing seed and layout_id required')
    # Scene owner validates its own supported layout domain. No new sampling.
    root = source_root() if source is None else Path(source).resolve()
    profile = json.loads((root / 'openend/episode.json').read_text())
    policy = {key: deepcopy(profile[key]) for key in ('no_progress_timeout_s', 'selection_order')}
    timeout = policy['no_progress_timeout_s']
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('Invalid live OpenEnd profile watchdog')
    if policy['selection_order'] not in ('distance_then_front', 'front_then_distance'):
        raise ValueError('Unknown original room selection policy')
    scene = []; runtime = []; preparations = []; factories = {}; gaps = []
    instructions = {
        '15': 'Collect all requested snacks and drinks for serving.',
        '16': 'Put the book on the coffee table, put the wallet in the coffee table drawer and close it, and hang the cap on the coat rack. Leave all three in their intended places.',
        '17': 'Dispose of the designated table waste in the correct bins and put the requested belongings in the storage basket. Leave earlier placements intact.',
        '18': 'Store the used tablecloth and cover the table with the clean tablecloth.',
        '20': 'Take the requested food from the refrigerator, heat it in the microwave, and arrange it with the requested cutlery on the meal tray.',
        '21': 'Prepare coffee and bread for the user.',
        '22': 'Collect each listed object and leave it inside its specified container.'}
    paths = ['openend/episode.json', 'openend/room.py', 'openend/dispatch.py',
             'openend/tasks/process_final/lifecycle.py',
             'short_tasks/agent_probe_v1/prepare_workspace.py',
             'short_tasks/agent_probe_v1/cap_v1/native_tools/openend_control_adapter_v1/session_adapter.py']
    if document_id == '16':
        for identity, kind in [('sofa_to_coffee_table', 'book_transfer'),
                               ('sofa_to_drawer', 'wallet_transfer'), ('hang_cap', 'cap_transfer')]:
            scene.append(dict(id=identity, type=kind, source='sofa',
                              **({'count':1} if kind == 'book_transfer' else {})))
            runtime.append(dict(instance_id=identity, type=kind, parameters={}))
        factories = {'sofa_to_coffee_table': 'approved_sofa_book_surface',
                     'sofa_to_drawer': 'approved_wallet_drawer'}
        for name, asset in [('sofa_to_coffee_table', 'book_red'), ('sofa_to_drawer', 'wallet')]:
            preparations.append(dict(task_name=name, asset=asset,
                scene_preparation='short_tasks.tasks.'+name+'.scene.prepare'))
            paths += ['short_tasks/tasks/'+name+'/'+f for f in ('adapter.py', 'scene.py')]
        paths += ['short_tasks/tasks/sofa_to_drawer/'+f for f in ('native_port.py', 'transfer.py', 'native_geometry.py')]
    elif document_id in ('19', '22'):
        if document_id == '19':
            if not {'asset', 'source'} <= set(p):
                raise ValueError('Explicit supported target asset and private initial source required')
            targets = [dict(asset=p['asset'], source=p['source'], destination='fixed_cpu_bag')]
            # Neither private source nor prim path is interpolated into instruction.
            instructions['19'] = 'Find the '+{'wallet':'wallet', 'glasses_case':'sunglasses case'}.get(p['asset'], 'requested object')+' and leave it inside the bag.'
        else:
            targets = p.get('targets')
        scene, runtime = _packing(targets, multiple=document_id == '22')
        if document_id == '22':
            instructions['22'] = 'Collect these items and leave each inside its assigned container: '+ '; '.join(
                {'wallet':'wallet','glasses_case':'sunglasses case'}[row['asset']]+' in the '+
                {'fixed_cpu_bag':'bag','storage_basket':'storage basket'}[row['destination']] for row in targets)+'.'
    elif document_id == '17':
        scene, runtime = _packing(p.get('belongings'), multiple=False)
        if any(t['destination'] != 'storage_basket' or t['source'] not in ('small_table', 'coffee_table.objects') for t in scene):
            raise ValueError('Table tidying selects table-source belongings and the storage basket')
        waste = p.get('waste')
        if not isinstance(waste, list) or not waste:
            raise ValueError('Explicit designated waste required')
        if any(not isinstance(w, dict) or set(w) != {'kind', 'count'} or
               w['kind'] not in ('paper','snack','can','cup') or type(w['count']) is not int or w['count'] <= 0 for w in waste):
            raise ValueError('Existing waste categories and positive counts required')
        scene.insert(0, dict(id='trash_group', type='trash_transfer', source='coffee_table.objects', items=waste))
        runtime.insert(0, dict(instance_id='trash_group', type='trash_transfer', parameters={'auto_bind':True}))
        instructions['17'] += ' Requested waste: '+', '.join(str(w['count'])+' '+w['kind'] for w in waste)+'. Requested belongings: '+', '.join(t['asset'] for t in p['belongings'])+'.'
        paths += ['openend/tasks/trash_transfer/registration.py', 'openend/tasks/trash_transfer/scoring.py', 'openend/tasks/trash_transfer/port.py']
    elif document_id == '20':
        variant = p.get('payload_variant'); cutlery = p.get('required_cutlery')
        if variant not in ('beef_box', 'leftover_box'):
            raise ValueError('Explicit existing refrigerated payload_variant required')
        if not isinstance(cutlery, list) or not cutlery or any(a not in ('meal_fork','meal_knife') for a in cutlery):
            raise ValueError('Explicit required fork/knife instances required')
        for identity, kind in [('fridge', 'fridge_to_microwave'), ('microwave', 'microwave_box_to_meal_tray')]:
            scene.append(dict(id=identity, type=kind, payload_variant=variant))
            runtime.append(dict(instance_id=identity, type=kind, parameters={'payload_variant':variant}))
        for index, asset in enumerate(cutlery, 1):
            scene.append(dict(id='cutlery_'+str(index), type='cutlery_to_meal_tray', asset=asset))
            runtime.append(dict(instance_id='cutlery_'+str(index), type='cutlery_to_meal_tray', parameters={'asset':asset}))
        instructions['20'] += ' Use the '+variant.replace('_',' ')+' and '+', '.join(a.replace('meal_','') for a in cutlery)+'.'
        paths += ['openend/tasks/meal_tray/'+f for f in ('registration.py', 'native_port.py', 'microwave_port.py', 'microwave_scoring.py', 'additional_scoring.py', 'final_state.py')]
    elif document_id == '21':
        if p.get('bread_variant') not in ('sugar','no_sugar'):
            raise ValueError('Explicit existing bread_variant required')
        scene = [dict(id='coffee',type='coffee'), dict(id='fryer',type='fryer')]
        runtime = [dict(instance_id='coffee',type='coffee',parameters={}),
                   dict(instance_id='fryer',type='fryer',parameters={'bread_variant':p['bread_variant']})]
        if p.get('heating_required') is not False or p.get('finish_on_small_plate') is not False:
            gaps = ['select_existing_mechanical_bread_instance_explicitly_no_heating_no_plate_return']
        factories = {'coffee':'approved_coffee_preparation', 'fryer':'approved_bread_manipulation_current'}
        instructions['21'] = 'Prepare coffee and retrieve the cup. Put the '+('white' if p['bread_variant']=='sugar' else 'yellow')+' cake into the fryer basket, close it, extract the basket and then retrieve the cake. Keep both completed results valid. This instance requires neither bread heating nor returning it to the small plate.'
        paths += ['short_tasks/tasks/prepare_coffee/'+f for f in ('port.py','evaluator.py')]
        paths += ['short_tasks/tasks/bake_bread/'+f for f in ('native_port.py','evaluator.py','opening_progress.py')]
        paths += ['short_tasks/tasks/bake_bread/'+f for f in ('prepare_initial.py','initial_policy.json','atomic_pose_source.json')]
        paths += ['evaluation/tasks/air_fryer/process_rules.py']
    elif document_id == '18':
        gaps = ['native22_two_cloth_dynamic_reader_and_contact_contract_unbound',
                'original_numeric_process_scoring_not_defined',
                'legacy_ideal_clamp_conflicts_with_original_8_8_and_8_9_no_attachment_contract']
    else:
        gaps = ['use_existing_stock_wooden_tray_multi_registration']
    if document_id in ('17','19','22'):
        paths += ['openend/tasks/'+family+'/'+f for family in ('wallet_transfer','glasses_transfer') for f in ('port.py','composite.py')]
        paths += ['openend/tasks/destination_transfers/registration.py', 'openend/tasks/scene_requirements/runtime_ports.py']
    return dict(document_id=document_id, runtime_id='long_task_'+document_id,
        name=NAMES[document_id], instruction=instructions[document_id], available=not gaps,
        unavailable=gaps, rule_version=RULE_VERSION, parameters=p,
        scene_request=dict(schema='openend.scene_requirements.request.v2', seed=p['seed'],
            layout_id=p['layout_id'], tasks=scene, extras=[]),
        runtime_request=dict(tasks=runtime, **policy),
        score_policy=dict(owner='openend.dispatch.MultiTaskDispatcher', aggregation='existing_total_score_math_fsum',
            watchdog_source='openend/episode.json', watchdog=policy,
            component_policy='original_native_component_scores', additional_contact_points=0,
            short_policy_wrappers_applied=False, maximum=None),
        native_factory_roles=factories, scene_preparations=preparations,
        completion_contract='all_required_current_component_predicates_at_original_room_settlement',
        initial_robot_policy='original_room_spawn_v2',
        source_hashes=_pins(root, paths), native_physics_validated=False)


def bind_manifest(compiled, manifest):
    """Bind generated identities without accepting caller-provided asset paths."""
    result = deepcopy(compiled)
    if not result['available']:
        raise ValueError('Long task unavailable: '+', '.join(result['unavailable']))
    if manifest.get('seed') != result['parameters']['seed'] or manifest.get('layout_id') != result['parameters']['layout_id']:
        raise ValueError('Generated scene differs from selected seed/layout')
    for task in result['runtime_request']['tasks']:
        if task['type'] != 'cutlery_to_meal_tray':
            continue
        asset = task['parameters']['asset']
        rows = [row for row in manifest['objects'] if row.get('id') == asset and
                row.get('meal_task_id', row.get('task_id')) == task['instance_id']]
        if len(rows) != 1 or not rows[0].get('path', '').startswith('/World/'):
            raise ValueError('One exact generated cutlery identity required: '+asset)
        task['parameters']['target_path'] = rows[0]['path']
    return result


def bind_compiled(compiled, manifest):
    result = bind_manifest(compiled, manifest)
    if any(t['type'] == 'trash_transfer' for t in result['runtime_request']['tasks']):
        from openend.tasks.trash_transfer.registration import expand_request
        result['runtime_request'] = expand_request(result['runtime_request'], manifest)
    return result
