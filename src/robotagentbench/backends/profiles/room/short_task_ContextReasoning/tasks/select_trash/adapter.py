"""Trusted single-task plugin; target category/identity stay on the evaluator side."""
from copy import deepcopy
import json
from pathlib import Path
from .task import NAME, RULE_VERSION, sample

HERE = Path(__file__).resolve().parent

def resolve(request):
    if request['rule_version'] != RULE_VERSION:
        raise ValueError('Explicit selection rule version required')
    if any(request.get(k) for k in ('targets', 'destination', 'background', 'position_constraints')):
        raise ValueError('Selection uses category/count only; no hidden target selector')
    if request.get('source', {'role': 'coffee_table'}) != {'role': 'coffee_table'}:
        raise ValueError('Coffee-table initial state required')
    options = request.get('options', {})
    if set(options) - {'category', 'count'}:
        raise ValueError('Unsupported selection option')
    catalog = json.loads((HERE.parents[1]/'asset_pools/trash_selection_v1/catalog.json').read_text())
    selection = sample(catalog, seed=request['seed'], **options)
    return dict(task_name=NAME, rule_version=RULE_VERSION, available=True,
        instruction=selection['public']['instruction']+' '+selection['public']['context'],
        scene_request=dict(tasks=[dict(id=NAME, type=NAME)]),
        runtime_components=[dict(instance_id=NAME, type=NAME, parameters={})],
        score_policy=dict(maximum=1, process_maximum=1, final_maximum=0,
                          arithmetic='gross_minus_deductions'),
        selection=selection)

def bind(plan, manifest):
    selection = plan['selection']
    rows = [o for o in manifest['objects'] if o.get('task_id') == NAME]
    expected = {a['instance_id']: a for a in selection['candidates']}
    if len(rows) != len(expected) or {o.get('instance_id') for o in rows} != set(expected):
        raise ValueError('Exactly the seeded unique candidate instances are required')
    for row in rows:
        wanted = expected[row['instance_id']]
        if row['path'] != wanted['path'] or row['id'] != wanted['all_assets_id'] or row.get('source') != 'coffee_table.objects':
            raise ValueError('Candidate role/asset differs from seeded selection')
        if Path(row['entry']).resolve() != Path(wanted['entry']).resolve():
            raise ValueError('Candidate entry differs from approved all_assets asset')
    plan['target_bindings'] = {k: a['path'] for k, a in expected.items()}
    return [dict(instance_id=NAME, type=NAME, parameters=deepcopy(selection))]

def normalize(plan, summaries):
    value = summaries[NAME]
    if value['maximum_score'] != 1 or value['score'] not in (0, 1):
        raise ValueError('Wrong selection score contract')
    points = value['score']
    return dict(success=bool(value['success']),
        scores=dict(process=points, current_final=0, deductions=0, total=points, maximum=1),
        failure_reason='wrong_candidate_touch' if value['reason']=='wrong_candidate_touch' else None,
        stages=[dict(name='touch_selection', complete=value['success'])],
        current_result=deepcopy(value))
