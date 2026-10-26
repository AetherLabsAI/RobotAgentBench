"""Two public shelf reasoning tasks, one private physical selection implementation."""
from copy import deepcopy
import json
import random
from .bank import BANK_DIR
from .task import RULE_VERSION, bind_scene

KINDS = {'select_drink': 'drink', 'select_snack': 'snack'}

def question(request, bank, *, kind=None):
    if any(request.get(k) for k in ('targets','destination','background','position_constraints')):
        raise ValueError('Reasoning selection does not accept hidden answer selectors')
    options=request.get('options', {})
    if set(options)-{'question_id'}:
        raise ValueError('Only public question_id is accepted')
    rows=[q for q in bank['questions'] if kind is None or q['kind']==kind]
    if 'question_id' in options:
        rows=[q for q in rows if q['id']==options['question_id']]
        if len(rows)!=1: raise ValueError('Question does not belong to this task')
        return rows[0]
    return random.Random(f"context-question:{request['task_name']}:{request['seed']}").choice(rows)

def plan_for(request, version, public, source):
    if request['rule_version']!=version: raise ValueError('Explicit frozen selection version required')
    if request.get('source', {'role':source})!={'role':source}:
        raise ValueError('Wrong selection source')
    name=request['task_name']
    return dict(task_name=name, rule_version=version, available=True,
        instruction=' '.join(public.get(k,'') for k in ('instruction','context','scenario')),
        public_question=deepcopy(public),
        scene_request=dict(tasks=[dict(id=name,type=name)]),
        runtime_components=[dict(instance_id=name,type=name,parameters={})],
        score_policy=dict(maximum=1,process_maximum=1,final_maximum=0,arithmetic='gross_minus_deductions'))

def resolve(request):
    if request['task_name'] not in KINDS: raise ValueError('Unknown shelf task')
    bank=json.loads((BANK_DIR/'public_questions.json').read_text())
    q=question(request,bank,kind=KINDS[request['task_name']])
    public=dict(question_id=q['id'], instruction=bank['shared_instruction'],
        context=bank['shared_context']+' '+q.get('context',''),scenario=q['scenario'])
    return plan_for(request,RULE_VERSION,public,'snack_shelf')

def bind(plan, manifest):
    from .inventory_plan import plan_inventory
    selected=plan_inventory(plan['public_question']['question_id'],plan['request']['seed'])
    selection=bind_scene(manifest,selected)
    name=plan['task_name']
    rows=[o for o in manifest['objects'] if o.get('task_role')=='touch_candidate']
    if any(o.get('task_id')!=name or o.get('source')!='snack_shelf' for o in rows):
        raise ValueError('Shelf ownership/source mismatch')
    plan['selection']=selection
    plan['target_bindings']={r['instance_id']:r['path'] for r in selection['candidates']}
    return [dict(instance_id=name,type=name,parameters=deepcopy(selection))]

def normalize(plan, summaries):
    value=summaries[plan['task_name']]
    if value['maximum_score']!=1 or value['score'] not in (0,1):
        raise ValueError('Wrong selection score contract')
    return dict(success=bool(value['success']),
        scores=dict(process=value['score'],current_final=0,deductions=0,total=value['score'],maximum=1),
        failure_reason='wrong_candidate_touch' if value['reason']=='wrong_candidate_touch' else None,
        stages=[dict(name='touch_selection',complete=value['success'])],current_result=deepcopy(value))
