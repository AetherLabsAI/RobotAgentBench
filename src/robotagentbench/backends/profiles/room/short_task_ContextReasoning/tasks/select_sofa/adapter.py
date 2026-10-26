"""Public sofa reasoning plugin. Answers remain private to the evaluator."""
from copy import deepcopy
import json
from pathlib import Path
from .task import bind_question
from short_task_ContextReasoning.tasks.select_shelf.adapter import question, plan_for, normalize

HERE=Path(__file__).resolve().parent
RULE_VERSION='context_reasoning.select_sofa.v1'

def resolve(request):
    bank=json.loads((HERE/'question_bank_v1/public_questions.json').read_text())
    q=question(request,bank)
    public=dict(question_id=q['id'],instruction=bank['shared_instruction'],
        context=bank['shared_context'],scenario=q['scenario'])
    return plan_for(request,RULE_VERSION,public,'sofa')

def bind(plan,manifest):
    public=json.loads((HERE/'question_bank_v1/public_questions.json').read_text())
    private=json.loads((HERE/'question_bank_v1/answers_private.json').read_text())
    selection,_=bind_question(manifest,public,private,plan['public_question']['question_id'])
    name=plan['task_name']
    rows=[o for o in manifest['objects'] if o.get('task_role')=='selection_candidate']
    if any(o.get('task_id')!=name or o.get('source')!='sofa' for o in rows):
        raise ValueError('Sofa ownership/source mismatch')
    plan['selection']=selection
    plan['target_bindings']={r['instance_id']:r['path'] for r in selection['candidates']}
    return [dict(instance_id=name,type=name,parameters=deepcopy(selection))]
