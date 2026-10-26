"""Sofa identity binding; reuse the existing selection state machine unchanged.

This is task-local logic, not a declaration of shared runtime registration.
"""
from copy import deepcopy
from pathlib import PurePosixPath
from short_task_ContextReasoning.tasks.select_trash.task import (
    SelectionEvaluator as TableSelectionEvaluator, touched_instances)

ASSETS = {'remote_tv':'remote_tv', 'wallet':'wallet',
    'glasses_case':'glasses_case', 'power_bank':'power_bank',
    'cap':'baseball_cap', 'umbrella':'umbrella',
    'water_bottle':'water_bottle', 'bill':'notebook_green'}


def bind_question(manifest, public_bank, private_bank, question_id):
    questions = [q for q in public_bank['questions'] if q['id'] == question_id]
    answers = [a for a in private_bank['answers'] if a['id'] == question_id]
    if len(questions) != 1 or len(answers) != 1:
        raise ValueError('Unique public question and private answer required')
    target = answers[0]['target']
    if target not in ASSETS:
        raise ValueError('Unknown semantic target')
    rows = [o for o in manifest['objects'] if o.get('task_role') == 'selection_candidate']
    if len(rows) != 8 or {o['id'] for o in rows} != set(ASSETS.values()):
        raise ValueError('All eight distinct original candidates are required')
    candidates = [dict(instance_id=o['instance_id'], path=o['path'], asset=o['id']) for o in rows]
    by_asset = {o['asset']:o['instance_id'] for o in candidates}
    plan = dict(candidates=candidates, target_instance_id=by_asset[ASSETS[target]],
                question_id=question_id)
    # Validate before returning even if no evaluator has yet been constructed.
    SofaSelectionEvaluator('binding-check', plan)
    public = dict(question_id=question_id, instruction=public_bank['shared_instruction'],
        context=public_bank['shared_context'], scenario=questions[0]['scenario'])
    return plan, public


class SofaSelectionEvaluator(TableSelectionEvaluator):
    """Only replace the table-specific 4/5-category initialization contract.

    arm/observe/finish, evidence validation, failure priority, time checking and
    scoring are inherited directly, not copied or altered.
    """
    def __init__(self, episode_id, plan):
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError('Episode identity required')
        rows=plan['candidates']; ids=[a['instance_id'] for a in rows]
        paths=[a['path'] for a in rows]
        if (len(rows)!=8 or len(set(ids))!=8 or len(set(paths))!=8
                or {a['asset'] for a in rows}!=set(ASSETS.values())
                or plan['target_instance_id'] not in ids):
            raise ValueError('Exactly eight uniquely bound candidates required')
        if any(not isinstance(p,str) or not p.startswith('/World/')
               or str(PurePosixPath(p))!=p or '..' in PurePosixPath(p).parts for p in paths):
            raise ValueError('Canonical world prim paths required')
        if any(p.startswith(q+'/') for p in paths for q in paths if p!=q):
            raise ValueError('Overlapping candidate roots')
        self.episode_id,self.ids,self.target=episode_id,set(ids),plan['target_instance_id']
        self.armed=False; self.last_time=None; self.last_evidence=None
        self.result=dict(score=0,maximum_score=1,success=False,terminated=False,
                         truncated=False,reason=None,evaluation_valid=True)
