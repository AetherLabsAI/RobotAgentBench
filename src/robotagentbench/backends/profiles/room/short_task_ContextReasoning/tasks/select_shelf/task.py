"""Shelf reasoning scoring; reuse the existing contact-selection state machine."""
from pathlib import PurePosixPath
from short_task_ContextReasoning.tasks.select_trash.task import SelectionEvaluator, touched_instances

RULE_VERSION='context_reasoning.select_shelf.v1'

def bind_scene(manifest, inventory):
    rows=[r for r in manifest['objects'] if r.get('task_role')=='touch_candidate']
    candidates=[dict(instance_id=r['instance_id'],path=r['path'],asset=r['id']) for r in rows]
    expected={o['instance_id']:o['product_family'] for layer in inventory['inventory']['layers'] for o in layer['objects']}
    if {r['instance_id']:r['asset'] for r in candidates}!=expected:
        raise ValueError('Every stocked instance must be bound, including background')
    matches=[r for r in candidates if r['asset'] in inventory['acceptable_products']]
    if len(matches)!=1 or matches[0]['instance_id']!=inventory['target_instance_id']:
        raise ValueError('Scene must contain exactly one acceptable answer')
    plan=dict(candidates=candidates,target_instance_id=matches[0]['instance_id'],question_id=inventory['question_id'],rule_version=RULE_VERSION)
    ShelfSelectionEvaluator('binding-check',plan)
    return plan

class ShelfSelectionEvaluator(SelectionEvaluator):
    """Only shelf identity validation differs; all scoring methods are inherited."""
    def __init__(self,episode_id,plan):
        if not isinstance(episode_id,str) or not episode_id:raise ValueError('Episode ID required')
        rows=plan['candidates'];ids=[r['instance_id'] for r in rows];paths=[r['path'] for r in rows]
        assets=[r['asset'] for r in rows]
        if not 12<=len(rows)<=16 or any(len(set(v))!=len(rows) for v in (ids,paths,assets)) or plan['target_instance_id'] not in ids:
            raise ValueError('Unique complete shelf inventory required')
        if any(not p.startswith('/World/') or str(PurePosixPath(p))!=p or '..' in PurePosixPath(p).parts for p in paths):
            raise ValueError('Canonical world paths required')
        if any(p.startswith(q+'/') for p in paths for q in paths if p!=q):raise ValueError('Overlapping roots')
        self.episode_id,self.ids,self.target=episode_id,set(ids),plan['target_instance_id']
        self.armed=False;self.last_time=None;self.last_evidence=None
        self.result=dict(score=0,maximum_score=1,success=False,terminated=False,truncated=False,reason=None,evaluation_valid=True)
