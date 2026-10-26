"""Task-owned category selection; no simulator imports or action implementation."""
from copy import deepcopy
import math
import random

CATEGORIES = ('recyclable', 'residual')
NAME = 'select_trash'
RULE_VERSION = 'context_reasoning.select_trash.v1'

def sample(catalog, *, seed, category=None, count=None):
    if type(seed) is not int or not 1 <= seed <= 4294967295:
        raise ValueError('Positive uint32 seed required')
    if category is not None and category not in CATEGORIES:
        raise ValueError('Unknown category')
    if count is not None and (type(count) is not int or count not in (4, 5)):
        raise ValueError('Exactly four or five candidates required')
    assets = catalog['assets']
    if len({a['id'] for a in assets}) != len(assets):
        raise ValueError('Duplicate asset identity')
    if any(a['id']=='rigid_plastic_tub' for a in assets):
        raise ValueError('Excluded tub may not enter the active pool')
    rng = random.Random(f'select-trash-v1:{seed}')
    category = category or rng.choice(CATEGORIES)
    count = count or rng.choice((4, 5))
    target_pool = [a for a in assets if a['category']==category]
    other_pool = [a for a in assets if a['category'] in CATEGORIES and a['category']!=category]
    if not target_pool or len(other_pool)<count-1:
        raise ValueError('Insufficient distinct candidates; never duplicate or relax category')
    selected = [deepcopy(rng.choice(target_pool)), *deepcopy(rng.sample(other_pool,count-1))]
    rng.shuffle(selected)
    for index, row in enumerate(selected):
        row['instance_id'] = f'item_{index:02d}'
        row['path'] = f'/World/ContextSelection/Item_{index:02d}'
    target = next(a['instance_id'] for a in selected if a['category']==category)
    return dict(task_name=NAME, rule_version=RULE_VERSION, seed=seed,
                category=category, candidates=selected, target_instance_id=target,
                maximum_score=1, public=public_prompt(category),
                runtime_registered=False, native_physics_validated=False)

def public_prompt(category):
    label = {'recyclable':'recyclable', 'residual':'non-recyclable'}[category]
    return dict(instruction=f'Touch the only {label} item on the coffee table with either gripper. '
                'Touching any other candidate ends the task unsuccessfully. Do not pick up or dispose of items.',
                context='For this task, clean empty glass containers, metal cookie tins, dry cardboard '
                'and newspapers are recyclable. Broken ceramics, used toothbrushes, spent composite '
                'pens and used erasers are non-recyclable. All containers here are empty and clean.')

class SelectionEvaluator:
    """Consumes one measured touch set per physics substep. No proximity proxy."""
    def __init__(self, episode_id, plan):
        if not isinstance(episode_id,str) or not episode_id:
            raise ValueError('Episode identity required')
        ids = [a['instance_id'] for a in plan['candidates']]
        target = plan['target_instance_id']
        if len(ids) not in (4,5) or len(set(ids))!=len(ids) or target not in ids:
            raise ValueError('Invalid candidate binding')
        matches = [a['instance_id'] for a in plan['candidates'] if a['category']==plan['category']]
        if matches != [target]:
            raise ValueError('Exactly one matching target required')
        self.episode_id,self.ids,self.target = episode_id,set(ids),target
        self.armed=False
        self.last_time=None
        self.last_evidence=None
        self.result=dict(score=0,maximum_score=1,success=False,terminated=False,
                         truncated=False,reason=None,evaluation_valid=True)

    def arm(self, time_s, *, initial_touches):
        if self.armed or self.result['terminated']:
            raise ValueError('Cannot arm twice')
        self._time(time_s)
        if initial_touches:
            raise ValueError('Reject initially touching scene before task starts')
        self.armed=True
        self.last_time=float(time_s)

    @staticmethod
    def _time(value):
        if type(value) not in (int,float) or not math.isfinite(value) or value<0:
            raise ValueError('Finite nonnegative simulation time required')

    def observe(self, *, episode_id, time_s, touched, evidence_valid=True):
        self._time(time_s)
        if episode_id!=self.episode_id or evidence_valid is not True or not self.armed:
            raise ValueError('Invalid, foreign or pre-acceptance evidence')
        touched=frozenset(touched)
        if not touched <= self.ids:
            raise ValueError('Unbound target in measured evidence')
        evidence=(float(time_s),touched)
        if self.last_evidence==evidence:
            return deepcopy(self.result)
        if time_s<=self.last_time:
            raise ValueError('Stale/conflicting physics evidence')
        if self.result['terminated'] or self.result['truncated']:
            return deepcopy(self.result)
        self.last_time=float(time_s)
        self.last_evidence=evidence
        # Same-substep ambiguity follows existing failure-before-success policy.
        wrong=touched-{self.target}
        if wrong:
            self.result.update(terminated=True,reason='wrong_candidate_touch',
                               touched=sorted(touched),terminal_time_s=time_s)
        elif self.target in touched:
            self.result.update(score=1,success=True,terminated=True,reason='correct_candidate_touch',
                               touched=[self.target],terminal_time_s=time_s)
        return deepcopy(self.result)

    def finish(self, reason):
        """Host owns the unchanged short-task deadline and stop lifecycle."""
        if not (self.result['terminated'] or self.result['truncated']):
            self.result.update(truncated=True,reason=reason)
        return deepcopy(self.result)

def touched_instances(pairs, finger_roots, bindings):
    from openend.tasks.meal_tray.task_contact_routing import registered_pair
    pairs=tuple(pairs)
    return frozenset(key for key,root in bindings.items()
        if any(bool(samples) and registered_pair(paths,finger_roots,(root,))
               for paths,samples in pairs))

class NativeEvidence:
    """Reuse trusted finger reporters and native FOUND/PERSIST/LOST processing.

    Host must arm after initial acceptance, then call after every physics substep.
    No scene writes, hold detection, 10-second wake timer or per-item grab scorer.
    """
    def __init__(self, stage, plan, episode_id):
        from openend.tasks.meal_tray.task_contact_routing import NativeContactRoutes
        self.bindings={a['instance_id']:a['path'] for a in plan['candidates']}
        self.native=NativeContactRoutes(stage,{k:(v,) for k,v in self.bindings.items()})
        self.evaluator=SelectionEvaluator(episode_id,plan)
        self.episode_id=episode_id

    def read_touches(self, report):
        if not isinstance(report,(tuple,list)) or len(report)!=2:
            raise ValueError('Native contact report required; missing is not no contact')
        self.native.contacts.ingest(report)
        return touched_instances(self.native.contacts.pairs(),self.native.fingers,self.bindings)

    def arm(self, time_s, report):
        self.evaluator.arm(time_s,initial_touches=self.read_touches(report))

    def after_physics(self, report, time_s):
        touched=self.read_touches(report)
        if not self.evaluator.armed:
            return None  # settle contacts never score
        return self.evaluator.observe(episode_id=self.episode_id,time_s=time_s,touched=touched)
