"""Document task 11; the existing whole-group reader owns every target."""
from collections import Counter
from .placement_feasibility import can_place, sample_categories
from copy import deepcopy
from openend.tasks.scene_requirements.planner import CLOTH_POOL, rng_for, count
from openend.tasks.cloth_clear.group import ClothClearGroup
from .push_progress_v1.scoring import AreaProgressClothClear, VERSION as AREA_RULE_VERSION
from short_tasks.tasks.stock_wooden_tray._contract import options, targets, role, plan, summaries, number, result, close_component

TASK_NAME='clear_tablecloth'
RULE_VERSION='openend.short_task.clear_tablecloth.v1'
GROUP_ID='cloth_group'
DEFAULT_TARGET_POOL=tuple(sorted((*CLOTH_POOL,'book_red')))


def resolve(request):
    opts=options(request,TASK_NAME,('count',))
    role(request,'source','coffee_table.cloth');role(request,'destination','coffee_table.outside_cloth')
    if 'targets' in request:
        if opts:raise ValueError('Exact targets cannot be mixed with random count')
        categories=targets(request,(*CLOTH_POOL,'book_red'))
    else:
        rng=rng_for(request.get('seed',1),GROUP_ID)
        n=count(opts['count']) if 'count' in opts else rng.randint(2,4)
        categories=sample_categories(rng,DEFAULT_TARGET_POOL,n)
    unsupported=None
    if len(categories)>4:
        unsupported=('cloth_scene_capacity','当前统一场景的桌布实例装配只支持最多4件，不能静默删去显式目标。')
    elif not can_place(categories):
        unsupported=('cloth_scene_capacity','指定目标按稳定姿态与现有间距无法全部放入桌布；不删减目标或缩放物体。')
    items=[dict(kind=k,count=n) for k,n in sorted(Counter(categories).items())]
    tasks=[dict(id=GROUP_ID,type='cloth_clear',source='coffee_table.cloth',items=items)]
    comps=[dict(instance_id=GROUP_ID,type='cloth_clear',parameters={})] if unsupported is None else []
    out=plan(TASK_NAME,RULE_VERSION,request,tasks,comps,3*len(categories),0.9*len(categories),2.1*len(categories),
        'Move all designated objects off the tablecloth onto the permitted tabletop area on the same coffee table. Leave the tablecloth in place and keep clear of the fixed wooden tray. The objects may be moved in any order.',
        ['所有目标初态由桌布承托，桌布及同桌合法目的地区由场景唯一绑定。',
         '桌布上的书只属于本任务，不注册普通书本归位任务。'],unavailable=unsupported)
    out['target_categories']=categories
    out['placement_progress_version']=AREA_RULE_VERSION
    return out


def bind(plan,manifest):
    from openend.tasks.cloth_clear.registration import resolve as upstream_resolve
    record=upstream_resolve(plan['runtime_components'][0],manifest)
    by_path={o['path']:o for o in manifest['objects']}
    if Counter(by_path[p]['id'] for p in record['target_paths'])!=Counter(plan['target_categories']):
        raise ValueError('Wrong cloth target membership')
    if any(by_path[p].get('task_id')!=GROUP_ID for p in record['target_paths']):raise ValueError('Cloth target owned by another task')
    plan['target_bindings']=[dict(target_path=p,category=by_path[p]['id']) for p in record['target_paths']]
    return deepcopy(plan['runtime_components'])


def normalize(plan,component_summaries):
    values=summaries(plan,component_summaries);s=values[GROUP_ID];items=s.get('items')
    if not isinstance(items,dict) or not items:raise ValueError('Whole-group per-target summaries required')
    bound=plan.get('target_bindings')
    if not bound or set(items)!={r['target_path'] for r in bound}:raise ValueError('Wrong cloth target identities')
    rows=[]
    categories={r['target_path']:r['category'] for r in bound}
    for path,v in items.items():
        if v.get('maximum_score')!=3:raise ValueError('Cloth rule maximum changed')
        if v.get('rule_version')!=AREA_RULE_VERSION:raise ValueError('Wrong cloth area-progress version')
        rows.append(dict(instance_id=path,target_path=path,category=categories[path],
            process=number(v['process_points']),current_final=number(v['final_state_points']),
            deductions=0,component_total=number(v['score']),process_complete=bool(v['operation_complete']),
            success=bool(v['current_placed'] and v['operation_complete'])))
    return result(plan,rows)


# Preserve the task-local class name for callers; scoring is area-based.
ReleasedClothClear=AreaProgressClothClear


class ShortClothGroup(ClothClearGroup):
    def __init__(self,bindings):
        super().__init__(bindings)
        self.items={p:ReleasedClothClear(s.binding,terminate_on_success=False) for p,s in self.items.items()}


def factory_override(plan,declaration):
    if declaration['type']!='cloth_clear':raise ValueError('Wrong cloth component')
    from openend.tasks.cloth_clear.port import ClothClearPort
    class ShortClothPort(ClothClearPort):
        def accept_initial(self):
            receipt=super().accept_initial()
            group=ShortClothGroup(self.bindings)
            self.score=group.observe(self.facts(self.latest,0));self.composite=group
            return receipt
        def consume_control(self,active):
            if type(active) is not bool:raise ValueError('Measured activation boolean required')
            self._short_started=getattr(self,'_short_started',False) or active
            if self._short_started:return super().consume_control(True)
            if not self.pending:raise RuntimeError('Missing initial inactive cloth evidence')
            for row in self.pending:
                self.composite.suspend_substep(row['time_s'],row['physics_dt_s'])
            self.pending=[];self.score=self.composite.summary()
            return self.score
        def finalize(self,reason):
            self.score=close_component(self.composite.summary(),reason,super().finalize)
            return self.score
    return ShortClothPort

