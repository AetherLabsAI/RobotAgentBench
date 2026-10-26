"""Document task 13: per-instance utensils, with release-gated current final."""
from copy import deepcopy
from openend.tasks.meal_tray.additional_scoring import AdditionalMealTask, CutleryFacts
from short_tasks.tasks.stock_wooden_tray._contract import options, targets, role, plan, summaries, number, result, close_component

TASK_NAME='prepare_cutlery'
RULE_VERSION='openend.short_task.prepare_cutlery.v1'
ASSETS=('meal_fork','meal_knife')


def resolve(request):
    options(request,TASK_NAME)
    role(request,'source','cutlery_holder');role(request,'destination','meal_tray_large')
    assets=targets(request,ASSETS,default=[dict(category='meal_knife',quantity=1),
        dict(category='meal_fork',quantity=2)],maximum=3)
    if assets.count('meal_knife')>1 or assets.count('meal_fork')>2:
        raise ValueError('The source holder contains one knife and two distinct forks')
    tasks=[dict(id=f'cutlery_item_{i+1}',type='cutlery_to_meal_tray',asset=asset) for i,asset in enumerate(assets)]
    out=plan(TASK_NAME,RULE_VERSION,request,tasks,[],5*len(assets),2*len(assets),3*len(assets),
        'Pick up the designated utensils from their holder slots: '+ ', '.join('fork' if a=='meal_fork' else 'knife' for a in assets)+'. Place all of them inside the large meal tray and release them.',
        ['来源餐具架、大餐盘和目标餐具由统一场景唯一绑定；目标初态不在大餐盘。',
         '默认一刀两叉，分别使用原餐具架的三个独立槽位；逐件绑定、全部完成才成功。',
         '仅注册餐具，不自动附带肉盒、沙拉、咖啡、蛋糕或饮料任务。'])
    out['target_categories']=assets
    return out


def bind(plan,manifest):
    from openend.tasks.meal_tray.native_port import one
    rows=[]
    for i,asset in enumerate(plan['target_categories']):
        identity=f'cutlery_item_{i+1}'
        matches=[o for o in manifest['objects'] if o.get('id')==asset
            and o.get('meal_task_id')==identity]
        if len(matches)!=1:raise ValueError('One exact task-owned cutlery instance required')
        obj=matches[0]
        rows.append(dict(instance_id=identity,type='cutlery_to_meal_tray',parameters=dict(asset=asset,target_path=obj['path'])))
    one(manifest,'meal_tray_large');one(manifest,'cutlery_holder')
    if len({r['parameters']['target_path'] for r in rows})!=len(rows):raise ValueError('Duplicate cutlery target')
    plan['runtime_components']=rows
    plan['target_bindings']=[dict(instance_id=r['instance_id'],category=r['parameters']['asset'],target_path=r['parameters']['target_path']) for r in rows]
    return deepcopy(rows)


def normalize(plan,component_summaries):
    values=summaries(plan,component_summaries);rows=[]
    for declaration in plan['runtime_components']:
        identity=declaration['instance_id'];s=values[identity]
        if s.get('task_type')!='cutlery_to_meal_tray' or s.get('maximum_score')!=5:raise ValueError('Wrong cutlery component')
        f=s['final_state'];complete=bool(s['operation_complete'])
        rows.append(dict(instance_id=identity,category=declaration['parameters']['asset'],target_path=declaration['parameters']['target_path'],
            process=number(s['operation_score']),current_final=number(f['score']),deductions=0,
            component_total=number(s['score']),process_complete=complete,success=bool(complete and f['condition_met'])))
    return result(plan,rows)


class ReleasedCutleryTask(AdditionalMealTask):
    def __init__(self,binding):
        super().__init__(binding)
        self.current_held=None
    def observe(self,placement,operation=None):
        super().observe(placement,operation)
        if operation is not None and isinstance(operation.facts,CutleryFacts):
            self.current_held=operation.facts.grasped
        self.final_state.met=bool(self.final_state.met and self.current_held is False
            and not placement.ground_contact)
        return self.summary()


def factory_override(plan,declaration):
    if declaration['type']!='cutlery_to_meal_tray':raise ValueError('Wrong cutlery component')
    from openend.tasks.meal_tray.native_port import CutleryMealPort
    class ShortCutleryPort(CutleryMealPort):
        def make_composite(self):return ReleasedCutleryTask(self.identity)
        def _placement(self,row,index):
            frame=super()._placement(row,index)
            if self.composite is not None and 'held' in row:self.composite.current_held=bool(row['held'])
            return frame
        def consume_control(self,active):
            if type(active) is not bool:raise ValueError('Measured activation boolean required')
            self._short_started=getattr(self,'_short_started',False) or active
            if self._short_started:return super().consume_control(True)
            if not self.pending:raise RuntimeError('Missing initial inactive cutlery evidence')
            self.pending=[];self.composite.pause();self.paused=True
            self.score=self.composite.summary()
            return self.score
        def finalize(self,reason):
            if self.pending:raise RuntimeError('Unconsumed native cutlery evidence at settlement')
            if self.latest is None or abs(self.latest['time_s']-self.clock.current_time)>1e-6:
                raise RuntimeError('Stale native cutlery settlement evidence')
            self.score=close_component(self.composite.summary(),reason,self.composite.finalize)
            return self.score
    return ShortCutleryPort
