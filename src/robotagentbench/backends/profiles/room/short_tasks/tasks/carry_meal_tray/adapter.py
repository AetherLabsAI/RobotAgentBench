"""Document task 14: empty tray requires no preparation tasks."""
from copy import deepcopy
from openend.tasks.destination_transfers.scoring import TrayScore
from short_tasks.tasks.stock_wooden_tray._contract import options, role, plan, summaries, number, result, close_component

TASK_NAME='carry_meal_tray'
RULE_VERSION='openend.short_task.carry_meal_tray.v1'
INSTANCE='meal_tray_carry'


def resolve(request):
    opts=options(request,TASK_NAME,('contents','retain_all_contents'))
    role(request,'source','scene_selected_support');role(request,'destination','coffee_table.meal_tray_zone')
    if 'targets' in request and request['targets']!=[{'category':'meal_tray_large','quantity':1}]:
        raise ValueError('Exactly one scene-owned large meal tray required')
    contents=opts.get('contents',[])
    if not isinstance(contents,list):raise ValueError('Contents must be an explicit list')
    if 'retain_all_contents' in opts and type(opts['retain_all_contents']) is not bool:
        raise ValueError('Contents retention must be a boolean')
    unavailable=None
    if contents or 'retain_all_contents' in opts:
        unavailable=('tray_contents_policy_undefined','文档尚未冻结内容物保留、掉落或扣分判据；首版只支持空盘搬运，不虚构内容物成功条件。')
    out=plan(TASK_NAME,RULE_VERSION,request,[dict(id=INSTANCE,type='meal_tray_transfer')],
        [dict(instance_id=INSTANCE,type='meal_tray_transfer',parameters={})] if unavailable is None else [],5,2,3,
        'Pick up the empty large meal tray from its initial support, carry it to the designated meal-tray area on the coffee table, and release it supported by the tabletop.',
        ['大餐盘初态在场景指定来源承托上、未被持有、未处于茶几目的地。',
         '目的地继承当前版本table_objects工作区，并使用实际茶几台面向上承托接触。',
         '空盘可直接搬运；不注册摆盘或食品准备任务。'],unavailable=unavailable)
    out['contents']=deepcopy(contents)
    out['destination_rule']='upstream.table_objects.actual_tabletop_support.v1'
    return out


def bind(plan,manifest):
    matches=[o for o in manifest['objects'] if o['id']=='meal_tray_large']
    if len(matches)!=1:raise ValueError('Exactly one large meal tray required')
    tasks=[t for t in manifest.get('tasks',[]) if t.get('id')==INSTANCE]
    if len(tasks)!=1 or tasks[0].get('type')!='meal_tray_transfer' or tasks[0].get('target_path')!=matches[0]['path']:
        raise ValueError('Scene has not bound the exact tray transfer task')
    plan['target_bindings']=[dict(instance_id=INSTANCE,target_path=matches[0]['path'],category='meal_tray_large')]
    return deepcopy(plan['runtime_components'])


def normalize(plan,component_summaries):
    values=summaries(plan,component_summaries);s=values[INSTANCE]
    if s.get('maximum_score')!=5:raise ValueError('Tray rule maximum changed')
    complete=bool(s['operation_complete'])
    rows=[dict(instance_id=INSTANCE,category='meal_tray_large',process=number(s['operation_score']),
        current_final=number(s['final_state_score']),deductions=0,component_total=number(s['score']),
        process_complete=complete,success=bool(complete and s['success']))]
    return result(plan,rows)


class ReleasedTrayScore(TrayScore):
    def observe(self,**facts):
        super().observe(**facts)
        if facts.get('final_sample',True) and facts['held']:self.final=False
        return self.summary()


def factory_override(plan,declaration):
    if declaration['type']!='meal_tray_transfer':raise ValueError('Wrong tray component')
    from openend.tasks.destination_transfers.tray_port import TrayPort
    class ShortTrayPort(TrayPort):
        def accept_initial(self):
            receipt=super().accept_initial()
            self.composite=ReleasedTrayScore();self.score=self.composite.summary()
            return receipt
        def consume_control(self,active):
            if type(active) is not bool:raise ValueError('Measured activation boolean required')
            self._short_started=getattr(self,'_short_started',False) or active
            return super().consume_control(self._short_started)
        def finalize(self,reason):
            self.score=close_component(self.composite.summary(),reason,super().finalize)
            return self.score
    return ShortTrayPort
