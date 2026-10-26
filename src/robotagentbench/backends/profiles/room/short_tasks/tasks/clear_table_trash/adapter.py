"""Document task 06: scene-owned marked garbage, one aggregate task."""
from collections import Counter
from copy import deepcopy
from hashlib import sha256
from openend.tasks.trash_transfer.registration import CATEGORIES, inventory
from openend.tasks.trash_transfer.scoring import TrashTransfer
from openend.tasks.scene_requirements.planner import rng_for, selections
from short_tasks.tasks.stock_wooden_tray._contract import options, targets, role, plan, summaries, number, result, close_component

TASK_NAME='clear_table_trash'
RULE_VERSION='openend.short_task.clear_table_trash.v1'
GROUP_ID='trash_group'


def resolve(request):
    opts=options(request,TASK_NAME,('count','categories'))
    role(request,'source','coffee_table');role(request,'destination','category_trash_bins')
    if 'targets' in request:
        if opts:raise ValueError('Exact targets cannot be combined with random count/categories')
        categories=targets(request,CATEGORIES)
    else:
        task=dict(type='trash_transfer',**opts)
        categories=selections(task,rng_for(request.get('seed',1),GROUP_ID))
    # The existing scene compiler freezes these exact seeded selections.
    items=[dict(kind=k,count=n) for k,n in sorted(Counter(categories).items())]
    scene=[dict(id=GROUP_ID,type='trash_transfer',source='coffee_table.objects',items=items)]
    out=plan(TASK_NAME,RULE_VERSION,request,scene,[],8*len(categories),5*len(categories),3*len(categories),
        'Clear the designated rubbish from the coffee table: '+ ', '.join({'paper':'crumpled paper','snack':'empty snack wrapper','can':'empty can','cup':'empty cup'}[k] for k in categories)+'. Put each item in the appropriate waste or recycling bin. The items may be cleared in any order.',
        ['目标由桌面有效承托，垃圾类别与来源由统一场景声明。',
         '纸团保留场景原有大小变体；空杯允许直立；背景物不得自动成为目标。'])
    out['target_categories']=categories
    return out


def bind(plan,manifest):
    objects,bins=inventory(manifest)
    own=sorted((o for o in objects if o.get('task_id')==GROUP_ID),key=lambda o:o['path'])
    if Counter(o['trash_category'] for o in own)!=Counter(plan['target_categories']):
        raise ValueError('Scene did not create exactly the requested marked garbage')
    components=[];bindings=[]
    for obj in own:
        category=obj['trash_category'];identity='trash_'+sha256(obj['path'].encode()).hexdigest()[:16]
        parameters=dict(target_path=obj['path'],category=category,destination_path=bins[CATEGORIES[category][1]])
        components.append(dict(instance_id=identity,type='trash_transfer',parameters=parameters))
        bindings.append(dict(instance_id=identity,category=category,target_path=obj['path'],destination_path=parameters['destination_path']))
    plan['runtime_components']=components;plan['target_bindings']=bindings
    return deepcopy(components)


def normalize(plan,component_summaries):
    values=summaries(plan,component_summaries);rows=[]
    for declaration in plan['runtime_components']:
        identity=declaration['instance_id'];s=values[identity]
        if s.get('task_type')!='trash_transfer' or s.get('category')!=declaration['parameters']['category'] or s.get('maximum_score')!=8:
            raise ValueError('Wrong garbage component/category/rule')
        p=s['pickup'];q=s['placement']
        process=number(p['earned_points']);final=number(q['earned'])
        deductions=sum(number(p.get(k,0)) for k in ('collision_deduction','scenery_floor_deduction'))+number(q['collision_deduction'])
        complete=bool(s.get('operation_complete',False))
        rows.append(dict(instance_id=identity,category=s['category'],target_path=declaration['parameters']['target_path'],
            process=process,current_final=final,deductions=deductions,component_total=number(s['score']),
            process_complete=complete,success=bool(s['success'] and complete and q['current_final'])))
    return result(plan,rows)


class ReleasedTrashTransfer(TrashTransfer):
    def observe(self,f,*,final_sample=True):
        super().observe(f,final_sample=final_sample)
        if f.held or f.on_floor:
            self.place.awarded.pop('disposed',None)
            self.place.current_final=False;self.current_success=False
        return self.summary()


def factory_override(plan,declaration):
    if declaration['type']!='trash_transfer':raise ValueError('Wrong garbage component')
    from openend.tasks.trash_transfer.port import TrashPort
    from openend.tasks.object_transfers.binding import Binding
    class ShortTrashPort(TrashPort):
        def create_composite(self,r):
            g=self.geometry
            self.binding=Binding(self.key,self.episode_id,self.instance_id,g.target,
                g.source_furniture(r['supports']),g.destination,g.cabinet,r['supports'],
                g.height(r['pose'],'rigid_origin'),r['time_s'],frozenset())
            self.composite=ReleasedTrashTransfer(self.binding,self.category,workspace_rates=True)
        def consume_control(self,active):
            if type(active) is not bool:raise ValueError('Measured activation boolean required')
            self._short_started=getattr(self,'_short_started',False) or active
            return super().consume_control(self._short_started)
        def finalize(self,reason):
            self.score=close_component(self.composite.summary(),reason,super().finalize)
            return self.score
    return ShortTrashPort
