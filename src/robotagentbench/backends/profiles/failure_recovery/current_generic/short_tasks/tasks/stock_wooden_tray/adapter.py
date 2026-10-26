"""Document task 03: one public task, 1–4 exact internal shelf targets."""
from copy import deepcopy
from dataclasses import replace
from evaluation.tasks.shelf_pick.openended_composite_v1.catalog import PRODUCTS
from evaluation.tasks.shelf_pick.process_rules import ProcessScorer
from openend.tasks.scene_requirements.shelf_composite import SnackTrayComposite, Frame
from ._contract import options, targets, role, plan, summaries, number, result, close_component

TASK_NAME='stock_wooden_tray'
RULE_VERSION='openend.short_task.stock_wooden_tray.v1'
PRODUCT_LABELS_EN = {
    'cayman_jacked': 'Cayman Jack cocktail', 'gatorade_m': 'orange Gatorade',
    'gatorade_t': 'blue Gatorade', 'starbucks': 'Starbucks coffee drink',
    'good2grow': 'Good2Grow juice', 'grapefruit_can': 'grapefruit sparkling water',
    'coke_c3': 'Coca-Cola', 'sprite_c1': 'Sprite', 'chomps_r': 'Chomps beef stick',
    'pring_bbq': 'Pringles BBQ chips', 'oreo_gf': 'gluten-free Oreo cookies',
    'honeymaid': 'Honey Maid crackers', 'zbar_t': 'Zbar snack bar',
    'skit_littles': 'Skittles candy',
}


def product_label_en(key):
    return PRODUCT_LABELS_EN[key] + ' (' + key + ')'


def resolve(request):
    options(request,TASK_NAME)
    role(request,'source','snack_shelf');role(request,'destination','coffee_table_wooden_tray')
    products=targets(request,PRODUCTS,maximum=4)
    tasks=[dict(id=f'stock_item_{i+1}',type='snack_drink_to_tray',product=product)
           for i,product in enumerate(products)]
    components=[dict(instance_id=t['id'],type=t['type'],parameters={'product':t['product']}) for t in tasks]
    out=plan(TASK_NAME,RULE_VERSION,request,tasks,components,7*len(products),4*len(products),3*len(products),
        'Pick up the designated products from the snack shelf: '+ ', '.join(product_label_en(p) for p in products)+'. Carry all of them to the coffee table and place them in the small wooden tray, not the large meal tray.',
        ['每件目标绑定不同实例，按品类和数量操作；小木托盘不是大餐盘。',
         '目标初态在货架承托上，未被持有、未预放到目的地。'])
    out['target_categories']=products
    return out


def bind(plan,manifest):
    from openend.tasks.meal_tray.product_claims import exact_object
    paths=[]
    for declaration in plan['runtime_components']:
        obj=exact_object(declaration['parameters'],manifest,declaration['instance_id'])
        paths.append(dict(instance_id=declaration['instance_id'],category=obj['id'],target_path=obj['path']))
    if len({r['target_path'] for r in paths})!=len(paths):raise ValueError('Two targets claim one object')
    plan['target_bindings']=paths
    return deepcopy(plan['runtime_components'])


def normalize(plan,component_summaries):
    values=summaries(plan,component_summaries);rows=[]
    for declaration in plan['runtime_components']:
        identity=declaration['instance_id'];s=values[identity]
        if s.get('task_kind')!='snack_drink_to_tray' or s.get('task_parameters')!=declaration['parameters']:
            raise ValueError('Wrong shelf component or product')
        if s.get('maximum_score')!=7:raise ValueError('Shelf rule maximum changed')
        pick=s.get('pick') or {};place=s.get('place') or {}
        process=number(pick.get('earned_points',0))
        deductions=sum(number(pick.get(k,0)) for k in ('collision_deduction','drop_back_deduction','wrong_grasp_deduction','hold_deduction'))
        current=number(place.get('final_state_points',0))
        complete=bool(s.get('operation_complete',False))
        met=bool(s.get('success',False) and complete and current==3)
        rows.append(dict(instance_id=identity,category=declaration['parameters']['product'],
            process=process,current_final=current,deductions=deductions,process_complete=complete,success=met,
            component_total=number(s.get('score',0)),phase=s.get('phase'),committed_target=s.get('committed_target')))
    out=result(plan,rows)
    evidence={identity:deepcopy(s['contact_evidence']) for identity,s in values.items()
              if 'contact_evidence' in s}
    out['contact_evidence']=evidence
    out['evidence_coverage_complete']=all(v.get('gap_steps',0)==0 for v in evidence.values())
    out['scoring_status']=('waiting_for_contact_evidence'
        if any(not v['complete'] for v in evidence.values()) else 'current')
    return out


class RecoverableShelfProcess(ProcessScorer):
    """Retain original charges/milestones; recoverable mistakes do not end task."""
    def observe_batch(self,frames):
        before=len(self.events);was_wrong=self.wrong_deduction>0
        super().observe_batch(frames)
        if self.failure in ('target_on_floor','wrong_drink_grasp'):
            fault=self.failure
            self.failure=None;self.terminal=False;self.reason=None
            for event in self.events[before:]:
                if event['kind']=='terminal_failure':event['kind']='recoverable_fault'
                if was_wrong and event['kind']=='wrong_drink_grasp':event['points']=0
            self.recoverable_fault=fault
        else:self.recoverable_fault=None
        return self.summary()


class StockComposite(SnackTrayComposite):
    def __init__(self,binding,*,room_managed=True):
        super().__init__(binding,room_managed=room_managed)
        self.pick=RecoverableShelfProcess(binding.episode_id,self.pick.dt,self.pick.task_id)
    def _advance(self,frame):
        super()._advance(frame)
        if self.completed_pick is not None:
            h=frame.place
            self.destination_state.met=bool(self.destination_state.complete and h.supported_by_destination
                and h.in_destination_region and not h.grasped and not h.on_floor)
            self.success=self.destination_state.met
    def summary(self):
        out=super().summary()
        out['short_task_rule_version']=RULE_VERSION
        return out


def factory_override(plan,declaration):
    if declaration['type']!='snack_drink_to_tray':raise ValueError('Wrong internal component')
    from openend.tasks.scene_requirements.shelf_port import SnackTrayPort
    class StockPort(SnackTrayPort):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.composite=StockComposite(self.scene.identity,room_managed=True)
            self.score=self.composite.summary()
        def accept_initial(self):
            import json
            if self.source.latest is None:raise RuntimeError('Native initial observation missing')
            raw=self.source.latest;p=self.source.active;c=raw['contacts']
            checks=dict(exact_owned_target=p in self.scene.identity.candidates,
                supported_by_original_board=bool(c['target_supported']),
                unheld=not bool(raw['grasped'][p]),
                not_on_floor=not bool(c['object_on_floor'][p]),
                not_initially_at_destination=not bool(raw['placement'][p]['held_supported_by_destination']))
            receipt=dict(accepted=all(checks.values()),checks=checks,evidence_source='native_physx',
                source_support=self.scene.identity.source_support,target=p,scene_pose_writes=0,
                rule_version=RULE_VERSION,full_trajectory_verified=False)
            (self.output/'initial_acceptance.json').write_text(json.dumps(receipt,indent=2)+'\n')
            if not receipt['accepted']:raise RuntimeError('Short Task shelf initial rejected: '+str(checks))
            self.source.initial_z={path:float(raw['poses'][path]['position_m'][2]) for path in self.scene.identity.candidates}
            # Keep a real zero-score initial frame so a caller can close before
            # its first action. Subsequent selected dense samples start at one.
            frame=self.source.frame(0)
            self.score=self.composite.observe(Frame(self.scene.identity,frame.pick,frame.place,self.clock.get_physics_dt()))
            self.samples=1
            return receipt
        def consume_control(self,active):
            if type(active) is not bool:raise ValueError('Measured activation boolean required')
            self._short_started=getattr(self,'_short_started',False) or active
            if self._short_started:return self.observe()
            self.pause();self.score=self.composite.summary()
            self._contact_status()
            return self.score
        def finalize(self,reason):
            self.score=close_component(self.composite.summary(),reason,super().finalize)
            return self.score
    return StockPort
