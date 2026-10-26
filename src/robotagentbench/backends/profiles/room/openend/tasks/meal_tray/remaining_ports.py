"""Native salad/drink pickup adapters, sharing the current large-tray final state.

No body/drive/solver writes. Only active tasks read detailed poses; contact
lifecycle remains live while asleep. Old scoring components remain unchanged.
"""
import json
from dataclasses import replace
import numpy as np
from .additional_scoring import AdditionalMealTask,OperationFrame,DRINKS
from .geometry import under
from .remaining_geometry import SaladSource,DrinkSource

def bind_views(port,view,paths):
    for path in set(paths)-set(port.views):
        v=view.create_rigid_body_view(path)
        if v.count!=1 or list(v.prim_paths)!=[path]:raise ValueError('Wrong native body '+path)
        port.views[path]=v

def measured_poses(port):
    result={}
    for path,v in port.views.items():
        t=np.asarray(v.get_transforms()).copy()[0]
        result[path]=dict(position_m=tuple(map(float,t[:3])),quaternion_wxyz=tuple(map(float,t[[6,3,4,5]])))
    return result

class SaladMealMixin:
    kind='salad_to_meal_tray';asset='salad_box';initial_workspace='fridge'

    def __init__(self,stage,manifest,output,episode_id,instance_id,parameters):
        from .native_port import one
        from layout_task_activation_v1.task_scene_registration_v1.fridge_to_microwave_v1.native_source import NativeSource
        from pxr import PhysicsSchemaTools
        super().__init__(stage,manifest,output,episode_id,instance_id,parameters)
        self.fridge=SaladSource(stage,one(manifest,'salad_box'));self.source_path=self.fridge.doors[1]
        # Reuse only the original measured force/contact classifier; do not
        # construct the old microwave destination reader or a second scorer.
        class ContactReader(NativeSource):
            def __init__(reader,scene):
                reader.scene=scene;reader.decode=PhysicsSchemaTools.intToSdfPath;reader.paths={}
                reader.branches={f'/World/Robot/gripper_{hand}_{side}_link{link}':(h,s)
                    for h,hand in enumerate(('l','r')) for s,side in enumerate(('inner','outer')) for link in (2,4)}
        self.fridge_contacts=ContactReader(self.fridge)

    def initialize(self,clock,view):
        super().initialize(clock,view)
        bind_views(self,view,[p for bodies,_,_ in self.fridge.joints.values() for p in bodies])

    def _tick(self,helper):
        c=super()._tick(helper)
        if self.composite is not None:
            def finger(p):return any(under(p,f'/World/Robot/gripper_{h}_{s}_link{i}') for h in ('l','r') for s in ('inner','outer') for i in range(1,5))
            touched=any(samples and any(finger(paths[i]) and any(under(paths[1-i],d) for d in self.fridge.doors) for i in (0,1)) for paths,samples in self.contacts.pairs())
            if touched:self.contact_selection.observe(time_s=self.clock.current_time,finger_contact=True,supports=frozenset(),external_contact=True,on_floor=False)
        return c

    def after_physics(self,helper_report):
        from evaluation.tasks.microwave_put.physical_geometry import inverse,transform
        from evaluation.tasks.refrigerator_take.isaac_evidence import outside_bounds
        from layout_task_activation_v1.task_scene_registration_v1.fridge_to_microwave_v1.native_source import joint_angle
        from evaluation.tasks.refrigerator_take.process_rules import CLOSED_DEG
        from openend.tasks.all_objects_integration.selective_read import contact_report
        super().after_physics(helper_report)
        row=self.latest;p=row['poses'];f=self.fridge
        points=inverse(transform(f.vertices,p[f.target]),p[f.doors[1]])
        out=outside_bounds(points,f.lower,f.upper)
        contacts=self.fridge_contacts.fridge_contacts(contact_report(self,helper_report),row['physics_dt_s'],out)
        angles=tuple(abs(joint_angle(f.joints[k],p)) for k in ('left','right'))
        row['salad']=dict(held=row['held'],out=out,collision=contacts['collision'],angles=angles,
            hold=bool(row['held'] and out and all(a<=CLOSED_DEG for a in angles)))

    def make_composite(self):
        from evaluation.tasks.refrigerator_take.process_rules import Frame,ProcessScorer
        task=AdditionalMealTask(self.identity);f=self.latest['salad']
        task.scorer=ProcessScorer(self.episode_id,self.geometry.target_body,initial_state_accepted=True)
        # Seed only from the real, accepted closed/unheld initial measurement;
        # otherwise a first touch on an already moving door would be a false reset.
        task.scorer.observe(Frame(self.episode_id,self.geometry.target_body,0,0.,f['angles'][1],f['angles'],f['held'],f['out'],False,False,True))
        task.operation_step=0
        return task

    def accept_initial(self):
        from evaluation.tasks.refrigerator_take.process_rules import CLOSED_DEG
        f=self.latest['salad']
        (self.output/'salad_initial_measurement.json').write_text(json.dumps(dict(
            facts=f,poses=self.latest['poses'],lower=self.fridge.lower.tolist(),upper=self.fridge.upper.tolist()),indent=2)+'\n')
        if f['held'] or f['out'] or any(a>CLOSED_DEG for a in f['angles']):raise ValueError('Salad initial must be inside closed fridge, unheld')
        return super().accept_initial()

    def operation_evidence(self,index):
        from evaluation.tasks.refrigerator_take.process_rules import Frame
        rows=[r['salad'] for r in self.pending];f=rows[-1]
        facts=Frame(self.episode_id,self.geometry.target_body,index,index*.05,f['angles'][1],f['angles'],f['held'],f['out'],any(r['collision'] for r in rows),all(r['hold'] for r in rows),True)
        return OperationFrame(self.identity,index,index*.05,facts)

class DrinkMealMixin:
    kind='drink_to_meal_tray';asset='snack_shelf';initial_workspace='shelf'
    @classmethod
    def validate_parameters(cls,parameters):
        if set(parameters)!={'product'} or parameters['product'] not in DRINKS:raise ValueError('Explicit supported drink product required')
    @classmethod
    def resolve_object(cls,parameters,manifest):
        cls.validate_parameters(parameters)
        rows=[o for o in manifest['objects'] if o['id']==parameters['product'] and o.get('task_role')=='requested_full_product' and o.get('task_group')==cls.kind]
        if len(rows)!=1:raise ValueError('Exactly one meal-owned shelf drink required')
        return rows[0]
    @classmethod
    def resolve_instance(cls,parameters,manifest,instance_id):
        from .product_claims import exact_object
        cls.validate_parameters(parameters)
        row=exact_object(parameters,manifest,instance_id)
        if row.get('task_group')!=cls.kind:raise ValueError('Drink belongs to another task group')
        return row

    def __init__(self,stage,manifest,output,episode_id,instance_id,parameters):
        from pxr import Usd,UsdPhysics,PhysxSchema,PhysicsSchemaTools
        from omni.physx.bindings._physx import ContactEventType
        from evaluation.tasks.shelf_pick.contact_stream import ContactStream
        from openend.tasks.holding_follow_v1.detection import GraspDetector
        super().__init__(stage,manifest,output,episode_id,instance_id,parameters)
        self.product=parameters['product'];self.stock=DrinkSource(stage,manifest,self.product,self.geometry.target_body)
        s=self.stock;self.source_path=s.support
        with Usd.EditContext(stage,stage.GetSessionLayer()):
            for path in s.stock:
                p=stage.GetPrimAtPath(path);a=p.GetAttribute('physxContactReport:threshold')
                if a and a.HasAuthoredValueOpinion() and a.Get()!=0:raise ValueError('Conflicting stock sensor')
                PhysxSchema.PhysxContactReportAPI.Apply(p).CreateThresholdAttr(0.)
        def sleeping(sid,actor):
            p=stage.GetPrimAtPath(PhysicsSchemaTools.intToSdfPath(actor))
            if not p:raise RuntimeError('Unknown actor')
            return bool(not p.HasAPI(UsdPhysics.RigidBodyAPI) or p.GetAttribute('physics:kinematicEnabled').Get() is True or self.interface.is_sleeping(sid,actor))
        target=self.geometry.target_body
        self.pick_contacts=ContactStream(PhysicsSchemaTools.intToSdfPath,sleeping,
            (ContactEventType.CONTACT_FOUND,ContactEventType.CONTACT_PERSIST,ContactEventType.CONTACT_LOST),
            tuple(p for p in s.stock if p!=target),s.workspace.top,self.config.support_face_tolerance_m,
            floor_paths=self.floors,shelf_paths=s.shelf_colliders,support_face=s.face,targets=(target,),
            support=s.support,object_supports=s.supports,support_tops=s.workspace.support_tops)
        self.detector=GraspDetector(tuple(s.stock));self.physical_index=0
        registered=manifest.get('registration',{}).get('request',{}).get('tasks',[])
        self.peers=frozenset(s.body_by_wrapper[o['path']] for t in registered if t['type']=='snack_drink_to_tray'
            for o in manifest['objects'] if o.get('task_id')==t['instance_id'] and o.get('task_role')=='requested_full_product' and o['path'] in s.body_by_wrapper)

    def initialize(self,clock,view):
        super().initialize(clock,view);bind_views(self,view,self.stock.stock)

    def _tick(self,helper):
        from openend.tasks.all_objects_integration.selective_read import contact_report
        c=super()._tick(helper);self.pick_contacts.ingest(contact_report(self,helper));return c

    def after_physics(self,helper_report):
        from evaluation.tasks.shelf_pick.geometry import rotation
        c=self._tick(helper_report);poses=measured_poses(self);g=self.geometry;s=self.stock
        s.workspace.validate_static(self.stage)
        p=poses[g.target_body];center=rotation(p['quaternion_wxyz']).apply(s.workspace.collision_vertices[g.target_body].mean(0))+p['position_m']
        pc=self.pick_contacts.summarize(g.target_body,center)
        held,_=self.detector.observe(self.clock.current_time,poses,pc['bilateral_contacts'])
        row=dict(time_s=self.clock.current_time,physics_dt_s=1/self.clock.hz,poses=poses,
            pairs=tuple(self.contacts.pairs()),held=bool(held[g.target_body]),pick_contacts=pc,all_held=held,
            **{k:v for k,v in c.items() if k!='bilateral'})
        self.latest=row;self.history.append(row)
        if self.composite is not None:self.pending.append(row)

    def make_composite(self):return AdditionalMealTask(self.identity,product=self.product)
    def accept_initial(self):
        receipt=super().accept_initial();self.initial_z=float(self.latest['poses'][self.geometry.target_body]['position_m'][2]);return receipt

    def operation_evidence(self,index):
        from evaluation.tasks.shelf_pick.process_rules import PhysicalFrame
        rows=self.pending if self.composite.previous_operation_control==index-1 else self.pending[-1:]
        result=[];target=self.geometry.target_body;dt=1/self.clock.hz
        for i,r in enumerate(rows):
            c=r['pick_contacts'];time=index*.05-(len(rows)-1-i)*dt
            result.append(PhysicalFrame(self.episode_id,self.physical_index,time,r['held'],
                not c['target_support_contact'],float(r['poses'][target]['position_m'][2]-self.initial_z),c['neighbor_contacts'],
                frozenset(p for p,v in r['all_held'].items() if v and p!=target and p not in self.peers),
                bool(c['object_on_floor'][target]),False,shelf_contact=c['shelf_contact'],active_target=target))
            self.physical_index+=1
        return OperationFrame(self.identity,index,index*.05,tuple(result),dt)
