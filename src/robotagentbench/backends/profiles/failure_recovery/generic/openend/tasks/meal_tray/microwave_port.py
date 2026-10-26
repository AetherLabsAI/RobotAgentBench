"""Room-native rotary -> extraction -> meal placement, one persistent task.

All stages use real finger contact and the existing 10 s touch wake, including
the knob, door, box and its physically supporting small plate.
World Hz always remains owned by the independent workspace selector.
"""
from dataclasses import asdict,replace
import json
import numpy as np
from .final_state import Binding
from .microwave_device import MicrowaveDevice
from .microwave_geometry import MicrowaveGeometry,PLATE,current_contact_rows,support_rows,under
from .microwave_scoring import MicrowaveTask,ExtractionSample


class MicrowaveMealMixin:
    kind='microwave_box_to_meal_tray';asset='beef_box'
    selection_mode='contact';initial_workspace='microwave';required_workspaces=('microwave',)

    @classmethod
    def validate_parameters(cls,parameters):
        if set(parameters)!={'payload_variant'} or parameters['payload_variant'] not in ('beef_box','leftover_box'):
            raise ValueError('Explicit beef_box or leftover_box required')

    @classmethod
    def resolve_object(cls,parameters,manifest):
        from .native_port import one
        cls.validate_parameters(parameters)
        return one(manifest,parameters['payload_variant'])

    def __init__(self,stage,manifest,output,episode_id,instance_id,parameters):
        from .native_port import one
        from pxr import Usd,UsdPhysics,PhysxSchema,PhysicsSchemaTools
        from omni.physx.bindings._physx import ContactEventType
        from evaluation.tasks.tidy_sofa.isaac_reader import Contacts
        super().__init__(stage,manifest,output,episode_id,instance_id,parameters)
        self.source_path=one(manifest,'microwave')['path']
        self.microwave=MicrowaveGeometry(stage,self.geometry)
        self.device=MicrowaveDevice(stage);self.origin=None;self.handoff=None
        self.payload_variant=parameters['payload_variant']
        # Parent contact filter did not include the optional small plate. Keep
        # its own contact lifecycle so plate carrying does not invent box grip.
        def sleeping(sid,actor):
            p=stage.GetPrimAtPath(PhysicsSchemaTools.intToSdfPath(actor))
            if not p:raise RuntimeError('Unknown native actor')
            return bool(not p.HasAPI(UsdPhysics.RigidBodyAPI) or self.interface.is_sleeping(sid,actor))
        self.contacts=Contacts(PhysicsSchemaTools.intToSdfPath,sleeping,
            (ContactEventType.CONTACT_FOUND,ContactEventType.CONTACT_PERSIST,ContactEventType.CONTACT_LOST),
            # Scoring/contact wake needs robot/box/plate/tray pairs. Knob-panel
            # internal mechanism pairs are not task contacts; the device reads
            # its own current raw report. Finger-to-oven remains via robot.
            (self.geometry.target,self.geometry.destination,self.robot,PLATE))
        if self.microwave.plate:
            with Usd.EditContext(stage,stage.GetSessionLayer()):
                p=stage.GetPrimAtPath(self.microwave.plate.target_body)
                a=p.GetAttribute('physxContactReport:threshold')
                if a and a.HasAuthoredValueOpinion() and a.Get()!=0.:raise ValueError('Conflicting plate contact sensor')
                PhysxSchema.PhysxContactReportAPI.Apply(p).CreateThresholdAttr(0.)
        lo,hi=one(manifest,'microwave')['visible_bounds_world_m']
        self.activation=(tuple((lo[i]+hi[i])/2 for i in (0,1)),
            ((lo[0],lo[1]),(hi[0],lo[1]),(hi[0],hi[1]),(lo[0],hi[1])))
        self.mechanism_trace=(self.output/'mechanism_events.jsonl').open('x');self._mechanism_event_count=0

    @property
    def workspace_id(self):return 'microwave'

    @property
    def requested_physics_hz(self):return self.physics_hz

    def selection_geometry(self):return self.activation

    def link_fridge(self,owner,instance_id):
        from .fridge_handoff_v1.handoff import FridgeHandoff
        if dict(owner.binding.objects)['target']!=self.geometry.target_body:
            raise ValueError('Fridge and microwave must bind the same native target')
        self.handoff=FridgeHandoff(owner.composite,episode_id=self.episode_id,
            target=self.geometry.target_body,variant=self.payload_variant,parent_id=instance_id)
        # No dependent high-frequency reader: the fridge task is independently
        # contact-started and finishes its own real put evidence first.
        self.fridge_owner=owner

    def selection_request(self):
        return self.filter_wake_request(super().selection_request())

    def filter_wake_request(self,request):
        if self.handoff is None:raise RuntimeError('Missing fridge handoff binding')
        if not self.handoff.ready and self.handoff.parent.success:
            from openend.tasks.glasses_transfer.contacts import reduce_pairs
            poses={}
            for path in (self.geometry.target_body,self.microwave.body):
                t=np.asarray(self.views[path].get_transforms()).copy()[0]
                poses[path]=dict(position_m=tuple(map(float,t[:3])),quaternion_wxyz=tuple(map(float,t[[6,3,4,5]])))
            c=reduce_pairs(tuple(self.contacts.pairs()),target=self.geometry.target,
                source=self.source_path,bag=self.geometry.destination,cabinet=None,
                floors=self.floors,robot=self.robot,pads=self.pads)
            if self.handoff.accept(self.clock.current_time,
                    inside_oven=self.microwave.inside(self.geometry.vertices,
                        poses[self.geometry.target_body],poses[self.microwave.body]),
                    supports=c['supports'],door_closed=bool(abs(self.device._measure()['door_angle_deg'])<=2.)):
                self.initial_supports=self.handoff.supports
                (self.output/'fridge_handoff.json').write_text(json.dumps(self.handoff.summary(),indent=2)+'\n')
        if not self.handoff.permits_touch(request):
            request.update(active=False,locked=False,touching=False,last_touch_s=None)
        return request

    def initialize(self,clock,view):
        super().initialize(clock,view);self.device.initialize(clock,view)
        from openend.tasks.holding_follow_v1.detection import GraspDetector
        self.plate_detector=GraspDetector((self.microwave.plate.target_body,)) if self.microwave.plate else None
        extra=[self.microwave.body]
        if self.microwave.plate:extra.append(self.microwave.plate.target_body)
        for path in extra:
            v=view.create_rigid_body_view(path)
            if v.count!=1 or list(v.prim_paths)!=[path]:raise ValueError('Wrong native microwave body '+path)
            self.views[path]=v
        com=np.asarray(self.views[self.geometry.target_body].get_coms()).copy()[0,:3]
        if com.shape!=(3,) or not np.isfinite(com).all():raise ValueError('Missing native payload COM')
        self.microwave.box_com=com

    def _report(self,helper):
        from openend.tasks.all_objects_integration.selective_read import contact_report
        return contact_report(self,helper)

    def _mechanics(self,report):
        if self.device.gate is None:return
        self.device.after_physics(report)
        events=self.device.runtime.cycle.events
        if len(events)!=self._mechanism_event_count:
            self.mechanism_trace.write(json.dumps(dict(time_s=self.clock.current_time,
                new_events=events[self._mechanism_event_count:],state=self.device.gate.summary()))+'\n')
            self.mechanism_trace.flush();self._mechanism_event_count=len(events)

    def _plate_facts(self,row):
        plate=self.microwave.plate
        if plate is None:return None
        from evaluation.tasks.microwave_take.isaac_process import carrying_facts
        return carrying_facts(support_rows(row['pairs']),row['poses'][plate.target_body],
            row['poses'][self.geometry.target_body],self.microwave.box_com,
            self.microwave.plate_footprint,payload_root=self.geometry.target)

    def _wake_plate(self,pairs):
        if not self.microwave.plate or self.composite is None:return
        # A touch on the plate is relevant only while it actually supports this
        # box. No unrelated plate touch can select the box's task.
        plate_support=any(any(under(paths[i],self.geometry.target) and under(paths[1-i],PLATE)
            and any((1 if i==0 else -1)*n[2]>.5 for n,_,_ in samples) for i in (0,1))
            for paths,samples in pairs)
        finger=plate_support and any(any(under(paths[i],PLATE) and
            any(under(paths[1-i],p) for pads in self.pads.values() for p in pads)
            for i in (0,1)) for paths,samples in pairs)
        if finger:
            self.contact_selection.observe(time_s=self.clock.current_time,finger_contact=True,
                supports=frozenset(),external_contact=True,on_floor=False)

    def _wake_device(self,pairs):
        if self.composite is None:return
        # Actual finger contact on the oven/knob/door, not its sector or a
        # nearby table. Box contact is already observed by the parent port.
        def finger(path):
            return any(under(path,self.robot+'/gripper_'+h+'_'+side+'_link'+str(i))
                for h in ('l','r') for side in ('inner','outer') for i in range(1,5))
        touched=any(samples and any(under(paths[i],self.source_path) and finger(paths[1-i])
                    for i in (0,1)) for paths,samples in pairs)
        if touched:
            self.contact_selection.observe(time_s=self.clock.current_time,finger_contact=True,
                supports=frozenset(),external_contact=True,on_floor=False)

    def after_physics(self,helper_report):
        report=self._report(helper_report);helper=dict(helper_report,_contact_report=report)
        self._mechanics(report);super().after_physics(helper)
        row=self.latest;self._wake_plate(row['pairs']);self._wake_device(row['pairs'])
        plate_held=False
        if self.plate_detector is not None:
            held,details=self.plate_detector.observe(self.clock.current_time,row['poses'])
            plate_held=held[self.microwave.plate.target_body]
            row['plate_position_follow']=details
        if self.composite is not None:
            from pxr import PhysicsSchemaTools
            door=self.device._measure()['door_angle_deg']
            g=self.geometry;m=self.microwave;p=row['poses'];b=self._plate_facts(row)
            c=m.clearance(g.vertices,p[g.target_body],p[m.body])
            pc=None if m.plate is None else m.clearance(m.plate.vertices,p[m.plate.target_body],p[m.body])
            walls=current_contact_rows(report,PhysicsSchemaTools.intToSdfPath)
            row['sample']=ExtractionSample(self.identity,row['time_s']-self.origin,row['physics_dt_s'],
                self.device.gate.eligible_completed,float(door),c,row['held'],row['held'],
                not bool(self.initial_supports & row['supports']),m.plate is not None,pc,
                bool(plate_held),
                bool(plate_held),
                False if b is None else not b['plate_supported_by_stand'],
                False if b is None else b['payload_on_plate'],m.score_contacts(walls,p[m.body]),True)

    def after_inactive(self,helper_report):
        report=self._report(helper_report);self._mechanics(report)
        super().after_inactive(dict(helper_report,_contact_report=report))
        pairs=tuple(self.contacts.pairs());self._wake_plate(pairs);self._wake_device(pairs)

    def make_composite(self):
        m=self.microwave;g=self.geometry
        self.origin=self.clock.current_time
        checks={}
        if self.handoff is None:raise RuntimeError('Microwave requires its fridge-to-microwave predecessor')
        # Box starts in the fridge; the predecessor owns that initial recipe.
        # The plate keeps the unchanged in-oven initial check.
        for name,geo in ([] if m.plate is None else [('plate',m.plate)]):
            checks[name+'_inside_oven']=all(m.inside(geo.vertices,r['poses'][geo.target_body],r['poses'][m.body]) for r in self.history)
        if not all(checks.values()):raise RuntimeError('Microwave initial rejected '+str(checks))
        from .fridge_handoff_v1.handoff import HandoffIlluminationGate
        self.device.accept_initial(self.identity,gate_factory=lambda *a,**k:
            HandoffIlluminationGate(*a,handoff=self.handoff,**k))
        (self.output/'fridge_handoff.json').write_text(json.dumps(self.handoff.summary(),indent=2)+'\n')
        return MicrowaveTask(self.identity,m.walls)

    def _placement(self,row,index):
        frame=super()._placement(row,index)
        frame=replace(frame,forbidden_plate_on_destination=self.microwave.plate_on_tray(row['poses'],row['pairs']))
        return self.device.gate.bind_frame(frame)

    def consume_control(self,active):
        if not active or len(self.pending)!=self.clock.hz//20:raise RuntimeError('Dense 20 Hz microwave interval required')
        index=round((self.clock.current_time-self.origin)*20)
        if abs(self.pending[-1]['time_s']-self.clock.current_time)>1e-6:raise RuntimeError('Stale microwave evidence')
        frame=self._placement(self.pending[-1],index)
        self.score=self.composite.observe(frame,tuple(r['sample'] for r in self.pending))
        self.index=index;self.pending=[];self.paused=False
        self.score.update(runtime_registered=True,evidence_source='native_physx',full_trajectory_verified=False)
        self.score['fridge_handoff']=self.handoff.summary()
        self.trace.write(json.dumps(dict(step=index,placement=asdict(frame),score=self.score),default=list)+'\n');self.trace.flush()
        return self.score

    def finalize(self,reason):
        self.pending=[];poses={}
        for path,v in self.views.items():
            t=np.asarray(v.get_transforms()).copy()[0]
            poses[path]=dict(position_m=tuple(map(float,t[:3])),quaternion_wxyz=tuple(map(float,t[[6,3,4,5]])))
        index=round((self.clock.current_time-self.origin)*20)
        if index>self.composite.final.last.step:
            frame=self._placement(dict(poses=poses,pairs=tuple(self.contacts.pairs())),index)
            self.composite.settle_current(frame,self.device.gate)
        self.score=self.composite.finalize(reason)
        self.score.update(runtime_registered=True,evidence_source='native_physx',full_trajectory_verified=False)
        self.score['fridge_handoff']=self.handoff.summary()
        return self.score

    def close(self):
        if getattr(self,'device',None) is not None:self.device.close()
        if getattr(self,'mechanism_trace',None) is not None:self.mechanism_trace.close()
        super().close()
