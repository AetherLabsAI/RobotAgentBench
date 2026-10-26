"""Scene-owned wallet adapter for MultiTaskRoom, native readout only.

Dense physical samples are buffered until the room admits the completed control
frame. Nonselected samples remain inactive, preserving physical time and state.
"""
from collections import deque
from dataclasses import asdict,replace
from pathlib import Path
import json,math
from openend.tasks.all_objects_integration.selective_read import contact_report, observe_selection, selection_request, inactive_object
import numpy as np
from .task_binding import Pose,Snapshot,WalletTaskBinding
from .composite import Facts,WalletTransfer
from openend.tasks.all_objects_integration.touch_wake import TouchWake as ContactSelection


class WalletPort:
    asset='wallet'
    physics_hz=240
    selection_mode='contact'

    @staticmethod
    def validate_parameters(parameters):
        from openend.tasks.destination_transfers.registration import destination
        destination(parameters)

    def make_geometry(self, stage, manifest, parameters):
        from openend.tasks.destination_transfers.registration import destination
        if destination(parameters) == 'storage_basket':
            from openend.tasks.destination_transfers.geometry import BasketGeometry
            return BasketGeometry(stage, manifest, self.asset)
        from .native_geometry import SceneGeometry
        return SceneGeometry(stage, manifest)

    def __init__(self,stage,manifest,output,episode_id,instance_id,parameters):
        self.validate_parameters(parameters)
        from pxr import Usd,UsdPhysics,PhysxSchema,PhysicsSchemaTools
        from omni.physx import get_physx_simulation_interface
        from omni.physx.bindings._physx import ContactEventType
        from evaluation.tasks.tidy_sofa.isaac_reader import Contacts,HANDS,PADS,ROBOT
        from openend.tasks.holding_follow_v1.detection import DetectionConfig, GraspDetector
        from .native_geometry import SceneGeometry
        self.stage=stage;self.manifest=manifest;self.episode_id=episode_id
        self.geometry=self.make_geometry(stage,manifest,parameters);g=self.geometry
        self.config=DetectionConfig();self.detector=GraspDetector((g.target,),self.config)
        self.hands=HANDS;self.pads=PADS;self.robot=ROBOT
        self.interface=get_physx_simulation_interface()
        if not stage.GetSessionLayer().anonymous:raise ValueError('Anonymous episode session required')
        with Usd.EditContext(stage,stage.GetSessionLayer()):
            bodies=[stage.GetPrimAtPath(g.body)]+[p for p in Usd.PrimRange(stage.GetPrimAtPath(ROBOT)) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
            for p in bodies:
                attr=p.GetAttribute('physxContactReport:threshold')
                if attr and attr.HasAuthoredValueOpinion() and attr.Get()!=0:raise ValueError('Conflicting contact instrumentation')
                PhysxSchema.PhysxContactReportAPI.Apply(p).CreateThresholdAttr(0.)
        def sleeping(sid,actor):
            p=stage.GetPrimAtPath(PhysicsSchemaTools.intToSdfPath(actor))
            if not p:raise RuntimeError('Unknown contact actor')
            return bool(not p.HasAPI(UsdPhysics.RigidBodyAPI) or self.interface.is_sleeping(sid,actor))
        self.contacts=Contacts(PhysicsSchemaTools.intToSdfPath,sleeping,
            (ContactEventType.CONTACT_FOUND,ContactEventType.CONTACT_PERSIST,ContactEventType.CONTACT_LOST),(g.target,ROBOT))
        self.output=Path(output);self.output.mkdir(exist_ok=False)
        self.trace=(self.output/'facts.jsonl').open('x')
        self.history=deque(maxlen=math.ceil(self.config.initial_history_s*240)+2)
        self.latest=None;self.pending=[];self.composite=None;self.score=None;self.index=-1;self.closed=False
        self.binding=None
        self.contact_selection=ContactSelection()
        (self.output/'geometry.json').write_text(json.dumps(dict(target=g.target,body=g.body,source=g.source,
            destination=g.destination,source_top=g.source_top,destination_top=g.top,exclusions=g.exclusions,
            initial_pose=g.initial_pose,detector=asdict(self.config),scene_pose_writes=0,physics_parameters_changed=False,
            insertion_predicate=g.top.get('semantics','legacy_origin_lowest_vertex_and_collider_centres'),
            whole_object_containment_verified=False,
            raw_mesh_vs_cooked_convex_contact_may_withhold_placement=True),indent=2)+'\n')

    @property
    def ended(self):return self.composite is not None and self.composite.ended

    def initialize(self,clock,view):
        self.clock=clock;self.last_time=clock.current_time
        self.views={}
        for key,path in [(self.geometry.target,self.geometry.body),*((p,p) for p in self.hands.values())]:
            v=view.create_rigid_body_view(path)
            if v.count!=1 or list(v.prim_paths)!=[path]:raise ValueError('Native body mismatch '+path)
            self.views[key]=v

    def reduce_contacts(self):
        from .contacts import reduce_pairs
        g=self.geometry
        result=reduce_pairs(self.contacts.pairs(),target=g.target,source=g.source,
            bag=getattr(g,'contact_destination',g.destination),cabinet=g.cabinet,floors=g.floors,robot=self.robot,pads=self.pads)
        result['supports']=frozenset(g.destination if p==getattr(g,'contact_destination',g.destination) else p for p in result['supports'])
        return result

    def after_physics(self,helper_report):
        if helper_report.get('events') or helper_report.get('docked_object_count') or helper_report.get('tray_dock_enabled'):
            raise RuntimeError('Helper-modified physical evidence')
        now=self.clock.current_time;dt=1/self.clock.hz
        if self.clock.hz not in (60,240) or abs(now-self.last_time-dt)>1e-6:raise RuntimeError('Missing native substep')
        self.geometry.validate_static(self.stage)
        poses={}
        for p,v in self.views.items():
            t=np.asarray(v.get_transforms()).copy()[0]
            pose=Pose(tuple(map(float,t[:3])),tuple(map(float,t[[6,3,4,5]])))
            poses[p]=asdict(pose)
        self.contacts.ingest(contact_report(self,helper_report));c=self.reduce_contacts()
        held,_=self.detector.observe(now,poses,c['bilateral'])
        g=self.geometry;pose=Pose(**poses[g.target])
        row=dict(time_s=now,pose=pose,held=bool(held[g.target]),physics_dt_s=dt,
                 footprint=g.footprint(pose),**{k:v for k,v in c.items() if k!='bilateral'})
        self.latest=row;self.history.append(row);self.last_time=now
        if self.composite is not None:
            observe_selection(self,row)
            self.pending.append(row)

    def after_inactive(self,helper_report):
        inactive_object(self,helper_report)

    def accept_initial(self):
        from openend.tasks.meal_tray.nonmotion_closeout_v1.initial_windows import initial_window
        if self.composite is not None or self.latest is None:raise RuntimeError('Initial acceptance once')
        rows,complete=initial_window(self.history,self.config.initial_history_s,self.clock.get_physics_dt())
        checks=dict(history_complete=complete and all(r['physics_dt_s']==self.clock.get_physics_dt() for r in rows),
            supported=bool(rows) and all(bool(r['supports']) for r in rows),
            unheld=not any(r['held'] or r['robot_contact'] or r['on_floor'] for r in rows),
            stable=bool(rows) and bool(np.ptp([r['pose'].position_m for r in rows],axis=0).max()<=self.config.initial_position_span_m))
        receipt=dict(accepted=all(checks.values()),checks=checks,evidence_source='native_physx',scene_pose_writes=0,
            initial_supports=sorted(self.latest['supports']),
            support_history_union=sorted(set().union(*(r['supports'] for r in rows))),source_is_task_constraint=False)
        (self.output/'initial_acceptance.json').write_text(json.dumps(receipt,indent=2)+'\n')
        if not receipt['accepted']:raise RuntimeError('Wallet initial rejected '+str(checks))
        g=self.geometry;r=self.latest
        self.binding=WalletTaskBinding(self.manifest,episode_id=self.episode_id,
            source_path=sorted(r['supports'])[0],initial_supports=r['supports'],bag_path=g.destination,cabinet_path=g.cabinet,
            initial=Snapshot(self.episode_id,g.target,0,r['time_s'],r['pose']))
        self.composite=WalletTransfer(self.binding,workspace_rates=getattr(self,'workspace_rates',False),
            allow_packed_support=getattr(self,'allow_packed_support',False));self.index=0
        self.score=self.composite.observe(self.facts(r,False,0));self.pending=[]
        return receipt

    def facts(self,row,active,index):
        g=self.geometry
        inside,rim=g.insertion_evidence(row['pose'])
        return Facts(self.episode_id,g.target,g.destination,index,inside_cavity=inside,
            rim_only=rim,active=active,**{k:v for k,v in row.items() if k not in ('finger_contact','footprint')})

    def selection_request(self):
        return selection_request(self)

    def consume_control(self,active):
        if not self.pending:raise RuntimeError('Missing wallet evidence batch')
        rows,self.pending=self.pending,[]
        for offset,r in enumerate(rows):
            if self.ended:break
            self.index+=1;f=self.facts(r,active,self.index)
            self.score=self.composite.observe(f,final_sample=offset==len(rows)-1)
            self.trace.write(json.dumps(dict(facts=asdict(f),score=self.score['score']),default=list)+'\n')
        self.score=dict(self.score,evidence_status='native_physx',native_full_trajectory_verified=False)
        self.trace.flush()
        return self.score

    def selection_geometry(self):
        if self.latest is None:return None
        p=self.latest['footprint']
        return tuple(sum(v[k] for v in p)/len(p) for k in (0,1)),p

    def pause(self):
        if not self.ended:self.composite.pick.pause();self.composite.place.pause()

    def finalize(self,reason):
        self.pending=[]
        self.score=dict(self.composite.finalize('success' if self.composite.reason=='success' else reason),
                        evidence_status='native_physx',native_full_trajectory_verified=False)
        return self.score

    def close(self):
        if not self.closed:self.trace.close();self.closed=True
