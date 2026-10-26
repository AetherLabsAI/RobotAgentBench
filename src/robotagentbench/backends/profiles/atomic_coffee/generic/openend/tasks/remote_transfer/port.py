"""Scene-owned remote adapter for MultiTaskRoom, native readout only.

Dense physical samples are buffered until the room admits the completed control
frame. Nonselected samples remain inactive, preserving physical time and state.
"""
from collections import deque
from dataclasses import asdict,replace
from pathlib import Path
import json,math
from openend.tasks.all_objects_integration.selective_read import contact_report, observe_selection, selection_request, inactive_object
import numpy as np
from .binding import Pose,Snapshot,RemoteBinding
from .composite import Facts,RemoteTransfer
from openend.tasks.all_objects_integration.touch_wake import TouchWake as ContactSelection


class RemotePort:
    asset='remote_tv'
    physics_hz=240
    selection_mode='contact'

    def __init__(self,stage,manifest,output,episode_id,instance_id,parameters):
        if parameters:raise ValueError('Remote transfer has no overrides')
        from pxr import Usd,UsdPhysics,PhysxSchema,PhysicsSchemaTools
        from omni.physx import get_physx_simulation_interface
        from omni.physx.bindings._physx import ContactEventType
        from evaluation.tasks.tidy_sofa.isaac_reader import Contacts,HANDS,PADS,ROBOT
        from openend.tasks.holding_follow_v1.detection import DetectionConfig, GraspDetector
        from .native_geometry import SceneGeometry
        self.stage=stage;self.manifest=manifest;self.episode_id=episode_id
        self.geometry=SceneGeometry(stage,manifest);g=self.geometry
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
            initial_pose=g.initial_pose,detector=asdict(self.config),scene_pose_writes=0,physics_parameters_changed=False),indent=2)+'\n')

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
        from evaluation.tasks.tidy_sofa.isaac_reader import under
        g=self.geometry;supports=set();floor=False;robot=False;finger=False;external=False;furniture=set()
        normals={h:[[0.,0.,0.],[0.,0.,0.]] for h in self.hands}
        for paths,samples in self.contacts.pairs():
            for i in (0,1):
                subject,other=paths[i],paths[1-i]
                if under(subject,self.robot):
                    furniture.update(p for p in g.furniture if under(other,p))
                if not under(subject,g.target):continue
                sign=1 if i==0 else -1
                if under(other,self.robot):
                    robot=True
                    for h,pads in self.pads.items():
                        for k,pad in enumerate(pads):
                            if under(other,pad):
                                finger=True
                                for n,_,_ in samples:
                                    for a in range(3):normals[h][k][a]+=sign*n[a]
                else:
                    external=True
                    floor|=any(under(other,p) for p in g.floors)
                    if any(sign*n[2]>.5 for n,_,_ in samples):
                        if under(other,g.destination):
                            # Reject shelf/side support: actual up-facing contact must be at measured top.
                            top=any(sign*n[2]>.5 and abs(p[2]-g.top['z'])<=self.config.support_face_tolerance_m for n,p,_ in samples)
                            supports.add(g.destination if top else other)
                        elif g.source is not None and under(other,g.source):supports.add(g.source)
                        else:supports.add(other)
        bilateral={g.target:{h:sum(a*b for a,b in zip(*v))<0 for h,v in normals.items()}}
        return dict(supports=frozenset(supports),on_floor=bool(floor),robot_contact=robot,
                    furniture_contacts=frozenset(furniture),bilateral=bilateral,external_contact=external,finger_contact=finger)

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
        if not receipt['accepted']:raise RuntimeError('Remote initial rejected '+str(checks))
        g=self.geometry;r=self.latest
        self.binding=RemoteBinding(self.manifest,episode_id=self.episode_id,source_region='measured_native_contacts',
            source_furniture_path=sorted(r['supports'])[0],initial_supports=r['supports'],destination_path=g.destination,
            initial=Snapshot(self.episode_id,g.target,0,r['time_s'],r['pose']))
        self.composite=RemoteTransfer(self.binding,workspace_rates=getattr(self,'workspace_rates',False));self.index=0
        self.score=self.composite.observe(self.facts(r,False,0));self.pending=[]
        return receipt

    def facts(self,row,active,index):
        g=self.geometry
        return Facts(self.episode_id,g.target,g.destination,index,tabletop=g.top['polygon'],
                     exclusions=tuple(g.exclusions),active=active,**{k:v for k,v in row.items() if k!='finger_contact'})

    def selection_request(self):
        return selection_request(self)

    def consume_control(self,active):
        if not self.pending:raise RuntimeError('Missing remote evidence batch')
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
