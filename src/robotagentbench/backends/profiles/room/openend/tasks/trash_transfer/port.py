"""All garbage classes share one native task port and independent scorers."""
from collections import deque
from dataclasses import asdict
from pathlib import Path
import json,math
from openend.tasks.all_objects_integration.ports import ObjectPort
from openend.tasks.all_objects_integration.touch_wake import TouchWake
from openend.tasks.object_transfers.binding import Binding
from openend.tasks.object_transfers.scoring import Facts
from .registration import validate_parameters, resolve_object
from .scoring import TrashTransfer


class TrashPort(ObjectPort):
    asset=None; key='trash_transfer'; mode='trash'
    validate_parameters=staticmethod(validate_parameters)
    resolve_object=staticmethod(resolve_object)
    preflight=staticmethod(resolve_object)

    def __init__(self,stage,manifest,output,episode_id,instance_id,parameters):
        obj=self.preflight(parameters,manifest)
        self.category=obj['trash_category']; self.asset=obj['id']
        from .geometry import TrashGeometry
        from pxr import Usd,UsdPhysics,PhysxSchema,PhysicsSchemaTools
        from omni.physx import get_physx_simulation_interface
        from omni.physx.bindings._physx import ContactEventType
        from evaluation.tasks.tidy_sofa.isaac_reader import Contacts,HANDS,PADS,ROBOT
        from openend.tasks.holding_follow_v1.detection import DetectionConfig, GraspDetector
        self.stage=stage;self.manifest=manifest;self.episode_id=episode_id;self.instance_id=instance_id
        self.geometry=TrashGeometry(stage,manifest,parameters);g=self.geometry
        self.config=DetectionConfig();self.detector=GraspDetector((g.target,),self.config)
        self.hands=HANDS;self.pads=PADS;self.robot=ROBOT;self.interface=get_physx_simulation_interface()
        if not stage.GetSessionLayer().anonymous:raise ValueError('Anonymous session required')
        # Existing readout instrumentation only. No body, drive or physics edits.
        with Usd.EditContext(stage,stage.GetSessionLayer()):
            bodies=[stage.GetPrimAtPath(g.body)]+[p for p in Usd.PrimRange(stage.GetPrimAtPath(ROBOT)) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
            for p in bodies:
                a=p.GetAttribute('physxContactReport:threshold')
                if a and a.HasAuthoredValueOpinion() and a.Get()!=0:raise ValueError('Conflicting contact instrumentation')
                PhysxSchema.PhysxContactReportAPI.Apply(p).CreateThresholdAttr(0.)
        def sleeping(sid,actor):
            p=stage.GetPrimAtPath(PhysicsSchemaTools.intToSdfPath(actor))
            if not p:raise RuntimeError('Unknown contact actor')
            return bool(not p.HasAPI(UsdPhysics.RigidBodyAPI) or self.interface.is_sleeping(sid,actor))
        self.contacts=Contacts(PhysicsSchemaTools.intToSdfPath,sleeping,
            (ContactEventType.CONTACT_FOUND,ContactEventType.CONTACT_PERSIST,ContactEventType.CONTACT_LOST),(g.target,ROBOT))
        self.output=Path(output);self.output.mkdir(exist_ok=False);self.trace=(self.output/'facts.jsonl').open('x')
        self.history=deque(maxlen=math.ceil(self.config.initial_history_s*240)+2)
        self.latest=None;self.pending=[];self.composite=None;self.score=None;self.index=-1;self.closed=False;self.binding=None
        self.contact_selection=TouchWake()
        (self.output/'geometry.json').write_text(json.dumps(dict(target=g.target,body=g.body,destination=g.destination,
            category=self.category,mode=self.mode,asset=self.asset,instance_id=instance_id,detector=asdict(self.config),
            native_full_trajectory_verified=False,scene_pose_writes=0,physics_parameters_changed=False,
            collider_parts=len(g.parts),initial_pose=g.initial_pose),indent=2)+'\n')

    def create_composite(self,r):
        g=self.geometry
        self.binding=Binding(self.key,self.episode_id,self.instance_id,g.target,
            g.source_furniture(r['supports']),g.destination,g.cabinet,r['supports'],
            g.height(r['pose'],'rigid_origin'),r['time_s'],frozenset())
        self.composite=TrashTransfer(self.binding,self.category,workspace_rates=getattr(self,'workspace_rates',False))

    def facts(self,r,active,index):
        g=self.geometry;inside,rim,fully=r['placement']
        return Facts(self.episode_id,self.instance_id,g.target,g.destination,index,r['time_s'],
            g.height(r['pose'],'rigid_origin'),'rigid_origin',r['held'],r['robot_contact'],r['finger_contact'],
            r['supports'],r['on_floor'],inside,rim,fully,r['supporting_hooks'],r['drawer_travel_m'],
            r['furniture_contacts'],frozenset(),r['external_contact'],active,True,r['physics_dt_s'])
