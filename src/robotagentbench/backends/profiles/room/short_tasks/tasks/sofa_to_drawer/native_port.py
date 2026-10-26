"""Native readers for task-owned combinations; common PhysX/holding interfaces."""
from collections import deque
from dataclasses import asdict
from pathlib import Path
import json, math
from openend.tasks.all_objects_integration.ports import ObjectPort
from openend.tasks.all_objects_integration.touch_wake import TouchWake as ContactSelection
from openend.tasks.object_transfers.binding import Binding
from openend.tasks.object_transfers.scoring import Facts
from .transfer import Transfer


class CombinationPort(ObjectPort):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def __init__(self,stage,manifest,output,episode_id,instance_id,parameters):
        self.validate_parameters(parameters)
        from .native_geometry import Geometry
        from pxr import Usd,UsdPhysics,PhysxSchema,PhysicsSchemaTools
        from omni.physx import get_physx_simulation_interface
        from omni.physx.bindings._physx import ContactEventType
        from evaluation.tasks.tidy_sofa.isaac_reader import Contacts,HANDS,PADS,ROBOT
        from openend.tasks.holding_follow_v1.detection import DetectionConfig, GraspDetector
        self.stage=stage;self.manifest=manifest;self.episode_id=episode_id;self.instance_id=instance_id
        self.geometry=Geometry(stage,manifest,self.asset,self.mode,parameters.get('target_path'));g=self.geometry
        self.config=DetectionConfig();self.detector=GraspDetector((g.target,),self.config)
        self.hands=HANDS;self.pads=PADS;self.robot=ROBOT;self.interface=get_physx_simulation_interface()
        if not stage.GetSessionLayer().anonymous:raise ValueError('Anonymous session required')
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
        self.contact_selection=ContactSelection()
        (self.output/'geometry.json').write_text(json.dumps(dict(target=g.target,body=g.body,destination=g.destination,
            mode=self.mode,asset=self.asset,instance_id=instance_id,detector=asdict(self.config),
            native_full_trajectory_verified=False,scene_pose_writes=0,physics_parameters_changed=False,
            collider_parts=len(g.parts),cube_parts=sorted(g.cube_parts),initial_pose=g.initial_pose),indent=2)+'\n')

    @classmethod
    def validate_parameters(cls, parameters):
        if parameters: raise ValueError('No physical or scoring overrides')

    def create_composite(self, row):
        g=self.geometry
        self.binding=Binding(self.key,self.episode_id,self.instance_id,g.target,
            g.source_furniture(row['supports']),g.destination,g.cabinet,
            row['supports'],g.height(row['pose'],'rigid_origin'),row['time_s'],frozenset())
        self.composite=Transfer(self.binding,self.asset,self.mode,
            workspace_rates=getattr(self,'workspace_rates',False))

    def facts(self, row, active, index):
        g=self.geometry; inside,rim,fully=row['placement']
        return Facts(self.episode_id,self.instance_id,g.target,g.destination,index,row['time_s'],
            g.height(row['pose'],'rigid_origin'),'rigid_origin',row['held'],row['robot_contact'],
            row['finger_contact'],row['supports'],row['on_floor'],inside,rim,fully,
            row['supporting_hooks'],row['drawer_travel_m'],row['furniture_contacts'],
            frozenset(),row['external_contact'],active,True,row['physics_dt_s'])

    def finalize(self, reason):
        if reason=='success':
            if not self.composite.summary()['success']:
                raise ValueError('Current native success required')
            self.composite.reason='success'
        return super().finalize({'no_new_net_progress_timeout':'no_score_increase_timeout'}.get(reason,reason))


class WalletDrawerPort(CombinationPort):
    asset='wallet'; mode='drawer'; key='wallet_to_drawer'


class NotebookDrawerPort(CombinationPort):
    asset='notebook_green'; mode='drawer'; key='note_to_drawer'
