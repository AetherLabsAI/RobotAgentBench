"""Native ports around existing approved scoring cores, not new reset APIs."""
from collections import deque
from dataclasses import asdict
from pathlib import Path
import json,math
import numpy as np
from openend.tasks.wallet_transfer.port import WalletPort
from openend.tasks.all_objects_integration.touch_wake import TouchWake as ContactSelection
from openend.tasks.object_transfers import binding,scoring,definitions

class ObjectPort(WalletPort):
    physics_hz=240
    selection_mode='contact'
    key=None
    mode=None
    def __init__(self,stage,manifest,output,episode_id,instance_id,parameters):
        self.validate_parameters(parameters)
        from .geometry import Geometry
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
    def validate_parameters(cls,parameters):
        if cls.asset=='paper_cup':
            if set(parameters)!={'target_path'} or not isinstance(parameters['target_path'],str) or not parameters['target_path'].startswith('/World/'):
                raise ValueError('Cup requires exact scene target_path')
        elif parameters:raise ValueError('No portable task overrides')

    def reduce_contacts(self):
        from .contacts import reduce
        g=self.geometry
        return reduce(self.contacts.pairs(),target=g.target,destination=g.destination,cabinet=g.cabinet,
            floors=g.floors,robot=self.robot,pads=self.pads,mode=self.mode)

    def initialize(self,clock,view):
        super().initialize(clock,view)
        self.extra_views={}
        for key,path in self.geometry.extra_bodies.items():
            v=view.create_rigid_body_view(path)
            if v.count!=1 or list(v.prim_paths)!=[path]:raise ValueError('Native auxiliary body mismatch '+path)
            self.extra_views[key]=v

    def after_physics(self,helper_report):
        super().after_physics(helper_report)
        # Sample geometry NOW at every physics substep, not later at a control endpoint.
        r=self.latest
        r['placement']=self.geometry.placement(r['pose'])
        r['drawer_travel_m']=0.
        if self.mode=='drawer':
            t=np.asarray(self.extra_views[self.geometry.destination].get_transforms()).copy()[0]
            inside,travel=self.geometry.drawer_measure(r['pose'],t)
            r['placement']=(inside,False,False);r['drawer_travel_m']=travel

    def initial_motion(self,rows):
        stable=bool(rows) and bool(np.ptp([r['pose'].position_m for r in rows],axis=0).max()<=self.config.initial_position_span_m)
        return dict(stable=stable),{}

    def accept_initial(self):
        from openend.tasks.meal_tray.nonmotion_closeout_v1.initial_windows import initial_window
        if self.composite is not None or self.latest is None:raise RuntimeError('Initial acceptance once')
        rows,complete=initial_window(self.history,self.config.initial_history_s,self.clock.get_physics_dt())
        motion,diagnostic=self.initial_motion(rows)
        checks=dict(history_complete=complete and all(r['physics_dt_s']==self.clock.get_physics_dt() for r in rows),
            supported=bool(rows) and all(bool(r['supports']) for r in rows),
            unheld=not any(r['held'] or r['robot_contact'] or r['on_floor'] for r in rows),
            **motion)
        receipt=dict(accepted=all(checks.values()),checks=checks,target=self.geometry.target,
            initial_supports=sorted(self.latest['supports']),evidence_source='native_physx',source_is_task_constraint=False,
            initial_motion_diagnostic=diagnostic)
        (self.output/'initial_acceptance.json').write_text(json.dumps(receipt,indent=2)+'\n')
        if not receipt['accepted']:raise RuntimeError(self.asset+' initial rejected '+str(checks))
        self.create_composite(self.latest)
        self.index=0;self.score=self.composite.observe(self.facts(self.latest,False,0));self.pending=[]
        (self.output/'binding.json').write_text(json.dumps(self.binding_receipt(),indent=2,default=list)+'\n')
        return receipt

    def binding_inventory(self):return self.manifest

    def create_composite(self,r):
        g=self.geometry;d=definitions.definition(self.key)
        self.binding=binding.bind(self.binding_inventory(),key=self.key,episode_id=self.episode_id,instance_id=self.instance_id,
            target_path=g.target,source_furniture=g.source_furniture(r['supports']),
            destinations={d.destination_role:dict(path=g.destination,furniture=g.cabinet)},initial_supports=r['supports'],
            initial_height_m=g.height(r['pose'],d.height_reference),height_reference=d.height_reference,initial_time_s=r['time_s'])
        self.composite=scoring.Transfer(self.binding,workspace_rates=getattr(self,'workspace_rates',False),
            allow_packed_support=getattr(self,'allow_packed_support',False))

    def binding_receipt(self):return asdict(self.binding)

    def consume_control(self,active):
        self.score=dict(super().consume_control(active),registered=True)
        return self.score

    def finalize(self,reason):
        self.score=dict(super().finalize(reason),registered=True)
        return self.score

    def facts(self,r,active,index):
        g=self.geometry;d=definitions.definition(self.key);inside,rim,fully=r['placement']
        return scoring.Facts(self.episode_id,self.instance_id,g.target,g.destination,index,r['time_s'],
            g.height(r['pose'],d.height_reference),d.height_reference,r['held'],r['robot_contact'],r['finger_contact'],
            r['supports'],r['on_floor'],inside,rim,fully,r['supporting_hooks'],r['drawer_travel_m'],
            r['furniture_contacts'],frozenset(),r['external_contact'],active,True,r['physics_dt_s'])

class WaterBottlePort(ObjectPort):
    asset='water_bottle';key='water_bottle_to_bag';mode='bag'
class BookPort(ObjectPort):
    asset='book_red';key='book_to_small_table';mode='surface'
    @staticmethod
    def resolve_object(parameters,manifest):
        if parameters:raise ValueError('No portable task overrides')
        rows=[o for o in manifest['objects'] if o['id']=='book_red' and o.get('task_role')!='cloth_clear_target']
        if len(rows)!=1:raise ValueError('One non-cloth book-transfer target required')
        return rows[0]
    def binding_inventory(self):
        # The old single-book component sees only its task's book; the room
        # retains every cloth target and its independent group ownership.
        return dict(self.manifest,objects=[o for o in self.manifest['objects']
            if not (o['id']=='book_red' and o.get('task_role')=='cloth_clear_target')])
class CapPort(ObjectPort):
    asset='baseball_cap';key='cap_to_rack';mode='hang'
    def initial_motion(self,rows):
        from .cap_initial import evaluate
        if len(rows)<2:return dict(no_sustained_drift=False),{}
        centres=[]
        for r in rows:
            p=self.geometry.world(r['pose'],self.geometry.visible)
            centres.append((p.min(0)+p.max(0))/2)
        diagnostic=evaluate(rows,centres,self.config.initial_position_span_m)
        return dict(no_sustained_drift=diagnostic['no_sustained_drift']),diagnostic
class NotebookPort(ObjectPort):
    asset='notebook_green';key='note_to_drawer';mode='drawer'
class PaperCupPort(ObjectPort):
    asset='paper_cup';key='paper_cup_to_bin';mode='trash'

class PowerBankPort(ObjectPort):
    asset='power_bank';mode='bag'
    def create_composite(self,r):
        from openend.tasks.powerbank_transfer.task_binding import PowerBankBinding,Snapshot
        from openend.tasks.powerbank_transfer.composite import PowerBankTransfer
        g=self.geometry
        self.binding=PowerBankBinding(self.manifest,episode_id=self.episode_id,source_path=g.source_furniture(r['supports']),
            bag_path=g.destination,cabinet_path=g.cabinet,initial_supports=r['supports'],
            initial=Snapshot(self.episode_id,g.target,0,r['time_s'],r['pose']))
        self.composite=PowerBankTransfer(self.binding,workspace_rates=getattr(self,'workspace_rates',False),
            allow_packed_support=getattr(self,'allow_packed_support',False))
    def binding_receipt(self):return self.binding.receipt()
    def facts(self,r,active,index):
        from openend.tasks.powerbank_transfer.composite import Facts
        g=self.geometry;inside,rim,_=r['placement']
        return Facts(self.episode_id,g.target,g.destination,index,r['time_s'],r['pose'],r['held'],r['robot_contact'],
            r['supports'],r['on_floor'],inside,rim,r['external_contact'],active,True,r['physics_dt_s'],r['furniture_contacts'])

FACTORIES={'powerbank_transfer':PowerBankPort,'water_bottle_transfer':WaterBottlePort,
 'book_transfer':BookPort,'cap_transfer':CapPort,'notebook_transfer':NotebookPort,'paper_cup_transfer':PaperCupPort}
