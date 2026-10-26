"""Native contact/pose ports for cutlery and shared coffee/fryer extraction.

Scene generation owns initial poses. This reader never writes body transforms,
drives, material/collision settings, or solver rates. Device physics is owned by
the existing preparation port, advanced once by MultiTaskRoom.
"""
from collections import deque
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import json
import math

from .final_state import Binding
from .scoring import MealTrayTask, PickupFrame
from .additional_scoring import AdditionalMealTask, OperationFrame, CutleryFacts
from .geometry import Geometry


def one(manifest, asset, target_path=None):
    rows=[o for o in manifest['objects'] if o['id']==asset and
          (target_path is None or o['path']==target_path)]
    if len(rows)!=1:raise ValueError('One exact scene-owned instance required: '+asset)
    return rows[0]


class MealPort:
    selection_mode='contact'
    physics_hz=240  # registration metadata; actual Hz belongs to the workspace
    kind=None
    asset=None

    @classmethod
    def validate_parameters(cls, parameters):
        if cls.kind=='cutlery_to_meal_tray':
            if (set(parameters)!={'asset','target_path'} or parameters['asset'] not in ('meal_fork','meal_knife')
                    or not isinstance(parameters['target_path'],str) or not parameters['target_path'].startswith('/World/')):
                raise ValueError('Explicit fork/knife and exact target_path required')
        elif parameters:
            raise ValueError('Scene identities only; no meal parameter overrides')

    @classmethod
    def resolve_object(cls, parameters, manifest):
        cls.validate_parameters(parameters)
        if cls.kind=='cutlery_to_meal_tray':
            return one(manifest,parameters['asset'],parameters['target_path'])
        if cls.kind=='bread_to_meal_tray':
            rows=[o for o in manifest['objects'] if o.get('meal_task_id') and
                  o['id'] in ('cake_plain_yellow','cake_frosted_white')]
            if len(rows)!=1:raise ValueError('Scene must mark exactly one selected bread meal target')
            return rows[0]
        return one(manifest,cls.asset)

    def __init__(self,stage,manifest,output,episode_id,instance_id,parameters):
        from pxr import Usd,UsdPhysics,PhysxSchema,PhysicsSchemaTools
        from omni.physx import get_physx_simulation_interface
        from omni.physx.bindings._physx import ContactEventType
        from evaluation.tasks.tidy_sofa.isaac_reader import Contacts,HANDS,PADS,ROBOT
        from openend.tasks.holding_follow_v1.detection import DetectionConfig, GraspDetector
        from openend.tasks.all_objects_integration.touch_wake import TouchWake
        from openend.tasks.scene_requirements.source_geometry import floor_paths
        target=self.resolve_object(parameters,manifest)
        tray=one(manifest,'meal_tray_large')
        self.geometry=Geometry(stage,target['path'],tray['path'])
        g=self.geometry
        self.stage=stage;self.episode_id=episode_id;self.instance_id=instance_id
        self.source_path=one(manifest,'cutlery_holder')['path'] if self.kind=='cutlery_to_meal_tray' else None
        self.device_owner=None;self.reader_dependencies=()
        self.binding=SimpleNamespace(objects=dict(target=g.target_body,destination=g.tray_body))
        self.config=DetectionConfig();self.detector=GraspDetector((g.target_body,),self.config)
        self.hands=HANDS;self.pads=PADS;self.robot=ROBOT
        self.floors=floor_paths(stage)
        if not self.floors:raise ValueError('Actual layout floor colliders required')
        self.interface=get_physx_simulation_interface()
        if not stage.GetSessionLayer().anonymous:raise ValueError('Anonymous episode sensor session required')
        reporters=[stage.GetPrimAtPath(p) for p in (g.target_body,g.tray_body)]
        reporters += [p for p in Usd.PrimRange(stage.GetPrimAtPath(ROBOT)) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        with Usd.EditContext(stage,stage.GetSessionLayer()):
            for p in reporters:
                a=p.GetAttribute('physxContactReport:threshold')
                if a and a.HasAuthoredValueOpinion() and a.Get()!=0:raise ValueError('Conflicting contact sensor threshold')
                PhysxSchema.PhysxContactReportAPI.Apply(p).CreateThresholdAttr(0.)
        def sleeping(sid,actor):
            p=stage.GetPrimAtPath(PhysicsSchemaTools.intToSdfPath(actor))
            if not p:raise RuntimeError('Unknown native contact actor')
            return bool(not p.HasAPI(UsdPhysics.RigidBodyAPI) or self.interface.is_sleeping(sid,actor))
        self.contacts=Contacts(PhysicsSchemaTools.intToSdfPath,sleeping,
            (ContactEventType.CONTACT_FOUND,ContactEventType.CONTACT_PERSIST,ContactEventType.CONTACT_LOST),
            (g.target,g.destination,ROBOT))
        self.contact_selection=TouchWake()
        self.output=Path(output);self.output.mkdir(exist_ok=False)
        self.trace=(self.output/'native_facts.jsonl').open('x')
        self.history=deque(maxlen=math.ceil(self.config.initial_history_s*240)+2)
        self.composite=None;self.score=None;self.latest=None;self.index=-1
        self.pending=[];self.closed=False;self.paused=False
        (self.output/'geometry.json').write_text(json.dumps(g.receipt(),indent=2)+'\n')

    @property
    def ended(self):return self.composite is not None and self.composite.ended

    @property
    def reader_dependencies(self):
        # Before preparation completes the parent owns its selected readout.
        # Touching an unprepared cup must not accumulate unselected operation
        # evidence in that parent's window.
        owner=getattr(self,'device_owner',None)
        if self.composite is not None and self.composite.summary().get('operation_complete',False):
            return ()
        clock=getattr(self,'clock',None)
        if clock is not None and owner is not None:
            from .task_contact_routing import rate_compatible
            if not rate_compatible(owner,clock.hz):return ()
        return (self._device_dependencies if owner is not None and
                getattr(getattr(owner,'composite',None),'success',False) else ())

    @reader_dependencies.setter
    def reader_dependencies(self,value):self._device_dependencies=tuple(value)

    def initialize(self,clock,view):
        self.clock=clock;self.last_time=clock.current_time;self.views={}
        for p in (self.geometry.target_body,self.geometry.tray_body,*self.hands.values()):
            v=view.create_rigid_body_view(p)
            if v.count!=1 or list(v.prim_paths)!=[p]:raise ValueError('Native meal body mismatch '+p)
            self.views[p]=v

    def _tick(self,helper):
        if helper.get('events') or helper.get('docked_object_count') or helper.get('tray_dock_enabled'):
            raise RuntimeError('Assisted manipulation cannot be scored')
        now=self.clock.current_time;dt=1/self.clock.hz
        if self.clock.hz not in (60,240) or abs(now-self.last_time-dt)>1e-6:
            raise RuntimeError('Missing meal native substep')
        from openend.tasks.all_objects_integration.selective_read import contact_report,observe_selection
        self.contacts.ingest(contact_report(self,helper))
        from openend.tasks.glasses_transfer.contacts import reduce_pairs
        g=self.geometry
        c=reduce_pairs(self.contacts.pairs(),target=g.target,source=self.source_path,bag=g.destination,
            cabinet=None,floors=self.floors,robot=self.robot,pads=self.pads)
        # Grasp detector keys use the actual native body, not its scene wrapper.
        c['bilateral']={g.target_body:c['bilateral'][g.target]}
        if self.composite is not None:observe_selection(self,dict(c,time_s=now))
        self.last_time=now
        return c

    def after_physics(self,helper_report):
        import numpy as np
        c=self._tick(helper_report);poses={}
        for p,v in self.views.items():
            t=np.asarray(v.get_transforms()).copy()[0]
            poses[p]=dict(position_m=tuple(map(float,t[:3])),quaternion_wxyz=tuple(map(float,t[[6,3,4,5]])))
        held,_=self.detector.observe(self.clock.current_time,poses,c['bilateral'])
        g=self.geometry
        row=dict(time_s=self.clock.current_time,physics_dt_s=1/self.clock.hz,poses=poses,
                 pairs=tuple(self.contacts.pairs()),held=bool(held[g.target_body]),
                 operation_ready=bool(self.reader_dependencies),
                 **{k:v for k,v in c.items() if k!='bilateral'})
        self.latest=row;self.history.append(row)
        if self.composite is not None:self.pending.append(row)

    def after_inactive(self,helper_report):
        if self.pending:raise RuntimeError('Unconsumed meal control evidence')
        self._tick(helper_report)
        for h in self.detector.history.values():h.clear()
        if self.composite is not None:
            self.composite.pause();self.paused=True

    def _placement(self,row,index):
        g=self.geometry;p=row['poses']
        pose=lambda key:(p[key]['position_m'],p[key]['quaternion_wxyz'])
        return g.frame(self.identity,index,pose(g.target_body),pose(g.tray_body),row['pairs'],self.floors)

    def accept_initial(self):
        import numpy as np
        from openend.tasks.meal_tray.nonmotion_closeout_v1.initial_windows import initial_window
        rows,complete=initial_window(self.history,self.config.initial_history_s,self.clock.get_physics_dt())
        g=self.geometry
        checks=dict(history_complete=complete,
            supported=bool(rows) and all(r['supports'] for r in rows),
            unheld=bool(rows) and not any(r['robot_contact'] or r['held'] or r['on_floor'] for r in rows),
            stable=bool(rows) and bool(np.ptp([r['poses'][g.target_body]['position_m'] for r in rows],axis=0).max()
                                      <=self.config.initial_position_span_m))
        if self.source_path is None:raise ValueError('Preparation source was not linked before initialization')
        self.identity=Binding(self.episode_id,self.instance_id,self.kind,g.target_body,self.source_path,g.tray_body)
        self.composite=self.make_composite()
        placement=self._placement(self.latest,0)
        checks['not_initially_on_meal_tray']=not(placement.supported_by_destination and placement.whole_inside_destination)
        receipt=dict(accepted=all(checks.values()),checks=checks,evidence_source='native_physx',
                     scene_pose_writes=0,physics_parameters_changed=False,full_trajectory_verified=False)
        (self.output/'initial_acceptance.json').write_text(json.dumps(receipt,indent=2)+'\n')
        if not receipt['accepted']:raise RuntimeError('Meal initial rejected '+str(checks))
        self.index=0;self.score=self.composite.observe(placement);self.pending=[]
        self.initial_supports=frozenset(self.latest['supports'])
        # Native initialization is not touch evidence for task selection.
        self._selection_time=self.clock.current_time
        return receipt

    def make_composite(self):
        return AdditionalMealTask(self.identity) if self.kind=='cutlery_to_meal_tray' else MealTrayTask(self.identity)

    def selection_request(self):
        from openend.tasks.all_objects_integration.selective_read import selection_request
        return selection_request(self)

    def consume_control(self,active):
        if not active or len(self.pending)!=self.clock.hz//20:
            raise RuntimeError('Complete selected 20 Hz meal interval required')
        if abs(self.pending[-1]['time_s']-self.clock.current_time)>1e-6:
            raise RuntimeError('Stale meal final-state evidence')
        row=self.pending[-1];index=self.index+1;placement=self._placement(row,index)
        evidence=None
        if self.kind=='cutlery_to_meal_tray':
            separated=not bool(self.initial_supports & row['supports'])
            evidence=OperationFrame(self.identity,index,index*.05,CutleryFacts(row['held'],separated,True))
        elif hasattr(self,'operation_evidence'):
            evidence=self.operation_evidence(index)
        elif (self.device_owner is not None and self.device_owner.composite.success
              and all(r['operation_ready'] for r in self.pending)):
            if self.composite.baseline is None:
                self.composite.accept_preparation(self.device_owner.composite)
            if self.kind=='coffee_to_meal_tray':
                s=self.device_owner.owner.service
                if s.latest['sample'].time_s!=self.clock.current_time:raise RuntimeError('Coffee owner was not read this frame')
                facts=s.facts(index,'1-4')
                s._clear_window()
            else:
                s=self.device_owner.source
                if s.latest['time_s']!=self.clock.current_time:raise RuntimeError('Fryer owner was not read this frame')
                facts=s.selected_facts(index)
                if not self.device_owner.composite.inherited_counter_support:
                    self.composite.inherited_counter_support=False
                self.composite.observe_inactive_other(facts.other)
            # Native owners consume each actual workspace substep. Grasp
            # windows use simulation seconds; control evidence remains 20 Hz.
            evidence=PickupFrame(self.identity,facts)
        self.score=self.composite.observe(placement,evidence)
        self.index=index;self.pending=[];self.paused=False
        self.score.update(evidence_source='native_physx',room_runtime_registered=True,full_trajectory_verified=False)
        self.trace.write(json.dumps(dict(step=index,simulation_time_s=self.clock.current_time,
            final_state=asdict(placement),score=self.score),default=list)+'\n');self.trace.flush()
        return self.score

    def pause(self):
        if self.composite is not None:self.composite.pause()
        self.paused=True

    def finalize(self,reason):
        import numpy as np
        self.pending=[]
        # One read at room settlement, NOT a full inactive per-substep reader.
        # Contact lifecycle has been maintained while inactive. Read both actual
        # current transforms so old placement credit cannot survive removal.
        poses={}
        for p in (self.geometry.target_body,self.geometry.tray_body):
            t=np.asarray(self.views[p].get_transforms()).copy()[0]
            poses[p]=dict(position_m=tuple(map(float,t[:3])),quaternion_wxyz=tuple(map(float,t[[6,3,4,5]])))
        row=dict(poses=poses,pairs=tuple(self.contacts.pairs()))
        self.index+=1
        self.composite.observe(self._placement(row,self.index))
        self.score=self.composite.finalize(reason)
        return self.score

    def close(self):
        if not self.closed:self.trace.close();self.closed=True


class CoffeeMealPort(MealPort):
    kind='coffee_to_meal_tray';asset='coffee_cup_blue_compact'

class BreadMealPort(MealPort):
    kind='bread_to_meal_tray';asset='cake_plain_yellow'

class CutleryMealPort(MealPort):
    kind='cutlery_to_meal_tray';asset='meal_fork'

FACTORIES={p.kind:p for p in (CoffeeMealPort,BreadMealPort,CutleryMealPort)}

# Task-owned extension consumed by the existing shared room factory.
from .microwave_port import MicrowaveMealMixin
class MicrowaveMealPort(MicrowaveMealMixin,MealPort):pass
FACTORIES[MicrowaveMealPort.kind]=MicrowaveMealPort

from .remaining_ports import SaladMealMixin,DrinkMealMixin
class SaladMealPort(SaladMealMixin,MealPort):pass
class DrinkMealPort(DrinkMealMixin,MealPort):pass
FACTORIES.update({p.kind:p for p in (SaladMealPort,DrinkMealPort)})


def preflight(tasks,manifest):
    selected=[t for t in tasks if t['type'] in FACTORIES]
    if not selected:return
    from .registration import bind
    bind(tasks,manifest)
    one(manifest,'meal_tray_large')
    for t in selected:
        FACTORIES[t['type']].resolve_object(t['parameters'],manifest)
        if t['type']=='cutlery_to_meal_tray':one(manifest,'cutlery_holder')


def link_ports(ports,tasks,manifest):
    from .registration import dependencies
    for edge in dependencies(tasks):
        child=ports[edge['instance_id']];owner=ports[edge['prerequisite_instance_id']]
        if edge['type']=='microwave_box_to_meal_tray':
            child.link_fridge(owner,edge['prerequisite_instance_id'])
            continue
        child.device_owner=owner;child.reader_dependencies=(edge['prerequisite_instance_id'],)
        child.source_path=dict(owner.binding.objects)['machine' if edge['prerequisite_type']=='coffee' else 'basket']
        owner.meal_preparation_kind=edge['prerequisite_type']
        if child.geometry.target_body!=dict(owner.binding.objects)['cup' if edge['prerequisite_type']=='coffee' else 'target']:
            raise ValueError('Preparation and meal task must bind the same native target')
