"""Live physical readout. Only session contact-report instrumentation is authored."""
from collections import deque
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from arena_api.policy_execution import PHYSICS_DT
from .initial_state import ROBOT
from .targets import DEFAULT
from .geometry import Geometry, rotation, HANDS, BASE_LINK
from .physical_detection import GraspDetector, DetectionConfig
from .contact_stream import ContactStream
from .process_rules import PhysicalFrame
from .shelf_collision import bind_shelf, RULE_VERSION


def serializable(value):
    if isinstance(value,dict):return {k:serializable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list,set,frozenset)):return [serializable(v) for v in value]
    return value


class IsaacShelfReader:
    """Per-step facts for the active target.

    With several acceptable targets (4-8) the active one is the primary
    instance until an acceptable target is physically grasped; the first
    confirmed grasp commits it for the rest of the episode. Other acceptable
    targets are neighbors for collision purposes and never wrong objects.
    """
    def __init__(self,stage,output,episode_id,neighbors,config=DetectionConfig(),spec=DEFAULT,targets=None,stock_roots=None,object_supports=None):
        from pxr import Usd, UsdPhysics, PhysxSchema, PhysicsSchemaTools
        from omni.physx import get_physx_simulation_interface
        from omni.physx.bindings._physx import ContactEventType
        self.stage=stage;self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False)
        self.spec=spec;self.target=spec.target
        # Episode-accurate acceptable targets (initial-state record); falls back to the static binding.
        self.targets=tuple(targets) if targets else tuple(spec.targets)
        if self.target not in self.targets:self.target=self.targets[0]
        self.episode_id=episode_id;self.neighbors=tuple(neighbors);self.config=config
        if self.target in self.neighbors or any(t not in (self.target,*self.neighbors) for t in self.targets):
            raise ValueError('Every acceptable target must be an activated stock instance')
        self.objects=(self.target,*self.neighbors)
        self.object_supports=dict(object_supports or {})
        self.geometry=Geometry(stage,spec,stock_roots=stock_roots,targets=self.targets,
                               supports=sorted(set(self.object_supports.values())))
        shelf_paths,support_face,shelf_binding=bind_shelf(stage,config.support_face_tolerance_m,spec.support)
        shelf_binding["target_collision_center_local"] = self.geometry.collision_vertices[self.target].mean(axis=0).tolist()
        shelf_binding["collision_center_local"] = {t:self.geometry.collision_vertices[t].mean(axis=0).tolist() for t in self.targets}
        shelf_binding["acceptable_targets"] = list(self.targets)
        self.shelf_binding = shelf_binding
        (self.output/"shelf-binding.json").write_text(json.dumps(shelf_binding,indent=2)+"\n")
        self.interface=get_physx_simulation_interface()
        # Only reporting schema/threshold, no collision/material/drive changes.
        with Usd.EditContext(stage,stage.GetSessionLayer()):
            bodies=[stage.GetPrimAtPath(p) for p in self.objects]
            bodies += [p for p in Usd.PrimRange(stage.GetPrimAtPath(ROBOT)) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
            for p in bodies:
                if not p or not p.HasAPI(UsdPhysics.RigidBodyAPI):raise ValueError('Missing contact-report body')
                PhysxSchema.PhysxContactReportAPI.Apply(p).CreateThresholdAttr(0.)
        def sleeping(stage_id,actor):
            p=stage.GetPrimAtPath(PhysicsSchemaTools.intToSdfPath(actor))
            if not p:raise RuntimeError('Unknown contact actor')
            return bool(not p.HasAPI(UsdPhysics.RigidBodyAPI) or self.interface.is_sleeping(stage_id,actor))
        self.stream=ContactStream(PhysicsSchemaTools.intToSdfPath,sleeping,
            (ContactEventType.CONTACT_FOUND,ContactEventType.CONTACT_PERSIST,ContactEventType.CONTACT_LOST),
            self.neighbors,self.geometry.top,config.support_face_tolerance_m,floor_paths=self.geometry.floor_paths,
            shelf_paths=shelf_paths,support_face=support_face,targets=self.targets,support=spec.support,
            object_supports=self.object_supports,support_tops=self.geometry.support_tops)
        self.detector=GraspDetector(self.objects,config)
        self.hashes={l.realPath:hashlib.sha256(Path(l.realPath).read_bytes()).hexdigest()
                     for l in stage.GetUsedLayers() if l.realPath and Path(l.realPath).is_file()}
        self.step=-1;self.latest=None;self.start_step=None;self.initial_z=None
        self.active=self.target;self.committed=False
        self.pending=[];self.closed=False
        # Initial-acceptance window in physics samples (public PHYSICS_DT; 240 Hz since 2026-09-15), never a fixed 120 Hz count.
        self.history=deque(maxlen=int(np.ceil(config.initial_history_s/PHYSICS_DT-1e-9))+2)
        self.trace=(self.output/'physical_facts.jsonl').open('x')
        (self.output/'detection-profile.json').write_text(json.dumps(dict(rule_version=RULE_VERSION,api_id=spec.api_id,target=self.target,
            acceptable_targets=list(self.targets),support=spec.support,config=asdict(config),
            geometry=self.geometry.description(),physics_modified=False,
            instrumentation='session-layer contact reports',numerical_tolerances_status='implementation candidate; requires physical validation'),indent=2)+'\n')

    def initialize(self,world):
        from evaluation.physics_readers import ReadOnlyRigidBody
        self.world=world;self.last_time=float(world.current_time);self.views={}
        for index,path in enumerate((*self.objects,*HANDS.values(),BASE_LINK)):
            view=ReadOnlyRigidBody(prim_path=path)
            view.initialize();self.views[path]=view

    def center(self,poses,path):
        return (rotation(poses[path]['quaternion_wxyz']).apply(self.geometry.collision_vertices[path].mean(axis=0))+poses[path]['position_m']).tolist()

    def after_step(self):
        if self.closed:raise RuntimeError('Reader closed')
        now=float(self.world.current_time)
        if abs(now-self.last_time-PHYSICS_DT)>1e-6:raise RuntimeError('Missing physical substep')
        self.geometry.validate_static(self.stage)
        poses={}
        for path,view in self.views.items():
            p,q=view.get_world_pose()
            if np.asarray(p).shape!=(3,) or not np.isfinite(p).all():raise ValueError('Invalid physical position')
            rotation(q)
            poses[path]=dict(position_m=p.tolist(),quaternion_wxyz=q.tolist())
        self.stream.ingest(self.interface.get_contact_report())
        contacts=self.stream.summarize(self.active,self.center(poses,self.active))
        grasped,grasp_details=self.detector.observe(now,poses,contacts['bilateral_contacts'])
        if not self.committed:
            held=[t for t in self.targets if grasped[t]]
            if held:
                self.committed=True
                if self.active not in held:
                    self.active=held[0]
                    contacts=self.stream.summarize(self.active,self.center(poses,self.active))
        self.step+=1
        self.latest=dict(time_s=now,reader_step=self.step,poses=poses,contacts=contacts,
                         grasped=grasped,grasp_details=grasp_details,active_target=self.active,target_committed=self.committed,
                         outside_workspace=self.geometry.outside_workspace(poses[self.active]['position_m'],poses[BASE_LINK]['position_m'],self.active))
        self.history.append(self.latest)
        if self.start_step is not None:self.pending.append(self.frame())
        self.trace.write(json.dumps(serializable(self.latest),allow_nan=False)+'\n')
        self.trace.flush();self.last_time=now

    def frame(self):
        if self.start_step is None:raise RuntimeError('Initial state not accepted')
        row=self.latest;c=row['contacts'];active=row['active_target']
        return PhysicalFrame(self.episode_id,self.step-self.start_step,row['time_s'],
            row['grasped'][active],not c['target_support_contact'],
            row['poses'][active]['position_m'][2]-self.initial_z[active],c['neighbor_contacts'],
            frozenset(p for p in self.neighbors if row['grasped'][p] and p not in self.targets),
            c['target_on_floor'],row['outside_workspace'],shelf_contact=c['shelf_contact'],active_target=active)

    def begin_episode(self):
        if self.start_step is not None or self.latest is None:raise RuntimeError('Invalid initial acceptance')
        if self.committed:raise RuntimeError('Target grasped before the episode started')
        self.start_step=self.step
        self.initial_z={t:self.latest['poses'][t]['position_m'][2] for t in self.targets}
        self.pending=[self.frame()]

    def take_batch(self):
        if not self.pending:raise RuntimeError('Missing physical frame batch')
        result=tuple(self.pending);self.pending=[]
        return result

    def close(self):
        if not self.closed:
            self.trace.close();self.closed=True
