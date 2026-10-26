"""Attach to running scene-owned PhysX. Never reset, step or author physics."""
import math
import numpy as np
from pxr import Usd, UsdPhysics
from arena_api.physics_defaults import PHYSICS_DT
from evaluation.tasks.shelf_pick.geometry import HANDS, BASE_LINK, rotation
from evaluation.tasks.shelf_pick.initial_state import ROBOT
from evaluation.tasks.shelf_pick.contact_stream import ContactStream
from evaluation.tasks.shelf_pick.physical_detection import GraspDetector, DetectionConfig
from evaluation.tasks.shelf_pick.process_rules import PhysicalFrame
from evaluation.tasks.coffee_table_place.bindings import PlacementBinding
from evaluation.tasks.coffee_table_place.contact_evidence import PlacementContacts
from evaluation.tasks.coffee_table_place.process_rules import HeldFacts
from .composite import Frame, SnackTrayComposite


def preflight(binding):
    stage=binding.stage
    robot=stage.GetPrimAtPath(ROBOT)
    paths=list(binding.stock)
    if robot:paths.extend(str(p.GetPath()) for p in Usd.PrimRange(robot) if p.HasAPI(UsdPhysics.RigidBodyAPI))
    missing=[p for p in paths if 'PhysxContactReportAPI' not in stage.GetPrimAtPath(p).GetAppliedSchemas()
             or stage.GetPrimAtPath(p).GetAttribute('physxContactReport:threshold').Get()!=0]
    return dict(ready=bool(robot) and not missing,missing_contact_report_bodies=missing,
                physics_modified=False,requires_scene_owner_instrumentation=bool(missing))


class NativeSnackTraySource:
    """Scene owner calls after_physics once per substep, then begin once available.

    Idle observation allows existing 60/240 clocks. Once scoring starts the
    caller must keep its existing 240 Hz lease until terminal; no pause on exit.
    """
    def __init__(self,binding,world,simulation_view=None):
        check=preflight(binding)
        if not check['ready']:raise RuntimeError(str(check))
        from pxr import PhysicsSchemaTools
        from omni.physx import get_physx_simulation_interface
        from omni.physx.bindings._physx import ContactEventType
        from evaluation.physics_readers import ReadOnlyRigidBody
        from evaluation.tasks.coffee_machine.openended_composite_v1.native_source import _BoundBodies
        self.binding=binding;self.world=world;self.interface=get_physx_simulation_interface()
        self.last_time=float(world.current_time);self.latest=None;self.process=None;self.closed=False
        self.active=binding.identity.candidates[0];self.locked=False
        self.config=DetectionConfig();self.detector=GraspDetector(tuple(binding.stock))
        self.views={}
        for p in (*binding.stock,*HANDS.values(),BASE_LINK):
            if simulation_view is None:
                v=ReadOnlyRigidBody(p);v.initialize()
            else:v=_BoundBodies(simulation_view,[p])
            self.views[p]=v
        def sleeping(stage_id,actor):
            prim=binding.stage.GetPrimAtPath(PhysicsSchemaTools.intToSdfPath(actor))
            if not prim:raise RuntimeError('Unresolved native actor')
            return bool(not prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.GetAttribute('physics:kinematicEnabled').Get() is True
                        or self.interface.is_sleeping(stage_id,actor))
        event_types=(ContactEventType.CONTACT_FOUND,ContactEventType.CONTACT_PERSIST,ContactEventType.CONTACT_LOST)
        neighbors=tuple(p for p in binding.stock if p!=self.active)
        self.contacts=ContactStream(PhysicsSchemaTools.intToSdfPath,sleeping,event_types,
            neighbors,binding.workspace.top,self.config.support_face_tolerance_m,
            floor_paths=binding.workspace.floor_paths,shelf_paths=binding.shelf_colliders,
            support_face=binding.face,targets=binding.identity.candidates,support=binding.identity.source_support,
            object_supports=binding.supports,support_tops=binding.workspace.support_tops)
        self.placements={p:PlacementContacts(PlacementBinding(binding.product.place_api,p,binding.tray,'tray'),
            PhysicsSchemaTools.intToSdfPath,sleeping,event_types,binding.tray_geometry.top,
            floor_paths=binding.workspace.floor_paths) for p in binding.identity.candidates}

    @property
    def requires_240hz(self):return self.process is not None and not self.process.ended

    def after_physics(self,helper_report):
        if self.closed:raise RuntimeError('Closed source')
        if helper_report['events'] or helper_report['docked_object_count'] or helper_report['tray_dock_enabled']:
            raise RuntimeError('Assisted manipulation is not physical evidence')
        now=float(self.world.current_time);dt=float(self.world.get_physics_dt())
        if not math.isfinite(now) or not any(abs(dt-v)<1e-10 for v in (1/60,PHYSICS_DT)):
            raise RuntimeError('Unexpected native clock')
        if self.requires_240hz and abs(dt-PHYSICS_DT)>1e-10:raise RuntimeError('Active task requires continuous native 240 Hz evidence')
        if abs(now-self.last_time-dt)>1e-6:raise RuntimeError('Missing or repeated native physics sample')
        self.binding.validate_static()
        poses={}
        for path,view in self.views.items():
            pos,q=view.get_world_pose();pos=np.asarray(pos);q=np.asarray(q)
            if pos.shape!=(3,) or not np.isfinite(pos).all():raise RuntimeError('Invalid native position')
            rotation(q);poses[path]=dict(position_m=pos,quaternion_wxyz=q)
        report=self.interface.get_contact_report()
        self.contacts.ingest(report)
        def summarize():
            pose=poses[self.active]
            center=rotation(pose['quaternion_wxyz']).apply(self.binding.workspace.collision_vertices[self.active].mean(axis=0))+pose['position_m']
            return self.contacts.summarize(self.active,center)
        contact=summarize()
        grasped,details=self.detector.observe(now,poses,contact['bilateral_contacts'])
        if not self.locked:
            held=[p for p in self.binding.identity.candidates if grasped[p]]
            if held:self.active=self.active if self.active in held else held[0];self.locked=True;contact=summarize()
        placement={p:r.read(report,rotation(poses[p]['quaternion_wxyz']).apply([0,0,1])) for p,r in self.placements.items()}
        row=dict(time_s=now,dt=dt,poses=poses,contacts=contact,placement=placement,
                 grasped=grasped,active=self.active,grasp_details=details)
        self.latest=row;self.last_time=now
        if self.requires_240hz:self.process.observe(self.frame(self.process.last.pick.sample_index+1))
        return self.summary()

    def begin(self):
        if self.closed or self.process is not None or self.latest is None:raise RuntimeError('One score start per native episode')
        # No pose/support/drift/pre-grasp initial gate (user decision).
        # Clock validity is an evidence requirement, not physical initial acceptance.
        if abs(self.latest['dt']-PHYSICS_DT)>1e-10:raise RuntimeError('Scoring requires the existing 240 Hz evidence clock')
        self.initial_z={p:float(self.latest['poses'][p]['position_m'][2]) for p in self.binding.identity.candidates}
        process=SnackTrayComposite(self.binding.identity)
        process.observe(self.frame(0))
        self.process=process
        return self.summary()

    def frame(self,n):
        r=self.latest;p=self.active;pose=r['poses'][p];c=r['contacts']
        pos,q=pose['position_m'],pose['quaternion_wxyz']
        outside=self.binding.workspace.outside_workspace(pos,r['poses'][BASE_LINK]['position_m'],p)
        grasp=bool(r['grasped'][p]);floor=bool(c['object_on_floor'][p])
        pf=PhysicalFrame(self.binding.identity.episode_id,n,n*PHYSICS_DT,grasp,
            not c['target_support_contact'],float(pos[2]-self.initial_z[p]),c['neighbor_contacts'],
            frozenset(o for o,v in r['grasped'].items() if v and o not in self.binding.identity.candidates),
            floor,outside,shelf_contact=c['shelf_contact'],active_target=p)
        lp,lq=self.binding.tray_geometry.pose_in_destination(pos,q)
        hf=HeldFacts(n,n*PHYSICS_DT,float(pos[2]),grasp,
            bool(r['placement'][p]['held_supported_by_destination']),self.binding.tray_geometry.in_region(p,pos,q),
            tuple(map(float,q)),tuple(lp),tuple(lq),floor,outside)
        return Frame(self.binding.identity,pf,hf)

    def summary(self):
        score=self.process.summary() if self.process else None
        if score is not None:
            score['evidence_source']='native_physx'
        return dict(evidence_source='native_physx',native_reader_connected=True,
            scoring_started=self.process is not None,score=score,
            needs_240hz=self.requires_240hz,full_trajectory_verified=False,runtime_registered=False)

    def close(self):self.closed=True
