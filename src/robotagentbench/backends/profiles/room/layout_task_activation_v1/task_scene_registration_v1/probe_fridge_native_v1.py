"""Diagnostic external torque, not robot manipulation or Task9 acceptance.

Original layout, packaged fridge, original helper. Only session-layer changes.
No pose/velocity setters, no helper-state injection, no source asset edits.
"""
import sys,json,traceback,hashlib,os
from pathlib import Path
ROOT=Path('/home/lingjun/robot_agent_arena/layout_task_activation_v1/task_scene_registration_v1')
API=Path('/home/lingjun/robot_agent_arena/api-protocol')
sys.path[:0]=[str(API),str(ROOT)]
import numpy as np
import omni.kit.app,omni.physx,omni.physics.tensors
from pxr import Usd,UsdGeom,UsdPhysics,PhysxSchema,UsdUtils,Gf,PhysicsSchemaTools
from arena_api.l1_fridge import Controller
FR='/World/Background/Shell/RefrigeratorRecovered'
OUT=Path('/pilot/result.json')
report=dict(status='started',test='native_external_torque_smoke_not_robot_task',
            rows=[],pose_velocity_writes=0,helper_state_writes=0,full_task_success=False)
owner=None;sim=None;contact_subscription=None
try:
    assert not OUT.exists()
    scene=ROOT/'refrigerator_layouts_v1/ID_01_LIVING_STRAIGHT.usda'
    isolated=os.environ.get('FRIDGE_SMOKE_ISOLATED','0')=='1'
    stage=(Usd.Stage.OpenMasked(str(scene),Usd.StagePopulationMask([FR,'/World/PhysicsScene','/World/PhysicsMaterials'])) if isolated
           else Usd.Stage.Open(str(scene)))
    stage.SetEditTarget(stage.GetSessionLayer())
    report['fixture']='isolated_fridge_no_room_floor_or_robot' if isolated else 'full_layout_robot_relocated_before_start'
    Path('/pilot/probe_source.py').write_bytes(Path(__file__).read_bytes())
    hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
            [str(scene),str(API/'all_assets/refrigerator/asset.usda')]}
    assert stage.GetPrimAtPath(FR)
    # Test-only initial staging: inherited robot base obstructed the door at 85.4 deg.
    # Keep the robot present, move it 1 m away along room -Y before attaching PhysX.
    # Update world-anchored fixed joints by exactly the same initial offset.
    robot=stage.GetPrimAtPath('/World/Robot')
    offset=Gf.Vec3d(0,-1,0)
    if robot:
        matrix=UsdGeom.XformCache().GetLocalToWorldTransform(robot)
    else:
        matrix=Gf.Matrix4d(1)
    old_position=matrix.ExtractTranslation();matrix.SetTranslateOnly(old_position+offset)
    if robot:
        parent=UsdGeom.XformCache().GetLocalToWorldTransform(robot.GetParent())
        UsdGeom.Xformable(robot).MakeMatrixXform().Set(matrix*parent.GetInverse())
    for joint in (Usd.PrimRange(robot) if robot else []):
        if not joint.IsA(UsdPhysics.Joint):continue
        a=joint.GetRelationship('physics:body0').GetTargets()
        b=joint.GetRelationship('physics:body1').GetTargets()
        if not a and b and str(b[0]).startswith('/World/Robot/'):
            attr=joint.GetAttribute('physics:localPos0')
            if attr and attr.Get() is not None:attr.Set(Gf.Vec3f(Gf.Vec3d(attr.Get())+offset))
    report['test_only_initial_robot_relocation']=dict(from_xyz=list(old_position),to_xyz=list(old_position+offset),
        persisted_to_room=False,reason='original_robot_base_blocks_door_handle_at_85.4_degrees')
    if isolated:report['test_only_initial_robot_relocation']=None
    physics=[p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if not physics:
        ps=UsdPhysics.Scene.Define(stage,'/World/PhysicsScene')
        ps.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1));ps.CreateGravityMagnitudeAttr(9.81)
        physics=[ps.GetPrim()]
    assert len(physics)==1
    px=PhysxSchema.PhysxSceneAPI.Apply(physics[0])
    px.CreateEnableGPUDynamicsAttr(False);px.CreateBroadphaseTypeAttr('MBP')
    px.CreateTimeStepsPerSecondAttr(240)
    report.update(scene=str(scene),physics_hz=240,helper_hz=60,rendered=False)
    owner=Controller(stage,physics_dt=1/240)
    report['hinges']={side:{name:stage.GetPrimAtPath(FR+'/'+side+'_door_hinge').GetAttribute(name).Get()
        for name in ('physics:axis','physics:lowerLimit','physics:upperLimit')}
        for side in ('left','right')}
    report['contacts']={}
    def on_contact(headers,points):
        for h in headers:
            if not h.num_contact_data:continue
            actors=[str(PhysicsSchemaTools.intToSdfPath(a)) for a in (h.actor0,h.actor1)]
            if not any(a.startswith(FR+'/') for a in actors):continue
            colliders=[str(PhysicsSchemaTools.intToSdfPath(a)) for a in (h.collider0,h.collider1)]
            key=' | '.join(sorted(colliders))
            depth=min(float(p.separation) for p in points[h.contact_data_offset:h.contact_data_offset+h.num_contact_data])
            old=report['contacts'].setdefault(key,dict(actors=actors,count=0,min_separation_m=depth,first_s=clock,last_s=clock))
            old.update(count=old['count']+1,min_separation_m=min(old['min_separation_m'],depth),last_s=clock)
    sim=omni.physx.get_physx_simulation_interface()
    contact_subscription=sim.subscribe_contact_report_events(on_contact)
    sid=UsdUtils.StageCache.Get().Insert(stage).ToLongInt();assert sim.attach_stage(sid)
    clock=0.;step=0
    sim.simulate(1/240,clock);sim.fetch_results();clock+=1/240;step+=1
    owner.after_physics_step()
    view=omni.physics.tensors.create_simulation_view('numpy',sid);view.set_subspace_roots('/')
    door=view.create_rigid_body_view(FR+'/right_door')
    cabinet=view.create_rigid_body_view(FR+'/body')
    assert door.count==cabinet.count==1
    start=cabinet.get_transforms().copy()
    indices=np.array([0],dtype=np.uint32)
    def advance(label,ticks,torque=0.,until=None):
        global clock,step
        for _ in range(ticks):
            if torque:
                door.apply_forces_and_torques_at_position(None,np.array([[0.,0.,torque]],dtype=np.float32),None,indices,True)
            sim.simulate(1/240,clock);sim.fetch_results();clock+=1/240;step+=1
            owner.after_physics_step()
            right=owner.ns['_fridge_pose']('right')[2]
            left=owner.ns['_fridge_pose']('left')[2]
            assert right is not None and left is not None
            report['rows'].append(dict(step=step,time_s=clock,phase=label,right_deg=right,left_deg=left,
                zone=owner.ns['FR_LATCH']['zone'],external_torque_Nm=torque,
                cabinet_displacement_m=float(np.linalg.norm(cabinet.get_transforms()[0,:3]-start[0,:3]))))
            if until and owner.ns['FR_LATCH']['zone']==until:return True
        return False
    advance('settle',240)
    assert abs(report['rows'][-1]['right_deg'])<2.,'Initial door unstable'
    assert advance('opening_external_torque',240,40.,'opening'),'External pull did not trigger original helper'
    assert advance('helper_open',240*5,until='open'),'Original helper did not reach open'
    advance('open_hold',240)
    report['open_hold_angle_deg']=report['rows'][-1]['right_deg']
    assert abs(report['open_hold_angle_deg']-89.)<2.,'Open hold differs from configured stop'
    assert advance('closing_external_torque',240,-150.,'closing'),'External push did not trigger closing'
    assert advance('helper_close',240*5,until='latch'),'Original helper did not relatch'
    advance('closed_hold',240)
    report['closed_angle_deg']=report['rows'][-1]['right_deg']
    assert abs(report['closed_angle_deg'])<2.,'Closed hold not near zero'
    report['max_cabinet_displacement_m']=max(r['cabinet_displacement_m'] for r in report['rows'])
    assert report['max_cabinet_displacement_m']<.001,'Cabinet moved'
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report.update(status='passed',source_layers_unchanged=True,source_hashes=hashes)
except Exception:
    report.update(status='failed',error=traceback.format_exc())
finally:
    if owner:
        report['helper']=owner.report();owner.close()
    contact_subscription=None
    if sim:sim.detach_stage()
    OUT.write_text(json.dumps(report,indent=2,default=list)+'\n')
    print('FRIDGE_NATIVE_RESULT',report['status'],report.get('error',''),flush=True)
    omni.kit.app.get_app().post_quit(0 if report['status']=='passed' else 1)
