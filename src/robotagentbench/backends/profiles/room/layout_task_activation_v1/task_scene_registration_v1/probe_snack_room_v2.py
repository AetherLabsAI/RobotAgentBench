"""Native multi-task room test; actual localization, no manipulation or video.
Navigation setup is unchanged from approved L1 constrained-planar probe v5.
The new shared room host selects and scores coffee/fryer, then idles to timeout.
"""
import sys,json,math,hashlib,traceback
from pathlib import Path
ROOT=Path('/home/lingjun/robot_agent_arena/layout_task_activation_v1/task_scene_registration_v1')
sys.path[:0]=[str(ROOT),str(ROOT/'room_demo_v6'),'/home/lingjun/robot_agent_arena/api-protocol']
import numpy as np
import omni.kit.app,omni.physx,omni.physics.tensors
from pxr import Usd,UsdGeom,UsdPhysics,PhysxSchema,UsdUtils,Gf,PhysicsSchemaTools
from action22 import JOINT_NAMES,StatePacker,validate
from navigation import PrimitiveController
from sector_activation_v1 import step
from multi_task_room_v2 import MultiTaskRoom
from coffee_hotwake_v1 import NativeSceneClock
OUTDIR=Path('/pilot');OUT=OUTDIR/'navigation_result.json'
report=dict(status='initializing',control_hz=20,rows=[],contacts=[],initial_pose_writes=1,
 rollout_pose_velocity_writes=0,navigation_model='existing_L1_native_planar_constraints_not_wheels',
 self_contacts=[],activation_source='measured_native_base_pose_sector_polygon',full_task_success=False,scoring_started=False,task_initialization_owner='unified_scene')
owner=None;sim=None
try:
 assert not OUT.exists()
 manifest=json.loads((ROOT/'coffee_initialized_rooms_v1/layout04/composition.json').read_text())
 stage=Usd.Stage.Open(manifest['scene']);stage.SetEditTarget(stage.GetSessionLayer())
 hashes={l.realPath:hashlib.sha256(Path(l.realPath).read_bytes()).hexdigest() for l in stage.GetUsedLayers() if l.realPath}
 sensor=json.loads((ROOT.parent/'hotwake_probe_inputs.json').read_text())['sensor']
 request=json.loads((ROOT/'multi_task_episode_snack_v1.json').read_text())
 owner=MultiTaskRoom(stage,manifest,OUTDIR/'room_evidence','native-multitask-navigation',request=request,sensor=sensor)
 report.update(request=request,sensor=sensor,source_sha256=hashes)
 scene=stage.GetPrimAtPath('/World/PhysicsScene')
 assert scene.GetAttribute('physxScene:timeStepsPerSecond').Get()==60
 assert scene.GetAttribute('physxScene:enableGPUDynamics').Get() is False
 base_path='/World/Robot/base_link';root=stage.GetPrimAtPath('/World/Robot')
 # One initial placement in the kitchen aisle, using existing L1 navigation configuration.
 # v1 at y=1.25 hit the shelf post while turning; retain its failure artifact.
 # Only choose a different initial navigation station, no collision/drive edits.
 matrix=Gf.Matrix4d(1);matrix.SetRotate(Gf.Rotation(Gf.Vec3d(0,0,1),90));matrix.SetTranslateOnly(Gf.Vec3d(-1.99,1.05,0))
 UsdGeom.Xformable(root).MakeMatrixXform().Set(matrix)
 anchor=UsdPhysics.FixedJoint(stage.GetPrimAtPath(base_path+'/FixedJoint'))
 anchor.CreateExcludeFromArticulationAttr(False);anchor.GetJointEnabledAttr().Set(True)
 # Keep L1's fixed-base initialization and parent articulation root.
 base_prim=stage.GetPrimAtPath(base_path)
 articulation_api=PhysxSchema.PhysxArticulationAPI.Apply(root)
 articulation_api.CreateEnabledSelfCollisionsAttr(True)
 articulation_api.CreateSolverPositionIterationCountAttr(32)
 articulation_api.CreateSolverVelocityIterationCountAttr(4)
 report['solver_velocity_iterations']=4
 nav={};drives={}
 base_matrix=UsdGeom.XformCache().GetLocalToWorldTransform(base_prim)
 initial_p=base_matrix.ExtractTranslation();initial_q=base_matrix.ExtractRotationQuat()
 anchor.CreateLocalPos0Attr(Gf.Vec3f(initial_p));anchor.CreateLocalRot0Attr(Gf.Quatf(initial_q))
 for kind,cls,axis,drive_type in [('hold',UsdPhysics.FixedJoint,None,None),('drive',UsdPhysics.PrismaticJoint,'X','linear'),('turn',UsdPhysics.RevoluteJoint,'Z','angular')]:
  j=cls.Define(stage,'/World/PositionHotwakeNavigation_'+kind)
  j.CreateBody1Rel().SetTargets([base_path]);j.CreateExcludeFromArticulationAttr(True)
  j.CreateJointEnabledAttr(False);j.CreateLocalPos0Attr(Gf.Vec3f(initial_p));j.CreateLocalRot0Attr(Gf.Quatf(initial_q))
  j.CreateLocalPos1Attr(Gf.Vec3f(0));j.CreateLocalRot1Attr(Gf.Quatf(1))
  if axis:
   j.CreateAxisAttr(axis);j.CreateLowerLimitAttr(-170 if kind=='turn' else -10);j.CreateUpperLimitAttr(170 if kind=='turn' else 10)
   drive=UsdPhysics.DriveAPI.Apply(j.GetPrim(),drive_type);drive.CreateTypeAttr('force');drive.CreateStiffnessAttr(0)
   drive.CreateDampingAttr(100000);drive.CreateMaxForceAttr(100000);drive.CreateTargetVelocityAttr(0);drives[kind]=drive
  nav[kind]=j
 for p in Usd.PrimRange(root):
  if p.HasAPI(UsdPhysics.RigidBodyAPI):PhysxSchema.PhysxContactReportAPI.Apply(p).CreateThresholdAttr(0)
 limits={}
 for name in JOINT_NAMES:
  p=stage.GetPrimAtPath('/World/Robot/joints/'+name)
  scale=1 if p.IsA(UsdPhysics.PrismaticJoint) else math.pi/180
  limits[name]=[float(p.GetAttribute('physics:lowerLimit').Get())*scale,float(p.GetAttribute('physics:upperLimit').Get())*scale]
 sid=UsdUtils.StageCache.Get().Insert(stage).ToLongInt();sim=omni.physx.get_physx_simulation_interface();native=omni.physx.get_physx_interface()
 assert sim.attach_stage(sid)
 step_count=0;simulation_time=0.;hz=60
 def physics_step():
  global step_count,simulation_time
  sim.simulate(1/hz,simulation_time);sim.fetch_results();step_count+=1;simulation_time+=1/hz
 physics_step()
 view=omni.physics.tensors.create_simulation_view('numpy',sid);view.set_subspace_roots('/')
 robot=view.create_articulation_view('/World/Robot')
 assert robot.count==1
 body_paths=[str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI) and not str(p.GetPath()).startswith('/World/Robot/') and p.GetAttribute('physics:rigidBodyEnabled').Get() is not False]
 bodies=view.create_rigid_body_view(body_paths)
 report['body_paths']=list(bodies.prim_paths)
 assert set(report['body_paths'])==set(body_paths),'Every nonrobot rigid body must be recorded'
 report['robot_link_names']=list(robot.shared_metatype.link_names)
 recorded_bodies=[];recorded_links=[]
 stage.Export(str(OUTDIR/'native_initial.usdc'))

 names=list(robot.shared_metatype.dof_names)
 native_targets=robot.get_dof_positions().copy()
 folded=json.loads((ROOT/'recorded_ready_pose_042_v1.json').read_text())
 for name,value in folded['targets_rad'].items():
  assert limits[name][0]<=value<=limits[name][1]
  native_targets[0,names.index(name)]=value
 report['navigation_posture']=folded
 robot.set_dof_positions(native_targets.copy(),np.array([0],dtype=np.uint32))
 robot.set_dof_velocities(np.zeros_like(native_targets),np.array([0],dtype=np.uint32))
 report.update(initial_joint_state_restores=1,initial_joint_velocity_zeroes=1)
 for name in JOINT_NAMES:
  i=names.index(name);native_targets[0,i]=min(limits[name][1],max(limits[name][0],native_targets[0,i]))
 joint20=[float(native_targets[0,names.index(n)]) for n in JOINT_NAMES]
 packer=StatePacker();last_vw=[0.,0.];current_mode='hold';activation={'active':False}
 def measured():
  q=robot.get_dof_positions()[0];pose=robot.get_root_transforms()[0]
  return packer.pack(dict(zip(names,map(float,q))),list(map(float,pose))),pose
 def snapshot():
  return b''.join(np.asarray(a).copy().tobytes() for a in (bodies.get_transforms(),bodies.get_velocities(),
   robot.get_link_transforms(),robot.get_dof_positions(),robot.get_dof_velocities(),robot.get_root_velocities()))
 clock=NativeSceneClock(stage,sim,lambda:simulation_time,snapshot)
 owner.initialize(clock,view);hz=clock.hz
 for _ in range(240):
  physics_step();owner.after_physics(dict(events=[],docked_object_count=0,tray_dock_enabled=False))
 report['initial_acceptance']=owner.accept_initial()
 identities={k:id(p.composite) for k,p in owner.ports.items()}
 report['scoring_started']=True
 no_helpers=dict(events=[],docked_object_count=0,tray_dock_enabled=False)
 def switch(kind):
  global current_mode
  state,pose=measured();before=pose.copy()
  assert max(abs(x) for x in last_vw)<.001
  anchor.GetJointEnabledAttr().Set(False)
  for joint in nav.values():joint.GetJointEnabledAttr().Set(False)
  nav[kind].GetLocalPos0Attr().Set(Gf.Vec3f(*map(float,pose[:3])))
  nav[kind].GetLocalRot0Attr().Set(Gf.Quatf(float(pose[6]),Gf.Vec3f(*map(float,pose[3:6]))))
  nav[kind].GetJointEnabledAttr().Set(True);sim.flush_changes();current_mode=kind
  assert np.array_equal(before,robot.get_root_transforms()[0])
 adjacent=set()
 for p in Usd.PrimRange(root):
  if p.IsA(UsdPhysics.Joint):
   a,b=[p.GetRelationship('physics:body'+str(i)).GetTargets() for i in (0,1)]
   if len(a)==len(b)==1:adjacent.add(frozenset([str(a[0]),str(b[0])]))
 def contact(headers,points):
  for h in headers:
   if not h.num_contact_data:continue
   actors=[str(PhysicsSchemaTools.intToSdfPath(a)) for a in (h.actor0,h.actor1)]
   robot_count=sum(a.startswith('/World/Robot/') for a in actors)
   if robot_count==2:
    if actors[0]==actors[1] or frozenset(actors) in adjacent:continue
    # Keep original gripper internal mechanisms; inspect arm vs arm/torso/head,
    # opposite grippers, and wrist camera vs other robot links.
    colliders=[str(PhysicsSchemaTools.intToSdfPath(a)) for a in (h.collider0,h.collider1)]
    same_gripper=any(all(('/gripper_'+side+'_') in a for a in actors) for side in ('l','r'))
    if not same_gripper:
     sep=min(float(p.separation) for p in points[h.contact_data_offset:h.contact_data_offset+h.num_contact_data])
     if sep<-.001:report['self_contacts'].append(dict(step=step_count,actors=actors,colliders=colliders,min_separation_m=sep))
    continue
   if robot_count!=1:continue
   colliders=[str(PhysicsSchemaTools.intToSdfPath(a)) for a in (h.collider0,h.collider1)]
   external=[c for c in colliders if not c.startswith('/World/Robot/')]
   if all('floor' in c.lower() or 'ground' in c.lower() for c in external):continue
   sep=min(float(p.separation) for p in points[h.contact_data_offset:h.contact_data_offset+h.num_contact_data])
   if sep<-.001:report['contacts'].append(dict(step=step_count,actors=actors,colliders=colliders,min_separation_m=sep))
 subscription=sim.subscribe_contact_report_events(contact)
 def advance(v,w,phase):
  global last_vw,activation,hz
  action=validate(joint20+[v,w],limits,last_vw,dt=.05);last_vw=[v,w]
  for outer,inner in [('idx41_gripper_l_outer_joint1','idx31_gripper_l_inner_joint1'),('idx81_gripper_r_outer_joint1','idx71_gripper_r_inner_joint1')]:
   native_targets[0,names.index(inner)]=-native_targets[0,names.index(outer)]
  robot.set_dof_position_targets(native_targets,np.array([0],dtype=np.uint32))
  if current_mode=='drive':drives['drive'].GetTargetVelocityAttr().Set(v)
  elif current_mode=='turn':drives['turn'].GetTargetVelocityAttr().Set(math.degrees(w))
  state_before,_=measured()
  actual_robot=dict(xy=state_before[-3:-1],yaw=math.degrees(state_before[-1]))
  velocity=list(map(float,robot.get_root_velocities()[0]))
  plan=owner.begin_control(actual_robot,command_vw=[v,w],base_velocity=velocity)
  hz=plan['physics_hz']
  for _ in range(plan['substeps']):
   physics_step();owner.after_physics(no_helpers)
  result=owner.finish_control(completed_substeps=plan['substeps'])
  state,pose=measured()
  report['rows'].append(dict(phase=phase,robot=actual_robot,result=result,action=action,
   state=state,time_s=simulation_time,base_velocity=velocity,**plan))
  recorded_bodies.append(bodies.get_transforms().copy());recorded_links.append(robot.get_link_transforms().copy())
  if len(report['rows'])%100==0:print('MULTITASK_NAV_TICK',len(report['rows']),phase,hz,flush=True)
  if report['contacts'] or report['self_contacts']:raise RuntimeError('Environment or nonadjacent self penetration detected; stop trial')
  assert abs(float(pose[2])-float(initial_p[2]))<.01
 for _ in range(40):advance(0,0,'far_settle')
 def run(kind,goal,label,direction=1):
  switch(kind);controller=PrimitiveController(dict(kind=kind,goal=goal,direction=direction))
  for _ in range(20*30):
   cur=measured()[0][-3:];world=controller.step(cur,dt=.05)
   v=math.cos(cur[2])*world[0]+math.sin(cur[2])*world[1]
   advance(float(v),float(world[2]),label)
   if controller.done:break
  else:raise RuntimeError('Original L1 primitive timeout')
  for _ in range(20):advance(0,0,label+'_stopped')
 cur=measured()[0][-3:];run('turn',[*cur[:2],0.],'leave_snack')
 cur=measured()[0][-3:];run('turn',[*cur[:2],math.pi/2],'reenter_snack')
 cur=measured()[0][-3:];run('turn',[*cur[:2],0.],'leave_snack_again')
 switch('hold')
 for _ in range(20):advance(0,0,'outside')
 report['transitions']=owner.transitions
 report['task_switches']=owner.dispatch.switches
 assert not owner.dispatch.ended
 assert all(id(p.composite)==identities[k] for k,p in owner.ports.items())
 assert owner.ports['snack'].index>0
 assert owner.ports['coffee'].index==owner.ports['fryer'].index==0
 assert any(r['phase']=='reenter_snack_stopped' and r['current_task']=='snack' for r in report['rows'])
 assert any(r['current_task'] is None and r['physics_hz']==60 for r in report['rows'])
 assert any(r['current_task']=='snack' and r['physics_hz']==240 for r in report['rows'])
 assert owner.ports['snack'].samples==12*owner.ports['snack'].index
 assert report['initial_acceptance']['snack']['initial_state_gate_enabled'] is False
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
 assert owner.dispatch.total_score==0
 report.update(status='native_snack_room_activation_passed',native_physics_steps=step_count,
  source_layers_unchanged=True,scorer_instances_retained=True,final_scoreboard=owner.dispatch.scores,
  termination_reason=owner.dispatch.termination_reason,
  idle_timeout_elapsed_s=owner.dispatch.last_time-owner.dispatch.started_at)

except Exception:
 report.update(status='failed',error=traceback.format_exc())
finally:
 if owner is not None:report['transitions']=owner.transitions
 print('MULTITASK_NAV_RESULT',report['status'],report.get('error',''),flush=True)
 try:
  if 'recorded_bodies' in globals() and recorded_bodies:
   np.savez_compressed(OUTDIR/'native_poses.npz',bodies=np.array(recorded_bodies),links=np.array(recorded_links))
  if owner is not None:owner.close()
  if sim is not None:sim.detach_stage()
  OUT.write_text(json.dumps(report,indent=2)+'\n')
 finally:omni.kit.app.get_app().post_quit(0 if report['status']!='failed' else 1)
