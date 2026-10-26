"""Native multi-task room test; actual localization, no manipulation or video.
Navigation setup is unchanged from approved L1 constrained-planar probe v5.
All 12 configured instances load; navigation selects coffee/fryer then idles.
This is not a portable-object manipulation or all-task success test.
"""
import sys,json,math,hashlib,traceback,os,time
from pathlib import Path
ROOT=Path('/home/lingjun/robot_agent_arena/layout_task_activation_v1/task_scene_registration_v1')
NEWROOT=Path('/home/lingjun/robot_agent_arena/openend')
sys.path[:0]=[str(NEWROOT.parent),str(ROOT/'room_demo_v6'),'/home/lingjun/robot_agent_arena/api-protocol']
import numpy as np
import omni.kit.app,omni.physx,omni.physics.tensors
from pxr import Usd,UsdGeom,UsdPhysics,PhysxSchema,UsdUtils,Gf,PhysicsSchemaTools
from action22 import JOINT_NAMES,StatePacker,validate
from navigation import PrimitiveController
from openend.room import MultiTaskRoom
from openend.clock import NativeSceneClock
from openend.tasks.all_objects_integration.startup import settle_and_accept
from openend.tasks.all_objects_integration.trial_input import load_config,read,sha,validate_actions
TASKROOT=NEWROOT/'tasks/all_objects_integration'
OUTDIR=Path('/pilot');OUT=OUTDIR/'navigation_result.json'
report=dict(status='initializing',control_hz=20,rows=[],contacts=[],initial_pose_writes=1,
 rollout_pose_velocity_writes=0,navigation_model='existing_L1_native_planar_constraints_not_wheels',
 self_contacts=[],activation_source='measured_native_base_pose_sector_polygon',full_task_success=False,scoring_started=False,task_initialization_owner='unified_scene')
owner=None;sim=None
trial=None;action_rows=None
timing=dict(physics_s=0.,readers_s=0.,control_s=0.,control_frames=0)
wall_start=time.perf_counter()
class ActionSourceFinished(Exception):pass
def checkpoint():
 report['timing']=dict(timing,wall_elapsed_s=time.perf_counter()-wall_start)
 temporary=OUT.with_suffix('.tmp')
 temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(OUT)
try:
 assert not OUT.exists()
 if os.environ.get('OPENEND_TRIAL_CONFIG'):
  trial=load_config(os.environ['OPENEND_TRIAL_CONFIG'],root=NEWROOT/'tasks/scene_requirements')
 manifest_path=Path(trial['manifest']) if trial else TASKROOT/'room_all_dynamic_v2/composition.json'
 request_path=Path(trial['request']) if trial else TASKROOT/'all_tasks.json'
 manifest=read(manifest_path)
 stage=Usd.Stage.Open(manifest['scene']);stage.SetEditTarget(stage.GetSessionLayer())
 hashes={l.realPath:hashlib.sha256(Path(l.realPath).read_bytes()).hexdigest() for l in stage.GetUsedLayers() if l.realPath}
 sensor=json.loads((ROOT.parent/'hotwake_probe_inputs.json').read_text())['sensor']
 request=read(request_path)
 report['trial_input']=trial
 from openend.tasks.all_objects_integration.workspace_physics import from_scene
 workspace_selector=from_scene(stage,manifest,sensor)
 owner=MultiTaskRoom(stage,manifest,OUTDIR/'room_evidence','native-multitask-navigation',request=request,sensor=sensor,workspace_selector=workspace_selector)
 report.update(request=request,sensor=sensor,source_sha256=hashes,
  loaded_instances=list(owner.ports),status='adapters_constructed',initial_settle_s=4)
 checkpoint()
 scene=stage.GetPrimAtPath('/World/PhysicsScene')
 assert scene.GetAttribute('physxScene:timeStepsPerSecond').Get()==60
 assert scene.GetAttribute('physxScene:enableGPUDynamics').Get() is False
 base_path='/World/Robot/base_link';root=stage.GetPrimAtPath('/World/Robot')
 # One initial placement in the kitchen aisle, using existing L1 navigation configuration.
 matrix=Gf.Matrix4d(1);matrix.SetRotate(Gf.Rotation(Gf.Vec3d(0,0,1),90));matrix.SetTranslateOnly(Gf.Vec3d(3.6,.9,0))
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
 if trial and trial['mode']=='actions':
  from action22 import ACTION_NAMES
  action_rows=validate_actions(read(trial['actions']),action_names=ACTION_NAMES,joint_limits=limits,
   validate_action=validate,manifest_sha256=sha(manifest_path),request_sha256=sha(request_path),
   source_layers=hashes,ready_sha256=sha(ROOT/'recorded_ready_pose_042_v1.json'),
   ready_targets=folded['targets_rad'])
  report['action_source_sha256']=sha(trial['actions'])
  report['requested_action_frames']=len(action_rows)
 timing['initialization_s']=time.perf_counter()-wall_start
 settle_wall=time.perf_counter()
 report['status']='settling';checkpoint()
 def initial_physics_step():
  physics_step()
  if (step_count-1)%240==0:
   report['initial_completed_steps']=step_count-1;checkpoint()
   print('MULTITASK_INITIAL',step_count-1,flush=True)
 report['initial_acceptance']=settle_and_accept(owner,initial_physics_step,
  dict(events=[],docked_object_count=0,tray_dock_enabled=False))
 timing['settle_s']=time.perf_counter()-settle_wall
 identities={k:id(p.composite) for k,p in owner.ports.items()}
 report['scoring_started']=True;report['status']='navigation_running';checkpoint()
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
 def advance(v,w,phase,targets=None):
  global last_vw,activation,hz
  control_start=time.perf_counter()
  action=validate((joint20 if targets is None else targets)+[v,w],limits,last_vw,dt=.05);last_vw=[v,w]
  if targets is not None:
   for name,value in zip(JOINT_NAMES,action[:20]):native_targets[0,names.index(name)]=value
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
   t=time.perf_counter();physics_step();timing['physics_s']+=time.perf_counter()-t
   t=time.perf_counter();owner.after_physics(no_helpers);timing['readers_s']+=time.perf_counter()-t
  result=owner.finish_control(completed_substeps=plan['substeps'])
  timing['control_s']+=time.perf_counter()-control_start;timing['control_frames']+=1
  state,pose=measured()
  report['rows'].append(dict(phase=phase,robot=actual_robot,result=result,action=action,
   state=state,time_s=simulation_time,base_velocity=velocity,**plan))
  recorded_bodies.append(bodies.get_transforms().copy());recorded_links.append(robot.get_link_transforms().copy())
  if len(report['rows'])%20==0:
   checkpoint();print('MULTITASK_NAV_TICK',len(report['rows']),phase,hz,flush=True)
  if report['contacts'] or report['self_contacts']:raise RuntimeError('Environment or nonadjacent self penetration detected; stop trial')
  assert abs(float(pose[2])-float(initial_p[2]))<.01
 if action_rows is not None:
  for row in action_rows:
   if owner.dispatch.ended:break
   if row['base_mode']!=current_mode:switch(row['base_mode'])
   a=row['action'];advance(a[-2],a[-1],row['phase'],a[:20])
  report.update(status='native_action_sequence_stopped' if owner.dispatch.ended else 'native_action_sequence_completed',
   action_source_exhausted=not owner.dispatch.ended,executed_action_frames=len(report['rows']),
   final_scoreboard=owner.dispatch.scores,task_switches=owner.dispatch.switches,
   termination_reason=owner.dispatch.termination_reason or 'source_exhausted',
   scorer_instances_retained=all(id(p.composite)==identities[k] for k,p in owner.ports.items()))
  assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
  report['source_layers_unchanged']=True
  from openend.tasks.all_objects_integration.scored_switch_audit import audit
  controls=[json.loads(line) for line in (OUTDIR/'room_evidence/controls.jsonl').read_text().splitlines()]
  report['scored_switch_audit']=audit(controls,report)
  # Exit this diagnostic branch without executing the old navigation sequence.
  # Finally still saves measured evidence and releases the simulation.
  raise ActionSourceFinished()
 # Task-local smoke test only. No navigation/manipulation claim: four seconds
 # of unchanged settling followed by one second of measured idle controls.
 for _ in range(20):advance(0,0,'binding_idle')
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
 assert owner.dispatch.total_score==0
 report.update(status='native_binding_idle_passed',source_layers_unchanged=True,
  scorer_instances_retained=all(id(p.composite)==identities[k] for k,p in owner.ports.items()),
  final_scoreboard=owner.dispatch.scores,full_task_success=False)
 raise ActionSourceFinished()
 def run(kind,goal,label,direction=1):
  switch(kind);controller=PrimitiveController(dict(kind=kind,goal=goal,direction=direction))
  for _ in range(20*30):
   cur=measured()[0][-3:];world=controller.step(cur,dt=.05)
   v=math.cos(cur[2])*world[0]+math.sin(cur[2])*world[1]
   advance(float(v),float(world[2]),label)
   if controller.done:break
  else:raise RuntimeError('Original L1 primitive timeout')
  for _ in range(20):advance(0,0,label+'_stopped')
 run('drive',[3.6,-.25,math.pi/2],'approach_back_facing',-1)
 cur=measured()[0][-3:];run('turn',[*cur[:2],0.],'sweep_coffee')
 cur=measured()[0][-3:];run('turn',[*cur[:2],math.pi/2],'turn_away')
 cur=measured()[0][-3:];run('turn',[*cur[:2],math.pi],'sweep_fryer')
 cur=measured()[0][-3:];run('turn',[*cur[:2],math.pi/2],'leave_fryer')
 cur=measured()[0][-3:];run('turn',[*cur[:2],0.],'reenter_coffee')
 cur=measured()[0][-3:];run('turn',[*cur[:2],math.pi/2],'leave_coffee')
 cur=measured()[0][-3:];run('drive',[cur[0],.9,math.pi/2],'leave')
 switch('hold')
 for _ in range(2405):
  if owner.dispatch.ended:break
  advance(0,0,'outside_until_no_progress_timeout')
 else:raise RuntimeError('Shared watchdog failed to stop')
 report['transitions']=owner.transitions
 report['task_switches']=owner.dispatch.switches
 assert owner.dispatch.ended and owner.dispatch.termination_reason=='no_score_increase_timeout'
 assert abs(owner.dispatch.last_time-owner.dispatch.started_at-120)<1e-6
 assert all(id(p.composite)==identities[k] for k,p in owner.ports.items())
 assert all(owner.ports[k].index>0 for k in ('coffee','fryer'))
 assert all(p.index==0 for k,p in owner.ports.items() if k not in ('coffee','fryer'))
 report['reader_substeps']=owner.reader_counts
 assert any(r['current_task']=='coffee' for r in report['rows'])
 assert any(r['current_task']=='fryer' for r in report['rows'])
 assert any(r['phase']=='reenter_coffee_stopped' and r['current_task']=='coffee' for r in report['rows'])
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
 assert owner.dispatch.total_score==0
 report.update(status='native_multitask_navigation_passed',native_physics_steps=step_count,
  source_layers_unchanged=True,scorer_instances_retained=True,final_scoreboard=owner.dispatch.scores,
  termination_reason=owner.dispatch.termination_reason,
  idle_timeout_elapsed_s=owner.dispatch.last_time-owner.dispatch.started_at)

except ActionSourceFinished:
 pass
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
  checkpoint()
 finally:omni.kit.app.get_app().post_quit(0 if report['status']!='failed' else 1)
