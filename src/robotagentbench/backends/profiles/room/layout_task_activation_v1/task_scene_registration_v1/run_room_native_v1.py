"""Isolated Action22 drive/turn trial using native planar constraints.

This is the reference constrained navigation model, NOT wheel dynamics.
Only initial placement writes a pose. During rollout, only joint drive targets,
navigation constraint anchors at measured stopped poses and wake requests change.
"""
import sys, os, json, math, hashlib, traceback
from pathlib import Path
sys.path.insert(0,'/pilot')
from action22 import VERSION, JOINT_NAMES, ACTION_NAMES, STATE_NAMES, StatePacker, validate
import numpy as np
from navigation import PrimitiveController
import omni.kit.app, omni.physx, omni.physics.tensors
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, UsdUtils, Gf, PhysicsSchemaTools

ROOT=Path('/pilot');OUT=ROOT/'navigation_result.json'
report=dict(status='initializing',action_version=VERSION,action_names=ACTION_NAMES,state_names=STATE_NAMES,
 control_hz=20,physics_hz=60,pose_source='native_physx_measured_base',navigation_model='native_planar_constraints_not_wheels',
 initial_pose_writes=1,rollout_pose_velocity_writes=0,full_openended_acceptance=False,rows=[],contacts=[])
sim=None
try:
 assert not OUT.exists(),'Unique output required'
 geometry=json.loads((ROOT/'calibration_drafts_layout01.json').read_text())
 src=ROOT/'room.usda';sha=hashlib.sha256(src.read_bytes()).hexdigest()
 stage=Usd.Stage.Open(str(src));stage.SetEditTarget(stage.GetSessionLayer())
 hz=report['physics_hz'];assert hz==60;substeps=hz//20
 report['clock_trial']='l1_240_reference' if hz==240 else 'layout_60_compatibility'
 scene=stage.GetPrimAtPath('/World/PhysicsScene')
 assert scene.GetAttribute('physxScene:timeStepsPerSecond').Get()==60
 assert scene.GetAttribute('physxScene:enableGPUDynamics').Get() is False
 scene.GetAttribute('physxScene:timeStepsPerSecond').Set(hz)
 base_path='/World/Robot/base_link';root=stage.GetPrimAtPath('/World/Robot')
 # One diagnostic initial placement in the sofa/table aisle. No source save.
 matrix=Gf.Matrix4d(1);matrix.SetRotate(Gf.Rotation(Gf.Vec3d(0,0,1),0));matrix.SetTranslateOnly(Gf.Vec3d(-.95,-1.05,0))
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
 step_count=0;simulation_time=0.
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
 stage.Export(str(ROOT/'native_initial.usda'))
 names=list(robot.shared_metatype.dof_names);ids=[names.index(n) for n in JOINT_NAMES]
 native_targets=robot.get_dof_positions().copy()
 robot.set_dof_positions(native_targets.copy(),np.array([0],dtype=np.uint32))
 robot.set_dof_velocities(np.zeros_like(native_targets),np.array([0],dtype=np.uint32))
 report['initial_joint_state_restores']=1;report['initial_joint_velocity_zeroes']=1
 for name,i in zip(JOINT_NAMES,ids):native_targets[0,i]=min(limits[name][1],max(limits[name][0],native_targets[0,i]))
 joint20=[float(native_targets[0,i]) for i in ids]
 packer=StatePacker();last_vw=[0.,0.];current_mode='hold'
 table_paths=['/World/Assets/CoffeeTable/black_box_3','/World/Assets/CoffeeTable/black_box_4']
 table_ids={p:PhysicsSchemaTools.sdfPathToInt(p) for p in table_paths}
 table=next(e for e in geometry['entries'] if e['id']=='table')['config'];active=False
 def measured():
  q=robot.get_dof_positions()[0];pose=robot.get_root_transforms()[0]
  return packer.pack(dict(zip(names,map(float,q))),list(map(float,pose))),pose
 def switch(kind):
  global current_mode
  state,pose=measured();before=pose.copy()
  velocity=robot.get_root_velocities()[0]
  report.setdefault('constraint_switches',[]).append(dict(mode=kind,linear_velocity=list(map(float,velocity[:3])),angular_velocity=list(map(float,velocity[3:]))))
  assert max(abs(x) for x in last_vw)<.001,'L1 switch requires stopped command'
  # L1 completion uses measured goal error + command speed, not this raw velocity.
  anchor.GetJointEnabledAttr().Set(False)
  for joint in nav.values():joint.GetJointEnabledAttr().Set(False)
  nav[kind].GetLocalPos0Attr().Set(Gf.Vec3f(*map(float,pose[:3])))
  nav[kind].GetLocalRot0Attr().Set(Gf.Quatf(float(pose[6]),Gf.Vec3f(*map(float,pose[3:6]))))
  nav[kind].GetJointEnabledAttr().Set(True);sim.flush_changes();current_mode=kind
  assert np.array_equal(before,robot.get_root_transforms()[0]),'Constraint switch moved root without stepping'
 def overlaps_rectangle(x,y,yaw):
  # Closed fan versus axis-aligned target rectangle, including arc intersections.
  poly=table['targetPolygon'];r=1.32;half=math.pi/4
  def within(px,py):return math.hypot(px-x,py-y)<=r+1e-9 and abs(math.atan2(math.sin(math.atan2(py-y,px-x)-yaw),math.cos(math.atan2(py-y,px-x)-yaw)))<=half+1e-9
  lo=[min(p[k] for p in poly) for k in (0,1)];hi=[max(p[k] for p in poly) for k in (0,1)]
  if lo[0]<=x<=hi[0] and lo[1]<=y<=hi[1]:return True
  if any(within(*p) for p in poly):return True
  for a in (yaw-half,yaw+half):
   delta=[r*math.cos(a),r*math.sin(a)];t0,t1=0.,1.
   for k,start in enumerate((x,y)):
    if abs(delta[k])<1e-12:
     if start<lo[k] or start>hi[k]:t0,t1=1.,0.;break
    else:
     u,v=sorted(((lo[k]-start)/delta[k],(hi[k]-start)/delta[k]));t0=max(t0,u);t1=min(t1,v)
   if t0<=t1:return True
  for i,p in enumerate(poly):
   q=poly[(i+1)%len(poly)];dx,dy=q[0]-p[0],q[1]-p[1];ax,ay=p[0]-x,p[1]-y
   A=dx*dx+dy*dy;B=2*(ax*dx+ay*dy);C=ax*ax+ay*ay-r*r;disc=B*B-4*A*C
   if A and disc>=0:
    for t in ((-B-math.sqrt(disc))/(2*A),(-B+math.sqrt(disc))/(2*A)):
     if 0<=t<=1 and within(p[0]+t*dx,p[1]+t*dy):return True
  return False
 def contact(headers,points):
  for h in headers:
   if not h.num_contact_data:continue
   actors=[str(PhysicsSchemaTools.intToSdfPath(a)) for a in (h.actor0,h.actor1)]
   if sum(a.startswith('/World/Robot/') for a in actors)!=1:continue
   colliders=[str(PhysicsSchemaTools.intToSdfPath(a)) for a in (h.collider0,h.collider1)]
   external=[c for c in colliders if not c.startswith('/World/Robot/')]
   if all('floor' in c.lower() or 'ground' in c.lower() for c in external):continue
   sep=min(float(p.separation) for p in points[h.contact_data_offset:h.contact_data_offset+h.num_contact_data])
   if sep<-.001:report['contacts'].append(dict(step=step_count,actors=actors,colliders=colliders,min_separation_m=sep))
 subscription=sim.subscribe_contact_report_events(contact)
 def advance(v,w,phase):
  global last_vw,active
  action=validate(joint20+[v,w],limits,last_vw,dt=.05);last_vw=[v,w]
  for outer,inner in [('idx41_gripper_l_outer_joint1','idx31_gripper_l_inner_joint1'),('idx81_gripper_r_outer_joint1','idx71_gripper_r_inner_joint1')]:native_targets[0,names.index(inner)]=-native_targets[0,names.index(outer)]
  robot.set_dof_position_targets(native_targets,np.array([0],dtype=np.uint32))
  if current_mode=='drive':drives['drive'].GetTargetVelocityAttr().Set(v)
  elif current_mode=='turn':drives['turn'].GetTargetVelocityAttr().Set(math.degrees(w))
  state_before,_=measured();x,y,yaw=state_before[-3:]
  hit=overlaps_rectangle(x,y,yaw)
  distance=math.hypot(x-table['center'][0],y-table['center'][1])
  if hit or math.hypot(x-table['target'][0],y-table['target'][1])<=table['nearWake']:active=True
  elif distance>max(table['exit'],1.82):active=False
  before={p:sim.is_sleeping(sid,i) for p,i in table_ids.items()}
  if active:
   for i in table_ids.values():sim.wake_up(sid,i)
  for _ in range(substeps):physics_step()
  state,pose=measured()
  recorded_bodies.append(bodies.get_transforms().copy())
  recorded_links.append(robot.get_link_transforms().copy())
  assert abs(float(pose[2])-float(initial_p[2]))<.01,'Unexpected root height drift'
  report['rows'].append(dict(tick=len(report['rows'])+1,physics_tick=step_count,simulation_time_s=simulation_time,
   base_pose_xyzw=list(map(float,pose)),base_velocity=list(map(float,robot.get_root_velocities()[0])),
   physics_hz_readback=scene.GetAttribute('physxScene:timeStepsPerSecond').Get(),physics_dt_s=1/hz,substeps_per_control=substeps,
   action=action,state=state,phase=phase,table_overlap=hit,table_wake_requested=active,
   table_sleeping_before=before,table_sleeping_after={p:sim.is_sleeping(sid,i) for p,i in table_ids.items()}))
  if report['contacts']:raise RuntimeError('Non-floor robot penetration detected; stop trial')
 for _ in range(40):advance(0,0,'settle')
 initial_state=report['rows'][-1]['state']
 # Short L1 closed-loop drive and stationary turn in the clear aisle.
 x,y,yaw=initial_state[-3:]
 goals=[dict(kind='drive',direction=1,goal=[x+.21*math.cos(yaw),y+.21*math.sin(yaw),yaw]),
        dict(kind='turn',goal=[x+.21,y,yaw+2.7]),
        dict(kind='turn',goal=[x+.21,y,-math.pi/2]),
        dict(kind='drive',direction=1,goal=[x+.21,y-.9,-math.pi/2])]
 for primitive in goals:
  switch(primitive['kind']);controller=PrimitiveController(primitive)
  for _ in range(20*30):
   current=measured()[0][-3:];world=controller.step(current,dt=.05)
   v=math.cos(current[2])*world[0]+math.sin(current[2])*world[1]
   advance(float(v),float(world[2]),primitive['kind'])
   if controller.done:break
  else:raise RuntimeError('L1 navigation primitive timeout')
  for _ in range(10):advance(0,0,'stop')
  if primitive is goals[0]:drive_state=report['rows'][-1]['state']
 switch('hold')
 for _ in range(100):advance(0,0,'hold')
 assert active is False,'Exit hysteresis did not clear'
 assert all(sim.is_sleeping(sid,i) for i in table_ids.values()),'Drawers did not naturally sleep after exit'
 report['exit_and_natural_sleep_passed']=True
 report['frequency_switch_scope']='60 Hz room navigation only; no manipulation or task-selected frequency switch'
 report['clock_switches']=[]
 for next_hz in ():
  old_hz=hz;before=robot.get_root_transforms()[0].copy();velocity=robot.get_root_velocities()[0]
  assert np.linalg.norm(velocity[:3])<.005 and np.linalg.norm(velocity[3:])<.01
  scene.GetAttribute('physxScene:timeStepsPerSecond').Set(next_hz);sim.flush_changes()
  hz=next_hz;substeps=hz//60
  assert np.array_equal(before,robot.get_root_transforms()[0]),'Clock switch moved root without stepping'
  report['clock_switches'].append(dict(from_hz=old_hz,to_hz=hz,after_control_tick=len(report['rows']),simulation_time_s=simulation_time,pose_unchanged_without_step=True))
  for _ in range(120):advance(0,0,'rate_probe_'+str(hz))
 final_state=report['rows'][-1]['state']
 report['translation_m']=math.dist(initial_state[-3:-1],drive_state[-3:-1]);report['yaw_change_rad']=final_state[-1]-drive_state[-1]
 report['max_yaw_excursion_rad']=max(abs(row['state'][-1]-initial_state[-1]) for row in report['rows'])
 report['total_displacement_m']=math.dist(initial_state[-3:-1],final_state[-3:-1])
 assert report['translation_m']>.15 and report['max_yaw_excursion_rad']>2.6 and report['total_displacement_m']>.85,'Measured base did not follow commands'
 report['source_unchanged']=sha==hashlib.sha256(src.read_bytes()).hexdigest();assert report['source_unchanged']
 report['sleep_to_wake_transitions']=sum(sum(row['table_sleeping_before'][p] and not row['table_sleeping_after'][p] for p in table_paths) for row in report['rows'] if row['table_wake_requested'])
 assert report['sleep_to_wake_transitions']>=2,'Need measured sleep-to-wake evidence from both drawers'
 report['status']='native_action22_navigation_smoke_passed'
except Exception:
 report['status']='failed';report['error']=traceback.format_exc()
finally:
 if 'recorded_bodies' in globals() and recorded_bodies:
  np.savez_compressed(ROOT/'native_poses.npz',bodies=np.array(recorded_bodies),links=np.array(recorded_links))
 if sim:sim.detach_stage()
 OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print('NAVIGATION_RESULT '+json.dumps({k:v for k,v in report.items() if k not in ('rows','action_names','state_names')}),flush=True)
 omni.kit.app.get_app().post_quit(0 if report['status']!='failed' else 1)
