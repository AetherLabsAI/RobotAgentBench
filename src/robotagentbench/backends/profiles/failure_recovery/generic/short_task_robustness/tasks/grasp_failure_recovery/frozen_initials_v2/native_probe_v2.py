"""Original full ShortTaskSession/driver, prepared cup reset, hold-only probe.

Not shared registration and not a recovery policy. No model or original motion
is read. Source scene/registry admission remains unchanged. The physical owner
sequence, settle length, camera resolutions and native baseline are reused.
"""
import argparse
import base64
import io
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback
from uuid import uuid4
from ..common import HERE,CURRENT,owned,digest,verify_sources,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--initial',required=True);p.add_argument('--output',required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--gpu-uuid',required=True);a=p.parse_args()
    if socket.gethostname() not in ('VM-0-8-ubuntu','VM-0-10-ubuntu'):raise ValueError('Only sim1/2')
    initial_path=owned(a.initial);initial=json.loads(initial_path.read_text())
    if initial['variant']!='table_waste_empty_grasp' or initial['preparation_accepted'] is not True:raise ValueError('Accepted prepared cup required')
    if digest(initial['scene'])!=initial['scene_sha256']:raise ValueError('Candidate changed')
    verify_sources(initial['source_layers']);verify_sources(initial['code_sources'])
    q=lambda *s:subprocess.check_output(['nvidia-smi',*s],text=True)
    occupancy=q('--id',a.gpu_uuid,'--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits')
    compute=q('--id',a.gpu_uuid,'--query-compute-apps=pid,process_name','--format=csv,noheader')
    pmon=q('pmon','-c','1');fields=occupancy.strip().split(',')
    active=[s.split() for s in pmon.splitlines() if s.strip() and not s.lstrip().startswith('#')]
    if q('--query-gpu=uuid','--format=csv,noheader').split()!=[a.gpu_uuid] or compute.strip() or int(fields[1])>2 or int(fields[2])!=0 or any(len(s)>1 and s[1]!='-' for s in active):
        raise RuntimeError('Existing isolated idle guard rejected; no process stopped')
    out=owned(a.output,fresh=True);out.mkdir(parents=True);cache=out/'cache'
    for name in ('cuda','data','documents','tmp'):(cache/name).mkdir(parents=True)
    os.environ.update(CUDA_VISIBLE_DEVICES='0',CUDA_DEVICE_ORDER='PCI_BUS_ID',XDG_CACHE_HOME=str(cache),
        CUDA_CACHE_PATH=str(cache/'cuda'),TMPDIR=str(cache/'tmp'))
    start=time.monotonic();app=sim=session=stage=view=None;products=[];annotators={};sid=None
    report=dict(status='initializing',variant=initial['variant'],seed=initial['seed'],profile=initial['frozen_profile_id'],
        physics_backend='cpu',runtime_registered=False,recovery_actions_tested=False,model_calls=0,
        initial_sha256=digest(initial_path),gpu=dict(index=a.gpu,uuid=a.gpu_uuid,occupancy=occupancy,pmon=pmon),cleanup_errors=[])
    def save():report['wall_seconds']=time.monotonic()-start;write_json(out/'native_report.json',report)
    try:
        save()
        from isaacsim import SimulationApp
        app=SimulationApp(dict(headless=True,renderer='RaytracedLighting',multi_gpu=False,active_gpu=0,physics_gpu=0,
            width=1280,height=800,sync_loads=True,limit_cpu_threads=8,fast_shutdown=False,
            extra_args=['--/log/file='+str(out/'kit.log'),'--/app/userConfigPath='+str(out/'kit_user.json'),
                '--/app/tokens/cache='+str(cache),'--/app/tokens/data='+str(cache/'data')]))
        import numpy as np
        import carb
        import omni.usd,omni.physx,omni.physics.tensors
        import omni.replicator.core as rep
        from PIL import Image
        from pxr import Usd,UsdGeom,UsdPhysics,UsdUtils,Gf
        from arena_api import robot_profiles as rp
        from arena_api.mobile_base_native import Controller
        from openend.clock import NativeSceneClock
        from short_tasks.runtime import build_session
        from short_tasks.scene import verify_manifest
        from short_tasks.agent_probe_v1 import native_owner,robot_parameter_alignment
        from short_tasks.agent_probe_v1.native_driver import NativeDriver
        from short_tasks.agent_probe_v1.host import SessionHost
        from ..common import audit_reset
        launch=initial['original_launch'];manifest=json.loads(Path(launch['manifest']).read_text())
        request=json.loads(Path(launch['request']).read_text());verify_manifest(manifest)
        if request['task_name']!='clear_table_trash':raise ValueError('Original trash request required')
        report['source_modules_sha256']={str(Path(m.__file__)):digest(m.__file__) for m in (native_owner,robot_parameter_alignment)}
        ctx=omni.usd.get_context()
        if not ctx.open_stage(initial['scene']):raise RuntimeError('Cannot open candidate scene')
        stage=ctx.get_stage();stage.SetEditTarget(stage.GetSessionLayer());audit_reset(stage,initial['robot_initialization'])
        from short_tasks.agent_probe_v1.collision_guard.runtime import NativeCollisionGuard
        collision_guard=NativeCollisionGuard(stage)
        settings=carb.settings.get_settings()
        for key,val in (('/physics/suppressReadback',False),('/physics/updateToUsd',True),('/physics/updateVelocitiesToUsd',True),('/physics/outputVelocitiesLocalSpace',False)):
            settings.set_bool(key,val)
        scene=stage.GetPrimAtPath('/World/PhysicsScene')
        if scene.GetAttribute('physxScene:enableGPUDynamics').Get() is not False or scene.GetAttribute('physxScene:timeStepsPerSecond').Get()!=60:
            raise ValueError('Original CPU/60Hz startup required')
        native_owner.initialize_base_pose(stage,initial['initial_base_xyyaw'])
        anchor=UsdPhysics.FixedJoint(stage.GetPrimAtPath('/World/Robot/base_link/FixedJoint'))
        if not anchor or anchor.GetJointEnabledAttr().Get() is False:raise ValueError('Missing original initialization anchor')
        base=UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath('/World/Robot/base_link'))
        anchor.CreateLocalPos0Attr(Gf.Vec3f(base.ExtractTranslation()));anchor.CreateLocalRot0Attr(Gf.Quatf(base.ExtractRotationQuat()))
        episode=str(uuid4());session=build_session(request,stage,manifest,out/'scoring',episode_id=episode,sensor=launch['sensor'])
        if not hasattr(session,'ports'):raise RuntimeError('Original task unavailable: '+str(session.result()))
        mobile=Controller(stage);report['robot_baseline']=robot_parameter_alignment.apply(stage)
        sid=UsdUtils.StageCache.Get().Insert(stage).ToLongInt();sim=omni.physx.get_physx_simulation_interface()
        if not sim.attach_stage(sid):raise RuntimeError('Native attach failed')
        current_time=0.;targets=robot=indices=None;pending_native=None
        def advance(dt):
            nonlocal current_time
            if collision_guard.ready and pending_native is not None:
                applied=collision_guard.filter(robot,names,pending_native,current_time)
                for name,value in applied.items():targets[0,names.index(name)]=value
                robot.set_dof_position_targets(targets,indices)
            sim.simulate(dt,current_time);sim.fetch_results();current_time+=dt
            return dict(events=[],docked_object_count=0,tray_dock_enabled=False)
        advance(1/60);view=omni.physics.tensors.create_simulation_view('numpy',sid);view.set_subspace_roots('/')
        robot=view.create_articulation_view('/World/Robot');names=list(robot.shared_metatype.dof_names)
        if robot.count!=1 or set(names)!=set(initial['robot_initialization']['positions']):raise ValueError('Exact native34 reset required')
        report['native_drive_verification']=robot_parameter_alignment.verify_native_forces(robot,names)
        targets=np.array([[initial['robot_initialization']['positions'][n] for n in names]],dtype=np.float32)
        indices=np.array([0],dtype=np.uint32);robot.set_dof_positions(targets.copy(),indices)
        robot.set_dof_velocities(np.zeros_like(targets),indices);robot.set_dof_position_targets(targets.copy(),indices)
        driven=[];limits={};units={}
        for n in names:
            prim=stage.GetPrimAtPath('/World/Robot/joints/'+n);linear=prim.IsA(UsdPhysics.PrismaticJoint)
            scale=1. if linear else math.pi/180;units[n]='m' if linear else 'rad'
            limits[n]=[float(prim.GetAttribute('physics:'+s+'Limit').Get())*scale for s in ('lower','upper')]
            if any(s.startswith('PhysicsDriveAPI:') for s in prim.GetAppliedSchemas()):driven.append(n)
        adapter=rp.NativeAdapter(rp.MOBILE22,names,units,driven,{n:float(targets[0,names.index(n)]) for n in driven},limits,
            mimic_followers=rp.inspect_mimic_followers(stage))
        target_rigids=[str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(initial['target']),Usd.TraverseInstanceProxies())
            if p.HasAPI(UsdPhysics.RigidBodyAPI) and p.GetAttribute('physics:rigidBodyEnabled').Get() is not False]
        if len(target_rigids)!=1:raise ValueError('Exactly one original physical cup body required: '+str(target_rigids))
        report['target_rigid_body']=target_rigids[0]
        target_body=view.create_rigid_body_view(target_rigids[0])
        if target_body.count!=1:raise ValueError('Original physical cup body was not bound')
        bodies=view.create_rigid_body_view([str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)
            and not str(p.GetPath()).startswith('/World/Robot/') and p.GetAttribute('physics:rigidBodyEnabled').Get() is not False])
        def snapshot():return b''.join(np.asarray(v).copy().tobytes() for v in (bodies.get_transforms(),bodies.get_velocities(),robot.get_link_transforms(),robot.get_dof_positions(),robot.get_dof_velocities(),robot.get_root_velocities()))
        clock=NativeSceneClock(stage,sim,lambda:current_time,snapshot);session.initialize(clock,view)
        for i in range(240):
            robot.set_dof_position_targets(targets,indices);session.after_physics(advance(1/60))
        report['settled_steps']=240;print('SETTLED',flush=True)
        mobile.bind();resolutions=native_owner.camera_resolutions('clear_table_trash')
        collision_guard.bind(robot,names)
        for name,path in native_owner.CAMERAS.items():
            product=rep.create.render_product(path,resolutions[name]);products.append(product)
            ann=rep.AnnotatorRegistry.get_annotator('rgb');ann.attach([product]);annotators[name]=ann
        def measured(label):
            qnow=dict(zip(names,map(float,robot.get_dof_positions()[0])));hand=initial['active_hand']
            report.setdefault('measurements',{})[label]=dict(joints=qnow,target_transforms=target_body.get_transforms().tolist(),
                target_velocities=target_body.get_velocities().tolist(),gripper_coordinates_rad={n:v for n,v in qnow.items() if '_gripper_'+hand+'_' in n},
                max_joint_target_residual=float(np.max(np.abs(robot.get_dof_positions()-targets))))
        def capture(label):
            before=snapshot();now=clock.current_time;previous=settings.get('/app/player/playSimulations');settings.set('/app/player/playSimulations',False)
            images={}
            try:
                for _ in range(4):rep.orchestrator.step(rt_subframes=1,pause_timeline=False,delta_time=0.,wait_for_render=True)
                for name,ann in annotators.items():
                    frame=np.asarray(ann.get_data())
                    if not native_owner.valid_camera_frame(name,frame,resolutions):raise RuntimeError('Invalid original camera image '+name)
                    buffer=io.BytesIO();Image.fromarray(frame[...,:3]).save(buffer,format='PNG');data=buffer.getvalue()
                    (out/(label+'_'+name+'.png')).write_bytes(data);images[name]=base64.b64encode(data).decode()
                if snapshot()!=before or clock.current_time!=now:raise RuntimeError('Rendering advanced physics')
            finally:settings.set('/app/player/playSimulations',previous)
            return images
        measured('settled');capture('settled');save()
        report['original_initial_acceptance']=session.accept_initial();report['initial_result']=session.result()
        def apply(native):
            nonlocal pending_native
            pending_native=dict(native)
        driver=NativeDriver(session,adapter,mobile,measured_joints=lambda:dict(zip(names,map(float,robot.get_dof_positions()[0]))),
            apply_native_targets=apply,simulate_fetch=advance,capture_png=lambda: capture('after_hold'))
        host=SessionHost(session,driver)
        # Same source 22D hold contract, no model/recovery movement.
        hold=[float(targets[0,names.index(n)]) for n in rp.get(rp.MOBILE22).order]+[0.,0.]
        reply=host.execute(dict(op='execute_actions',request_id=str(uuid4()),episode_id=episode,expected_step=0,actions=[hold[:] for _ in range(20)]))
        measured('after_hold');report['final_result']=session.result()
        if report['final_result'].get('success') or report['final_result'].get('scores',{}).get('process') or report['final_result'].get('scores',{}).get('current_final'):
            raise RuntimeError('Idle failed grasp must not earn task progress')
        report.update(status='completed',hold_control_steps=reply['actions_executed'],physics_hz=clock.hz,
            original_native_initial_accepted=True,physical_hold_executed=True,source_motion_read=False)
        verify_sources(initial['source_layers']);verify_sources(initial['code_sources'])
    except BaseException as exc:report.update(status='invalid',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        save()
        for name,callback in (('session',lambda:session and session.close()),('annotators',lambda:[ann.detach() for ann in annotators.values()]),
                ('products',lambda:[p.destroy() for p in products]),('view',lambda:view and view.invalidate()),
                ('physics',lambda:sim and sid is not None and sim.detach_stage()),('app',lambda:app and app.close())):
            try:callback()
            except BaseException as exc:report['cleanup_errors'].append(name+': '+str(exc))
        save()
    print(json.dumps({k:report.get(k) for k in ('status','error','profile','hold_control_steps','wall_seconds')}),flush=True)
    os._exit(0 if report['status']=='completed' and not report['cleanup_errors'] else 1)


if __name__=='__main__':main()
