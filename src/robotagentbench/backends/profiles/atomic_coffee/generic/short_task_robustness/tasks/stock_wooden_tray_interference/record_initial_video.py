"""Actual native frames of existing 4 s / 60 Hz initialization, no task actions.

Reuse the human diagnostic's unchanged CPU attach, recorded ready-pose reset,
head camera and 20 fps capture. No human trigger or score injection.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
import xml.etree.ElementTree as ET

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--launch',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--gpu',type=int,required=True)
    args=ap.parse_args();out=args.output.resolve()
    if out==HERE or not out.is_relative_to(HERE) or out.exists():raise ValueError('Fresh task output required')
    gpu=subprocess.check_output(['nvidia-smi','--id',str(args.gpu),'--query-gpu=uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True).strip().split(',')
    inventory=ET.fromstring(subprocess.check_output(['nvidia-smi','--id',str(args.gpu),'-q','-x'],text=True))
    processes=inventory.find('.//gpu/processes')
    if int(gpu[1])!=0 or int(gpu[2])!=0 or processes is None or processes.findall('process_info'):
        raise RuntimeError('GPU not strictly idle; no process stopped')
    out.mkdir();(out/'video_frames').mkdir()
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu[0].strip(),CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='4',
        XDG_CACHE_HOME=str(out/'cache'),CUDA_CACHE_PATH=str(out/'cache/cuda'))
    launch=json.loads(args.launch.read_text());manifest=json.loads(Path(launch['manifest']).read_text())
    report=dict(status='initializing',physics_backend='cpu_physx',gpu_use='rendering_only',
        gpu_uuid=gpu[0].strip(),full_task_verified=False,model_run=False,human_triggered=False,
        score_injection=False,target_pose_writes=0,cleanup_errors=[],frames=0)
    start=time.monotonic();app=sim=view=rgb=product=None;hashes={}
    def checkpoint():
        report['wall_s']=time.monotonic()-start
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    def timeout(*_):raise TimeoutError('Existing 900 s native diagnostic budget exhausted')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(900);checkpoint()
    try:
        from isaacsim import SimulationApp
        app=SimulationApp(dict(headless=True,renderer='RaytracedLighting',multi_gpu=False,fast_shutdown=False,
            active_gpu=args.gpu,physics_gpu=0,width=640,height=480,sync_loads=True,limit_cpu_threads=8,
            extra_args=['--/log/file='+str(out/'kit.log'),'--/app/userConfigPath='+str(out/'kit_user.json'),
                        '--/app/tokens/cache='+str(out/'cache'),'--/app/tokens/data='+str(out/'data')]))
        import numpy as np
        import carb
        import omni.usd,omni.physx,omni.physics.tensors
        import omni.replicator.core as rep
        from PIL import Image
        from pxr import Usd,UsdGeom,UsdPhysics,UsdUtils,Gf
        from arena_api import robot_profiles as rp
        from .cpu_contract import validate_cpu_scene
        context=omni.usd.get_context()
        if not context.open_stage(manifest['scene']):raise ValueError('Cannot open generated room')
        stage=context.get_stage();stage.SetEditTarget(stage.GetSessionLayer())
        hashes={l.realPath:hashlib.sha256(Path(l.realPath).read_bytes()).hexdigest() for l in stage.GetUsedLayers() if l.realPath}
        if hashes.get(manifest['scene'])!=launch['scene_sha256']:raise ValueError('Prepared scene changed')
        validate_cpu_scene(stage)
        if stage.GetPrimAtPath('/World/PhysicsScene').GetAttribute('physxScene:timeStepsPerSecond').Get()!=60:
            raise ValueError('Existing 60 Hz initialization required')
        settings=carb.settings.get_settings()
        for k,v in (('/physics/suppressReadback',False),('/physics/updateToUsd',True),
                    ('/physics/updateVelocitiesToUsd',True),('/physics/outputVelocitiesLocalSpace',False)):
            settings.set_bool(k,v)
        x,y,yaw=launch['initial_base_xyyaw'];matrix=Gf.Matrix4d(1)
        matrix.SetRotate(Gf.Rotation(Gf.Vec3d(0,0,1),math.degrees(yaw)));matrix.SetTranslateOnly(Gf.Vec3d(x,y,0))
        UsdGeom.Xformable(stage.GetPrimAtPath('/World/Robot')).MakeMatrixXform().Set(matrix)
        anchor=UsdPhysics.FixedJoint(stage.GetPrimAtPath('/World/Robot/base_link/FixedJoint'))
        base=UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath('/World/Robot/base_link'))
        anchor.CreateLocalPos0Attr(Gf.Vec3f(base.ExtractTranslation()));anchor.CreateLocalRot0Attr(Gf.Quatf(base.ExtractRotationQuat()))
        sid=UsdUtils.StageCache.Get().Insert(stage).ToLongInt();sim=omni.physx.get_physx_simulation_interface()
        if not sim.attach_stage(sid):raise ValueError('Native CPU attach failed')
        elapsed=0.
        def advance():
            nonlocal elapsed
            sim.simulate(1/60,elapsed);sim.fetch_results();elapsed+=1/60
        advance();view=omni.physics.tensors.create_simulation_view('numpy',sid)
        view.set_subspace_roots('/');view.initialize_kinematic_bodies()
        robot=view.create_articulation_view('/World/Robot');names=list(robot.shared_metatype.dof_names)
        targets=robot.get_dof_positions().copy();ix=np.array([0],dtype=np.uint32)
        for name,value in json.loads(Path(launch['ready_pose']).read_text())['targets_rad'].items():targets[0,names.index(name)]=value
        for inner,outer in rp.FOLLOWERS.items():targets[0,names.index(inner)]=-targets[0,names.index(outer)]
        robot.set_dof_positions(targets.copy(),ix);robot.set_dof_velocities(np.zeros_like(targets),ix)
        robot.set_dof_position_targets(targets.copy(),ix)
        target_path=manifest['environment_reservations'][0]['target_path'];target=view.create_rigid_body_view(target_path)
        product=rep.create.render_product('/World/Robot/head_link2/Head_Camera',(640,480))
        rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach([product]);times=[];poses=[]
        def capture():
            before=target.get_transforms().copy();playing=settings.get('/app/player/playSimulations')
            settings.set('/app/player/playSimulations',False)
            try:
                for _ in range(4):rep.orchestrator.step(rt_subframes=1,pause_timeline=False,delta_time=0.,wait_for_render=True)
                frame=np.asarray(rgb.get_data()).copy()
                if frame.shape!=(480,640,4) or frame[:,:,:3].std()<1:raise ValueError('Invalid original head camera frame')
                if not np.array_equal(before,target.get_transforms()):raise ValueError('Capture advanced physics')
                Image.fromarray(frame[:,:,:3]).save(out/'video_frames'/('frame_%05d.png'%len(times)))
                times.append(elapsed);poses.append(before[0].tolist());report['frames']=len(times)
            finally:settings.set('/app/player/playSimulations',playing)
        report['status']='recording_initialization';checkpoint();capture()
        for i in range(240):
            robot.set_dof_position_targets(targets,ix);advance()
            if (i+1)%3==0:capture()
            if (i+1)%60==0:checkpoint()
        (out/'video_times.json').write_text(json.dumps(times,indent=2)+'\n')
        (out/'target_poses.json').write_text(json.dumps(poses,indent=2)+'\n')
        subprocess.run(['ffmpeg','-nostdin','-v','error','-framerate','20','-i',str(out/'video_frames/frame_%05d.png'),
            '-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'initialization.mp4')],check=True)
        report.update(status='recorded',simulation_s=4.,fps=20,video='initialization.mp4',
            target_translation_m=float(np.linalg.norm(np.array(poses[-1][:3])-poses[0][:3])),
            native_settling_recorded=True,initial_state_acceptance=False,
            note='Initialization observation only; no action, human intervention, or scoring acceptance.')
    except BaseException as exc:report.update(status='error',error=str(exc),traceback=traceback.format_exc())
    finally:
        signal.alarm(0)
        if rgb:
            try:rgb.detach();product.destroy()
            except BaseException as exc:report['cleanup_errors'].append(str(exc))
        if sim:
            try:sim.detach_stage()
            except BaseException as exc:report['cleanup_errors'].append(str(exc))
        report['source_layers_unchanged']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
        checkpoint()
        if app:app.close()
        report['native_shutdown_complete']=True;checkpoint()
    print(json.dumps(report),flush=True)
    return 0 if report['status']=='recorded' and not report['cleanup_errors'] else 1

if __name__=='__main__':
    import sys
    code=main();sys.stdout.flush();sys.stderr.flush();os._exit(code)
