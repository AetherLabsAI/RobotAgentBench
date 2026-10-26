"""Frozen-route native diagnostic; NOT a scored episode or model rollout.

Uses the unchanged prepared shelf room and existing owner's CPU 60 Hz/4 s
settling + ready-pose restoration. Only the new human and its approved
temporary target constraint are authored in an anonymous session layer.
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
from .reporting import report_json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--launch',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--gpu',type=int,required=True)
    ap.add_argument('--record-video',action='store_true',help='Record actual head-camera frames at control boundaries')
    ap.add_argument('--full-hd',action='store_true',help='User-requested 1920x1080 render; unchanged camera pose, physics and action timing')
    ap.add_argument('--gpu0-graphics-only',action='store_true',help='Requisitioned GPU0: allow at most 32 MiB of graphics-only contexts, no compute allocation')
    ap.add_argument('--runtime-backend',action='store_true',help='Explicit candidate acceptance through real ShortTaskSession; does not open public availability')
    ap.add_argument('--public-session',action='store_true',help='Build through unmodified registry/build_session, including source verification and contact router')
    ap.add_argument('--sensor-file',type=Path,help='Existing owner calibration containing sensor; mandatory for public-session')
    ap.add_argument('--robot-approach-source',type=Path,help='Task-owned continuous recorded 20 Hz actions; human activation never pauses or reverses the robot')
    ap.add_argument('--near-target-entry',action='store_true',help='User-approved shorter existing entry segment; destination must be away from the nearby gripper')
    ap.add_argument('--side-grip',action='store_true',help='New side-entry forearm and fitted cylindrical grip diagnostic')
    ap.add_argument('--reviewed-grip',action='store_true',help='Use the user-reviewed palm-first 010 pose and its CPU closure bake')
    ap.add_argument('--short-wrist',action='store_true',help='Use the reviewed 011 hand plus 25 mm wrist, including shortened collision geometry')
    ap.add_argument('--lateral-entry',action='store_true',help='012 short hand with collision-planned closure, approaching diagonally from the shelf side')
    ap.add_argument('--avoid-arm-retreat',action='store_true',help='Select the side retreat against observed whole robot and scene geometry')
    ap.add_argument('--despawn-after-exit',action='store_true',help='Hide and disable the human only after the complete visible retreat')
    ap.add_argument('--product',choices=['sprite_c1','gatorade_t'],default='sprite_c1')
    ap.add_argument('--relocation-seed',type=int,help='Reproducible empty-position candidate order; defaults to room seed')
    ap.add_argument('--contact-closure',action='store_true',help='Common hand envelope; individual digits stop on native target contact')
    ap.add_argument('--overhead-entry',action='store_true',help='Descend from free space below the next shelf board, without moving stock')
    ap.add_argument('--allow-gpu4-idle-1mib',action='store_true',help='Explicitly authorized diagnostic: GPU4 only, <=1 MiB, zero utilization and no compute/graphics processes')
    ap.add_argument('--allow-gpu5-idle-1mib',action='store_true',help='Explicitly authorized fallback: GPU5 only, <=1 MiB, zero utilization and no compute/graphics processes')
    ap.add_argument('--allow-gpu4-idle-2mib',action='store_true',help='New explicit authorization: GPU4 only, <=2 MiB, zero utilization, no compute/graphics processes')
    args=ap.parse_args()
    render_width,render_height=(1920,1080) if args.full_hd else (640,480)
    if args.public_session and (not args.runtime_backend or args.sensor_file is None):
        raise ValueError('Public session requires runtime backend and existing sensor calibration')
    if not (args.contact_closure and args.despawn_after_exit):
        raise ValueError('Frozen diagnostic requires reviewed contact closure and exit despawn')
    if args.overhead_entry:raise ValueError('Do not replace the frozen entry route')
    if args.overhead_entry:args.contact_closure=True
    if args.contact_closure:args.avoid_arm_retreat=True
    if args.avoid_arm_retreat:args.lateral_entry=True
    if args.lateral_entry:args.short_wrist=True
    if args.short_wrist:args.reviewed_grip=True
    if args.reviewed_grip:args.side_grip=True
    out=args.output.resolve()
    if out.exists() or not out.is_relative_to(HERE):raise ValueError('Fresh task-owned output required')
    gpu=subprocess.check_output(['nvidia-smi','--id',str(args.gpu),'--query-gpu=uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True).strip().split(',')
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    residual_allowed=False
    if args.gpu0_graphics_only:
        if args.gpu!=0:raise ValueError('This requisition is limited to GPU0')
        inventory=ET.fromstring(subprocess.check_output(['nvidia-smi','--id','0','-q','-x'],text=True))
        process_list=inventory.find('.//gpu/processes')
        residual_allowed=(process_list is not None and int(gpu[1])<=32 and int(gpu[2])==0
            and all(p.findtext('type')=='G' for p in process_list.findall('process_info'))
            and gpu[0].strip() not in processes)
    if args.allow_gpu5_idle_1mib:
        if args.gpu!=5:raise ValueError('Fallback residual-memory authorization is limited to GPU5')
        inventory=ET.fromstring(subprocess.check_output(['nvidia-smi','--id','5','-q','-x'],text=True))
        process_list=inventory.find('.//gpu/processes')
        residual_allowed=(process_list is not None and not process_list.findall('process_info')
                          and int(gpu[1])<=1 and int(gpu[2])==0)
    if args.allow_gpu4_idle_1mib or args.allow_gpu4_idle_2mib:
        if args.gpu!=4:raise ValueError('Residual-memory authorization is limited to GPU4')
        inventory=ET.fromstring(subprocess.check_output(['nvidia-smi','--id','4','-q','-x'],text=True))
        process_list=inventory.find('.//gpu/processes')
        residual_allowed=(process_list is not None and not process_list.findall('process_info')
                          and int(gpu[1])<=(2 if args.allow_gpu4_idle_2mib else 1) and int(gpu[2])==0)
    if (int(gpu[1])!=0 and not residual_allowed) or int(gpu[2])!=0 or gpu[0].strip() in processes:
        raise RuntimeError('GPU not idle; no process stopped')
    out.mkdir()
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu[0].strip(),CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='4',
        XDG_CACHE_HOME=str(out/'cache'),CUDA_CACHE_PATH=str(out/'cache/cuda'))
    launch=json.loads(args.launch.read_text())
    manifest=json.loads(Path(launch['manifest']).read_text())
    report=dict(status='initializing',physics_backend='cpu_physx',gpu_use='rendering_only',
                full_task_verified=False,model_run=False,target_pose_writes=0,score_injection=False,
                gpu_uuid=gpu[0].strip(),cleanup_errors=[],
                gpu0_graphics_only_admission=args.gpu0_graphics_only,
                gpu_admission=dict(memory_mib=int(gpu[1]),utilization_percent=int(gpu[2]),
                    gpu4_residual_memory_authorized=args.allow_gpu4_idle_1mib or args.allow_gpu4_idle_2mib,
                    gpu5_residual_memory_authorized=args.allow_gpu5_idle_1mib,
                    authorized_residual_mib=2 if args.allow_gpu4_idle_2mib else (1 if args.allow_gpu4_idle_1mib or args.allow_gpu5_idle_1mib else 0)))
    start=time.monotonic()
    def checkpoint():
        report['wall_s']=time.monotonic()-start
        (out/'report.json').write_text(report_json(report)+'\n')
    def timeout(*_):raise TimeoutError('Existing 900 s native diagnostic budget exhausted')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(900)
    app=sim=grasp=port=session=None
    checkpoint()
    try:
        from isaacsim import SimulationApp
        app=SimulationApp(dict(headless=True,renderer='RaytracedLighting',multi_gpu=False,fast_shutdown=False,
            active_gpu=args.gpu,physics_gpu=0,width=render_width,height=render_height,sync_loads=True,limit_cpu_threads=8,
            extra_args=['--/log/file='+str(out/'kit.log'),'--/app/userConfigPath='+str(out/'kit_user.json'),
                        '--/app/tokens/cache='+str(out/'cache'),'--/app/tokens/data='+str(out/'data')]))
        import numpy as np
        import carb
        import omni.usd,omni.physx,omni.physics.tensors
        from pxr import Usd,UsdGeom,UsdPhysics,UsdUtils,Gf,PhysicsSchemaTools,PhysxSchema
        from omni.physx.bindings._physx import ContactEventType
        from arena_api import robot_profiles as rp
        from .human_asset import NativeHuman,HumanAsset,TemporaryHumanGrasp
        from .adapter import factory_override
        from .motion import RelocationMotion,seeded_destinations
        from .diagnostic_variant import bind_existing_drink
        from .cpu_contract import validate_cpu_scene,validate_human_rigid_prims
        context=omni.usd.get_context()
        if not context.open_stage(manifest['scene']):raise RuntimeError('Prepared scene load failed')
        stage=context.get_stage();stage.SetEditTarget(stage.GetSessionLayer())
        hashes={l.realPath:hashlib.sha256(Path(l.realPath).read_bytes()).hexdigest() for l in stage.GetUsedLayers() if l.realPath}
        settings=carb.settings.get_settings()
        for key,value in (('/physics/suppressReadback',False),('/physics/updateToUsd',True),
                          ('/physics/updateVelocitiesToUsd',True),('/physics/outputVelocitiesLocalSpace',False)):
            settings.set_bool(key,value)
        validate_cpu_scene(stage)
        if stage.GetPrimAtPath('/World/PhysicsScene').GetAttribute('physxScene:timeStepsPerSecond').Get()!=60:
            raise ValueError('Existing 60 Hz owner required')
        x,y,yaw=launch['initial_base_xyyaw']
        matrix=Gf.Matrix4d(1);matrix.SetRotate(Gf.Rotation(Gf.Vec3d(0,0,1),math.degrees(yaw)))
        matrix.SetTranslateOnly(Gf.Vec3d(x,y,0))
        UsdGeom.Xformable(stage.GetPrimAtPath('/World/Robot')).MakeMatrixXform().Set(matrix)
        anchor=UsdPhysics.FixedJoint(stage.GetPrimAtPath('/World/Robot/base_link/FixedJoint'))
        base=UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath('/World/Robot/base_link'))
        anchor.CreateLocalPos0Attr(Gf.Vec3f(base.ExtractTranslation()))
        anchor.CreateLocalRot0Attr(Gf.Quatf(base.ExtractRotationQuat()))
        # Bind an existing instance in a task-local manifest copy. No scene
        # pose, physical parameter or shared task registration is modified.
        from .initialization import prepare_reservation
        plan=json.loads((Path(launch['manifest']).parent/'task_plan.json').read_text())
        manifest=prepare_reservation(plan,stage,manifest)
        reservation=manifest['environment_reservations'][0]
        reference=reservation.get('recorded_route_reference')
        if reference:
            if not np.allclose(reference['base_xyyaw'],launch['initial_base_xyyaw'],atol=1e-8,rtol=0):
                raise ValueError('Prepared station differs from the route planning reference')
            if not args.robot_approach_source or args.robot_approach_source.resolve()!=Path(reference['source']).resolve():
                raise ValueError('Reserved diagnostic requires its audited continuous action source')
            if hashlib.sha256((args.robot_approach_source/'recording_20hz.npz').read_bytes()).hexdigest()!=reference['source_sha256']:
                raise ValueError('Recorded route planning evidence changed')
        if 'source_product_range' not in reservation:raise ValueError('Source product range required')
        manifest,diagnostic_id,obj=bind_existing_drink(manifest,args.product)
        (out/'diagnostic_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        if args.public_session:
            from short_tasks.runtime import build_session
            from .startup_request import request_for_generated_room
            request=request_for_generated_room(plan,manifest)
            if request['targets'][0]['category']!=args.product:
                raise ValueError('Diagnostic product differs from generated task target')
            session=build_session(request,stage,manifest,out/'scoring',episode_id='human_diagnostic',
                sensor=json.loads(args.sensor_file.read_text())['sensor'])
            if not hasattr(session,'ports'):
                raise RuntimeError('Public session unavailable: '+str(session.result()))
            port=session.ports[diagnostic_id]
            actor=session.environment_actor
            report['public_session_built']=True
            report['verified_sources']=session.plan['source_receipt']
        else:
            Port=factory_override({},dict(type='snack_drink_to_tray'))
            port=Port(stage,manifest,out/'scoring','human_diagnostic',diagnostic_id,{'product':args.product})
        target=port.scene.identity.candidates[0]
        report['target_path']=target;report['shelf_level']=port.scene.identity.source_support
        from .adaptive_hand import AdaptiveHand
        from .contact_closure import ContactClosure,digit_for_bone
        from .frozen_native_route import world_motion,check_live_route
        asset=AdaptiveHand();contact_controller=ContactClosure();contact_digits=[]
        if args.near_target_entry:
            from .near_entry import shorten_entry
            reservation=shorten_entry(reservation,asset)
            report['near_target_entry']=reservation['diagnostic_entry_variant']
        motion,orient,destination=world_motion(reservation)
        dest=destination[:2];entry=np.asarray(motion.points[0]);grip_closure=1.
        report.update(initialization_reservation=reservation,
            relocation_sampling=dict(policy='consume_frozen_initialization_route',runtime_replanning=False,
                                     product=args.product,seed=reservation['seed']),
            entry_design=reservation['route_evidence'].get('entry'),
            grip_geometry=dict(method='common_envelope_independent_native_contact_stop',
                               product_specific_finger_fit=False,cpu_bake=str(asset.path)))
        initial=np.eye(4);initial[:3,:3]=orient;initial[:3,3]=entry
        if args.runtime_backend:
            from .adapter import create_environment_actor
            if not args.public_session:
                actor=create_environment_actor(stage,manifest,out/'runtime_actor','human_diagnostic')
            human=actor.backend.human
            report['human_waiting']=actor.backend.evidence['waiting']
        else:
            human=NativeHuman(stage,'/World/HumanInterference',initial,asset)
            report['human_waiting']=human.wait_for_trigger()
        report['human_asset']=validate_human_rigid_prims([stage.GetPrimAtPath(p) for p in human.links.values()])
        grasp=(actor.backend.grasp if args.runtime_backend else
               TemporaryHumanGrasp(stage,human.root+'/ExternalGrasp',human.links['hand.R'],target))
        from .release_collision import ReleasedTargetFilter
        released_target_filter=(actor.backend.release_filter if args.runtime_backend else
            ReleasedTargetFilter(stage,human.root,list(human.links.values()),target))
        stage.GetSessionLayer().Export(str(out/'actor_session.usda'))
        report.update(status='attaching_physics',motion_waypoints=motion.points,destination_xy=dest)
        checkpoint()
        sid=UsdUtils.StageCache.Get().Insert(stage).ToLongInt()
        sim=omni.physx.get_physx_simulation_interface()
        if not sim.attach_stage(sid):raise RuntimeError('Native attach failed')
        class Clock:
            current_time=0.
            hz=60
            def get_physics_dt(self):return 1/60
        clock=Clock()
        def advance():
            sim.simulate(1/60,clock.current_time);sim.fetch_results();clock.current_time+=1/60
            return dict(events=[],docked_object_count=0,tray_dock_enabled=False)
        advance()
        view=omni.physics.tensors.create_simulation_view('numpy',sid);view.set_subspace_roots('/')
        view.initialize_kinematic_bodies();human.bind(view)
        robot=view.create_articulation_view('/World/Robot')
        def native_robot_links():
            from .human_asset import pose_matrix
            return {'/World/Robot/'+name:pose_matrix(row[:3],row[3:7]) for name,row in
                    zip(robot.shared_metatype.link_names,robot.get_link_transforms()[0])}
        names=list(robot.shared_metatype.dof_names);targets=robot.get_dof_positions().copy()
        for name,value in json.loads(Path(launch['ready_pose']).read_text())['targets_rad'].items():
            targets[0,names.index(name)]=value
        for inner,outer in rp.FOLLOWERS.items():targets[0,names.index(inner)]=-targets[0,names.index(outer)]
        ix=np.array([0],dtype=np.uint32)
        robot.set_dof_positions(targets.copy(),ix);robot.set_dof_velocities(np.zeros_like(targets),ix)
        robot.set_dof_position_targets(targets.copy(),ix)
        target_view=view.create_rigid_body_view(target)
        if args.runtime_backend:
            if not args.public_session:
                from short_tasks.runtime import ShortTaskSession
                candidate_plan=dict(plan,available=True)  # Explicit candidate test only.
                session=ShortTaskSession(candidate_plan,manifest,{diagnostic_id:port},
                    episode_id='human_diagnostic',environment_actor=actor)
            session.initialize(clock,view)
        else:
            port.initialize(clock,view)
        report['status']='settling';checkpoint()
        for i in range(240):
            robot.set_dof_position_targets(targets,ix)
            (session.after_physics if session else port.after_physics)(advance())
        report['initial_acceptance']=(session.accept_initial() if session else port.accept_initial())
        report['initial_target_pose']=target_view.get_transforms()[0].tolist()
        stage.GetSessionLayer().Export(str(out/'settled_session.usda'))
        report['native_frozen_route_recheck']=check_live_route(stage,manifest,reservation,asset,
            robot_link_poses=native_robot_links(),
            source_support_verified=bool(report['initial_acceptance'].get('accepted') and
                port.source.latest['contacts']['target_supported']))
        (out/'initialization_reservation.json').write_text(json.dumps(reservation,indent=2)+'\n')
        report['runtime_replanning']=False
        stock_view=view.create_rigid_body_view(list(port.scene.stock))
        stock_before=stock_view.get_transforms().copy()
        approach=None
        if args.robot_approach_source:
            if not args.robot_approach_source.resolve().is_relative_to(HERE):
                raise ValueError('Task-owned audited source required')
            from .recorded_approach import RecordedApproach
            units,limits,driven={},{},[]
            for name in names:
                joint=stage.GetPrimAtPath('/World/Robot/joints/'+name)
                scale=1. if joint.IsA(UsdPhysics.PrismaticJoint) else math.pi/180
                units[name]='m' if scale==1 else 'rad'
                limits[name]=[float(joint.GetAttribute('physics:'+key).Get())*scale
                              for key in ('lowerLimit','upperLimit')]
                if any(s.startswith('PhysicsDriveAPI:') for s in joint.GetAppliedSchemas()):driven.append(name)
            adapter=rp.NativeAdapter(rp.MOBILE22,names,units,driven,
                {n:float(targets[0,names.index(n)]) for n in driven},limits,
                mimic_followers=rp.inspect_mimic_followers(stage))
            approach=RecordedApproach(args.robot_approach_source,adapter)
            first=approach.native(0)
            if any(abs(first[n]-float(targets[0,names.index(n)]))>1e-6 for n in first):
                raise ValueError('Recorded approach initial commands do not match prepared ready pose')
            report['robot_approach']=dict(approach.evidence,
                initial_base_xyyaw=launch['initial_base_xyyaw'],
                executor='existing_NativeAdapter_to_native_position_drives',
                production_session=False,robot_runtime_pose_writes=0)
        # Record from before robot motion, with the existing head camera settings.
        import omni.replicator.core as rep
        from PIL import Image
        product=rep.create.render_product('/World/Robot/head_link2/Head_Camera',(render_width,render_height))
        report['render_resolution']=[render_width,render_height]
        rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach([product])
        video_times=[]
        if args.record_video:(out/'video_frames').mkdir()
        def capture(name):
            before=target_view.get_transforms().copy()
            playing=settings.get('/app/player/playSimulations')
            settings.set('/app/player/playSimulations',False)
            try:
                for _ in range(4):
                    rep.orchestrator.step(rt_subframes=1,pause_timeline=False,delta_time=0.,wait_for_render=True)
                frame=np.asarray(rgb.get_data()).copy()
                if frame.shape!=(render_height,render_width,4):raise RuntimeError('Invalid head camera image dimensions')
                if not np.array_equal(before,target_view.get_transforms()):raise RuntimeError('Render advanced target physics')
                Image.fromarray(frame[:,:,:3]).save(out/(name+'.png'))
            finally:settings.set('/app/player/playSimulations',playing)
        def video_frame():
            capture('video_frames/frame_%05d'%len(video_times))
            video_times.append(clock.current_time)
        capture('before')
        if args.record_video:video_frame()
        if args.runtime_backend:
            from .session_probe import run
            run(session,approach,robot,targets,names,ix,advance,capture,video_frame,
                args.record_video,out,report,checkpoint)
            return 0 if report['runtime_backend_verified'] or report.get('idle_session_verified') else 1
        # User-approved pre-contact gate: elapsed time alone never starts the
        # hand. This hold-only diagnostic can verify idle suppression; a real
        # robot approach must come from the unchanged action execution owner.
        from .proximity import NativeTriggerFacts, ProximityTrigger
        trigger_reader=NativeTriggerFacts(port.source,view)
        trigger_gate=ProximityTrigger(target)
        from .intervention_rules import HumanCollisionLedger
        ledger=HumanCollisionLedger({str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath('/World/Robot'))
                                    if p.HasAPI(UsdPhysics.RigidBodyAPI)},set(human.links.values()))
        approach_contact_pairs={};approach_contacts=[]
        report['status']='waiting_for_gripper';checkpoint()
        trigger_trace=[]
        wait_steps=3*(approach.trigger_last_frame+1) if approach else 300
        action_trace=[]
        for waiting_step in range(wait_steps+1):
            facts=trigger_reader.read()
            decision=trigger_gate.observe(facts,route_clear=True)
            trigger_trace.append(decision)
            # Save before route admission too, so a safety rejection retains
            # the genuine proximity event instead of losing its evidence.
            if decision['start']:
                (out/'trigger_trace.json').write_text(json.dumps(trigger_trace,indent=2)+'\n')
                report['proximity_trigger']=decision
                report['robot_source_frame_at_trigger']=waiting_step//3
            if decision['start']:
                # Recheck actual settled geometry at activation, not merely at
                # room generation. No route replanning or robot pose changes.
                if args.near_target_entry or reference:
                    from .near_entry import check_away_from_gripper
                    from evaluation.tasks.shelf_pick.geometry import rotation
                    inverse=rotation(port.source.latest['poses'][target]['quaternion_wxyz']).inv()
                    report['relocation_away_from_gripper']=check_away_from_gripper(facts,
                        inverse.apply(np.asarray(motion.points[4])-motion.points[1]))
                report['native_activation_route_recheck']=check_live_route(stage,manifest,reservation,asset,
                    source_support_verified=facts['target_supported'],robot_link_poses=native_robot_links())
                break
            if waiting_step<wait_steps:
                if waiting_step%3==0:
                    port.begin_selected_control(True)
                    if approach:
                        frame,command=approach.command_at_step(waiting_step)
                        for name,value in command.items():targets[0,names.index(name)]=value
                        action_trace.append(dict(time_s=clock.current_time,source_frame=frame,
                                                 native_drive_targets=command))
                robot.set_dof_position_targets(targets,ix)
                port.after_physics(advance())
                for header in sim.get_contact_report()[0]:
                    pair_ids=tuple(sorted((int(header.actor0),int(header.actor1))))
                    if 0 in pair_ids:continue
                    paths=tuple(str(PhysicsSchemaTools.intToSdfPath(a)) for a in pair_ids)
                    if not any(p.startswith(human.root+'/') for p in paths):continue
                    if header.type==ContactEventType.CONTACT_LOST:approach_contact_pairs.pop(pair_ids,None)
                    else:approach_contact_pairs[pair_ids]=paths
                ledger.observe(waiting_step,clock.current_time,list(approach_contact_pairs.values()),complete=True)
                if approach_contact_pairs:
                    approach_contacts.append(dict(time_s=clock.current_time,pairs=list(approach_contact_pairs.values())))
                if waiting_step%3==2:
                    port.consume_control(True)
                    if args.record_video:video_frame()
                if waiting_step%60==0:
                    report['approach_physical_step']=waiting_step;checkpoint()
        (out/'robot_actions.json').write_text(json.dumps(action_trace,indent=2)+'\n')
        report['approach_human_contacts']=approach_contacts
        (out/'trigger_trace.json').write_text(json.dumps(trigger_trace,indent=2)+'\n')
        report['proximity_trigger']=decision
        report['waiting_physical_steps']=waiting_step
        report['approach_final_joints']=dict(zip(names,map(float,robot.get_dof_positions()[0])))
        report['approach_final_facts']=facts
        report['human_collision_deduction']=ledger.deduction
        report['collision_episodes']=ledger.contact_episodes
        if not trigger_gate.fired:
            report.update(status='waiting_no_trigger',human_motion_started=False,
                          automatic_trigger_verified=False,
                          idle_trigger_suppression_verified=True)
            return 0  # Finally still closes native resources; no forced start.
        report['human_appearance']=dict(human.activate_on_trigger(),at_simulation_s=clock.current_time,
                                        reason='measured_25cm_proximity_and_route_admitted')
        report['human_motion_started']=True
        if approach:
            report['robot_approach']['intervention_response']='continue_original_commands_at_20hz'
        report['status']='moving_human';checkpoint()
        port.set_external_control(True)
        capture('triggered')
        motion_start_s=clock.current_time
        trace=[];active_pairs=dict(approach_contact_pairs);human_target=False;acquired=False;released=False
        closure_cap=None;last_closure=0.
        # Keep native simulation/rendering alive for two seconds after hiding.
        # Previously retirement was at i=180, but the last video sample was
        # i=179: the PNG showed disappearance while the MP4 never did.
        motion_steps=301 if args.despawn_after_exit else 181
        for i in range(motion_steps):
            global_robot_step=waiting_step+i
            if global_robot_step%3==0:
                port.begin_selected_control(True)
                if approach:
                    frame,command=approach.command_at_step(global_robot_step)
                    for name,value in command.items():targets[0,names.index(name)]=value
                    action_trace.append(dict(time_s=clock.current_time,phase='concurrent_human_motion',
                        source_frame=frame,native_drive_targets=command))
            elapsed=i/60;sample=motion.sample(elapsed)
            if sample.phase=='carry' and not acquired:
                grasp.acquire(human.views['hand.R'].get_transforms()[0],target_view.get_transforms()[0],
                              hand_target_contact=human_target,
                              hand_closed=contact_controller.ready(contact_digits) if contact_controller else
                                (last_closure>=grip_closure*motion.sample(elapsed-1/60).closure if args.side_grip else last_closure>0))
                acquired=True
            if sample.phase=='release' and acquired and not released:
                grasp.release();released=True
            if released_target_filter.update(phase=sample.phase,released=released,constraint_active=grasp.active,
                    previous_closure=last_closure,supported=bool(port.source.latest['contacts']['target_supported'])):
                report['released_target_filter']=dict(released_target_filter.evidence(),at_motion_s=elapsed)
                report['released_target_filter']['cache_invalidation']=released_target_filter.invalidate_reader_cache(port.source)
                # The local display ledger has actor pairs instead of collider
                # pairs. Expire only the now-filtered human/target records.
                active_pairs={k:pair for k,pair in active_pairs.items() if target not in pair}
                stage.GetSessionLayer().Export(str(out/'release_filter_session.usda'))
            m=np.eye(4);m[:3,:3]=orient;m[:3,3]=sample.wrist_position
            closure=sample.closure*grip_closure if args.side_grip else (min(sample.closure,closure_cap) if closure_cap is not None else sample.closure)
            if contact_controller:closure=contact_controller.command(sample.closure,sample.phase)
            if 'human_retired' not in report:human.command(m,closure)
            last_closure=closure
            robot.set_dof_position_targets(targets,ix)
            port.after_physics(advance())
            if global_robot_step%3==2:port.consume_control(True)
            headers,data=sim.get_contact_report()[:2]
            raw_target_contacts=[]
            for h in headers:
                # Diagnostic tracks resolved native actor IDs; no fake contacts.
                key=tuple(sorted((int(h.actor0),int(h.actor1))))
                if 0 in key:
                    report['unresolved_contact_headers']=report.get('unresolved_contact_headers',0)+1
                    continue
                paths=tuple(str(PhysicsSchemaTools.intToSdfPath(a)) for a in key)
                if not any(p.startswith(human.root+'/') for p in paths):continue
                if target in paths and h.type!=ContactEventType.CONTACT_LOST:raw_target_contacts.append(paths)
                if h.type==ContactEventType.CONTACT_LOST:active_pairs.pop(key,None)
                else:active_pairs[key]=paths
            target_pairs=[pair for pair in active_pairs.values() if target in pair]
            if contact_controller:
                contact_digits=sorted({digit_for_bone(bone) for bone,path in human.links.items()
                    if digit_for_bone(bone) and any(path in pair for pair in target_pairs)})
                contact_controller.observe(contact_digits,sample.phase,elapsed)
                report['contact_closure']=dict(caps=dict(contact_controller.caps),events=contact_controller.events,
                    current_digits=contact_digits,ready=contact_controller.ready(contact_digits))
            if args.avoid_arm_retreat:
                unexpected=[pair for pair in active_pairs.values() if target not in pair
                    and not any(p.startswith('/World/Robot/') for p in pair)]
                if unexpected:raise RuntimeError('Human motion touched scene geometry during '+sample.phase+': '+str(unexpected))
            human_target=any(any(p==human.links['hand.R'] or '/Bodies/f_' in p or '/Bodies/thumb_' in p
                                for p in pair) for pair in target_pairs)
            # Stop closing once a finger actually meets the drink. Continuing
            # the authored full curl through the can caused the first trial's
            # tilt; do not fix that by weakening collisions or changing friction.
            finger_contact=any(any('/Bodies/f_' in p or '/Bodies/thumb_' in p for p in pair) for pair in target_pairs)
            if not args.side_grip and closure_cap is None and sample.phase=='approach' and closure>0 and finger_contact:
                closure_cap=closure
            robot_contacts=[pair for pair in active_pairs.values() if any(p.startswith('/World/Robot/') for p in pair)]
            ledger.observe(waiting_step+i,clock.current_time,list(active_pairs.values()),complete=True)
            trace.append(dict(time_s=elapsed,phase=sample.phase,target_pose=target_view.get_transforms()[0].tolist(),
                human_target_contact=human_target,robot_human_contacts=robot_contacts,
                human_target_pairs=target_pairs,hand_closure=closure,
                contact_digits=contact_digits,
                target_pair_filter_enabled=released_target_filter.enabled,raw_human_target_contacts=raw_target_contacts,
                human_contact_pairs=list(active_pairs.values()),collision_deduction=ledger.deduction,
                supported=bool(port.source.latest['contacts']['target_supported']),constraint_active=grasp.active,
                process_points=port.composite.summary()['pick'].get('earned_points',0)))
            if i%30==0:
                report['motion_step']=i;checkpoint()
            if sample.phase=='done' and args.despawn_after_exit and 'human_retired' not in report:
                if grasp.active or not released:raise RuntimeError('Cannot despawn before releasing the target')
                if robot_contacts:raise RuntimeError('Cannot despawn while touching the robot')
                report['human_retired']=dict(human.retire_after_exit(),at_motion_s=elapsed,
                    after_retreat=True,grasp_active=False)
            if i in (54,108,180):capture({54:'grasp',108:'carry',180:'released'}[i])
            if args.record_video and i%3==2:video_frame()
        capture('final')
        if args.despawn_after_exit:
            report['post_retirement_observation']=dict(
                native_physics_steps=motion_steps-181,duration_s=(motion_steps-181)/60,
                video_frames_after_retirement=sum(t>motion_start_s+3+1/60 for t in video_times) if video_times else 0,
                repeated_still_frames=False)
        stock_after=stock_view.get_transforms().copy()
        stock_deltas={p:float(np.linalg.norm(stock_after[j,:3]-stock_before[j,:3]))
                      for j,p in enumerate(stock_view.prim_paths) if p!=target}
        report.update(status='motion_completed',acquired=acquired,released=released,
            final_target_pose=trace[-1]['target_pose'],final_supported=trace[-1]['supported'],
            human_motion_process_points=max(r['process_points'] for r in trace),
            robot_human_contact_steps=sum(bool(r['robot_human_contacts']) for r in trace),
            contact_limited_closure=closure_cap,collision_episodes=ledger.contact_episodes,
            human_collision_deduction=ledger.deduction,background_displacements_m=stock_deltas,
            final_xy_error_m=float(np.linalg.norm(np.array(trace[-1]['target_pose'][:2])-np.array(dest))))
        report['observed_robot_route_clear']=not approach_contacts and not any(r['robot_human_contacts'] for r in trace)
        report['robot_motion_policy']='continuous_recorded_commands' if approach else 'idle_suppression_diagnostic'
        report['route_admitted']=report['observed_robot_route_clear'] and report['final_supported']
        if approach:
            frames=[row['source_frame'] for row in action_trace]
            report['continuous_action_clock_verified']=frames==list(range(len(frames)))
            if not report['continuous_action_clock_verified']:
                raise RuntimeError('Robot action source paused, skipped or reversed a frame')
        report['released_target_filter']=dict(report.get('released_target_filter',{}),**released_target_filter.evidence())
        report['post_filter_native_contact_steps']=sum(bool(r['raw_human_target_contacts']) for r in trace if r['target_pair_filter_enabled'])
        qx,qy,qz,qw=trace[-1]['target_pose'][3:]
        report['final_target_tilt_deg']=math.degrees(math.acos(max(-1.,min(1.,1.-2.*(qx*qx+qy*qy)))))
        (out/'trace.json').write_text(json.dumps(trace,indent=2)+'\n')
        if not trace[-1]['supported']:raise RuntimeError('Relocated drink lacks native shelf support')
        port.set_external_control(False)
        if args.record_video:
            (out/'video_times.json').write_text(json.dumps(video_times,indent=2)+'\n')
            subprocess.run(['ffmpeg','-nostdin','-v','error','-framerate','20','-i',
                str(out/'video_frames/frame_%05d.png'),'-c:v','libx264','-pix_fmt','yuv420p',
                '-movflags','+faststart',str(out/'human_interference.mp4')],check=True)
            report['video']=dict(path='human_interference.mp4',frames=len(video_times),fps=20,
                source='native_head_camera',simulation_times='video_times.json',interpolated=False)
    except BaseException as exc:
        report.update(status='error',error=str(exc),traceback=traceback.format_exc())
        if 'trace' in locals():(out/'trace.json').write_text(json.dumps(trace,indent=2)+'\n')
    finally:
        signal.alarm(0)
        if 'action_trace' in locals():(out/'robot_actions.json').write_text(json.dumps(action_trace,indent=2)+'\n')
        if args.record_video and 'video_times' in locals() and video_times and not (out/'human_interference.mp4').exists():
            (out/'video_times.json').write_text(json.dumps(video_times,indent=2)+'\n')
            encoded=subprocess.run(['ffmpeg','-nostdin','-v','error','-framerate','20','-i',
                str(out/'video_frames/frame_%05d.png'),'-c:v','libx264','-pix_fmt','yuv420p',
                '-movflags','+faststart',str(out/'human_interference.mp4')],check=False)
            key='video' if report.get('status') in ('motion_completed','idle_completed') else 'partial_video'
            report[key]=dict(path='human_interference.mp4',frames=len(video_times),fps=20,
                            encoded=encoded.returncode==0,source='native_head_camera',
                            simulation_times='video_times.json',interpolated=False)
        if 'rgb' in locals():
            try:rgb.detach();product.destroy()
            except BaseException as exc:report['cleanup_errors'].append(dict(operation='render',error=str(exc)))
        for label,fn in [('session',lambda:session.close() if session else None),
                         ('grasp',lambda:grasp.close() if grasp else None),('port',lambda:port.close() if port and not session else None),
                         ('physics',lambda:sim.detach_stage() if sim else None)]:
            try:fn()
            except BaseException as exc:report['cleanup_errors'].append(dict(operation=label,error=str(exc)))
        if 'hashes' in locals():report['source_layers_unchanged']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
        checkpoint()
        if app:app.close()
        report['native_shutdown_complete']=True;checkpoint()
        if report['cleanup_errors']:
            return 1  # Also overrides an early diagnostic success return.
    print(report_json(report),flush=True)
    return 0 if report['status']=='motion_completed' and not report['cleanup_errors'] else 1


if __name__=='__main__':
    # Same shutdown pattern as the existing native owner: Kit is closed above;
    # avoid a second Python-GC destruction of already released native plugins.
    code=main()
    import sys
    sys.stdout.flush();sys.stderr.flush();os._exit(code)
