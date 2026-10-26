"""Real CPU coffee-bag relocation diagnostic; not a scored/model episode.

Existing atomic initialization, physics policies, source joint20 approach and
head camera are reused. Human only moves via native kinematic targets; bag
only via contact and the reviewed temporary external constraint.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--initial',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--authorized-shared-gpu2',action='store_true')
    p.add_argument('--authorized-independent-approach',action='store_true')
    a=p.parse_args();out=a.output.resolve()
    if out.exists() or not out.is_relative_to(HERE):raise ValueError('Fresh task-local output required')
    if a.gpu!=2 or not a.authorized_shared_gpu2:raise ValueError('This preview has authorization only for shared GPU2')
    uuid=subprocess.check_output(['nvidia-smi','--id','2','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    admission=subprocess.check_output(['nvidia-smi','--id','2','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
    out.mkdir();(out/'frames').mkdir()
    os.environ.update(CUDA_VISIBLE_DEVICES=uuid,CUDA_DEVICE_ORDER='PCI_BUS_ID',XDG_CACHE_HOME=str(out/'cache'),CUDA_CACHE_PATH=str(out/'cache/cuda'))
    report=dict(status='starting',gpu=2,gpu_admission=admission,sharing_authorized=True,
        physics='cpu_physx',scored_episode=False,full_pour_verified=False,model_run=False,
        bag_pose_writes=0,cleanup_errors=[])
    started=time.monotonic();app=world=reader=human=grasp=physics_runtime=rgb=product=None
    def save(name,value):
        (out/name).write_text(json.dumps(value,indent=2,default=lambda v:v.tolist() if hasattr(v,'tolist') else list(v),allow_nan=False)+'\n')
    def checkpoint():report['wall_s']=time.monotonic()-started;save('report.json',report)
    def timeout(*_):raise TimeoutError('Existing native preview 900-second watchdog')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(900);checkpoint()
    try:
        from isaacsim import SimulationApp
        app=SimulationApp(dict(headless=True,renderer='RaytracedLighting',multi_gpu=False,fast_shutdown=False,
            active_gpu=a.gpu,physics_gpu=0,width=640,height=480,sync_loads=True,limit_cpu_threads=8,
            extra_args=['--/log/file='+str(out/'kit.log'),'--/app/userConfigPath='+str(out/'kit_user.json'),
                        '--/app/tokens/cache='+str(out/'cache'),'--/app/tokens/data='+str(out/'data'),
                        '--/app/tokens/documents='+str(out/'documents')]))
        import numpy as np
        import carb
        import omni.usd,omni.physics.tensors,omni.physx
        import omni.replicator.core as rep
        from PIL import Image
        from pxr import Usd,UsdGeom,UsdPhysics,PhysxSchema,PhysicsSchemaTools
        from omni.physx.bindings._physx import ContactEventType
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.core.utils.types import ArticulationAction
        from arena_api import l1_scene,l1_fridge,physics_defaults,policy_execution as pe
        from arena_api import robot_collision_filters,finger_pad_material,finger_pad_contact
        from arena_api import microwave_physics,scene_obstacles,coffee_box_physics
        from arena_api.soft_body_runtime import EpisodePhysicsRuntime
        from arena_api.robot_initialization import apply as reset_robot
        from arena_api.task_runtime import RobotInitialization
        from arena_randomization import shelf_supports
        from evaluation.tasks.coffee_machine.isaac_evidence import CoffeeReader
        from evaluation.tasks.coffee_machine.specs import BAG
        from evaluation.tasks.coffee_machine.facts import COUNTER
        from evaluation.tasks.coffee_machine.tools.convert_trajectory import convert
        from short_task_robustness.tasks.stock_wooden_tray_interference.adaptive_hand import AdaptiveHand
        from short_task_robustness.tasks.stock_wooden_tray_interference.contact_closure import ContactClosure,digit_for_bone
        from short_task_robustness.tasks.stock_wooden_tray_interference.human_asset import NativeHuman,TemporaryHumanGrasp,pose_matrix
        from short_task_robustness.tasks.coffee_bag_interference.human_collision import CoffeeHumanPairs
        from short_task_robustness.tasks.coffee_bag_interference.motion import RelocationMotion
        from short_task_robustness.tasks.stock_wooden_tray_interference.retreat_geometry import check_transport
        from short_task_robustness.tasks.coffee_bag_interference.preview_geometry import plan,robot_visual_boxes
        from short_task_robustness.tasks.coffee_bag_interference.proximity import NativeCoffeeBagTriggerFacts,CoffeeBagApproachTrigger
        from short_task_robustness.tasks.coffee_bag_interference.acquisition import acquisition_ready
        initial=json.loads((a.initial.parent/'initial.json').read_text())
        if 'relocation_pair' not in initial:
            raise ValueError('Rebuild random initial state: latest variant requires joint start/end planning')
        if initial['role']!='source' and not a.authorized_independent_approach:
            raise ValueError('Paired/random initial state requires the authorized independent diagnostic, not source replay')
        if a.authorized_independent_approach:
            from short_task_robustness.tasks.coffee_bag_interference.preview_actions import approach
            candidate=approach(initial)
        else: candidate=convert('1-2',initial['source_episode'])
        save('preview_actions.json',candidate)
        context=omni.usd.get_context()
        if not context.open_stage(str(a.initial)):raise RuntimeError('Cannot load source initial scene')
        for _ in range(30):app.update()
        stage=context.get_stage();stage.SetEditTarget(stage.GetSessionLayer())
        hashes={l.realPath:hashlib.sha256(Path(l.realPath).read_bytes()).hexdigest() for l in stage.GetUsedLayers() if l.realPath}
        policies={'l1_scene':l1_scene.apply(stage)}
        controller=l1_fridge.Controller(stage,physics_dt=pe.PHYSICS_DT)
        root=stage.GetPrimAtPath('/World/Robot')
        if not root.HasAPI(PhysxSchema.PhysxArticulationAPI):PhysxSchema.PhysxArticulationAPI.Apply(root)
        for name,module in [('collisions',robot_collision_filters),('material',finger_pad_material),('contact',finger_pad_contact),
                ('microwave',microwave_physics),('shelf',shelf_supports),('obstacles',scene_obstacles),('coffee',coffee_box_physics)]:
            policies[name]=module.apply(stage)
        physics_defaults.apply_robot(stage)
        asset=AdaptiveHand();reservation=plan(stage,asset,initial['seed'],initial);save('reservation.json',reservation)
        motion=RelocationMotion(*reservation['waypoints']);orientation=np.asarray(reservation['orientation'])
        m=np.eye(4);m[:3,:3]=orientation;m[:3,3]=motion.points[0]
        human=NativeHuman(stage,'/World/CoffeeBagHuman',m,asset);report['human_initial_absence']=human.wait_for_trigger()
        grasp=TemporaryHumanGrasp(stage,human.root+'/ExternalGrasp',human.links['hand.R'],BAG)
        release_filter=CoffeeHumanPairs(stage,human.root,list(human.links.values()),BAG)
        report['human_collision_policy']=release_filter.evidence()
        reader=CoffeeReader(stage,out/'atomic_evidence','coffee-bag-preview','1-2')
        world=World(physics_dt=pe.PHYSICS_DT,rendering_dt=1/30,stage_units_in_meters=1.,physics_prim_path='/World/PhysicsScene',backend='numpy',device='cpu')
        robot=world.scene.add(SingleArticulation('/World/Robot',name='arena_robot'))
        physics_runtime=EpisodePhysicsRuntime(carb.settings.get_settings());physics_runtime.prepare(stage,world)
        world.reset();init=RobotInitialization(initial['robot_initialization']['positions'],initial['robot_initialization']['hold_targets'])
        save('robot_initialization.json',reset_robot(robot,stage,init));policies['runtime']=physics_defaults.verify_runtime(world,robot)
        policies['episode_physics']=physics_runtime.verify(world);save('policies.json',policies)
        reader.initialize(world);view=SimulationManager.get_physics_sim_view();view.initialize_kinematic_bodies();human.bind(view)
        target=view.create_rigid_body_view(BAG);facts=NativeCoffeeBagTriggerFacts(stage,view,reader)
        names=robot.dof_names;indices=np.array([names.index(n) for n in init.hold_targets]);values=np.array(list(init.hold_targets.values()))
        def command(targets):
            robot.apply_action(ArticulationAction(joint_positions=np.array([targets[n] for n in init.hold_targets]),joint_indices=indices))
        def advance():
            world.step(render=False);controller.after_physics_step();reader.after_step()
        report['status']='settling';checkpoint()
        for i in range(round(initial['settle_seconds']/pe.PHYSICS_DT)):
            if i%pe.SUBSTEPS==0:command(init.hold_targets)
            advance()
        report['initial_contacts']=reader.latest['contacts'][BAG]
        report['initial_tilt_deg']=reader.latest['tilt_deg'][BAG]
        report['initial_beans_in_bag']=reader.beans_in_bag()
        if COUNTER not in reader.latest['contacts'][BAG]['supports'] or reader.latest['contacts'][BAG]['robot']:
            raise RuntimeError('No valid initial native bag support/contact')
        native_robot=view.create_articulation_view('/World/Robot')
        robot_poses={'/World/Robot/'+name:pose_matrix(row[:3],row[3:]).T for name,row in
            zip(native_robot.shared_metatype.link_names,native_robot.get_link_transforms()[0])}
        robot_bounds=robot_visual_boxes(stage,release_filter.grippers,robot_poses)
        clear,reason=check_transport(motion,asset,orientation,robot_bounds,reservation['fixture_boxes'],reservation['target_bounds'])
        report['settled_route_check']=dict(clear=clear,reason=reason)
        if not clear:raise RuntimeError('Source robot blocks frozen human route: '+str(reason))
        trigger=CoffeeBagApproachTrigger(BAG,reservation=reservation);trigger.arm();closure=ContactClosure()
        product=rep.create.render_product('/World/Robot/head_link2/Head_Camera',(640,480))
        rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach([product]);settings=carb.settings.get_settings()
        frame_times=[]
        def capture():
            before=target.get_transforms().copy();playing=settings.get('/app/player/playSimulations')
            settings.set('/app/player/playSimulations',False)
            try:
                for _ in range(4):rep.orchestrator.step(rt_subframes=1,pause_timeline=False,delta_time=0.,wait_for_render=True)
                image=np.asarray(rgb.get_data()).copy()
                if image.shape!=(480,640,4) or not np.array_equal(before,target.get_transforms()):raise RuntimeError('Invalid native no-step capture')
                Image.fromarray(image[:,:,:3]).save(out/'frames'/('frame_%05d.png'%len(frame_times)))
                frame_times.append(float(world.current_time))
            finally:settings.set('/app/player/playSimulations',playing)
        capture();report['status']='robot_approaching';checkpoint()
        time0=float(world.current_time);human_start=None;digits=[];pairs={};retired=False;released=False;last_closure={d:0. for d in closure.last}
        trace=[];baseline=target.get_transforms()[0].copy();human_contacts=[]
        for step in range(200):
            command(candidate['actions'][step])
            for sub in range(pe.SUBSTEPS):
                elapsed=None if human_start is None else float(world.current_time)-human_start
                if elapsed is not None and not retired:
                    sample=motion.sample(elapsed)
                    native_contact=any(BAG in row and any(h in row for h in human.links.values()) for row in pairs.values())
                    if acquisition_ready(phase=sample.phase,already_acquired=grasp.ever_acquired,
                            measured_contact=native_contact,opposed_contact=closure.ready(digits),closure_values=last_closure):
                        grasp.acquire(human.views['hand.R'].get_transforms()[0],target.get_transforms()[0],
                            hand_target_contact=native_contact,hand_closed=closure.ready(digits))
                        report['acquisition']=dict(time_s=float(world.current_time),phase=sample.phase,
                            native_digits=list(digits),closed_fingers=last_closure.copy(),hand_target_contact=True)
                        checkpoint()
                    if sample.phase=='carry' and not grasp.ever_acquired:
                        raise RuntimeError('Cannot carry without measured opposed-contact acquisition during closure')
                    if sample.phase=='release' and grasp.active:grasp.release();released=True
                    if release_filter.update(phase=sample.phase,released=released,constraint_active=grasp.active,
                            previous_closure=last_closure,supported=COUNTER in reader.latest['contacts'][BAG]['supports']):
                        # The original atomic stream owns no openend placements.
                        from short_task_robustness.tasks.stock_wooden_tray_interference.release_collision import discard_filtered_pairs
                        discard_filtered_pairs(reader.contacts.active,list(human.links.values()),BAG)
                        reader.contacts.identity.clear()
                        pairs={key:row for key,row in pairs.items() if not (BAG in row and any(h in row for h in human.links.values()))}
                    m[:3,3]=sample.wrist_position;last_closure=closure.command(sample.closure,sample.phase);human.command(m,last_closure)
                advance()
                headers=reader.interface.get_contact_report()[0]
                for h in headers:
                    key=tuple(sorted((int(h.actor0),int(h.actor1))))
                    if 0 in key:continue
                    row=tuple(str(PhysicsSchemaTools.intToSdfPath(v)) for v in key)
                    if not any(v.startswith(human.root+'/') for v in row):continue
                    if h.type==ContactEventType.CONTACT_LOST:pairs.pop(key,None)
                    else:pairs[key]=row
                digits=sorted({digit_for_bone(b) for b,path in human.links.items() if digit_for_bone(b)
                    and any(path in row and BAG in row for row in pairs.values())})
                collision=[row for row in pairs.values() if any(v.startswith('/World/Robot/') for v in row)]
                if collision:human_contacts.append(dict(time_s=float(world.current_time),pairs=collision))
                if human_start is None:
                    event=trigger.observe(facts.read(),route_clear=clear)
                    if event['start']:
                        report['trigger']=event;report['trigger_time_s']=float(world.current_time)
                        human.activate_on_trigger();human_start=float(world.current_time);report['status']='human_relocating';checkpoint()
                else:
                    phase=motion.sample(float(world.current_time)-human_start).phase
                    closure.observe(digits,phase,float(world.current_time)-human_start)
                    if phase=='done' and not retired:
                        if grasp.active or not released or collision:raise RuntimeError('Unsafe human retirement')
                        report['human_retirement']=human.retire_after_exit();retired=True;checkpoint()
                if sub==pe.SUBSTEPS-1:
                    trace.append(dict(time_s=float(world.current_time),pose=target.get_transforms()[0].tolist(),digits=digits,
                        closure=last_closure,phase='waiting' if human_start is None else motion.sample(float(world.current_time)-human_start).phase,
                        supported=COUNTER in reader.latest['contacts'][BAG]['supports'],human_pairs=list(pairs.values()),robot_human_pairs=collision))
            capture()
            if step%20==0:report['control_step']=step;checkpoint()
            if retired and float(world.current_time)-human_start>=5.:break
        save('trace.json',trace);save('video_times.json',frame_times)
        from .training_domain import contains_origin
        final=target.get_transforms()[0];report.update(status='preview_completed' if retired else 'not_triggered',
            acquired=grasp.ever_acquired,released=released,human_retired=retired,initial_target_pose=baseline.tolist(),
            final_target_pose=final.tolist(),measured_relocation_m=float(np.linalg.norm(final[:2]-baseline[:2])),
            final_supported=COUNTER in reader.latest['contacts'][BAG]['supports'],final_tilt_deg=reader.latest['tilt_deg'][BAG],
            beans_in_bag_final=reader.beans_in_bag(),robot_human_contacts=human_contacts,closure_caps=closure.caps,
            source_robot_commands_continued=True,frame_count=len(frame_times),planned_distance_m=reservation['distance_m'],
            training_domain=reservation['training_domain'],
            final_in_training_domain=contains_origin(final[:3],reservation['training_domain']))
        from .joint_initialization import direction_check
        if human_start is not None:
            trigger_direction=report['trigger']['direction_check']
            report['final_direction_check']=direction_check(trigger_direction['start_origin_m'],
                final[:3].tolist(),trigger_direction['reference_position_m'])
            if not report['final_direction_check']['allowed']:
                raise RuntimeError('Actual final relocation approaches the triggering jaw')
        if not report['final_in_training_domain']:raise RuntimeError('Actual relocated bag left original training-data domain')
        subprocess.run(['ffmpeg','-nostdin','-v','error','-framerate','20','-i',str(out/'frames/frame_%05d.png'),
            '-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'coffee_bag_preview.mp4')],check=True)
    except BaseException as e:
        report.update(status='error',error=str(e),traceback=traceback.format_exc())
        if 'trace' in locals():save('trace.json',trace)
        if (out/'frames/frame_00000.png').exists():
            subprocess.run(['ffmpeg','-nostdin','-v','error','-framerate','20','-i',str(out/'frames/frame_%05d.png'),'-c:v','libx264','-pix_fmt','yuv420p',str(out/'partial_preview.mp4')])
    finally:
        signal.alarm(0)
        for label,fn in [('grasp',lambda:grasp.close() if grasp else None),('reader',lambda:reader.close() if reader else None),
                ('render',lambda:(rgb.detach(),product.destroy()) if rgb is not None else None),
                ('world',lambda:world.stop() if world else None),('physics',lambda:physics_runtime.close() if physics_runtime else None)]:
            try:fn()
            except BaseException as e:report['cleanup_errors'].append(dict(operation=label,error=str(e)))
        if 'hashes' in locals():report['source_layers_unchanged']=all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==h for path,h in hashes.items())
        checkpoint()
        if app:app.close()
        report['native_shutdown_complete']=True;checkpoint()
    return 0 if report['status']=='preview_completed' else 1


if __name__=='__main__':
    code=main()
    import sys
    sys.stdout.flush();sys.stderr.flush();os._exit(code)
