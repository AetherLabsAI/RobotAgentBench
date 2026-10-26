"""Task-local native diagnostic; NOT a scored episode or model rollout.

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

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--launch',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--gpu',type=int,required=True)
    ap.add_argument('--record-video',action='store_true',help='Record actual head-camera frames at control boundaries')
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
    ap.add_argument('--allow-gpu4-idle-2mib',action='store_true',help='New explicit authorization: GPU4 only, <=2 MiB, zero utilization, no compute/graphics processes')
    args=ap.parse_args()
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
                gpu_admission=dict(memory_mib=int(gpu[1]),utilization_percent=int(gpu[2]),
                    gpu4_residual_memory_authorized=args.allow_gpu4_idle_1mib or args.allow_gpu4_idle_2mib,
                    authorized_residual_mib=2 if args.allow_gpu4_idle_2mib else (1 if args.allow_gpu4_idle_1mib else 0)))
    start=time.monotonic()
    def checkpoint():
        report['wall_s']=time.monotonic()-start
        (out/'report.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    def timeout(*_):raise TimeoutError('Existing 900 s native diagnostic budget exhausted')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(900)
    app=sim=grasp=port=None
    checkpoint()
    try:
        from isaacsim import SimulationApp
        app=SimulationApp(dict(headless=True,renderer='RaytracedLighting',multi_gpu=False,fast_shutdown=False,
            active_gpu=args.gpu,physics_gpu=0,width=640,height=480,sync_loads=True,limit_cpu_threads=8,
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
        manifest,diagnostic_id,obj=bind_existing_drink(manifest,args.product)
        (out/'diagnostic_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        Port=factory_override({},dict(type='snack_drink_to_tray'))
        port=Port(stage,manifest,out/'scoring','human_diagnostic',diagnostic_id,{'product':args.product})
        target=port.scene.identity.candidates[0]
        report['target_path']=target;report['shelf_level']=port.scene.identity.source_support
        box=np.array(obj['visible_bounds_world_m']);center=box.mean(axis=0)
        # This first diagnostic room is axis aligned. General rotated-layout
        # admission is deliberately not claimed by this probe.
        board=stage.GetPrimAtPath(port.scene.identity.source_support)
        bb=UsdGeom.BBoxCache(0,['default','render']).ComputeWorldBound(board).ComputeAlignedRange()
        shelf=np.array([bb.GetMin(),bb.GetMax()])
        others=[o['visible_bounds_world_m'] for o in manifest['objects']
                if '/tier3/' in o.get('instance_id','') and o['path']!=obj['path']]
        seed=args.relocation_seed if args.relocation_seed is not None else manifest['seed']
        destinations=seeded_destinations(shelf[:,:2],box[:,:2],[np.array(b)[:,:2] for b in others],seed)
        dest=destinations[0]
        report['relocation_sampling']=dict(seed=seed,footprint_candidates=len(destinations),
            policy='seeded_empty_same_level_footprints_then_transport_and_retreat_rejection',product=args.product)
        if not args.avoid_arm_retreat:raise ValueError('Random relocation requires full route admission')
        asset=HumanAsset(HERE/'assets/derived/left_hand_side_entry_012.json') if args.lateral_entry else (
            HumanAsset(HERE/'assets/derived/left_hand_short_wrist_011.json') if args.short_wrist else (
            HumanAsset(HERE/'assets/derived/left_arm_reviewed_grip_010c.json') if args.reviewed_grip else (
            HumanAsset(HERE/'assets/derived/left_arm_grip_forearm.json') if args.side_grip else HumanAsset())))
        if args.product=='gatorade_t' and not args.contact_closure:
            asset=HumanAsset(HERE/'assets/derived/left_hand_side_entry_bottle_018.json')
        contact_controller=None;contact_digits=[]
        if args.contact_closure:
            from .adaptive_hand import AdaptiveHand
            from .contact_closure import ContactClosure,digit_for_bone
            asset=AdaptiveHand();contact_controller=ContactClosure()
        points,bones=asset.sample(0)
        palm=points[asset.collision_groups()['hand.R']]
        # Reach from the side, below the next board: the earlier vertical arm
        # visibly crossed the upper shelf. Pose design changes only this actor.
        orient=np.array([[0,1,0],[0,0,1],[1,0,0]],dtype=float)
        if args.side_grip:orient=np.array([[0,-1,0],[0,0,1],[-1,0,0]],dtype=float)
        # Align the anatomical forearm axis along the shelf front instead of
        # letting its original bent rest pose sweep through background stock.
        elbow=bones['forearm.R'][:3,3]
        angle=math.atan2(elbow[2],-elbow[1])
        # Read the schema's existing contact envelope; do not reduce collision
        # offsets to make the hand fit. Geometrically tilt the NEW actor until
        # its arm envelope stays in front of its palm's approach plane.
        schema_probe=stage.DefinePrim('/World/HumanGeometryProbe','Xform')
        contact_margin=float(PhysxSchema.PhysxCollisionAPI.Apply(schema_probe).GetContactOffsetAttr().Get())
        stage.RemovePrim('/World/HumanGeometryProbe')
        arm_points=points[asset.collision_groups()['forearm.R']+asset.collision_groups().get('upper_arm.R',[])]
        def arm_clear(a):
            return (arm_points[:,1]*math.sin(a)+arm_points[:,2]*math.cos(a)).max()+contact_margin <= \
                   (palm[:,1]*math.sin(a)+palm[:,2]*math.cos(a)).max()
        if not arm_clear(angle):
            lo,hi=angle,math.pi/2
            if not arm_clear(hi):raise ValueError('No front-plane-clear anatomical arm pose')
            for _ in range(32):
                mid=(lo+hi)/2
                if arm_clear(mid):hi=mid
                else:lo=mid
            angle=hi
        rx=np.array([[1,0,0],[0,math.cos(angle),-math.sin(angle)],[0,math.sin(angle),math.cos(angle)]])
        orient=orient@rx
        base_orientation=orient.copy()
        if args.lateral_entry:
            # User-requested side entry: rotate only the new actor around the
            # upright can axis. No robot/camera/target transform changes.
            turn=math.radians(65)
            rz=np.array([[math.cos(turn),-math.sin(turn),0],
                         [math.sin(turn),math.cos(turn),0],[0,0,1]])
            orient=rz@orient
            report['entry_design']='short_wrist_shelf_right_diagonal_65deg'
        palm_world=palm@orient.T
        grip=np.array([center[0]-palm_world[:,0].mean(),box[0,1]-palm_world[:,1].max(),center[2]-palm_world[:,2].mean()])
        grip_closure=1.
        if args.contact_closure:
            radius=float(max(box[1,:2]-box[0,:2])/2)
            grip_center=asset.palm_center(radius)
            grip=center-orient@grip_center
            report['grip_geometry']=dict(method='common_envelope_independent_native_contact_stop',
                measured_radius_m=radius,center_local=grip_center.tolist(),cpu_bake=str(asset.path),
                product_specific_finger_fit=False,physical_validation=False)
            from .retreat_geometry import choose_side_entry,choose_overhead_entry
            stock_bounds=[np.array(o['visible_bounds_world_m']) for o in manifest['objects']
                if o.get('instance_id','').startswith('shelf/') and o['path']!=obj['path']]
            if args.overhead_entry:
                ceilings=[]
                def existing_contact_offset(root):
                    values=[]
                    for p in Usd.PrimRange(root):
                        if not p.HasAPI(UsdPhysics.CollisionAPI):continue
                        value=PhysxSchema.PhysxCollisionAPI(p).GetContactOffsetAttr().Get()
                        # An unapplied optional Physx API returns None. Use the
                        # schema fallback already read on our temporary probe;
                        # do not apply APIs or author offsets on shared stock.
                        values.append(float(contact_margin if value is None else value))
                    if not values or not np.isfinite(values).all() or min(values)<0:
                        raise ValueError('Cannot resolve existing contact envelope for '+str(root.GetPath()))
                    return max(values)
                stock_offsets=[existing_contact_offset(stage.GetPrimAtPath(o['path'])) for o in manifest['objects']
                    if o.get('instance_id','').startswith('shelf/') and o['path']!=obj['path']]
                for support in board.GetParent().GetChildren():
                    b=UsdGeom.BBoxCache(0,['default','render','proxy']).ComputeWorldBound(support).ComputeAlignedRange()
                    if not b.IsEmpty() and b.GetMin()[2]>box[1,2]:
                        ceilings.append((float(b.GetMin()[2]),existing_contact_offset(support)))
                if not ceilings:raise ValueError('Upper shelf bound missing; overhead entry not admitted')
                ceiling,ceiling_offset=min(ceilings)
                report['overhead_envelopes']=dict(hand_contact_offset_m=contact_margin,
                    stock_contact_offsets_m=stock_offsets,ceiling_contact_offset_m=ceiling_offset)
                (out/'overhead_geometry.json').write_text(json.dumps(dict(base_orientation=base_orientation.tolist(),
                    grip_center=grip_center.tolist(),target_box=box.tolist(),stock_bounds=[b.tolist() for b in stock_bounds],
                    ceiling_z=ceiling,**report['overhead_envelopes']),indent=2)+'\n')
                checkpoint()
                orient,grip,adaptive_entry,entry_report=choose_overhead_entry(
                    asset,base_orientation,grip_center,box,stock_bounds,ceiling,
                    contact_margin,stock_offsets,ceiling_offset)
            else:
                orient,grip,adaptive_entry,entry_report=choose_side_entry(asset,base_orientation,grip_center,box,stock_bounds)
            report['entry_design']=entry_report
        elif args.reviewed_grip:
            if abs(asset.data['target_height_m']-(box[1,2]-box[0,2]))>1e-5:
                raise ValueError('Reviewed grip target height differs from the existing room target')
            grip=center-orient@np.array(asset.data['grip_center_local'])
            report['grip_geometry']=dict(reviewed_pose=asset.data['reviewed_pose'],
                center_local=asset.data['grip_center_local'],closure=1.,
                cpu_bake=str(asset.path),physical_validation=False)
        elif args.side_grip:
            fit=json.loads((HERE/'assets/derived/grip_fit.json').read_text())
            if fit['selected'] is None:raise ValueError('No admitted geometric grip')
            if abs(fit['radius_m']-(box[1,0]-box[0,0])/2)>1e-6:raise ValueError('Grip fit uses another target size')
            grip=center-orient@np.array(fit['selected']['center_local'])
            grip_closure=fit['selected']['closure']
            report['grip_geometry']=fit['selected']
        wrist_offset=grip-center
        dest_grip=np.array([dest[0],dest[1],center[2]])+wrist_offset
        # Clearance and entry lengths are derived from this target's size.
        height=box[1,2]-box[0,2]
        entry=grip-np.array([0,2*height,0]);exit_pose=dest_grip-np.array([0,2*height,0])
        if args.side_grip:
            # The forearm remains lateral. Approach the opening along its
            # palm normal, not along the extended fingers (which sweep the can).
            approach=orient@np.array([0.,0.,-1.])
            entry=grip+2*height*approach;exit_pose=dest_grip+2*height*approach
        if args.contact_closure:entry=adaptive_entry
        lift=height
        if args.product=='gatorade_t':
            # Bottle height is not a safe lift distance under the next board.
            # Derive the new actor's path from actual shelf/hand clearances.
            ceilings=[]
            for support in board.GetParent().GetChildren():
                b=UsdGeom.BBoxCache(0,['default','render','proxy']).ComputeWorldBound(support).ComputeAlignedRange()
                if not b.IsEmpty() and b.GetMin()[2]>box[1,2]:ceilings.append(float(b.GetMin()[2]))
            if not ceilings:raise ValueError('No upper shelf clearance bound for bottle transport')
            hand_top=max(float((asset.sample(c)[0]@orient.T+grip)[:,2].max()) for c in (0.,1.))
            free=min(ceilings)-max(float(box[1,2]),hand_top)
            if free<=0:raise ValueError('Bottle/hand cannot fit below next shelf board')
            lift=min(height,free/2)
            report['bottle_lift_geometry']=dict(ceiling_z=min(ceilings),free_height_m=free,lift_m=lift)
        raised=grip+np.array([0,0,lift]);dest_raised=dest_grip+np.array([0,0,lift])
        motion=RelocationMotion(entry,grip,raised,dest_raised,dest_grip,exit_pose)
        initial=np.eye(4);initial[:3,:3]=orient;initial[:3,3]=entry
        human=NativeHuman(stage,'/World/HumanInterference',initial,asset)
        report['human_asset']=validate_human_rigid_prims([stage.GetPrimAtPath(p) for p in human.links.values()])
        grasp=TemporaryHumanGrasp(stage,human.root+'/ExternalGrasp',human.links['hand.R'],target)
        stage.GetSessionLayer().Export(str(out/'actor_session.usda'))
        report.update(status='attaching_physics',motion_waypoints=motion.points,destination_xy=dest)
        checkpoint()
        sid=UsdUtils.StageCache.Get().Insert(stage).ToLongInt()
        sim=omni.physx.get_physx_simulation_interface()
        if not sim.attach_stage(sid):raise RuntimeError('Native attach failed')
        class Clock:
            current_time=0.
            def get_physics_dt(self):return 1/60
        clock=Clock()
        def advance():
            sim.simulate(1/60,clock.current_time);sim.fetch_results();clock.current_time+=1/60
            return dict(events=[],docked_object_count=0,tray_dock_enabled=False)
        advance()
        view=omni.physics.tensors.create_simulation_view('numpy',sid);view.set_subspace_roots('/')
        view.initialize_kinematic_bodies();human.bind(view)
        robot=view.create_articulation_view('/World/Robot')
        names=list(robot.shared_metatype.dof_names);targets=robot.get_dof_positions().copy()
        for name,value in json.loads(Path(launch['ready_pose']).read_text())['targets_rad'].items():
            targets[0,names.index(name)]=value
        for inner,outer in rp.FOLLOWERS.items():targets[0,names.index(inner)]=-targets[0,names.index(outer)]
        ix=np.array([0],dtype=np.uint32)
        robot.set_dof_positions(targets.copy(),ix);robot.set_dof_velocities(np.zeros_like(targets),ix)
        robot.set_dof_position_targets(targets.copy(),ix)
        target_view=view.create_rigid_body_view(target)
        port.initialize(clock,view)
        report['status']='settling';checkpoint()
        for i in range(240):
            robot.set_dof_position_targets(targets,ix)
            port.after_physics(advance())
        report['initial_acceptance']=port.accept_initial()
        report['initial_target_pose']=target_view.get_transforms()[0].tolist()
        if args.avoid_arm_retreat:
            from .retreat_geometry import choose_retreat,check_transport
            bounds_cache=UsdGeom.BBoxCache(0,['default','render','proxy'])
            robot_bounds=[];scene_bounds=[]
            for prim in stage.Traverse():
                path=str(prim.GetPath())
                is_robot=path.startswith('/World/Robot/')
                if not is_robot:continue
                if not prim.IsA(UsdGeom.Boundable):continue
                bb=bounds_cache.ComputeWorldBound(prim).ComputeAlignedRange()
                if bb.IsEmpty():continue
                bounds=np.array([bb.GetMin(),bb.GetMax()])
                if not np.isfinite(bounds).all():raise ValueError('Invalid retreat obstacle '+path)
                # Distant room geometry cannot meet the bounded retreat.
                robot_bounds.append(bounds)
            # Do not represent a hollow shelf as one solid box. Check actual
            # native shelf contacts during execution; use per-stock bounds
            # for the separate conservative route precheck.
            scene_bounds=[np.array(o['visible_bounds_world_m']) for o in manifest['objects']
                if o.get('instance_id','').startswith('shelf/') and o['path']!=obj['path']]
            open_points,_=asset.sample(0.)
            rejected=[];admitted=False
            for candidate in destinations:
                dest=candidate
                dest_grip=np.array([dest[0],dest[1],center[2]])+wrist_offset
                dest_raised=dest_grip+np.array([0,0,lift])
                final_box=box+np.array([dest[0]-center[0],dest[1]-center[1],0])
                proposal=RelocationMotion(entry,grip,raised,dest_raised,dest_grip,dest_grip)
                valid,transport_info=check_transport(proposal,asset,orient,robot_bounds,scene_bounds,box)
                if not valid:
                    rejected.append(dict(destination=dest,reason=transport_info))
                    if transport_info['phase']=='approach':break
                    continue
                try:
                    exit_pose,retreat_report=choose_retreat(dest_grip,open_points@orient.T,
                        robot_bounds,scene_bounds,2*height,list(asset.collision_groups().values()),final_box)
                except ValueError as exc:
                    rejected.append(dict(destination=dest,reason=str(exc)));continue
                admitted=True;break
            report['relocation_sampling'].update(rejected=rejected,admitted=admitted)
            checkpoint()
            if not admitted:raise ValueError('No geometry-admitted random relocation path; see rejected candidates')
            (out/'retreat_obstacles.json').write_text(json.dumps(dict(robot=[b.tolist() for b in robot_bounds],
                scene=[b.tolist() for b in scene_bounds],start=dest_grip.tolist(),
                open_points_world_offset=(open_points@orient.T).tolist(),length_m=float(2*height),
                hand_groups=list(asset.collision_groups().values()),target_bounds=final_box.tolist()),indent=2)+'\n')
            motion=RelocationMotion(entry,grip,raised,dest_raised,dest_grip,exit_pose)
            report['destination_xy']=dest
            report['transport_geometry']=transport_info
            report['retreat_geometry']=dict(retreat_report,robot_bound_count=len(robot_bounds),
                scene_bound_count=len(scene_bounds),exit_wrist=exit_pose.tolist())
            report['motion_waypoints']=motion.points
            # Initialization ends with one explicit protected destination.
            # The triggered actor consumes these fixed points, not a new
            # random search. This is metadata, not another scoring object.
            from .reservation import freeze_reservation,with_reservation,slot_is_clear
            reservation=freeze_reservation(target_path=target,
                shelf_level_path=port.scene.identity.source_support,seed=seed,
                shelf_bounds=shelf,target_bounds=box,destination_xy=dest,
                obstacle_bounds=scene_bounds,waypoints=motion.points,
                route_evidence=dict(transport_clear=True,retreat_clear=True,
                    transport=transport_info,retreat=retreat_report))
            manifest=with_reservation(manifest,reservation)
            (out/'diagnostic_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            (out/'initialization_reservation.json').write_text(json.dumps(reservation,indent=2)+'\n')
            report['initialization_reservation']=reservation
            motion=RelocationMotion(*reservation['waypoints'])
            if not slot_is_clear(reservation,scene_bounds):raise ValueError('Reserved slot occupied; no runtime fallback')
        stock_view=view.create_rigid_body_view(list(port.scene.stock))
        stock_before=stock_view.get_transforms().copy()
        report['status']='moving_human';checkpoint()
        port.set_external_control(True)
        # Match the owner's existing read-only head-camera capture configuration.
        import omni.replicator.core as rep
        from PIL import Image
        product=rep.create.render_product('/World/Robot/head_link2/Head_Camera',(640,480))
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
                if frame.shape!=(480,640,4):raise RuntimeError('Invalid original head camera image')
                if not np.array_equal(before,target_view.get_transforms()):raise RuntimeError('Render advanced target physics')
                Image.fromarray(frame[:,:,:3]).save(out/(name+'.png'))
            finally:settings.set('/app/player/playSimulations',playing)
        capture('before')
        def video_frame():
            capture('video_frames/frame_%05d'%len(video_times))
            video_times.append(clock.current_time)
        if args.record_video:video_frame()
        trace=[];active_pairs={};human_target=False;acquired=False;released=False
        closure_cap=None;last_closure=0.
        from .intervention_rules import HumanCollisionLedger
        ledger=HumanCollisionLedger({str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath('/World/Robot'))
                                    if p.HasAPI(UsdPhysics.RigidBodyAPI)},set(human.links.values()))
        # Keep native simulation/rendering alive for two seconds after hiding.
        # Previously retirement was at i=180, but the last video sample was
        # i=179: the PNG showed disappearance while the MP4 never did.
        motion_steps=301 if args.despawn_after_exit else 181
        for i in range(motion_steps):
            if i%3==0:port.begin_selected_control(True)
            elapsed=i/60;sample=motion.sample(elapsed)
            if sample.phase=='carry' and not acquired:
                grasp.acquire(human.views['hand.R'].get_transforms()[0],target_view.get_transforms()[0],
                              hand_target_contact=human_target,
                              hand_closed=contact_controller.ready(contact_digits) if contact_controller else
                                (last_closure>=grip_closure*motion.sample(elapsed-1/60).closure if args.side_grip else last_closure>0))
                acquired=True
            if sample.phase=='release' and acquired and not released:
                grasp.release();released=True
            m=np.eye(4);m[:3,:3]=orient;m[:3,3]=sample.wrist_position
            closure=sample.closure*grip_closure if args.side_grip else (min(sample.closure,closure_cap) if closure_cap is not None else sample.closure)
            if contact_controller:closure=contact_controller.command(sample.closure,sample.phase)
            if 'human_retired' not in report:human.command(m,closure)
            last_closure=closure
            robot.set_dof_position_targets(targets,ix)
            port.after_physics(advance())
            if i%3==2:port.consume_control(True)
            headers,data=sim.get_contact_report()[:2]
            for h in headers:
                # Diagnostic tracks resolved native actor IDs; no fake contacts.
                key=tuple(sorted((int(h.actor0),int(h.actor1))))
                if 0 in key:
                    report['unresolved_contact_headers']=report.get('unresolved_contact_headers',0)+1
                    continue
                paths=tuple(str(PhysicsSchemaTools.intToSdfPath(a)) for a in key)
                if not any(p.startswith(human.root+'/') for p in paths):continue
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
            ledger.observe(i,clock.current_time,list(active_pairs.values()),complete=True)
            trace.append(dict(time_s=elapsed,phase=sample.phase,target_pose=target_view.get_transforms()[0].tolist(),
                human_target_contact=human_target,robot_human_contacts=robot_contacts,
                human_target_pairs=target_pairs,hand_closure=closure,
                contact_digits=contact_digits,
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
                video_frames_after_retirement=sum(t>video_times[0]+3+1/60 for t in video_times) if video_times else 0,
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
        report['idle_robot_route_clear']=not any(r['robot_human_contacts'] for r in trace)
        report['route_admitted']=report['idle_robot_route_clear'] and report['final_supported']
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
        if 'rgb' in locals():
            try:rgb.detach();product.destroy()
            except BaseException as exc:report['cleanup_errors'].append(dict(operation='render',error=str(exc)))
        for label,fn in [('grasp',lambda:grasp.close() if grasp else None),('port',lambda:port.close() if port else None),
                         ('physics',lambda:sim.detach_stage() if sim else None)]:
            try:fn()
            except BaseException as exc:report['cleanup_errors'].append(dict(operation=label,error=str(exc)))
        if 'hashes' in locals():report['source_layers_unchanged']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
        checkpoint()
        if app:app.close()
        report['native_shutdown_complete']=True;checkpoint()
    print(json.dumps(report),flush=True)
    return 0 if report['status']=='motion_completed' and not report['cleanup_errors'] else 1


if __name__=='__main__':
    # Same shutdown pattern as the existing native owner: Kit is closed above;
    # avoid a second Python-GC destruction of already released native plugins.
    code=main()
    import sys
    sys.stdout.flush();sys.stderr.flush();os._exit(code)
