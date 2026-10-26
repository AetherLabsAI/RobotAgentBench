"""Native cap handoff evidence, independent of historical grasp calibration.

The Sheet goal remains a physical brim grasp followed by a released, stable
hang on the canonical rack.  Export only derives geometry from the saved scene;
scoring reads the unchanged physical state and passive contact capture directly.
"""
from pathlib import Path
import json
import numpy as np

SCHEMA = 'real2sim.native_hat_hang_metrics.v1'
TARGET = '/World/Assets/BaseballCap'
BRIM = TARGET + '/Obj/Meshes/Sketchfab_model/root/GLTF_SceneRootNode/baseballcap_Scene_Node_0/Cloth_1/Pattern2D_2289809_Node_9/Object_20/Object_7'
RACK = '/World/Assets/CoatRack'

def rotations(wxyz):
    q = np.asarray(wxyz, dtype=float)
    if not np.isfinite(q).all() or np.any(abs(np.linalg.norm(q, axis=-1)-1) > .002):
        raise ValueError('Invalid recorded rigid quaternion')
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w,x,y,z = np.moveaxis(q, -1, 0)
    return np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)], axis=-1).reshape(q.shape[:-1]+(3,3))

def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {key:data[key].copy() for key in data.files}

def export(core, asset_root):
    """Create geometry-only place metrics; never alter action or physical state."""
    from pxr import Usd, UsdGeom, UsdPhysics, Sdf, UsdUtils
    from scipy.spatial import ConvexHull
    core, asset_root = Path(core), Path(asset_root)
    layer = Sdf.Layer.CreateAnonymous()
    layer.ImportFromString((core/'initial_scene.usda').read_text())
    UsdUtils.ModifyAssetPaths(layer, lambda value: str((asset_root/value).resolve()) if value and not Path(value).is_absolute() else value)
    stage = Usd.Stage.Open(layer)
    root = stage.GetPrimAtPath(TARGET)
    rigid = np.asarray(UsdGeom.Xformable(root).ComputeLocalToWorldTransform(0).RemoveScaleShear()).T
    all_points, brim_points = [], None
    for node in Usd.PrimRange(root):
        if not node.IsA(UsdGeom.Mesh) or UsdGeom.Imageable(node).ComputeVisibility() == 'invisible':
            continue
        points = np.asarray(UsdGeom.Mesh(node).GetPointsAttr().Get(), dtype=float)
        matrix = np.asarray(UsdGeom.Xformable(node).ComputeLocalToWorldTransform(0)).T
        points = (np.c_[points,np.ones(len(points))] @ (np.linalg.inv(rigid)@matrix).T)[:,:3]
        all_points.append(points)
        if str(node.GetPath()) == BRIM:
            brim_points = points
    if brim_points is None:
        raise ValueError('Canonical visible brim mesh is absent')
    points = np.concatenate(all_points)
    points = points[ConvexHull(points).vertices]
    rb = UsdPhysics.RigidBodyAPI(root)
    dynamic = bool(rb and rb.GetRigidBodyEnabledAttr().Get() is not False and not rb.GetKinematicEnabledAttr().Get())
    rack = stage.GetPrimAtPath(RACK)
    bound = UsdGeom.BBoxCache(0,['default','render'],useExtentsHint=False).ComputeWorldBound(rack).ComputeAlignedBox()
    collisions = sum(n.HasAPI(UsdPhysics.CollisionAPI) and n.GetAttribute('physics:collisionEnabled').Get() is not False for n in Usd.PrimRange(rack))
    state = load(core/'full_state.npz')
    env = json.loads((core/'runtime_environment.json').read_text())
    np.savez_compressed(core/'place_metrics.npz', schema=np.array(SCHEMA),task_id=np.array('benchmark_tidy_sofa_6_7'),target_body_path=np.array(TARGET),active_hand=np.array(env['ACTIVE_HAND']),fps=state['fps'],rendered_frames=np.array(len(state['frame_index'])),visible_points_body_local_m=points,brim_points_body_local_m=brim_points,brim_mesh_path=np.array(BRIM),target_dynamic=np.array(dynamic),rack_min_xyz_m=np.array(bound.GetMin()),rack_max_xyz_m=np.array(bound.GetMax()),rack_collision_count=np.array(collisions),rack_visible=np.array(UsdGeom.Imageable(rack).ComputeVisibility()!='invisible'))
    (core/'native_hat_metrics_provenance.json').write_text(json.dumps(dict(schema=SCHEMA,source_scene='initial_scene.usda',source_state='full_state.npz',source_contacts='target_contact_observation.json',source_trace='pick_contact_trace.npz',geometry_only_export=True,action_or_state_modified=False,legacy_seat_parameters_fabricated=False),indent=2))

def score_native(state_path, metrics, expected_furniture_min, expected_furniture_max):
    from real2sim.teleop.coat_rack_contract import HANG_BRANCH_SHAFT_PATH, HANG_BRANCH_TIP_PATH, HANG_BRANCH_JUNCTION_XYZ_M, HANG_BRANCH_TIP_CENTER_XYZ_M, BRANCH_RADIUS_M
    from real2sim.production.score_task6_code_control import _segment_intersects_aabb
    core = Path(state_path).parent
    state = load(state_path)
    trace = load(core/'pick_contact_trace.npz')
    observation = json.loads((core/'target_contact_observation.json').read_text())
    if str(metrics['schema'].item()) != SCHEMA:
        raise ValueError('Wrong native hat evidence schema')
    n = len(state['frame_index']); fps = float(state['fps'])
    if not np.array_equal(state['frame_index'], np.arange(n)):
        raise ValueError('Native frame indices must be contiguous from zero')
    for name in ('target_dynamic', 'rack_visible'):
        value = np.asarray(metrics[name])
        if value.shape != () or value.dtype.kind != 'b':
            raise ValueError('Invalid boolean geometry evidence: ' + name)
    count = np.asarray(metrics['rack_collision_count'])
    if count.shape != () or count.dtype.kind not in 'iu' or int(count) < 0:
        raise ValueError('Invalid rack collision count')
    if n < 30 or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid physical frame clock')
    if any(str(data[key].item()) != value for data,key,value in [(state,'task_id','benchmark_tidy_sofa_6_7'),(state,'target_prim',TARGET),(metrics,'task_id','benchmark_tidy_sofa_6_7'),(metrics,'target_body_path',TARGET),(trace,'target_body_path',TARGET)]):
        raise ValueError('Native hat task/target identity mismatch')
    if int(metrics['rendered_frames'])!=n or not np.isclose(float(metrics['fps']),fps) or observation['frames']!=n or observation['target']!=TARGET or not observation.get('physics_rollout_performed') or observation.get('pose_writes') is not False:
        raise ValueError('Native capture identity, clock or provenance mismatch')
    paths=state['rigid_body_paths'].astype(str).tolist()
    if paths.count(TARGET)!=1:
        raise ValueError('Target is not exactly one physical body')
    idx=paths.index(TARGET); xyz=np.asarray(state['prop_xyz_m'][:,idx],float); rot=rotations(state['prop_wxyz'][:,idx]); points=np.asarray(metrics['visible_points_body_local_m'],float); brim=np.asarray(metrics['brim_points_body_local_m'],float)
    if xyz.shape!=(n,3) or not np.isfinite(xyz).all() or points.ndim!=2 or points.shape[1]!=3 or len(points)<4 or brim.ndim!=2 or brim.shape[1]!=3 or len(brim)<4 or not np.isfinite(points).all() or not np.isfinite(brim).all() or str(metrics['brim_mesh_path'].item())!=BRIM:
        raise ValueError('Invalid native cap geometry or recorded poses')
    if np.asarray(trace['fabric_obj_xyz']).shape!=xyz.shape or not np.allclose(trace['fabric_obj_xyz'],xyz,atol=1e-6,rtol=0):
        raise ValueError('Physical contact trace does not match state')
    bounds=[(points@m.T+p) for m,p in zip(rot,xyz)];lo=np.array([p.min(0) for p in bounds]);hi=np.array([p.max(0) for p in bounds]);centers=(lo+hi)/2
    dimensions=hi[0]-lo[0]
    if np.min(dimensions)<.01 or np.max(dimensions)>.65:
        raise ValueError('Native visible cap scale is invalid')
    hand=str(metrics['active_hand'].item());hands=trace['contact_hands'].astype(str).tolist()
    if hand not in ['l','r'] or hands.count(hand)!=1:
        raise ValueError('Invalid measured gripping hand')
    h=hands.index(hand);column=['l','r'].index(hand);contacts=np.asarray(trace['pad_contacts_by_hand'][:,h]);forces=np.asarray(trace['pad_force_vectors_by_hand'][:,h]);pads=np.asarray(trace['pad_xyz_by_hand'][:,h]);actual_grip=np.asarray(trace['grip_joint_position_by_hand'][:,h]);grip=1-np.asarray(state['command_grip_rad'][:,column])/.7843981633974483
    if contacts.shape!=(n,2) or forces.shape!=(n,2,3) or pads.shape!=(n,2,3) or grip.shape!=(n,) or actual_grip.shape!=(n,) or not all(np.isfinite(v).all() for v in [contacts,forces,pads,grip,actual_grip]):
        raise ValueError('Invalid native gripper evidence')
    for key in ['command_head_rad','command_waist_rad_m']:
        if state[key].shape!=(n,2) or not np.isfinite(state[key]).all():raise ValueError('Robot posture trace is missing or nonfinite')
    opening=np.flatnonzero(grip<.8)
    if not len(opening):raise ValueError('No physical opening command')
    release=int(opening[0]);tail=slice(n-20,n);window=min(50,release)
    if window<5:raise ValueError('No measurable initial grasp interval')
    norm=np.linalg.norm(forces,axis=2);bilateral=(contacts>0).all(1)&(norm>.05).all(1);opposition=np.sum(forces[:,0]*forces[:,1],axis=1)/np.maximum(norm[:,0]*norm[:,1],1e-12)
    brim_frames=[set(),set()];rack_frames=set();unwanted=[];max_pad_penetration=0.;max_rack_penetration=0.;rack_paths={HANG_BRANCH_SHAFT_PATH,HANG_BRANCH_TIP_PATH}
    pad_paths=[f'/World/Robot/gripper_{hand}_{part}_link4' for part in ['inner','outer']]
    for event in observation['contacts']:
        frame=int(event['frame'])
        numeric=[event['impulse_norm_sum_ns'],event['minimum_separation_m'],*event['position_min_m'],*event['position_max_m']]
        if not np.isfinite(numeric).all() or event['impulse_norm_sum_ns']<0:raise ValueError('Nonfinite or negative contact evidence')
        if frame<0 or event['impulse_norm_sum_ns']<=1e-5:continue
        if frame>=n:raise ValueError('Contact frame outside recorded state')
        pair={event['actor0'],event['actor1']}
        if TARGET in pair:
            other=next(v for v in pair if v!=TARGET)
            penetration=max(0.,-float(event['minimum_separation_m']))
            if other.startswith('/World/Robot/'):
                max_pad_penetration=max(max_pad_penetration,penetration)
            if other.startswith(RACK+'/'):
                max_rack_penetration=max(max_rack_penetration,penetration)
            if other in rack_paths:rack_frames.add(frame)
            if frame<window and other in pad_paths:
                point=(np.asarray(event['position_min_m'])+np.asarray(event['position_max_m']))/2
                local=(point-xyz[frame])@rot[frame]
                if float(np.linalg.norm(brim-local,axis=1).min())<=.010:brim_frames[pad_paths.index(other)].add(frame)
        elif any(v.startswith('/World/Robot/') for v in pair) and not all(v.startswith('/World/Robot/') for v in pair):
            unwanted.append(dict(frame=frame,actors=sorted(pair)))
    self_contacts = []
    unexpected_target_links = []
    for event in observation['contacts']:
        if int(event['frame']) < 0 or event['impulse_norm_sum_ns'] <= 1e-5:
            continue
        pair = {event['actor0'], event['actor1']}
        if all(path.startswith('/World/Robot/') for path in pair):
            self_contacts.append(event['frame'])
        if TARGET in pair and any(path.startswith('/World/Robot/') and path not in pad_paths for path in pair):
            unexpected_target_links.append(event['frame'])
    overlap=np.array([_segment_intersects_aabb(np.array(HANG_BRANCH_JUNCTION_XYZ_M),np.array(HANG_BRANCH_TIP_CENTER_XYZ_M),a-BRANCH_RADIUS_M,b+BRANCH_RADIUS_M) for a,b in zip(lo,hi)])
    speed=np.linalg.norm(np.diff(centers,axis=0),axis=1)*fps;span=float(np.linalg.norm(centers[tail]-centers[-1],axis=1).max());separation=float(np.linalg.norm(pads[tail]-centers[tail,None,:],axis=2).min());brim_count=len(brim_frames[0]&brim_frames[1]);both=bilateral[:window]
    gates=dict(native_dynamic_cap=bool(metrics['target_dynamic']),canonical_visible_collision_rack=bool(metrics['rack_visible']) and int(metrics['rack_collision_count'])>0 and np.allclose(metrics['rack_min_xyz_m'],expected_furniture_min,atol=.002,rtol=0) and np.allclose(metrics['rack_max_xyz_m'],expected_furniture_max,atol=.002,rtol=0),physical_bilateral_visible_brim_grasp=brim_count>=3 and int(both.sum())>=max(3,window//2) and float(np.mean(opposition[:window][both]<=-.5))>=.8,physical_open_release=bool(np.max(grip[:release])>=.8 and np.max(grip[tail])<=.1 and np.min(np.abs(actual_grip[tail]))>=.9*.7843981633974483),released_from_both_pads=not bool(np.any(contacts[tail]>0)),branch_intersects_visible_cap=bool(overlap[tail].all()),physical_branch_support_after_release=sum(f>=release for f in rack_frames)>=2,physical_branch_support_in_tail=any(f>=n-20 for f in rack_frames),hangs_above_floor=bool(lo[tail,2].min()>=.65),hand_separated_after_release=separation>=.10,stable_hang_tail_speed=float(speed[-20:].max())<=.12,stable_hang_tail_span=span<=.04,no_single_frame_teleport=float(np.linalg.norm(np.diff(centers,axis=0),axis=1).max())<=.10,no_robot_furniture_or_distractor_contact=not unwanted,rigid_contact_penetration_bounded=max_pad_penetration<=.003 and max_rack_penetration<=.003)
    gates.update(no_robot_self_contact=not self_contacts, target_contact_only_active_gripper_pads=not unexpected_target_links)
    gates={k:bool(v) for k,v in gates.items()};failed=sorted(k for k,v in gates.items() if not v)
    margins=dict(frames=n,fps=fps,release_frame=release,initial_bilateral_brim_contact_frames=brim_count,tail_min_visible_z_m=float(lo[tail,2].min()),tail_max_center_speed_mps=float(speed[-20:].max()),tail_span_m=span,tail_min_pad_separation_m=separation,max_robot_cap_penetration_m=max_pad_penetration,max_rack_cap_penetration_m=max_rack_penetration,unwanted_robot_contacts=unwanted[:10],posture_basis='finite measured posture; historical robot calibration is provenance, not a Sheet goal gate')
    return dict(schema='real2sim.task6_code_control_score.v1',ok=not failed,benchmark_subtask='6.7',runtime_task_id='benchmark_tidy_sofa_6_7',failed_gates=failed,identity_errors=[],gates=gates,strict_place_reasons=failed,strict_place_margins=margins,metrics=margins)
