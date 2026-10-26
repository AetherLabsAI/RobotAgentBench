"""Task-local sofa/table empty-grasp CPU proposals from fixed tool configuration.

No nearest-trajectory frame selection and no target displacement to fit IK.
Calibration is an explicit task-owned input, never a request-supplied module.
"""
import hashlib
import json
import os
from pathlib import Path
import random
from copy import deepcopy
from ..common import HERE, owned, digest, physics, finish, write_json
from .catalog import definition
from .tipped_initial import load_base


def draw(variant, seed):
    d = definition(variant)
    if variant not in ('sofa','table_waste_empty_grasp'):
        raise ValueError('Only the two newly approved empty-grasp cases')
    if type(seed) is not int or not 1 <= seed < 2**32:
        raise ValueError('Positive uint32 seed required')
    rng = random.Random(int.from_bytes(hashlib.sha256(
        f'failure-recovery-v2:{variant}:{seed}:empty-grasp'.encode()).digest(),'big'))
    return dict(gap_m=rng.uniform(*d['gap_range_m']), miss_sign=rng.choice((-1,1)),
                orientation_jitter=False, axis='robot_local_y', seed=seed)


def fingertip_goal(target, tips, rotation, base_rotation, request):
    import numpy as np
    target,tips,rotation,base_rotation = map(np.asarray,(target,tips,rotation,base_rotation))
    if target.ndim != 2 or target.shape[1] != 3 or tips.ndim != 2 or tips.shape[1] != 3:
        raise ValueError('Finite 3D collider-envelope point arrays required')
    if not len(target) or not len(tips) or any(not np.isfinite(x).all() for x in (target,tips,rotation,base_rotation)):
        raise ValueError('Finite nonempty geometry required')
    if any(r.shape != (3,3) or not np.allclose(r.T@r,np.eye(3),atol=1e-7) or
           not np.isclose(np.linalg.det(r),1.,atol=1e-7) for r in (rotation,base_rotation)):
        raise ValueError('Fixed rigid tool/base rotations required')
    if request['miss_sign'] not in (-1,1) or not .02 <= request['gap_m'] <= .05:
        raise ValueError('Only the approved signed 2–5 cm gap')
    axis = base_rotation[:,1]*request['miss_sign']
    offsets = tips@rotation.T
    translation = (target.min(0)+target.max(0))/2-offsets.mean(0)
    translation += axis*((target@axis).max()+request['gap_m']-
                         ((offsets+translation)@axis).min())
    goal = np.eye(4); goal[:3,:3]=rotation; goal[:3,3]=translation
    return goal, axis


def support_clear_goal(goal, target, tips, hand_points, base_rotation, *, maximum_gap):
    """Choose Z inside the EXISTING 3D gap budget, not a new height range.

Target and fixed tool rotation never change. The lowest closed-hand envelope
must clear the target's original support plane. A midpoint is a numerical pose
proposal, not a new randomized parameter. Native contact remains unverified.
"""
    import numpy as np
    goal=goal.copy();rotated=tips@goal[:3,:3].T
    basis=base_rotation
    target_local=target@basis
    tip_local=(rotated+goal[:3,3])@basis
    planar=np.maximum(np.maximum(target_local.min(0)[:2]-tip_local.max(0)[:2],
        tip_local.min(0)[:2]-target_local.max(0)[:2]),0.)
    budget=maximum_gap**2-float(planar@planar)
    if budget<0:return None,dict(reason='planar_gap_exceeds_confirmed_3d_budget')
    max_vertical=float(np.sqrt(budget))
    lower=max(float(goal[2,3]),float(target[:,2].min()-(hand_points@goal[:3,:3].T)[:,2].min()))
    upper=float(target[:,2].max()+max_vertical-rotated[:,2].min())
    if not lower<upper:
        return None,dict(reason='fixed_direction_cannot_clear_support_within_confirmed_gap',
            tool_z_domain_m=[lower,upper])
    goal[2,3]=lower+(upper-lower)/2
    return goal,dict(tool_z_domain_m=[lower,upper],chosen_tool_z_m=float(goal[2,3]),
        policy='minimum_support_plane_to_existing_3d_gap_budget_midpoint',
        support_plane_z_m=float(target[:,2].min()),extra_clearance_parameter=None,
        fixed_orientation_preserved=True)


def envelope_distance(a,b,basis):
    import numpy as np
    a,b=a@basis,b@basis
    delta=np.maximum(np.maximum(a.min(0)-b.max(0),b.min(0)-a.max(0)),0.)
    return float(np.linalg.norm(delta))


def build(base_key, output, *, variant, configuration):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from pxr import Usd, UsdGeom
    from ..poses import world, place
    from ..procedural_grasp import Kinematics, complete_closed_reset, collider_points
    from short_tasks.tasks.robot_pose_alignment_v1.geometry import checker
    d = definition(variant)
    if variant not in ('sofa','table_waste_empty_grasp'):
        raise ValueError('Approved empty-grasp variant required')
    task = d['original_task']
    base, launch, manifest, bound, ready = load_base(base_key, task)
    config_path = owned(configuration)
    if not config_path.is_relative_to(HERE/'expansion_v2/configurations'):
        raise ValueError('Fixed task-owned calibration required')
    cfg = json.loads(config_path.read_text())
    if cfg.get('variant') != variant or cfg.get('orientation_jitter') is not False or not cfg.get('provenance'):
        raise ValueError('Explicit provenance and fixed task grasp orientation required')
    if variant == 'sofa':
        objects = [o for o in manifest['objects'] if o.get('task_id') == 'sofa_to_coffee_table']
        if bound['asset'] != 'remote_tv' or len(objects)!=1 or objects[0]['id']!='remote_tv':
            raise ValueError('This first sofa recovery binds one TV remote')
        target = objects[0]['path']
    else:
        if len(bound['runtime_components']) != 1:
            raise ValueError('This first waste recovery binds one upright empty paper cup')
        params = bound['runtime_components'][0]['parameters']
        if params.get('category') != 'cup':raise ValueError('Cup binding required')
        target = params['target_path']
    out = owned(output,fresh=True); out.mkdir(parents=True)
    stage = Usd.Stage.CreateNew(str(out/'initial.usda'))
    stage.GetRootLayer().subLayerPaths=[os.path.relpath(manifest['scene'],out)]
    UsdGeom.SetStageMetersPerUnit(stage,1.); UsdGeom.SetStageUpAxis(stage,'Z')
    stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
    before = physics(stage)
    if variant == 'sofa':
        from short_tasks.scene import prepare_runtime_scene
        effective = prepare_runtime_scene(bound,stage,manifest)
        sampled_target = world(stage,target).copy()
        stage.GetSessionLayer().Clear()
        place(stage,target,sampled_target)
        manifest = effective
    from short_tasks.agent_probe_v1.native_owner import initialize_base_pose
    original_target = world(stage,target).copy()
    points=np.concatenate([r[3] for r in collider_points(stage,target)])
    from . import operating_station
    if d.get('robot_station_policy')!='post_failure_operating_station.v1':
        raise ValueError('New near-operating station definition has not been confirmed')
    bases,station_policy=operating_station.proposals(stage,manifest,task,launch)
    room_gate=operating_station.RoomGate(stage,manifest)
    write_json(out/'station_domain.json',station_policy)
    issued=dict(positions=ready['targets_rad'],hold_targets=ready['targets_rad'])
    hand_reference_rotation=Rotation.from_euler('z',launch['initial_base_xyyaw'][2]).as_matrix()
    reference_base=np.eye(4);reference_base[:3,:3]=hand_reference_rotation
    reference_base[:2,3]=launch['initial_base_xyyaw'][:2]
    reference_lateral=float((np.linalg.inv(reference_base)@np.r_[(points.min(0)+points.max(0))/2,1.])[1])
    request=draw(variant,manifest['seed'])
    attempts=[];accepted=False;diagnostic=None
    for proposal in bases:
        chosen_base=proposal['initial_base_xyyaw']
        base_receipt=initialize_base_pose(stage,chosen_base)
        robot=world(stage,'/World/Robot/base_link')
        target_local=np.linalg.inv(robot)@np.r_[(points.min(0)+points.max(0))/2,1.]
        # The source look-at station can put a target exactly on the midline;
        # there is no left/right target side there. Preserve attempted hand.
        side=target_local[1] if target_local[1]!=0. else reference_lateral
        if side==0.:raise ValueError('No target-side or attempted-hand identity')
        hand='l' if side>0 else 'r'
        if hand not in cfg['fixed_tool_rotation_base']:
            attempts.append(dict(candidate=proposal['candidate_index'],base_xyyaw=chosen_base,
                hand=hand,accepted=False,reason='missing_source_backed_fixed_hand_orientation'))
            write_json(out/'station_attempts.json',attempts)
            # This proposal is unsupported, not silently assigned the other
            # hand. Coverage is reported and never claimed as both-hand ready.
            continue
        init,closure=complete_closed_reset(stage,issued,hand)
        model=Kinematics(stage,hand,init['positions'])
        preflight=room_gate.check(model,init['positions'],omit_active_arm=True)
        if preflight:
            attempts.append(dict(candidate=proposal['candidate_index'],base_xyyaw=chosen_base,
                hand=hand,accepted=False,reason=preflight,phase='unchanged_non_active_links_preflight'))
            write_json(out/'station_attempts.json',attempts)
            continue
        # Base heading follows the EXISTING operating-workspace rule. The hand
        # does not rotate with that changed base: its task-world grasp direction
        # remains the original fixed direction in the unchanged furniture frame.
        rotation=hand_reference_rotation@np.asarray(cfg['fixed_tool_rotation_base'][hand])
        initial_frames=model.frames(init['positions']);inverse=np.linalg.inv(initial_frames[model.tool])
        tips=[]
        for branch in ('inner','outer'):
            for _,body,local,_ in collider_points(stage,f'/World/Robot/gripper_{hand}_{branch}_link4'):
                tips.extend((inverse@initial_frames[body]@local.T).T[:,:3])
        tips=np.asarray(tips)
        goal,axis=fingertip_goal(points,tips,rotation,model.base[:3,:3],request)
        hand_points=np.concatenate([(inverse@initial_frames[body]@local.T).T[:,:3]
            for _,body,local in room_gate.robot if body.startswith('/World/Robot/gripper_'+hand+'_')])
        goal,height_receipt=support_clear_goal(goal,points,tips,hand_points,model.base[:3,:3],
            maximum_gap=d['gap_range_m'][1])
        if goal is None:
            attempts.append(dict(candidate=proposal['candidate_index'],base_xyyaw=chosen_base,
                hand=hand,accepted=False,phase='fixed_direction_gap_geometry',**height_receipt))
            write_json(out/'station_attempts.json',attempts)
            continue
        def assess(coordinates,fit):
            actual=model.frames(coordinates)[model.tool]
            actual_tips=tips@actual[:3,:3].T+actual[:3,3]
            gap=float((actual_tips@axis).min()-(points@axis).max())
            distance=envelope_distance(points,actual_tips,model.base[:3,:3])
            rejection=(room_gate.check(model,coordinates)
                if fit['converged'] and .02<=gap<=.05 and .02<=distance<=.05
                else 'not_checked_ik_or_gap_rejected')
            if rejection is None:
                # Recheck a winner through the unchanged full source gate.
                rejection=checker(stage,manifest,chosen_base)({'targets_rad':coordinates})
            return gap,distance,rejection
        q,ik=model.solve(goal)
        measured,distance_3d,reason=assess(q,ik)
        numerical_attempts=[dict(ik,numerical_start='task_ready',reason=reason)]
        if ik['converged'] and reason is not None:
            from .ik_branches import solutions
            for branch_q,branch_ik in solutions(model,goal):
                gap,dist,rejection=assess(branch_q,branch_ik)
                numerical_attempts.append(dict(branch_ik,reason=rejection))
                if rejection is None:
                    q,ik,measured,distance_3d,reason=branch_q,branch_ik,gap,dist,rejection
                    break
        init['positions']=q
        for name in model.names:init['hold_targets'][name]=q[name]
        if ik['converged'] and .02<=measured<=.05 and .02<=distance_3d<=.05 and diagnostic is None:
            diagnostic=dict(robot_initialization=deepcopy(init),gripper_closure=closure,
                initial_base_xyyaw=list(chosen_base),robot_base_initialization=base_receipt,
                active_hand=hand,candidate=proposal['candidate_index'],
                procedural_grasp=dict(request=request,ik=ik,goal_tcp_world=goal.tolist(),
                    fingertip_envelope_signed_gap_m=measured,
                    conservative_3d_envelope_distance_m=distance_3d,height=height_receipt,
                    room_rejection=reason,exact_mesh_surface_distance_verified=False,accepted_cpu=False))
        accepted=bool(ik['converged'] and .02<=measured<=.05 and .02<=distance_3d<=.05 and reason is None)
        attempts.append(dict(candidate=proposal['candidate_index'],base_xyyaw=chosen_base,
            hand=hand,accepted=accepted,ik=ik,measured_gap_m=measured,
            conservative_3d_envelope_distance_m=distance_3d,height=height_receipt,
            numerical_attempts=numerical_attempts,reason=reason))
        write_json(out/'station_attempts.json',attempts)
        if accepted:break
    if not accepted:
        if physics(stage)!=before:raise ValueError('Physical parameters changed during station search')
        if diagnostic is not None:
            diagnostic['robot_base_initialization']=initialize_base_pose(stage,diagnostic['initial_base_xyyaw'])
            record=dict(variant=variant,seed=manifest['seed'],target=target,
                **diagnostic,original_launch=launch,original_bindings=bound,
                preparation_accepted=False,status='rejected_pose_visual_diagnostic_only',
                original_geometry_checker_passed=False,station_policy=station_policy,
                maximum_score=bound['score_policy']['maximum'],completion_policy=d['completion'],
                source_motion_read=False,source_checkpoint=None,object_relocated_for_ik=False,
                seed_retried=False,configuration_sha256=digest(config_path),
                code_sources={str(Path(__file__).resolve()):digest(__file__),str(config_path):digest(config_path),
                    str(Path(operating_station.__file__).resolve()):digest(operating_station.__file__)})
            diagnostic_record=finish(stage,out,record,before)
            diagnostic_record['layout_scope']='full_original_short_task_random_room'
            write_json(out/'initial.json',diagnostic_record)
        else:stage.GetRootLayer().Save()
        write_json(out/'rejection.json',dict(variant=variant,seed=manifest['seed'],
            accepted=False,phase='cpu_only',candidate_count=len(bases),
            target_unchanged=bool(np.array_equal(world(stage,target),original_target)),
            physics_unchanged=True,source_motion_read=False,seed_retried=False,
            hand_configuration_coverage=sorted(cfg['fixed_tool_rotation_base']),
            source_working_sector_preserved=True,native_test_run=False))
        raise ValueError('Operating-station domain exhausted; retained every proposal; no object/seed replacement')
    if not np.array_equal(world(stage,target),original_target):raise ValueError('Target moved during empty-grasp solve')
    record=dict(variant=variant,seed=manifest['seed'],target=target,active_hand=hand,
        target_local_y_m=float(target_local[1]),
        hand_selection_policy=d['active_hand_policy'],robot_initialization=init,
        robot_base_initialization=base_receipt,
        initial_base_xyyaw=chosen_base,station_proposals=attempts,
        station_selection='first_valid_existing_operating_sector_candidate',
        station_policy=station_policy,
        opening_station_policy_replaced_with_user_approved_operating_station=True,
        original_operating_sector_unchanged=True,
        hand_configuration_coverage=sorted(cfg['fixed_tool_rotation_base']),
        hand_world_orientation_fixed=True,
        original_geometry_checker_passed=True,
        original_launch=launch,original_bindings=bound,maximum_score=bound['score_policy']['maximum'],
        completion_policy=d['completion'],gripper_closure=closure,
        procedural_grasp=dict(request=request,ik=ik,goal_tcp_world=goal.tolist(),
            fingertip_envelope_signed_gap_m=measured,room_rejection=reason,
            conservative_3d_envelope_distance_m=distance_3d,height=height_receipt,
            numerical_attempts=numerical_attempts,
            exact_mesh_surface_distance_verified=False,accepted_cpu=accepted),
        source_motion_read=False,source_checkpoint=None,object_relocated_for_ik=False,
        seed_retried=False,configuration_sha256=digest(config_path),preparation_accepted=accepted,
        code_sources={str(Path(__file__).resolve()):digest(__file__),str(config_path):digest(config_path),
            str(Path(operating_station.__file__).resolve()):digest(operating_station.__file__)})
    result=finish(stage,out,record,before)
    result['layout_scope']='full_original_short_task_random_room'
    write_json(out/'initial.json',result)
    if not accepted:raise ValueError('Empty-grasp CPU proposal rejected; retained diagnostic; no resampling')
    return result


def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--base-key',required=True)
    p.add_argument('--variant',choices=('sofa','table_waste_empty_grasp'),required=True)
    p.add_argument('--configuration',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    value=build(a.base_key,a.output,variant=a.variant,configuration=a.configuration)
    print(json.dumps({k:value[k] for k in ('variant','seed','active_hand','preparation_accepted','scene_sha256')}))


if __name__=='__main__':main()
