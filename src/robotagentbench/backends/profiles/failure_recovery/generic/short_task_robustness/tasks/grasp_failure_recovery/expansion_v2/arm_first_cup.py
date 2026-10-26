"""Solve an empty closed hand first; only then place the same upright cup.

Human-approved order for table waste missed grasp only. Cup XY must remain in
the FULL original living-plan placement region. No source trajectories, relaxed
joint limits, collision flags, camera changes or inherited task progress.
"""
import argparse
import json
import os
from pathlib import Path
from copy import deepcopy
from ..common import HERE, CURRENT, owned, digest, physics, finish, write_json
from .tipped_initial import load_base
from .empty_grasp import draw, envelope_distance


def cup_after_hand(cup, tips, base_rotation, request):
    """Pure XY translation, never a rotation/height/scale change or arm edit."""
    import numpy as np
    cup,tips,basis=map(lambda p:np.asarray(p,dtype=float),(cup,tips,base_rotation))
    if any(p.ndim!=2 or p.shape[1]!=3 or not len(p) or not np.isfinite(p).all() for p in (cup,tips)):
        raise ValueError('Finite original cup and solved fingertip envelopes required')
    if basis.shape!=(3,3) or not np.allclose(basis.T@basis,np.eye(3),atol=1e-7) or not np.isclose(np.linalg.det(basis),1.):
        raise ValueError('Rigid actual robot base orientation required')
    if request['miss_sign'] not in (-1,1) or not .02<=request['gap_m']<=.05:
        raise ValueError('Only the approved signed 2–5 cm gap')
    axis=basis[:,1]*request['miss_sign']
    if not np.isclose(axis[2],0.,atol=1e-7):raise ValueError('Upright robot lateral axis required')
    shift=(tips.min(0)+tips.max(0)-cup.min(0)-cup.max(0))/2
    shift[2]=0.
    shift+=axis*(float((tips@axis).min())-request['gap_m']-float(((cup+shift)@axis).max()))
    return shift,axis


def inside_region(points, delta, region):
    """Original living_plan u/v rectangle; test the whole footprint, not pivot."""
    import numpy as np
    p=np.asarray(points,dtype=float)
    if p.ndim!=2 or p.shape[1]!=3 or not len(p) or not np.isfinite(p).all():raise ValueError('Finite footprint required')
    canonical=np.c_[p,np.ones(len(p))]@np.linalg.inv(delta)
    uv=canonical[:,:2]@np.column_stack((region['u'],region['v']))
    return bool(np.all(uv>=[region['start'],region['left']]) and
                np.all(uv<=[region['end'],region['right']]))


def build(base_key, output, configuration):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from pxr import Usd,UsdGeom
    from ..poses import world,place
    from ..procedural_grasp import Kinematics,complete_closed_reset,collider_points
    from short_tasks.agent_probe_v1.native_owner import initialize_base_pose
    from short_tasks.tasks.clear_table_trash.initial_station import table_context
    from short_tasks.tasks.robot_pose_alignment_v1.geometry import checker
    from . import operating_station,ik_branches
    from .catalog import definition
    variant='table_waste_empty_grasp';d=definition(variant)
    _,launch,manifest,bound,ready=load_base(base_key,'clear_table_trash')
    parts=bound['runtime_components']
    if len(parts)!=1 or parts[0]['parameters']['category']!='cup':raise ValueError('Exactly one used cup required')
    target=parts[0]['parameters']['target_path']
    row=next(o for o in manifest['objects'] if o['path']==target)
    if row['id']!='paper_cup':raise ValueError('The same original paper cup is required')
    if any(o['path']!=target and o.get('source')=='coffee_table.objects' for o in manifest['objects']):
        raise ValueError('This first one-cup sample requires no other portable table-region item; do not skip original pairClear protection')
    config_path=owned(configuration);cfg=json.loads(config_path.read_text())
    if not config_path.is_relative_to(HERE/'expansion_v2/configurations') or cfg.get('variant')!=variant or cfg.get('orientation_jitter') is not False:
        raise ValueError('Task-owned fixed cup hand directions required')
    if set(cfg['fixed_tool_rotation_base'])!={'l','r'}:raise ValueError('Separately derived left and right directions required')
    out=owned(output,fresh=True);out.mkdir(parents=True)
    stage=Usd.Stage.CreateNew(str(out/'initial.usda'))
    stage.GetRootLayer().subLayerPaths=[os.path.relpath(manifest['scene'],out)]
    UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,'Z')
    stage.SetDefaultPrim(stage.GetPrimAtPath('/World'));before=physics(stage)
    original=world(stage,target).copy()
    composition=Path(manifest['short_task_composition']['cache'])/'placement.json'
    plans=json.loads(composition.read_text())['living_plan']['coffee_table']
    original_cup_plan=next(p for p in plans if p['instance_id']==row['instance_id'])
    if not np.isclose(original_cup_plan['rollRad'],np.pi/2,atol=1e-7):
        raise ValueError('The source cup must already be upright; no hidden roll/yaw replacement')
    cup=np.concatenate([r[3] for r in collider_points(stage,target)])
    bbox=UsdGeom.BBoxCache(0,['default','render'],False,True)
    vr=bbox.ComputeWorldBound(stage.GetPrimAtPath(target)).ComputeAlignedRange()
    from short_tasks.tasks.retrieve_from_drawer.prepare_initial import corners
    visual=corners(np.array([vr.GetMin(),vr.GetMax()]))
    if not np.isclose(np.ptp(visual[:,2]),original_cup_plan['size'][2],atol=1e-7):
        raise ValueError('Measured source cup height disagrees with its upright placement receipt')
    delta,table,baseline=table_context(stage,manifest)
    source=CURRENT/'openend/tasks/scene_requirements/placement_sources/room_preview_plan_seed42.json'
    region=json.loads(source.read_text())['coffee_table']
    # The same original source rectangle, rigidly transferred with the table.
    uv=np.array([(region['start']+region['end'])/2,(region['left']+region['right'])/2])
    centre_xy=np.column_stack((region['u'],region['v']))@uv
    anchor=(np.r_[centre_xy,0.,1.]@delta)[:3]
    issued=dict(positions=ready['targets_rad'],hold_targets=ready['targets_rad'])
    proposals,station_policy=operating_station.proposals(stage,manifest,'clear_table_trash',launch)
    gate=operating_station.RoomGate(stage,manifest)
    # Only the stale pre-placement target is omitted from computational search.
    # Its collision API is NEVER altered. The final original gate includes it.
    gate.obstacles=[o for o in gate.obstacles if not(o[0]==target or o[0].startswith(target+'/'))]
    hand_reference=Rotation.from_euler('z',launch['initial_base_xyyaw'][2]).as_matrix()
    request=draw(variant,manifest['seed']);attempts=[];winner=None
    for station in proposals:
        base_xyyaw=station['initial_base_xyyaw'];base_receipt=initialize_base_pose(stage,base_xyyaw)
        for hand in ('l','r'):
            initial,closure=complete_closed_reset(stage,issued,hand)
            model=Kinematics(stage,hand,initial['positions'])
            rejection=gate.check(model,initial['positions'],omit_active_arm=True)
            if rejection:
                attempts.append(dict(candidate=station['candidate_index'],hand=hand,reason=rejection,phase='non_active_room_gate'))
                continue
            frames=model.frames(initial['positions']);inverse=np.linalg.inv(frames[model.tool])
            tips=np.concatenate([(inverse@frames[body]@local.T).T[:,:3]
                for branch in ('inner','outer') for _,body,local,_ in
                collider_points(stage,f'/World/Robot/gripper_{hand}_{branch}_link4')])
            hand_points=np.concatenate([(inverse@frames[body]@local.T).T[:,:3]
                for _,body,local in gate.robot if body.startswith('/World/Robot/gripper_'+hand+'_')])
            rotation=hand_reference@np.asarray(cfg['fixed_tool_rotation_base'][hand])
            rotated=tips@rotation.T;hand_rotated=hand_points@rotation.T
            # TCP request derives from workspace centre, asset height and support,
            # NOT the original cup's XY or an extracted failed trajectory frame.
            goal=np.eye(4);goal[:3,:3]=rotation
            goal[:2,3]=anchor[:2]-rotated.mean(0)[:2]
            goal[2,3]=max(float((cup[:,2].min()+cup[:,2].max())/2-rotated[:,2].mean()),
                float(cup[:,2].min()-hand_rotated[:,2].min()))
            q,ik=model.solve(goal)
            from itertools import chain
            solutions=chain([(q,ik)],ik_branches.solutions(model,goal) if ik['converged'] else ())
            for q,fit in solutions:
                reason=None;placement=None
                if not fit['converged']:reason='arm_ik_not_converged'
                else:reason=gate.check(model,q)
                if reason is None:
                    # Arm is now frozen: no more IK or joint change is allowed.
                    solved=model.frames(q)[model.tool]
                    actual_tips=tips@solved[:3,:3].T+solved[:3,3]
                    shift,axis=cup_after_hand(cup,actual_tips,model.base[:3,:3],request)
                    moved=cup+shift
                    local=np.linalg.inv(model.base)@np.r_[(moved.min(0)+moved.max(0))/2,1.]
                    side_hand='l' if local[1]>0 else 'r' if local[1]<0 else None
                    measured=float((actual_tips@axis).min()-(moved@axis).max())
                    distance=envelope_distance(moved,actual_tips,model.base[:3,:3])
                    if side_hand!=hand:reason='placed_cup_not_on_active_hand_side'
                    elif not inside_region(np.concatenate((moved,visual+shift)),delta,region):reason='cup_outside_original_placement_region'
                    elif not(.02<=measured<=.05 and .02<=distance<=.05):reason='cup_fingertip_gap_rejected'
                    else:
                        box=np.array([moved.min(0),moved.max(0)])
                        reason=next(('cup_room_overlap:'+p for p,b,t in gate.obstacles if gate.intersects(box,b) and
                            (t is None or gate.triangles_hit_box(t,box))),None)
                    placement=dict(translation_world_m=shift.tolist(),signed_gap_m=measured,
                        conservative_3d_envelope_distance_m=distance,active_hand_by_final_cup_side=side_hand)
                    if reason is None:
                        proposed=original.copy();proposed[:3,3]+=shift;place(stage,target,proposed)
                        reason=checker(stage,manifest,base_xyyaw)({'targets_rad':q})
                        if reason is None:
                            final=deepcopy(initial);final['positions']=q
                            for name in model.names:final['hold_targets'][name]=q[name]
                            winner=dict(robot_initialization=final,gripper_closure=closure,
                                initial_base_xyyaw=base_xyyaw,robot_base_initialization=base_receipt,
                                active_hand=hand,target_local_y_m=float(local[1]),
                                original_target_world=original.tolist(),proposed_target_world=proposed.tolist(),
                                procedural_grasp=dict(request=request,ik=fit,goal_tcp_world=goal.tolist(),
                                    room_rejection=None,accepted_cpu=True,**placement))
                        else:place(stage,target,original)
                attempts.append(dict(candidate=station['candidate_index'],hand=hand,ik=fit,reason=reason,
                    placement=placement,arm_solved_before_cup_placement=True))
                write_json(out/'station_attempts.json',attempts)
                if winner is not None:break
            if winner is not None:break
        if winner is not None:break
    if winner is None:
        stage.GetRootLayer().Save()
        write_json(out/'rejection.json',dict(accepted=False,attempts=len(attempts),source_motion_read=False,
            target_restored=True,physics_unchanged=physics(stage)==before,native_test_run=False))
        raise ValueError('Arm-first domain exhausted; no range/physics relaxation')
    code_files=[Path(__file__),config_path,Path(operating_station.__file__),Path(ik_branches.__file__),source,baseline,composition]
    record=dict(variant=variant,seed=manifest['seed'],target=target,**winner,
        original_launch=launch,original_bindings=bound,maximum_score=bound['score_policy']['maximum'],
        completion_policy=d['completion'],preparation_accepted=True,original_geometry_checker_passed=True,
        station_policy=station_policy,station_proposals=attempts,hand_world_orientation_fixed=True,
        hand_configuration_coverage=['l','r'],source_motion_read=False,source_checkpoint=None,
        initialization_order=['solve_empty_closed_arm','freeze_arm','place_same_upright_cup','full_geometry_gate'],
        object_relocated_for_ik=True,cup_translation_only=True,cup_original_height_preserved=True,
        arm_changed_after_cup_placement=False,original_cup_placement_region_preserved=True,
        seed_retried=False,configuration_sha256=digest(config_path),
        code_sources={str(p):digest(p) for p in code_files})
    result=finish(stage,out,record,before);result['layout_scope']='full_original_short_task_random_room'
    write_json(out/'initial.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--base-key',required=True)
    p.add_argument('--configuration',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();v=build(a.base_key,a.output,a.configuration)
    print(json.dumps({k:v[k] for k in ('variant','active_hand','preparation_accepted','initialization_order')}))


if __name__=='__main__':main()
