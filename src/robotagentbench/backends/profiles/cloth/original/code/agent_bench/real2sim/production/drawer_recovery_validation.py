"""Measured basket support and arm recovery gates for the four drawer tasks."""
from __future__ import annotations
import json
import numpy as np
from real2sim.teleop.contact_step_aggregation import aggregate_target_impulses
from real2sim.teleop.task6_drawer_sequence_contract import drawer_home_arm_rad, arm_recovery_phase, DRAWER_RECOVERY_OUTPUT_FRAMES, GRIP_OPEN_START_FRAME


def evaluate_drawer_recovery(state, task):
    if task not in {'4.4', '4.10', '6.5', '6.15'}:
        raise ValueError('unsupported drawer task')
    pick=task.startswith('4.')
    frames=len(state['frame_index'])
    required=1900 if pick else DRAWER_RECOVERY_OUTPUT_FRAMES
    if frames < required:
        raise ValueError('drawer recovery requires the complete action and settle window')
    archive=json.loads(str(np.asarray(state['drawer_contacts_json']).item()))
    durations=np.asarray(archive['frame_durations_s'],dtype=float)
    if durations.shape != (frames,) or not np.isfinite(durations).all() or np.any(durations<=0):
        raise ValueError('invalid all-substep contact clock')
    paths=np.asarray(state['rigid_body_paths']).astype(str).tolist()
    target=str(np.asarray(state['target_prim']).item())
    basket='/World/Assets/CoffeeTable/black_box_1'
    xyz=np.asarray(state['prop_xyz_m'],dtype=float)[:,paths.index(basket)]
    if xyz.shape!=(frames,3) or not np.isfinite(xyz).all():raise ValueError('invalid basket state')
    def bilateral(body, hand):
        result=aggregate_target_impulses(archive['contacts'],target_path=body,
            pad_paths=[f'/World/Robot/gripper_{hand}_{side}_link4' for side in ('outer','inner')],frame_durations_s=durations)
        vectors=result['mean_force_vectors_n'];force=np.linalg.norm(vectors,axis=-1)
        cosine=np.sum(vectors[:,0]*vectors[:,1],axis=-1)/np.maximum(force[:,0]*force[:,1],1e-12)
        return (force.min(axis=-1)>=.5)&(cosine<-.25)
    right=bilateral(basket,'r');start,end=(134,1510) if pick else (34,GRIP_OPEN_START_FRAME)
    losses=np.flatnonzero(~right[start:end])+start
    names=np.asarray(state['robot_dof_names']).astype(str).tolist()
    arm_names=[f'idx{prefix}{j}_arm_{hand}_joint{j}' for hand,prefix in [('l',2),('r',6)] for j in range(1,8)]
    q=np.asarray(state['robot_joint_position_rad_or_m'],dtype=float)[:,[names.index(n) for n in arm_names]]
    if q.shape!=(frames,14) or not np.isfinite(q).all():raise ValueError('invalid actual arm positions')
    home_error=float(np.max(np.abs(q[-20:]-drawer_home_arm_rad(pick=pick))))
    starts={h:next(t for t in range(frames) if arm_recovery_phase(t,pick=pick,hand=h,task='benchmark_'+task.replace('.','_')) is not None) for h in ('l','r')}
    hits=[]
    def hand_of(path):
        return 'l' if any(v in path for v in ('arm_l','gripper_l')) else 'r' if any(v in path for v in ('arm_r','gripper_r')) else None
    for e in archive['contacts']:
        if float(e['impulse_norm_sum_ns'])<=1e-4:continue
        f=int(e['frame']);a,b=e['actor0'],e['actor1']
        for robot,other in ((a,b),(b,a)):
            if not robot.startswith('/World/Robot/'):continue
            hand=hand_of(robot)
            if hand and f>=starts[hand] and other not in (basket,target):
                hits.append(e);break
            if pick and other==target and f>=starts['l'] and not robot.startswith('/World/Robot/gripper_l_'):
                hits.append(e);break
    delta=xyz[-1]-xyz[0]
    left_retained=bool(np.all(bilateral(target,'l')[-20:])) if pick else True
    gates={
        'right_bilateral_support_through_manipulation':len(losses)==0,
        'basket_physically_pulled':float(np.max(xyz[:,0]-xyz[0,0]))>=.12,
        'basket_reinserted_before_arm_return':float(np.linalg.norm(delta))<=.035,
        'basket_supported_at_original_height':abs(float(delta[2]))<=.005,
        'basket_stable_after_return':float(np.max(np.linalg.norm(xyz[-20:]-xyz[-1],axis=-1)))<=.002,
        'both_actual_arms_at_task_standby':home_error<=.002,
        'collision_free_arm_recovery':not hits,
        'picked_object_retained_at_standby':left_retained,
    }
    return dict(schema='real2sim.drawer_recovery_validation.v1',success=all(gates.values()),gates=gates,
        right_support_loss_frames=losses.tolist(),arm_home_error_rad=home_error,
        basket_final_delta_m=delta.tolist(),recovery_contact_count=len(hits),recovery_contact_examples=hits[:10])
