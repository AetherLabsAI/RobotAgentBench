"""Complete fixed left-hand direction from verified native paired arm frames.

This is NOT an invented Euler-angle mirror or a claimed left-hand recording.
The right direction remains byte-for-byte unchanged. Native left/right tool
basis reflection is calibrated and checked at two distinct legal configurations.
Only fixed direction matrices and provenance are saved, never reset arm poses.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from ..common import HERE,owned,digest,write_json
from .tipped_initial import load_base


def complete(base_key, configuration, output):
    import numpy as np
    from pxr import Usd
    from ..procedural_grasp import Kinematics,complete_closed_reset
    from short_tasks.agent_probe_v1.native_owner import initialize_base_pose
    from arena_api.robot_profiles import GRIPPER_INTERNAL
    path=owned(configuration)
    if not path.is_relative_to(HERE/'expansion_v2/configurations'):raise ValueError('Task configuration required')
    cfg=json.loads(path.read_text())
    if cfg.get('variant')!='table_waste_empty_grasp' or set(cfg['fixed_tool_rotation_base'])!={'r'}:
        raise ValueError('Explicit existing right-only cup direction required')
    _,launch,manifest,_,ready=load_base(base_key,'clear_table_trash')
    stage=Usd.Stage.Open(manifest['scene']);stage.SetEditTarget(stage.GetSessionLayer())
    initialize_base_pose(stage,launch['initial_base_xyyaw'])
    q,_=complete_closed_reset(stage,dict(positions=ready['targets_rad'],hold_targets=ready['targets_rad']),'r')
    q=dict(q['positions'])
    for name in GRIPPER_INTERNAL:q[name]=0.
    rnames=[f'idx6{i}_arm_r_joint{i}' for i in range(1,8)]
    lnames=[f'idx2{i}_arm_l_joint{i}' for i in range(1,8)]
    right_model=Kinematics(stage,'r',q);left_model=Kinematics(stage,'l',q)
    def paired(right_values,signs):
        coordinates=dict(q)
        for r,l,v,sign in zip(rnames,lnames,right_values,signs):
            if not right_model.bounds[r][0]<=v<=right_model.bounds[r][1]:raise ValueError('Invalid right numerical frame sample')
            if not left_model.bounds[l][0]<=sign*v<=left_model.bounds[l][1]:raise ValueError('Native arm limits do not admit this exact reflected sample')
            coordinates[r]=float(v);coordinates[l]=float(sign*v)
        frames=right_model.frames(coordinates)
        rb=right_model.base[:3,:3].T@frames['/World/Robot/gripper_r_center_link'][:3,:3]
        lb=left_model.base[:3,:3].T@frames['/World/Robot/gripper_l_center_link'][:3,:3]
        rp=right_model.base[:3,:3].T@(frames['/World/Robot/gripper_r_center_link'][:3,3]-right_model.base[:3,3])
        lp=left_model.base[:3,:3].T@(frames['/World/Robot/gripper_l_center_link'][:3,3]-left_model.base[:3,3])
        return rb,lb,rp,lp
    S=np.diag([1.,-1.,1.])
    # Infer coordinate SIGN CONVENTIONS from the native paired chains, rather
    # than assuming all seven angles negate. Every tested sample stays legal.
    import itertools
    found=None;patterns_checked=0
    for signs in itertools.product((-1,1),repeat=7):
        intervals=[]
        for r,l,sign in zip(rnames,lnames,signs):
            a,b=right_model.bounds[r];c,d=left_model.bounds[l]
            if sign<0:c,d=-d,-c
            intervals.append((max(a,c),min(b,d)))
        if any(a>=b for a,b in intervals):continue
        patterns_checked+=1
        mid=[(a+b)/2 for a,b in intervals]
        r1,l1,rp,lp=paired(mid,signs);C=r1.T@S@l1
        if not np.allclose(S@rp,lp,atol=1e-7,rtol=0):continue
        samples=[mid]
        for j,(lo,hi) in enumerate(intervals):
            for fraction in (.25,.75):
                values=list(mid);values[j]=lo+(hi-lo)*fraction;samples.append(values)
        error=0.;good=True
        for values in samples[1:]:
            rr,ll,rp,lp=paired(values,signs)
            error=max(error,float(np.max(np.abs(S@rr@C-ll))),float(np.max(np.abs(S@rp-lp))))
            if error>1e-7:good=False;break
        if good:
            found=(C,signs,error,len(samples));break
    if found is None:
        raise ValueError('Paired native tool bases do not have a consistent reflection; no fallback')
    C,signs,error,sample_count=found
    if not np.allclose(C.T@C,np.eye(3),atol=1e-7,rtol=0) or not np.isclose(np.linalg.det(C),-1.,atol=1e-7):
        raise ValueError('Expected native improper tool-basis reflection')
    right=np.asarray(cfg['fixed_tool_rotation_base']['r'])
    left=S@right@C
    if not np.allclose(left.T@left,np.eye(3),atol=1e-7,rtol=0) or not np.isclose(np.linalg.det(left),1.,atol=1e-7):
        raise ValueError('Mirrored fixed hand direction must remain a proper rotation')
    result=deepcopy(cfg)
    result['fixed_tool_rotation_base']['l']=left.tolist()
    result['hand_coverage']=['l','r']
    result['provenance'].append(dict(hand='l',source='native_paired_arm_tool_coordinate_geometry',
        recorded_left_grasp_claimed=False,source_configuration_sha256=digest(path),
        native_geometry_scene_sha256=digest(manifest['scene']),
        base_reflection=S.tolist(),tool_coordinate_reflection=C.tolist(),
        independent_legal_frame_samples=sample_count,native_joint_signs=list(signs),
        sign_patterns_checked=patterns_checked,independent_frame_error=error,
        right_hand_direction_unchanged=True,arm_reset_written=False))
    result['pair_calibration_code_sha256']=digest(__file__)
    result['native_grasp_orientation_accepted']=False
    out=owned(output,fresh=True)
    if not out.is_relative_to(HERE/'expansion_v2/configurations'):raise ValueError('Task-owned configuration output required')
    write_json(out,result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--base-key',required=True)
    p.add_argument('--configuration',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();v=complete(a.base_key,a.configuration,a.output)
    print(json.dumps(dict(hand_coverage=v['hand_coverage'],provenance=v['provenance'][-1])))


if __name__=='__main__':main()
