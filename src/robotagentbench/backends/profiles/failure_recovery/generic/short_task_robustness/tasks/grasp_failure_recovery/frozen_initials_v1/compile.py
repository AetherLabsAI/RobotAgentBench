"""One-time offline export; the ONLY frozen-bank module allowed to solve IK."""
import json
import math
from copy import deepcopy
from pathlib import Path
from ..common import HERE,owned,digest,write_json,verify_sources,physics,finish
from .bank import SCHEMA,CASES,checksum,validate
from .cup import context,base_matrix,place_and_check

INPUTS={
    'shelf_drink':('procedural_v1/shelf_adopt303_cpu/initial/initial.json',
        'procedural_v1/shelf_adopt303_native_r2/native/native_report.json'),
    'coffee_bag':('procedural_v1/bag_adopt301_cpu/initial/initial.json',
        'procedural_v1/bag_adopt301_native_r2/native/native_report.json'),
    'sofa':('expansion_v2/candidates/sofa_empty401_operating_v8/initial.json',None),
    'table_waste_empty_grasp':('expansion_v2/candidates/cup_armfirst401_visible_v4/initial.json',None),
}


def extract(variant,source,report):
    p=HERE/source;v=json.loads(p.read_text())
    if v['variant']!=variant or v.get('preparation_accepted') is not True:raise ValueError('Only accepted CPU candidates may be frozen')
    verify_sources(v['source_layers']);verify_sources(v['code_sources'])
    validation=dict(cpu=True,native=False,random_revalidation_required=True)
    if report:
        r=json.loads((HERE/report).read_text())
        if r.get('status')!='completed' or r.get('hold_only_verified') is not True or r.get('original_native_initial_accepted') is not True or r.get('cleanup_errors'):
            raise ValueError('Native-template report did not pass')
        validation.update(native=True,native_template_hold_steps=r['hold_control_steps'],
            native_initial_score=r['initial_score'],native_idle_score=r['idle_score'],
            report=report,report_sha256=digest(HERE/report))
    return dict(id=f'{variant}_{v["active_hand"]}_{v["seed"]}',variant=variant,
        active_hand=v['active_hand'],robot_initialization=deepcopy(v['robot_initialization']),
        initial_base_xyyaw=v.get('initial_base_xyyaw'),source_robot_base=v.get('source_robot_base'),
        cpu_accepted=True,validation=validation,source_record=source,source_record_sha256=digest(p),
        source_motion_needed=False,maximum_score=v['maximum_score'],completion_policy=v['completion_policy'])


def left_cup(base_key,right_record,work):
    """One offline native-limit solve at the already accepted operating station."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    from ..procedural_grasp import Kinematics,complete_closed_reset,collider_points
    from ..expansion_v2.ik_branches import solutions
    from ..expansion_v2.empty_grasp import draw
    from ..expansion_v2.operating_station import RoomGate
    from short_tasks.agent_probe_v1.native_owner import initialize_base_pose
    out=owned(work,fresh=True);out.mkdir(parents=True);c=context(base_key,out);stage=c['stage'];before=physics(stage)
    xyyaw=right_record['initial_base_xyyaw'];base_receipt=initialize_base_pose(stage,xyyaw)
    issued=dict(positions=c['ready']['targets_rad'],hold_targets=c['ready']['targets_rad'])
    initial,closure=complete_closed_reset(stage,issued,'l');model=Kinematics(stage,'l',initial['positions'])
    config_path=HERE/'expansion_v2/configurations/cup_directions_geometry_v5.json'
    cfg=json.loads(config_path.read_text())
    rotation=Rotation.from_euler('z',c['launch']['initial_base_xyyaw'][2]).as_matrix()@np.asarray(cfg['fixed_tool_rotation_base']['l'])
    frames=model.frames(initial['positions']);inv=np.linalg.inv(frames[model.tool])
    tips=np.concatenate([(inv@frames[body]@local.T).T[:,:3] for branch in ('inner','outer')
        for _,body,local,_ in collider_points(stage,f'/World/Robot/gripper_l_{branch}_link4')])
    gate=RoomGate(stage,c['manifest'])
    hand=np.concatenate([(inv@frames[body]@local.T).T[:,:3] for _,body,local in gate.robot
        if body.startswith('/World/Robot/gripper_l_')])
    r=c['region'];uv=np.array([(r['start']+r['end'])/2,(r['left']+r['right'])/2])
    xy=np.column_stack((r['u'],r['v']))@uv;anchor=(np.r_[xy,0.,1.]@c['delta'])[:3]
    rotated=tips@rotation.T;goal=np.eye(4);goal[:3,:3]=rotation
    goal[:2,3]=anchor[:2]-rotated.mean(0)[:2]
    goal[2,3]=max(float((c['cup'][:,2].min()+c['cup'][:,2].max())/2-rotated[:,2].mean()),
        float(c['cup'][:,2].min()-(hand@rotation.T)[:,2].min()))
    seed=next(s for s in range(401,501) if draw('table_waste_empty_grasp',s)['miss_sign']==-1)
    request=draw('table_waste_empty_grasp',seed)
    q,fit=model.solve(goal);attempts=[]
    from itertools import chain
    for q,fit in chain([(q,fit)],solutions(model,goal)):
        candidate=deepcopy(initial);candidate['positions']=q
        for n in model.names:candidate['hold_targets'][n]=q[n]
        receipt=(place_and_check(c,candidate,'l',xyyaw,request,gate=gate) if fit['converged']
            else dict(accepted_cpu=False,room_rejection='offline_ik_not_converged'))
        attempts.append(dict(ik=fit,check=receipt));write_json(out/'offline_attempts.json',attempts)
        if receipt['accepted_cpu']:
            record=dict(variant='table_waste_empty_grasp',seed=seed,target=c['target'],active_hand='l',
                robot_initialization=candidate,gripper_closure=closure,initial_base_xyyaw=xyyaw,
                robot_base_initialization=base_receipt,original_launch=c['launch'],original_bindings=c['bound'],
                maximum_score=c['bound']['score_policy']['maximum'],completion_policy=right_record['completion_policy'],
                preparation_accepted=True,original_geometry_checker_passed=True,procedural_grasp=receipt,
                source_motion_read=False,offline_ik_calls=len(attempts),online_ik_required=False,
                code_sources={str(Path(__file__)):digest(__file__),str(config_path):digest(config_path)})
            value=finish(stage,out,record,before);return out/'initial.json',value,c['delta']
    raise ValueError('Offline left profile not accepted; no mirrored/zero-pose fallback')


def main():
    import numpy as np
    from pxr import Usd
    from short_tasks.tasks.clear_table_trash.initial_station import table_context
    result=dict(schema=SCHEMA,cases={name:dict(profiles=[],online_ik_fallback=False) for name in CASES},
        source_tree='full_only',runtime_joint_interpolation=False,scope='frozen_initials_not_new_scoring')
    for name,(source,report) in INPUTS.items():
        profile=extract(name,source,report);result['cases'][name]['profiles'].append(profile)
    cup=result['cases']['table_waste_empty_grasp']['profiles'][0]
    right=json.loads((HERE/INPUTS['table_waste_empty_grasp'][0]).read_text())
    stage=Usd.Stage.Open(right['scene'])
    delta,_,_=table_context(stage,right['original_launch']['manifest'] if isinstance(right['original_launch'].get('manifest'),dict)
        else json.loads(Path(right['original_launch']['manifest']).read_text()))
    cup.update(base_canonical_column=(np.linalg.inv(delta.T)@base_matrix(right['initial_base_xyyaw'])).tolist(),
        supported_miss_signs=[1],asset_id='paper_cup',anchor='canonical_actual_coffee_table_geometry')
    try:
        p,left,left_delta=left_cup('tipped_cup_base401_r3',right,HERE/'frozen_initials_v1/offline/cup_left_v1')
        lp=extract('table_waste_empty_grasp',str(p.relative_to(HERE)),None)
        lp.update(base_canonical_column=(np.linalg.inv(left_delta.T)@base_matrix(left['initial_base_xyyaw'])).tolist(),
            supported_miss_signs=[-1],asset_id='paper_cup',anchor='canonical_actual_coffee_table_geometry')
        result['cases']['table_waste_empty_grasp']['profiles'].append(lp)
    except ValueError as exc:
        result['cases']['table_waste_empty_grasp']['left_coverage_error']=str(exc)
    result['cases']['table_waste_tipped_cup'].update(status='blocked_missing_native_supported_rest_and_failure_history',
        candidate='expansion_v2/candidates/tipped_side401_cpu_r3/initial.json')
    for name,case in result['cases'].items():
        case['hand_coverage']=sorted({p['active_hand'] for p in case['profiles']})
        case['fast_scene_generator_connected']=name=='table_waste_empty_grasp'
    result['content_sha256']=checksum(result);validate(result)
    out=owned(HERE/'frozen_initials_v1/configurations/bank_v1.json',fresh=True);write_json(out,result)
    print(json.dumps({name:dict(profiles=len(c['profiles']),hands=c['hand_coverage']) for name,c in result['cases'].items()}))


if __name__=='__main__':main()
