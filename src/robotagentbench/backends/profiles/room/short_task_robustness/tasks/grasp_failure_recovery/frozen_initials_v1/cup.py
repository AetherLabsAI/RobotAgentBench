"""Frozen native34 + analytic 2–5cm placement. No optimizer or IK search."""
import argparse
import json
import math
import os
import time
from copy import deepcopy
from pathlib import Path
from ..common import HERE,CURRENT,owned,digest,physics,finish,write_json
from .bank import load,select
from ..expansion_v2.empty_grasp import draw,envelope_distance
from ..expansion_v2.arm_first_visible_cup_v4 import cup_after_hand,inside_region


def context(base_key,out):
    import numpy as np
    from pxr import Usd,UsdGeom
    from ..poses import world
    from ..procedural_grasp import collider_points
    from ..expansion_v2.tipped_initial import load_base
    from short_tasks.tasks.clear_table_trash.initial_station import table_context
    from short_tasks.tasks.retrieve_from_drawer.prepare_initial import corners
    _,launch,manifest,bound,ready=load_base(base_key,'clear_table_trash')
    parts=bound['runtime_components']
    if len(parts)!=1 or parts[0]['parameters']['category']!='cup':raise ValueError('Exactly one original paper cup required')
    target=parts[0]['parameters']['target_path'];row=next(o for o in manifest['objects'] if o['path']==target)
    if row['id']!='paper_cup' or any(o['path']!=target and o.get('source')=='coffee_table.objects' for o in manifest['objects']):
        raise ValueError('First frozen cup contract is one paper cup, no omitted pairClear protection')
    stage=Usd.Stage.CreateNew(str(out/'initial.usda'));stage.GetRootLayer().subLayerPaths=[os.path.relpath(manifest['scene'],out)]
    UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,'Z');stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
    composition=Path(manifest['short_task_composition']['cache'])/'placement.json'
    plan=next(p for p in json.loads(composition.read_text())['living_plan']['coffee_table'] if p['instance_id']==row['instance_id'])
    if not np.isclose(plan['rollRad'],np.pi/2,atol=1e-7):raise ValueError('Source cup must be upright; do not redraw or silently rotate it')
    cup=np.concatenate([r[3] for r in collider_points(stage,target)])
    bbox=UsdGeom.BBoxCache(0,['default','render'],False,True).ComputeWorldBound(stage.GetPrimAtPath(target)).ComputeAlignedRange()
    visual=corners(np.array([bbox.GetMin(),bbox.GetMax()]))
    if not np.isclose(np.ptp(visual[:,2]),plan['size'][2],atol=1e-7):raise ValueError('Upright cup receipt/geometry mismatch')
    delta,table,baseline=table_context(stage,manifest)
    source=CURRENT/'openend/tasks/scene_requirements/placement_sources/room_preview_plan_seed42.json'
    return dict(stage=stage,launch=launch,manifest=manifest,bound=bound,ready=ready,target=target,
        original=world(stage,target).copy(),cup=cup,visual=visual,delta=delta,table=table,
        region=json.loads(source.read_text())['coffee_table'],references=[source,baseline,composition])


def base_matrix(xyyaw):
    import numpy as np
    from scipy.spatial.transform import Rotation
    b=np.eye(4);b[:3,:3]=Rotation.from_euler('z',xyyaw[2]).as_matrix();b[:2,3]=xyyaw[:2]
    return b


def hand_geometry(stage,model,q):
    import numpy as np
    from ..procedural_grasp import collider_points
    frames=model.frames(q);inverse=np.linalg.inv(frames[model.tool])
    tips=np.concatenate([(inverse@frames[body]@local.T).T[:,:3] for branch in ('inner','outer')
        for _,body,local,_ in collider_points(stage,f'/World/Robot/gripper_{model.hand}_{branch}_link4')])
    actual=frames[model.tool]
    return tips@actual[:3,:3].T+actual[:3,3],frames


def place_and_check(c,initial,hand,xyyaw,request,*,gate=None):
    import numpy as np
    from ..procedural_grasp import Kinematics
    from ..poses import place
    from ..expansion_v2.camera_projection_complete import project_target
    from ..expansion_v2.operating_station import RoomGate
    from short_tasks.tasks.robot_pose_alignment_v1.geometry import checker
    stage=c['stage'];model=Kinematics(stage,hand,initial['positions'])
    tips,frames=hand_geometry(stage,model,initial['positions'])
    shift,axis=cup_after_hand(c['cup'],tips,model.base[:3,:3],request)
    moved=c['cup']+shift
    local=np.linalg.inv(model.base)@np.r_[(moved.min(0)+moved.max(0))/2,1.]
    side='l' if local[1]>0 else 'r' if local[1]<0 else None
    gap=float((tips@axis).min()-(moved@axis).max());dist=envelope_distance(moved,tips,model.base[:3,:3])
    reason=None
    if side!=hand:reason='wrong_active_hand_side'
    elif not inside_region(np.concatenate((moved,c['visual']+shift)),c['delta'],c['region']):reason='cup_outside_original_region'
    elif not(.02<=gap<=.05 and .02<=dist<=.05):reason='gap_rejected'
    gate=gate or RoomGate(stage,c['manifest'])
    obstacles=[o for o in gate.obstacles if not(o[0]==c['target'] or o[0].startswith(c['target']+'/'))]
    if reason is None:
        box=np.array([moved.min(0),moved.max(0)])
        reason=next(('cup_room_overlap:'+p for p,b,t in obstacles if gate.intersects(box,b) and
            (t is None or gate.triangles_hit_box(t,box))),None)
    proposed=c['original'].copy();proposed[:3,3]+=shift;projection={}
    if reason is None:
        place(stage,c['target'],proposed)
        projection=project_target(stage,frames,c['target'])
        if not any(e['whole_target_envelope_inside_original_frustum'] for e in projection.values()):reason='cup_not_wholly_in_original_camera_frustum'
        else:reason=checker(stage,c['manifest'],xyyaw)({'targets_rad':initial['positions']})
    if reason is not None:place(stage,c['target'],c['original'])
    return dict(accepted_cpu=reason is None,room_rejection=reason,request=request,
        translation_world_m=shift.tolist(),fingertip_envelope_signed_gap_m=gap,
        conservative_3d_envelope_distance_m=dist,original_camera_projection=projection,
        target_local_y_m=float(local[1]),proposed_target_world=proposed.tolist(),
        exact_mesh_surface_distance_verified=False,ik_calls=0)


def build(base_key,output,*,seed=None):
    import numpy as np
    from short_tasks.agent_probe_v1.native_owner import initialize_base_pose
    from ..expansion_v2.catalog import definition
    from . import bank as contract
    started=time.monotonic();out=owned(output,fresh=True);out.mkdir(parents=True)
    c=context(base_key,out);stage=c['stage'];before=physics(stage)
    episode_seed=c['manifest']['seed'] if seed is None else seed
    bank_path=HERE/'frozen_initials_v1/configurations/bank_v1.json';bank=load(bank_path)
    request=draw('table_waste_empty_grasp',episode_seed)
    profile=select(bank,'table_waste_empty_grasp',episode_seed,miss_sign=request['miss_sign'])
    base=c['delta'].T@np.asarray(profile['base_canonical_column'])
    if not np.isclose(base[2,3],0.,atol=1e-7):raise ValueError('Frozen station does not match current floor height')
    xyyaw=[float(base[0,3]),float(base[1,3]),math.atan2(base[1,0],base[0,0])]
    base_receipt=initialize_base_pose(stage,xyyaw)
    initialization=deepcopy(profile['robot_initialization'])
    result=place_and_check(c,initialization,profile['active_hand'],xyyaw,request)
    if not result['accepted_cpu']:
        stage.GetRootLayer().Save();write_json(out/'rejection.json',dict(result,offline_bank_extension_required=True,
            online_ik_fallback=False,seed_retried=False))
        raise ValueError('Frozen pose rejected in this actual room; no optimizer/seed fallback: '+str(result['room_rejection']))
    d=definition('table_waste_empty_grasp')
    from .. import common,procedural_grasp,poses
    from ..expansion_v2 import empty_grasp,arm_first_visible_cup_v4,camera_projection_complete,operating_station
    from short_tasks.tasks.robot_pose_alignment_v1 import geometry
    sources=[Path(__file__),Path(contract.__file__),bank_path,*c['references']]
    sources += [Path(m.__file__) for m in (common,procedural_grasp,poses,empty_grasp,
        arm_first_visible_cup_v4,camera_projection_complete,operating_station,geometry)]
    record=dict(variant='table_waste_empty_grasp',seed=episode_seed,room_seed=c['manifest']['seed'],
        target=c['target'],active_hand=profile['active_hand'],robot_initialization=initialization,
        initial_base_xyyaw=xyyaw,robot_base_initialization=base_receipt,
        original_launch=c['launch'],original_bindings=c['bound'],original_target_world=c['original'].tolist(),
        proposed_target_world=result['proposed_target_world'],maximum_score=c['bound']['score_policy']['maximum'],
        completion_policy=d['completion'],preparation_accepted=True,original_geometry_checker_passed=True,
        initialization_policy='frozen_native34_then_continuous_analytic_cup_placement.v1',
        frozen_profile_id=profile['id'],frozen_template_validation=profile['validation'],
        new_episode_native_acceptance_inherited=False,procedural_grasp=result,
        source_motion_read=False,source_checkpoint=None,online_ik_calls=0,
        robot_joint_interpolation=False,independent_joint_jitter=False,
        arm_changed_after_cup_placement=False,cup_translation_only=True,
        original_cup_placement_region_preserved=True,seed_retried=False,
        code_sources={str(p):digest(p) for p in sources})
    value=finish(stage,out,record,before)
    value.update(layout_scope='full_original_room_with_fresh_continuous_cup_draw',
        generation_wall_seconds=time.monotonic()-started)
    write_json(out/'initial.json',value);return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--base-key',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int);a=p.parse_args();v=build(a.base_key,a.output,seed=a.seed)
    print(json.dumps({k:v[k] for k in ('seed','room_seed','active_hand','frozen_profile_id','online_ik_calls','generation_wall_seconds')}))


if __name__=='__main__':main()
