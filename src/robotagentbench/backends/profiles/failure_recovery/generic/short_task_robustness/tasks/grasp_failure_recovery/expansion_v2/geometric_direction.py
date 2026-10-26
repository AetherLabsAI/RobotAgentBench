"""Human-approved natural cup side-grasp directions from native hand geometry.

No training motion or assumed left/right mirror. Approach is horizontal toward
the cup, jaw closing is horizontal across its upright axis, existing wrist
camera stays on the upper side of the hand. Matrices are frozen per hand.
"""
import argparse
import json
from pathlib import Path
from ..common import HERE,owned,digest,write_json
from .tipped_initial import load_base


def side_direction(inner_tip,outer_tip,palm_origin,camera_origin):
    import numpy as np
    inner,outer,palm,camera=map(lambda v:np.asarray(v,dtype=float),
        (inner_tip,outer_tip,palm_origin,camera_origin))
    if any(p.shape!=(3,) or not np.isfinite(p).all() for p in (inner,outer,palm,camera)):
        raise ValueError('Finite native hand-local points required')
    approach=(inner+outer)/2-palm
    if np.linalg.norm(approach)==0:raise ValueError('Missing native palm-to-fingertip direction')
    approach/=np.linalg.norm(approach)
    closing=outer-inner;closing-=approach*float(closing@approach)
    if np.linalg.norm(closing)==0:raise ValueError('Degenerate native opposing fingertip geometry')
    closing/=np.linalg.norm(closing)
    normal=np.cross(approach,closing)
    local_basis=np.column_stack((approach,closing,normal))
    choices=[]
    for sign in (1.,-1.):
        desired_approach=np.array([1.,0.,0.])
        desired_closing=np.array([0.,sign,0.])
        desired=np.column_stack((desired_approach,desired_closing,np.cross(desired_approach,desired_closing)))
        rotation=desired@local_basis.T
        choices.append((float((rotation@camera)[2]),rotation,sign))
    camera_height,rotation,sign=max(choices,key=lambda r:r[0])
    if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-7,rtol=0) or not np.isclose(np.linalg.det(rotation),1.,atol=1e-7):
        raise ValueError('Native hand direction must yield a proper rotation')
    return rotation,dict(approach_local=approach.tolist(),jaw_axis_local=closing.tolist(),
        palm_normal_local=normal.tolist(),desired_approach_base=[1.,0.,0.],
        desired_jaw_axis_base=[0.,sign,0.],wrist_camera_height_relative_to_tcp_m=camera_height,
        definition='horizontal_side_grasp_of_upright_cup_wrist_camera_on_upper_side')


def freeze(base_key,output):
    import numpy as np
    from pxr import Usd,UsdGeom
    from ..procedural_grasp import Kinematics,complete_closed_reset,collider_points
    from short_tasks.agent_probe_v1.native_owner import initialize_base_pose,CAMERAS
    _,launch,manifest,bound,ready=load_base(base_key,'clear_table_trash')
    if bound['target_categories']!=['cup']:raise ValueError('Explicit upright used-cup task required')
    stage=Usd.Stage.Open(manifest['scene']);stage.SetEditTarget(stage.GetSessionLayer())
    initialize_base_pose(stage,launch['initial_base_xyyaw'])
    initial=dict(positions=ready['targets_rad'],hold_targets=ready['targets_rad'])
    initial,_=complete_closed_reset(stage,initial,'l')
    initial,_=complete_closed_reset(stage,initial,'r')
    rotations={};provenance=[]
    for hand,name in (('l','left'),('r','right')):
        model=Kinematics(stage,hand,initial['positions']);frames=model.frames(initial['positions'])
        inverse=np.linalg.inv(frames[model.tool])
        tip_centres=[]
        for branch in ('inner','outer'):
            pts=np.concatenate([(inverse@frames[body]@local.T).T[:,:3]
                for _,body,local,_ in collider_points(stage,f'/World/Robot/gripper_{hand}_{branch}_link4')])
            tip_centres.append(pts.mean(0))
        palm_path=f'/World/Robot/gripper_{hand}_base_link'
        palm=(inverse@frames[palm_path])[:3,3]
        cache=UsdGeom.XformCache()
        authored_palm=np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(palm_path))).T
        authored_camera=np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(CAMERAS[name]))).T
        camera=(inverse@frames[palm_path]@np.linalg.inv(authored_palm)@authored_camera)[:3,3]
        rotation,evidence=side_direction(*tip_centres,palm,camera)
        rotations[hand]=rotation.tolist()
        provenance.append(dict(hand=hand,source='actual_native_closed_finger_palm_and_camera_geometry',
            tip_centres_local_m=[p.tolist() for p in tip_centres],
            palm_origin_local_m=palm.tolist(),camera_origin_local_m=camera.tolist(),
            **evidence,source_motion_read=False,native_physics_verified=False))
    result=dict(schema='failure_recovery.fixed_grasp_direction.v1',variant='table_waste_empty_grasp',
        fixed_tool_rotation_base=rotations,orientation_jitter=False,hand_coverage=['l','r'],
        provenance=provenance,direction_definition_human_authorized=True,
        code_sha256=digest(__file__),native_geometry_scene_sha256=digest(manifest['scene']),
        source_data_needed_during_episode_generation=False,arm_positions_frozen=False,
        source_checkpoint_replayed=False,native_grasp_orientation_accepted=False)
    out=owned(output,fresh=True)
    if not out.is_relative_to(HERE/'expansion_v2/configurations'):raise ValueError('Task-owned fixed config output required')
    write_json(out,result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--base-key',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();v=freeze(a.base_key,a.output)
    print(json.dumps(dict(hand_coverage=v['hand_coverage'],provenance=v['provenance'])))


if __name__=='__main__':main()
