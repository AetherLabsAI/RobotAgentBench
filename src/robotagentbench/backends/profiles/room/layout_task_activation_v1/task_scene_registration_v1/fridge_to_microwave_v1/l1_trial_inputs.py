"""Read-only L1 action preflight, shared by the task-owned opening trial.

Only the actual 20 Hz drive commands are replay inputs. Recorded poses are
initialization/comparison evidence, never per-frame simulation commands.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path('/home/lingjun/robot_agent_arena/layout_task_activation_v1/task_scene_registration_v1')
API = Path('/home/lingjun/robot_agent_arena/api-protocol')
SOURCE = API/'evaluation/tasks/refrigerator_take/sources/ep_beefw3_001/trajectory'
MW = API/'evaluation/tasks/microwave_put/references/l1_beef_camv1_001569_20260913'


def load():
    raw = dict(np.load(SOURCE/'raw_npz', allow_pickle=False))
    exported = np.load(SOURCE/'actions_20hz.npz', allow_pickle=False)
    grouped = np.column_stack([exported[k] for k in ('armL','armR','waist','head','lgrip','rgrip')])
    assert np.array_equal(grouped, raw['action'])
    assert float(raw['fps'][0]) == float(exported['fps'][0]) == 20
    assert raw['action'].shape == (1128,20) and np.isfinite(raw['action']).all()
    assert len(set(raw['action_names'])) == 20
    assert np.ptp(raw['base_pose'],axis=0).max() == 0
    return raw


def matrix(pose):
    from pxr import Gf
    m = Gf.Matrix4d(1)
    m.SetRotate(Gf.Quatd(float(pose[6]),Gf.Vec3d(*map(float,pose[3:6]))).GetNormalized())
    m.SetTranslateOnly(Gf.Vec3d(*map(float,pose[:3])))
    return m


def audit(stage, manifest):
    from pxr import Gf, UsdGeom, UsdPhysics
    from room_fridge_helper_v1 import FRIDGE
    raw = load()
    cache = UsdGeom.XformCache()
    # Recorded untouched left door is the common rigid frame, not the moving
    # right door. Row-vector USD composition: old-world -> local -> new-world.
    assert np.max(np.ptp(raw['door_pose'][:,1,:3],axis=0)) < 1e-5, 'Expected untouched left door'
    old = matrix(raw['door_pose'][0,1])
    current = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(FRIDGE+'/left_door'))
    delta = old.GetInverse()*current
    axes = np.asarray(delta)[:3,:3]
    assert np.allclose(axes@axes.T,np.eye(3),atol=1e-5), 'Nonrigid source/room mapping'
    assert np.allclose(axes[2],[0,0,1],atol=1e-4), 'Not a planar room transform'
    base = matrix(raw['base_pose'][0])*delta
    target = next(r for r in manifest['objects'] if r['id']=='beef_box')
    target_current = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(target['path']))
    target_source = matrix(raw['box_pose'][0,0])*delta
    gap = (target_current.ExtractTranslation()-target_source.ExtractTranslation()).GetLength()
    bad=[]
    for j,n in enumerate(raw['action_names']):
        p=stage.GetPrimAtPath('/World/Robot/joints/'+str(n))
        assert p and (p.IsA(UsdPhysics.PrismaticJoint) or p.IsA(UsdPhysics.RevoluteJoint)),str(n)
        bounds=np.asarray([p.GetAttribute('physics:'+b+'Limit').Get() for b in ('lower','upper')],float)
        if p.IsA(UsdPhysics.RevoluteJoint):bounds=np.deg2rad(bounds)
        # Exact float32 wire equivalence at a limit is not a relaxed limit.
        v=raw['action'][:,j].astype(np.float32)
        ix=np.flatnonzero((v<np.float32(bounds[0])) | (v>np.float32(bounds[1])))
        if len(ix):bad.append(dict(joint=str(n),first_frame=int(ix[0]),count=len(ix)))
    initial=np.load(MW/'initial_state.npz',allow_pickle=False)
    mw=np.load(MW/'control_state_60hz.npz',allow_pickle=False)
    old_names=raw['joint_names'].tolist(); new_names=initial['dof_names'].tolist()
    right=[f'idx{61+i}_arm_r_joint{i+1}' for i in range(7)]
    jump=[abs(float(raw['q'][-1,old_names.index(n)]-initial['q'][new_names.index(n)])) for n in right]
    anchor=stage.GetPrimAtPath('/World/Robot/base_link/FixedJoint')
    report=dict(source_episode='ep_beefw3_001',source_control_hz=20,source_frames=1128,
        opening_frames=168,opening_duration_s=8.4,action_values_changed=False,
        source_sha256={str(SOURCE/n):hashlib.sha256((SOURCE/n).read_bytes()).hexdigest()
                       for n in ('raw_npz','actions_20hz.npz','states_20hz.npz')},
        source_to_room_matrix=np.asarray(delta).tolist(),trial_base_matrix=np.asarray(base).tolist(),
        target_initial_difference_m=float(gap),target_slot=target['slot'],
        target_reset_planned=False,command_limit_violations=bad,
        fixed_anchor_attributes={a.GetName():str(a.Get()) for a in anchor.GetAttributes()} if anchor else {},
        microwave_source_frames=len(mw['target']),microwave_source_hz=60,
        boundary_max_right_arm_jump_deg=float(np.rad2deg(max(jump))),
        boundary_max_right_arm_jump_joint=right[int(np.argmax(jump))],
        missing_continuous_segments=['held-box navigation between appliances','arm/grasp transition to microwave entry'],
        combined_replay_ready=False,full_trajectory_verified=False,
        next_test='opening-only native room diagnostic; preserve randomized boxes and original physics')
    return raw,base,report


if __name__=='__main__':
    import argparse
    from pxr import Usd
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    manifest=json.loads((ROOT/'coffee_initialized_rooms_v1/layout03/composition.json').read_text())
    stage=Usd.Stage.Open(manifest['scene'])
    _,_,report=audit(stage,manifest)
    with a.output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k not in ('fixed_anchor_attributes','source_sha256')}))
