"""Read-only bounds for the exact previously measured folded ready posture."""
import hashlib,itertools,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
REFERENCE=Path(__file__).resolve().parent/'ready_reference'
def matrix(pose):
    result=np.eye(4);result[:3,:3]=Rotation.from_quat(pose[3:]).as_matrix().T;result[3,:3]=pose[:3];return result
def load(stage):
    from pxr import UsdGeom
    report=json.loads((REFERENCE/'navigation_result.json').read_text())
    if report['status']!='native_action_sequence_completed':raise ValueError('Ready posture evidence incomplete')
    current=Path('/home/lingjun/robot_agent_arena/layout_task_activation_v1/task_scene_registration_v1/recorded_ready_pose_042_v1.json')
    if json.loads(current.read_text())!=report['navigation_posture']:raise ValueError('Ready posture reference changed')
    poses=np.load(REFERENCE/'native_poses.npz')['links'][:,0]
    names=report['robot_link_names'];base=names.index('base_link')
    cache=UsdGeom.XformCache();target_base=np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath('/World/Robot/base_link')))
    transforms={name:[] for name in names}
    for frame in poses:
        reference_inverse=np.linalg.inv(matrix(frame[base]))
        for name,pose in zip(names,frame):transforms[name].append(matrix(pose)@reference_inverse@target_base)
    return transforms
def bounds(stage,prim,world_box,transforms,cache):
    from pxr import UsdPhysics
    body=prim
    while body and not body.IsPseudoRoot() and not body.HasAPI(UsdPhysics.RigidBodyAPI):body=body.GetParent()
    if not body or body.GetName() not in transforms:raise ValueError('Unbound ready-posture collider '+str(prim.GetPath()))
    corners=np.array(list(itertools.product(*zip(*world_box))))
    local=np.c_[corners,np.ones(8)]@np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(body)))
    points=np.concatenate([(local@m)[:,:3] for m in transforms[body.GetName()]])
    return np.array([points.min(0),points.max(0)])
