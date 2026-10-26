"""Shelf-source reset coordinates and read-only recorded initial link envelope.

No drives or physical attributes are authored here. This is a source initial
geometry check, not a substitute for native settling in the generated room.
"""
import json
import math
from pathlib import Path

SOURCE = Path(__file__).with_name('source_ready_pose.json')


def read():
    value = json.loads(SOURCE.read_text())
    if value['status'] != 'source_shelf_recorded_initial_posture' or value['source_frame_index'] != 0:
        raise ValueError('Expected shelf initial observation, not a trajectory frame')
    targets = value['targets_rad']
    if len(targets) != 34 or set(targets) != set(value['source_recorded_joint_names']):
        raise ValueError('Incomplete shelf-source reset coordinates')
    if not all(math.isfinite(v) for v in targets.values()):
        raise ValueError('Non-finite shelf-source reset')
    return value


def load(stage):
    """Map source links to the current robot base without copying world XY/Z."""
    import numpy as np
    from pxr import UsdGeom, UsdPhysics
    from scipy.spatial.transform import Rotation
    value = read()
    for name, position in value['targets_rad'].items():
        joint = stage.GetPrimAtPath('/World/Robot/joints/' + name)
        if not joint or not (joint.IsA(UsdPhysics.PrismaticJoint) or joint.IsA(UsdPhysics.RevoluteJoint)):
            raise ValueError('Shelf-source joint absent in room: ' + name)
        scale = 1. if joint.IsA(UsdPhysics.PrismaticJoint) else math.pi/180
        limits = [joint.GetAttribute('physics:'+k+'Limit').Get() for k in ('lower','upper')]
        if any(v is None for v in limits) or not limits[0]*scale <= position <= limits[1]*scale:
            raise ValueError('Shelf-source pose violates existing room joint limits: ' + name)
    def matrix(xyz, q):
        m = np.eye(4)
        m[:3,:3] = Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix().T
        m[3,:3] = xyz
        return m
    paths = value['source_link_paths']
    if len(paths) != len(set(paths)) or len(paths) != len(value['source_link_xyz_m']) or len(paths) != len(value['source_link_wxyz']):
        raise ValueError('Invalid source link observation')
    recorded = {path: matrix(p, q) for path,p,q in zip(paths,value['source_link_xyz_m'],value['source_link_wxyz'])}
    base = '/World/Robot/base_link'
    current = stage.GetPrimAtPath(base)
    if not current or base not in recorded:
        raise ValueError('Shelf-source base link missing')
    target_base = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(current))
    delta = np.linalg.inv(recorded[base]) @ target_base
    result = {}
    for path, transform in recorded.items():
        body = stage.GetPrimAtPath(path)
        if not body or not body.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError('Source link does not match room articulation: ' + path)
        result[body.GetName()] = [transform @ delta]
    return result
