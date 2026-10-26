"""Pre-physics pose/reset composition; never writes dynamics or drive settings."""
import os
from copy import deepcopy


def world(stage, path):
    import numpy as np
    from pxr import UsdGeom
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsActive():
        raise ValueError('Missing active pose instance: ' + path)
    return np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).T


def place(stage, path, matrix):
    import numpy as np
    from pxr import Gf, UsdGeom
    prim = stage.GetPrimAtPath(path)
    local = np.linalg.inv(world(stage, str(prim.GetParent().GetPath()))) @ matrix
    xf = UsdGeom.Xformable(stage.OverridePrim(path))
    xf.ClearXformOpOrder()
    xf.AddTransformOp(opSuffix='failureInitial').Set(Gf.Matrix4d(local.T.tolist()))


def stage_with_layers(out, layers):
    from pxr import Sdf, Usd, UsdGeom
    from arena_api import l1_scene
    stage = Usd.Stage.CreateNew(str(out/'initial.usda'))
    stage.GetRootLayer().subLayerPaths = [os.path.relpath(p, out) for p in layers]
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
    stage.GetPrimAtPath('/World').CreateAttribute(l1_scene.MARKER, Sdf.ValueTypeNames.String).Set(l1_scene.POLICY)
    return stage


def reset(names, values, *, source):
    if len(names) != len(values) or len(set(names)) != len(names):
        raise ValueError('Complete, uniquely ordered joint coordinates required')
    positions = dict(zip(names, map(float, values)))
    # Passive linkage follows the unchanged source robot; only source driven
    # coordinates are held. Measured passive values may still be reset once.
    driven = {k:v for k,v in positions.items() if '_arm_' in k or '_body_' in k
              or '_head_' in k or k in ('idx41_gripper_l_outer_joint1', 'idx81_gripper_r_outer_joint1')}
    return dict(positions=positions, hold_targets=deepcopy(driven), source=source)


def trace_pose(row):
    import numpy as np
    from scipy.spatial.transform import Rotation
    m = np.eye(4)
    w,x,y,z = row['quaternion']
    m[:3,:3] = Rotation.from_quat([x,y,z,w]).as_matrix()
    m[:3,3] = row['position']
    return m
