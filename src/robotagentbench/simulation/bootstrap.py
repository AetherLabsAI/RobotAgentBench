"""Camera profiles and pre-physics base placement from the native owner."""
import math

CAMERAS = dict(head='/World/Robot/head_link2/Head_Camera',
    left='/World/Robot/gripper_l_base_link/Left_Camera',
    right='/World/Robot/gripper_r_base_link/Right_Camera')


CAMERA_RESOLUTIONS = dict(head=(1280,800), left=(848,480), right=(848,480))


def camera_resolutions(task_name, instruction=None):
    return dict(CAMERA_RESOLUTIONS)


def valid_camera_frame(name, frame, resolutions=None):
    width, height = (CAMERA_RESOLUTIONS if resolutions is None else resolutions)[name]
    return frame.shape == (height,width,4) and str(frame.dtype) == 'uint8'


def initialize_base_pose(stage, initial_base_xyyaw):
    """Place the base frame, accounting for an already-authored local FK frame.

    This is pre-physics initialization only. Prepared Context scenes can keep
    the old Robot container transform while authoring a new base_link local
    transform. Assigning the requested BASE pose directly to Robot applies that
    displacement twice. Preserve the articulation's local transforms and solve
    the container transform so base_link has exactly the declared world pose.
    """
    from pxr import Gf, UsdGeom
    root=stage.GetPrimAtPath('/World/Robot')
    base=stage.GetPrimAtPath('/World/Robot/base_link')
    if not root or not base:raise ValueError('Robot and base frame required')
    if UsdGeom.Xformable(base).GetResetXformStack():
        raise ValueError('Base frame ignores Robot container; cannot initialize via container')
    cache=UsdGeom.XformCache()
    root_before=cache.GetLocalToWorldTransform(root)
    base_before=cache.GetLocalToWorldTransform(base)
    base_in_robot=base_before*root_before.GetInverse()
    x,y,yaw=initial_base_xyyaw
    desired=Gf.Matrix4d(1)
    desired.SetRotate(Gf.Rotation(Gf.Vec3d(0,0,1),math.degrees(yaw)))
    desired.SetTranslateOnly(Gf.Vec3d(x,y,0))
    xf=UsdGeom.Xformable(root);reset=xf.GetResetXformStack()
    parent=Gf.Matrix4d(1) if reset else cache.GetLocalToWorldTransform(root.GetParent())
    xf.MakeMatrixXform().Set(base_in_robot.GetInverse()*desired*parent.GetInverse())
    xf.SetResetXformStack(reset)
    actual=UsdGeom.XformCache().GetLocalToWorldTransform(base)
    if not Gf.IsClose(actual,desired,1e-9):
        raise ValueError('Declared base world pose was not restored')
    return dict(method='base_world_frame_compensation_before_physics',
        requested_xyyaw=list(initial_base_xyyaw),
        base_world_before=list(map(list,base_before)),base_in_robot=list(map(list,base_in_robot)),
        base_world_after=list(map(list,actual)),source_joint_frames_unchanged=True)
