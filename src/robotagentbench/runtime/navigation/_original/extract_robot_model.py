"""Extract the robot kinematic model from a room USD (CPU USD only, no simulation).

Usage: bash api-protocol/evaluation/tools/run_cpu_usd.sh tools/extract_robot_model.py <room.usda> <out.json>
"""
import json
import sys

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics

ROBOT = '/World/Robot'


def mat(m):  # Gf row-vector matrix -> numpy column-vector 4x4
    return np.array([[m[i][j] for j in range(4)] for i in range(4)], dtype=float).T


def frame(pos, rot):
    q = Gf.Quatd(float(rot.GetReal()), *[float(v) for v in rot.GetImaginary()])
    m = Gf.Matrix4d().SetRotate(q)
    m.SetTranslateOnly(Gf.Vec3d(*[float(v) for v in pos]))
    return mat(m)


def extract(stage, scene):
    root = stage.GetPrimAtPath(ROBOT)
    cache = UsdGeom.XformCache()
    root_world = mat(cache.GetLocalToWorldTransform(root))
    links = {}
    for child in root.GetChildren():
        if child.HasAPI(UsdPhysics.RigidBodyAPI):
            links[child.GetName()] = mat(cache.GetLocalToWorldTransform(child))
    joints = []
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        b0 = joint.GetBody0Rel().GetTargets()
        b1 = joint.GetBody1Rel().GetTargets()
        kind = prim.GetTypeName().replace('Physics', '').replace('Joint', '').lower() or 'generic'
        row = dict(name=prim.GetName(), type=kind, path=str(prim.GetPath()),
                   parent=str(b0[0]).split('/')[-1] if b0 else None,
                   child=str(b1[0]).split('/')[-1] if b1 else None,
                   frame0=frame(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get()).tolist(),
                   frame1=frame(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get()).tolist(),
                   exclude_from_articulation=bool(joint.GetExcludeFromArticulationAttr().Get()),
                   enabled=joint.GetJointEnabledAttr().Get() is not False)
        if kind in ('revolute', 'prismatic'):
            scale = np.pi / 180 if kind == 'revolute' else 1.0
            row['axis'] = prim.GetAttribute('physics:axis').Get()
            row['lower'] = float(prim.GetAttribute('physics:lowerLimit').Get()) * scale
            row['upper'] = float(prim.GetAttribute('physics:upperLimit').Get()) * scale
            v = prim.GetAttribute('physxJoint:maxJointVelocity')
            row['max_velocity'] = float(v.Get()) * scale if v and v.Get() is not None else None
            for token in ('angular', 'linear'):
                t = prim.GetAttribute(f'drive:{token}:physics:targetPosition')
                if t and t.Get() is not None:
                    row['drive_target'] = float(t.Get()) * scale
                s = prim.GetAttribute(f'state:{token}:physics:position')
                if s and s.Get() is not None:
                    row['state_position'] = float(s.Get()) * scale
        joints.append(row)
    cameras = {}
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Camera):
            cam = UsdGeom.Camera(prim)
            cameras[prim.GetName()] = dict(
                path=str(prim.GetPath()), link=prim.GetParent().GetName(),
                local=mat(UsdGeom.Xformable(prim).GetLocalTransformation()).tolist(),
                focal_length=float(cam.GetFocalLengthAttr().Get()),
                horizontal_aperture=float(cam.GetHorizontalApertureAttr().Get()),
                vertical_aperture=float(cam.GetVerticalApertureAttr().Get()))
    model = dict(schema='openend_explore.robot_model.v1', source_scene=scene, robot_path=ROBOT,
                 root_world=root_world.tolist(),
                 links={k: (np.linalg.inv(root_world) @ v).tolist() for k, v in links.items()},
                 joints=joints, cameras=cameras)
    return model


def main(scene, out):
    model = extract(Usd.Stage.Open(scene), scene)
    with open(out, 'w') as f:
        json.dump(model, f)
    print('links', len(model['links']), 'joints', len(model['joints']), 'cameras', list(model['cameras']))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
