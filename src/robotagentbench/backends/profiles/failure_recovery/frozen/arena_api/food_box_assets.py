"""Shared approved fridge assets. No control, mass, friction or force tuning."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT/'resources/shared_assets/fridge_food_boxes_v1'
SCENE_LAYER = DIRECTORY/'scene_bindings.usda'
REVISION = 'fridge_food_boxes_v1'
ASSETS = {'food': DIRECTORY/'foodbox.usda', 'salad': DIRECTORY/'saladbox.usda'}
INSTANCES = {'/World/Assets/MicrowaveFoodbox': 'leftover_box',
             '/World/Assets/Task108Foodbox': 'leftover_box',
             '/World/Assets/MicrowaveSaladbox': 'salad_box',
             '/World/Assets/Task108Saladbox': 'salad_box'}


def verify_files():
    data = json.loads((DIRECTORY/'manifest.json').read_text())
    if data['revision'] != REVISION:
        raise ValueError('Unexpected food-box revision')
    for name, expected in data['files'].items():
        if hashlib.sha256((DIRECTORY/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Shared fridge food-box asset changed: '+name)
    return data


def verify_instance(stage, path, variant):
    """Reject stale reset geometry/scale instead of silently accepting old data."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    data = verify_files()['variants'][variant]
    root = stage.GetPrimAtPath(path)
    if not root or not root.IsActive():
        raise ValueError('Missing active food box: '+path)
    body = UsdPhysics.RigidBodyAPI(root)
    if not body or body.GetKinematicEnabledAttr().Get() or body.GetRigidBodyEnabledAttr().Get() is False:
        raise ValueError('Food box must retain native dynamic body: '+path)
    tf = UsdGeom.Xformable(root).ComputeLocalToWorldTransform(0)
    if not Gf.IsClose(Gf.Transform(tf).GetScale(), Gf.Vec3d(1), 1e-6):
        raise ValueError('Stale reset applies extra box scaling; rebuild initial state: '+path)
    expected = Usd.Stage.Open(str(DIRECTORY/data['asset']))
    source = expected.GetDefaultPrim()
    inverse = tf.GetInverse()
    suffixes = set()
    for p in Usd.PrimRange(source):
        suffix = str(p.GetPath())[len(str(source.GetPath())):]
        actual = stage.GetPrimAtPath(path+suffix)
        if not actual or actual.GetTypeName() != p.GetTypeName():
            raise ValueError('Food-box geometry differs: '+path+suffix)
        for attr in p.GetAttributes():
            name = attr.GetName()
            if name in ('physics:velocity','physics:angularVelocity'):
                continue
            if name.startswith(('physics:','physx')) or name in ('points','faceVertexCounts','faceVertexIndices'):
                if actual.GetAttribute(name).Get() != attr.Get():
                    raise ValueError('Food-box native attribute differs: '+path+suffix+'.'+name)
        if p.IsA(UsdGeom.Mesh):
            suffixes.add(suffix)
            local = UsdGeom.Xformable(actual).ComputeLocalToWorldTransform(0)*inverse
            if not Gf.IsClose(local, UsdGeom.Xformable(p).ComputeLocalToWorldTransform(0), 1e-7):
                raise ValueError('Stale reset moved food-box geometry: '+path+suffix)
            wanted = p.GetRelationship('material:binding:physics').GetTargets()
            if wanted and actual.GetRelationship('material:binding:physics').GetTargets() != wanted:
                raise ValueError('Missing original food-box physics material: '+path+suffix)
    actual_meshes = {str(p.GetPath())[len(path):] for p in Usd.PrimRange(root) if p.IsA(UsdGeom.Mesh)}
    if actual_meshes != suffixes:
        raise ValueError('Old and canonical food-box meshes coexist: '+path)
    return data


def apply(stage):
    verify_files()
    session = stage.GetSessionLayer()
    path = str(SCENE_LAYER)
    session.subLayerPaths = [path]+[p for p in session.subLayerPaths if p != path]
    checked = []
    for path, variant in INSTANCES.items():
        p = stage.GetPrimAtPath(path)
        if p and p.IsActive():
            verify_instance(stage, path, variant)
            checked.append(path)
    return dict(revision=REVISION, instance_scale=1, active_instances_checked=checked,
                physical_reset_revalidated=False)
