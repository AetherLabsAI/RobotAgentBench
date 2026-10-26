"""Approved frame-zero reset for food/salad; never modify recorded actions."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from .initial_state import TASK_DIR, ROBOT, FRIDGE, TARGET

SOURCES = {'food':'food_000165', 'salad':'salad_000006'}
SOURCE_ROOT = TASK_DIR/'sources/fridge_variant_candidates_20260917'


def wire_targets(targets, limits):
    """Canonical double boundary ONLY when the native float32 command is identical."""
    result=dict(targets)
    encoded=[]
    for name,value in result.items():
        lo,hi=limits[name]
        boundary=lo if value<lo else hi if value>hi else None
        if boundary is not None:
            if np.float32(value)!=np.float32(boundary):
                raise ValueError('Genuine source command outside limits: '+name)
            result[name]=boundary
            encoded.append(dict(joint=name,source_double=value,wire_double=boundary,float32_identical=True))
    return result,encoded


def load(kind):
    if kind not in SOURCES:
        raise ValueError('Unknown recorded variant')
    directory = SOURCE_ROOT/SOURCES[kind]
    receipt = json.loads((TASK_DIR/'reports/source_recording_audit_20260917.json').read_text())[kind]
    for relative, expected in receipt['sha256'].items():
        if hashlib.sha256((directory/relative).read_bytes()).hexdigest() != expected:
            raise ValueError('Recorded source changed: '+relative)
    with np.load(directory/'raw_npz',allow_pickle=False) as data:
        raw = {k:data[k].copy() for k in data.files}
    return directory, receipt, raw


def robot_plan(kind):
    from arena_api.task_runtime import RobotInitialization
    _,_,raw = load(kind)
    return RobotInitialization(
        positions=dict(zip(raw['joint_names'].tolist(),map(float,raw['q'][0]))),
        hold_targets=dict(zip(raw['action_names'].tolist(),map(float,raw['action'][0]))))


def build(source, output, kind):
    from pxr import Gf, Sdf, Usd, UsdGeom, PhysxSchema
    from .random_reset import build as build_boxes, physical_snapshot, EXTRAS
    from .randomization import SLOTS
    from arena_api.food_box_assets import verify_instance
    output=Path(output).resolve()
    if not output.is_relative_to(TASK_DIR.resolve()) or output==TASK_DIR.resolve():
        raise ValueError('Task-local output required')
    directory, receipt, raw = load(kind)
    scene,plan = build_boxes(source,output/'native_boxes',0,fixed=True)
    layer=Sdf.Layer.CreateAnonymous()
    layer.subLayerPaths=[str(scene)]
    stage=Usd.Stage.Open(layer)
    source_stage=Usd.Stage.Open(str(scene))
    for key in ('upAxis','metersPerUnit','framesPerSecond','timeCodesPerSecond'):
        stage.SetMetadata(key,source_stage.GetMetadata(key))
    stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
    baseline=physical_snapshot(stage)
    layout=json.loads((directory/'raw_layout.json').read_text())['layout']
    def pose(path, values):
        prim=stage.GetPrimAtPath(path)
        if not prim or not prim.IsActive():
            raise ValueError('Missing recorded pose destination: '+path)
        xf=UsdGeom.Xformable(prim)
        oldscale=Gf.Transform(xf.ComputeLocalToWorldTransform(0)).GetScale()
        if not Gf.IsClose(oldscale,Gf.Vec3d(1),1e-6):
            raise ValueError('Recorded pose must not overwrite instance scale: '+path)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble,'recordedFrame0').Set(Gf.Vec3d(*map(float,values[:3])))
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble,'recordedFrame0').Set(
            Gf.Quatd(float(values[6]),Gf.Vec3d(*map(float,values[3:6]))).GetNormalized())
        xf.SetResetXformStack(True)
    pose(ROBOT,raw['base_pose'][0])
    original={'beef':'/World/Assets/BeefTray','food':'/World/Assets/BinFoodR','salad':'/World/Assets/SaladBoxR'}
    paths=raw['box_paths'].tolist()
    for box_kind, original_path in original.items():
        body=TARGET if box_kind=='beef' else EXTRAS[box_kind]+'/Asset'
        root=TARGET if box_kind=='beef' else EXTRAS[box_kind]
        values=raw['box_pose'][0,paths.index(original_path)]
        pose(body,values)
        entry=layout[box_kind]
        level=SLOTS[entry['slot']]['level']
        # Reader uses compartment and actual physical body, not old sampled pose.
        plan['boxes'][box_kind]=dict(entry,level=level,root=root,body=body,
            world_position_m=values[:3].tolist(),world_quaternion_xyzw=values[3:].tolist(),
            source_compartment='shelf4' if level=='up' else 'bin34')
    # Frame-zero joint state only; retain drive/material/solver/collision settings.
    for side in ('left','right'):
        prim=stage.GetPrimAtPath(FRIDGE+'/'+side+'_door_hinge')
        state=PhysxSchema.JointStateAPI.Apply(prim,'angular')
        state.CreatePositionAttr(float(raw['door_theta'][0]) if side=='right' else 0.)
        state.CreateVelocityAttr(0.)
    if physical_snapshot(stage)!=baseline:
        raise ValueError('Source reset changed physical parameters')
    for box_kind,variant in (('food','leftover_box'),('salad','salad_box')):
        verify_instance(stage,EXTRAS[box_kind]+'/Asset',variant)
    destination=output/'scene.usda'
    layer.Export(str(destination))
    plan.update(source_policy='user-approved original recording frame zero',seed=None,
        source_episode=receipt['episode'],source_raw_sha256=receipt['sha256']['raw_npz'],
        scene_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        robot=dict(world_pose_xyz_quaternion_xyzw=raw['base_pose'][0].tolist()),
        physics_parameters_changed=False,physical_initial_state_accepted=False)
    with (output/'recorded_initial.json').open('x') as f:
        json.dump(plan,f,indent=2,allow_nan=False)
    return destination,plan
