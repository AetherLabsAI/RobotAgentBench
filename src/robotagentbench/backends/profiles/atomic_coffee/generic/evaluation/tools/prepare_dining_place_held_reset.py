"""Reset directly to an observed pad/plate contact pose; no pickup/search.

The source frame need not be a free-hold success: its support is disclosed and
removed by initializing above the table. The new 5-second physical test alone
decides whether it is a useful held initial state. Assets/drive gains stay fixed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

MODE='direct_inhand_v1'


def matrix(value):
    from pxr import Gf
    q=value['quaternion_wxyz']
    result=Gf.Matrix4d().SetRotate(Gf.Quatd(q[0],Gf.Vec3d(*q[1:])))
    result.SetTranslateOnly(Gf.Vec3d(*value['position_m']))
    return result


def parent_pose_for_body(local, world_pose):
    # A rigid pose excludes the asset's mm-to-m scale. Never invert that scale
    # into the parent: doing so would enlarge the current plate by 1000x.
    return local.RemoveScaleShear().GetInverse()*world_pose


def prepare(base, output, checkpoint_run):
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics
    from evaluation.tools.prepare_dining_place_initial import prepare as base_prepare, pose, PLATE, PLATE_ROOT, BOX, HAND
    from evaluation.tasks.dining_place.kinematics import body_transform
    source=Path(checkpoint_run).resolve(strict=True)
    source_plan=json.loads((source.parent/'initial.json').read_text())
    report=json.loads((source/'preclamp_report.json').read_text())
    trace=source/'preclamp_trace.jsonl'
    # Last physically observed state, not a command or a fabricated held pose.
    with trace.open() as stream:
        last=None
        for line in stream:
            if line.strip(): last=json.loads(line)
    if not last or last['step']!=report['last_clamp_sample']['step']:
        raise ValueError('Missing matching physical contact checkpoint')
    if not all(p['fresh'] and p['force_n']>=.05 for p in last['pads'].values()):
        raise ValueError('Checkpoint has no fresh bilateral plate contact')
    command=float(report['last_clamp_sample']['command_angle_rad'])
    actual=(last['joints']['idx81_gripper_r_outer_joint1']-last['joints']['idx71_gripper_r_inner_joint1'])/2
    out=Path(output).resolve()
    plan=base_prepare(base,out,aperture=actual,settle_aperture=command,
                      calibration='legacy',staged_reset=False)
    stage=Usd.Stage.Open(str(out/'initial.usda'))
    station=matrix(plan['robot_station'])
    # Reuse measured right-arm and pad coordinates, including passive fingers.
    # Left arm remains at the task's existing initial pose.
    reset={n:v['value'] for n,v in plan['joint_targets'].items()}
    reset.update({n:v for n,v in last['joints'].items() if '_r_' in n or n=='idx02_body_joint2'})
    reset['idx01_body_joint1']=plan['joint_targets']['idx01_body_joint1']['value']
    commands={n:dict(v) for n,v in plan['joint_targets'].items()}
    for n in commands:
        if '_arm_r_' in n or n=='idx02_body_joint2': commands[n]=dict(source_plan['joint_targets'][n])
    commands['idx81_gripper_r_outer_joint1']['value']=command
    commands['idx71_gripper_r_inner_joint1']['value']=-command
    for name,value in reset.items():
        prim=stage.GetPrimAtPath('/World/Robot/joints/'+name)
        linear=prim.IsA(UsdPhysics.PrismaticJoint)
        native=value if linear else math.degrees(value)
        low,high=[prim.GetAttribute('physics:'+s+'Limit').Get() for s in ('lower','upper')]
        if not math.isfinite(native) or not low-.001<=native<=high+.001:
            raise ValueError('Contact checkpoint outside native joint limits: '+name)
        axis='linear' if linear else 'angular'
        state=PhysxSchema.JointStateAPI.Apply(prim,axis)
        state.CreatePositionAttr(native);state.CreateVelocityAttr(0.)
        if name in commands and prim.HasAPI(UsdPhysics.DriveAPI,axis):
            drive=UsdPhysics.DriveAPI(prim,axis)
            target=commands[name]['value']
            drive.CreateTargetPositionAttr(target if linear else math.degrees(target))
            drive.CreateTargetVelocityAttr(0.)
    hand=body_transform(stage,HAND,reset,station)
    relative=matrix(last['bodies']['plate'])*matrix(last['hand']).GetInverse()
    plate=relative*hand
    box_relative=Gf.Matrix4d(*[v for row in source_plan['payload_to_plate'] for v in row])
    def author(path, transform):
        xf=UsdGeom.Xformable(stage.GetPrimAtPath(path))
        xf.ClearXformOpOrder();xf.AddTransformOp().Set(transform);xf.SetResetXformStack(True)
    body_xf=UsdGeom.Xformable(stage.GetPrimAtPath(PLATE))
    original_scale=Gf.Transform(body_xf.ComputeLocalToWorldTransform(0)).GetScale()
    local=body_xf.GetLocalTransformation()
    author(PLATE_ROOT,parent_pose_for_body(local,plate))
    actual_world=body_xf.ComputeLocalToWorldTransform(0)
    if (Gf.Transform(actual_world).GetScale()-original_scale).GetLength()>1e-9:
        raise RuntimeError('Direct reset changed the plate asset scale')
    if (actual_world.ExtractTranslation()-plate.ExtractTranslation()).GetLength()>1e-8:
        raise RuntimeError('Direct reset plate pose did not compose')
    author(BOX,box_relative*plate)
    # The flat-bottom plate is initialized high above the table; no staging support.
    clearance=float(plate.ExtractTranslation()[2]-plan['surface']['support_top_m'])
    if clearance<.10: raise ValueError('Held reset must start clear of the table')
    source_files=(trace,source/'preclamp_report.json',source.parent/'initial.json')
    plan.update(grasp_setup=MODE,grasp_calibration='observed_pad_plate_relative_pose',
        staged_reset=False,robot_warmup_steps=0,initial_wrist_ik=None,table_support=None,
        joint_targets=commands,joint_reset_coordinates=reset,
        plate_to_hand=[list(row) for row in relative],payload_to_plate=[list(row) for row in box_relative],
        hand_fk=pose(hand),plate=pose(plate),payload=pose(box_relative*plate),
        source_contact_checkpoint=dict(run=str(source),step=last['step'],pads=last['pads'],
            original_table_support=last.get('plate_support'),original_free_hold_accepted=False,
            hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}),
        plate_origin_above_table_m=clearance,
        preserved_plate_world_scale=list(original_scale),
        boundary='Direct pre-episode articulation/plate/box reset; no pre-seat search, pickup, attachment or validation-time object writes')
    stage.GetRootLayer().customLayerData={'arenaTask':'10.2','initialPlanJson':json.dumps(plan,sort_keys=True),'physicsValidated':False}
    stage.GetRootLayer().Save()
    plan['scene_sha256']=hashlib.sha256((out/'initial.usda').read_bytes()).hexdigest()
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in plan['source_layer_sha256'].items()):
        raise RuntimeError('Shared source changed during reset preparation')
    (out/'initial.json').write_text(json.dumps(plan,indent=2,allow_nan=False)+'\n')
    (out/'contact_checkpoint.json').write_text(json.dumps(last,indent=2,allow_nan=False)+'\n')
    return plan


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint-run',type=Path,required=True)
    args=parser.parse_args()
    result=prepare(args.base,args.output,args.checkpoint_run)
    print(json.dumps({k:result[k] for k in ('grasp_setup','plate','payload','plate_origin_above_table_m','physics_validated')},indent=2))
