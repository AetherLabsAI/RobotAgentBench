"""Author a task-local held-plate initial candidate; never edit shared assets."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from evaluation.tasks.dining_place import initial_state as task
from evaluation.tasks.dining_place.kinematics import body_transform
from evaluation.tasks.dining_place.source_reference import load_reference, initial_reference
from evaluation.tools.prepare_dining_place_reference import inspect_destination

PLATE_ROOT = '/World/Assets/MicrowaveTakePlate'
PLATE = PLATE_ROOT + '/Asset'
BOX = '/World/Assets/MicrowaveBeefbox'
HAND = '/World/Robot/gripper_r_center_link'


def pose(matrix):
    from pxr import Gf
    q = matrix.RemoveScaleShear().ExtractRotationQuat()
    return dict(position_m=list(matrix.ExtractTranslation()), quaternion_wxyz=[q.GetReal(), *q.GetImaginary()])


def native_rim_relative(azimuth_deg=0.):
    """Seat the current parallel conical rim at the distal pad faces.

    v12 ring: r=105..162.5 mm, top=20..30, bottom=14..24.
    Align its local tangent plane with the native jaw closing axis (hand X),
    and extend toward the finger tips (+hand Z), not toward the wrist (-Z).
    No mesh, collision, drive or material edits are involved.
    """
    from pxr import Gf
    angle = math.atan2(.010, .0575)
    sn, cs = math.sin(angle), math.cos(angle)
    radius = .145
    height = .017 + (radius-.105)*.010/.0575
    x = -(-radius*sn + height*cs)
    z = -.010 - (-radius*cs - height*sn)
    relative=Gf.Matrix4d(0,1,0,0, sn,0,cs,0, cs,0,-sn,0, x,0,z,1)
    if not math.isfinite(azimuth_deg) or abs(azimuth_deg)>90:
        raise ValueError('Initial grasp azimuth must be within +/-90 degrees')
    # Rotate around the contact patch, not the plate origin. The L2 reference
    # grips from the side; explicit azimuth also permits the front-grip audit.
    pivot=Gf.Vec3d(0,0,-.010)
    return (relative*Gf.Matrix4d().SetTranslate(-pivot)
            *Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1,0,0),azimuth_deg))
            *Gf.Matrix4d().SetTranslate(pivot))


def level_initial_wrist(stage, joints, robot_world, relative):
    """Task-reset IK only: keep hand position, level the current plate plane."""
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from pxr import Gf
    names = [f'idx{61+i}_arm_r_joint{i+1}' for i in range(7)]
    original = np.asarray([joints[n] for n in names])
    hand = body_transform(stage,HAND,joints,robot_world)
    normal = (relative*hand).TransformDir(Gf.Vec3d(0,0,1)).GetNormalized()
    correction = Gf.Matrix4d().SetRotate(Gf.Rotation(normal,Gf.Vec3d(0,0,1)))
    desired = hand*correction
    desired.SetTranslateOnly(hand.ExtractTranslation())
    desired_rotation = Rotation.from_matrix(np.asarray(desired)[:3,:3].T)
    low,high=[],[]
    for name in names:
        prim=stage.GetPrimAtPath(task.ROBOT+'/joints/'+name)
        low.append(math.radians(prim.GetAttribute('physics:lowerLimit').Get()))
        high.append(math.radians(prim.GetAttribute('physics:upperLimit').Get()))
    def residual(values,regularize=True):
        coordinates={**joints,**dict(zip(names,values))}
        actual=body_transform(stage,HAND,coordinates,robot_world)
        position=np.asarray(actual.ExtractTranslation())-np.asarray(desired.ExtractTranslation())
        rotation=Rotation.from_matrix(np.asarray(actual)[:3,:3].T)
        error=np.r_[position*20,(desired_rotation.inv()*rotation).as_rotvec()]
        return np.r_[error,(values-original)*.0001] if regularize else error
    solved=least_squares(residual,original,bounds=(low,high),max_nfev=180,
        xtol=1e-10,ftol=1e-10,gtol=1e-10)
    error=residual(solved.x,False)
    if np.linalg.norm(error[:3])/20 > .0005 or np.linalg.norm(error[3:])>.002:
        raise ValueError('Initial level-grasp IK failed; no candidate is runnable')
    joints.update(dict(zip(names,map(float,solved.x))))
    return dict(hand_position_error_m=float(np.linalg.norm(error[:3])/20),
        hand_orientation_error_deg=float(np.degrees(np.linalg.norm(error[3:]))),
        original_right_arm_rad=original.tolist(),reset_right_arm_rad=solved.x.tolist(),
        reason='Current plate level and clear of wrist at initial grasp; not a retargeted placement trajectory')


def prepare(base, output, *, aperture=.065, seat_shift=(0.,0.,0.), settle_aperture=.025, calibration='native_rim', staged_reset=False, grasp_azimuth_deg=0., grasp_setup='legacy_staged'):
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics
    from tools.build_l1_scene import prepare as environment
    from evaluation.tasks.microwave_take.initial_state import verify_plate_asset
    verify_plate_asset()
    if calibration not in ('native_rim', 'legacy'):
        raise ValueError('Unknown initial grasp calibration')
    if grasp_setup not in ('legacy_staged','l2_closed_loop','table_supported_v1','l2_inhand_source_v1'):
        raise ValueError('Unknown grasp setup')
    if grasp_setup=='l2_inhand_source_v1':
        from evaluation.tasks.dining_place.l2_inhand_setup import PREOPEN
        if not staged_reset or calibration!='legacy' or any(seat_shift) or grasp_azimuth_deg:
            raise ValueError('L2 source initialization requires legacy calibration, staged reset and no extra pose tuning')
        aperture=PREOPEN
    if grasp_setup in ('l2_closed_loop','table_supported_v1') and (not staged_reset or calibration!='native_rim' or aperture<.05):
        raise ValueError('L2 closed-loop setup requires staged reset, native rim and a preopened jaw')
    if grasp_setup=='table_supported_v1' and grasp_azimuth_deg!=0:
        raise ValueError('Table-supported candidate requires the audited front grasp')
    root = Path(__file__).resolve().parents[2]
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError(out)
    if not 0 <= settle_aperture <= aperture <= .62 or not all(math.isfinite(v) and abs(v) <= .05 for v in seat_shift):
        raise ValueError('Invalid initial aperture or local seat translation')
    out.mkdir(parents=True)
    env = out/'environment/scene_room.usda'
    environment(base, env, seed=7, appearance=False)
    assets = root/'evaluation/tasks/microwave_take/resources'
    from evaluation.tasks.microwave_take.initial_state import PLATE_LAYER
    from arena_api.beef_asset import DIRECTORY as BEEF_ASSET_DIR
    layer = PLATE_LAYER
    stage = Usd.Stage.CreateNew(str(out/'initial.usda'))
    stage.GetRootLayer().subLayerPaths = [os.path.relpath(layer,out), os.path.relpath(env,out)]
    UsdGeom.SetStageMetersPerUnit(stage,1)
    UsdGeom.SetStageUpAxis(stage,'Z')
    stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
    source_hashes = {l.realPath: hashlib.sha256(Path(l.realPath).read_bytes()).hexdigest()
                     for l in stage.GetUsedLayers() if l.realPath and Path(l.realPath) != out/'initial.usda'}
    manifest,motion=load_reference()
    source=initial_reference(motion)
    surface=inspect_destination(stage)
    candidate=task.height_candidate(source,surface['support_top_m'],surface['lift_limits_m'])
    joints=dict(candidate['arm_targets_rad'])
    joints.update({k:v['value'] for k,v in candidate['stationary_joint_targets'].items()})
    # Start just open enough for the rim; native PD closes during unscored setup.
    joints.update(idx41_gripper_l_outer_joint1=.62,idx31_gripper_l_inner_joint1=-.62,
                  idx81_gripper_r_outer_joint1=aperture,idx71_gripper_r_inner_joint1=-aperture)
    robot_world=Gf.Matrix4d(1)
    table_min,table_max=surface['world_bounds_m']
    # West long edge, on the clear southern half of the table. The scene and
    # microwave remain fixed. Bind this station to the measured tabletop frame.
    robot_world.SetTranslate(Gf.Vec3d(table_min[0]-.60,table_min[1]+.68,0))
    robot=UsdGeom.Xformable(stage.GetPrimAtPath(task.ROBOT))
    robot.ClearXformOpOrder();robot.AddTransformOp().Set(robot_world);robot.SetResetXformStack(True)
    native_relative=native_rim_relative(grasp_azimuth_deg)
    ik=level_initial_wrist(stage,joints,robot_world,native_relative) if calibration=='native_rim' else None
    table_support=None
    if grasp_setup=='table_supported_v1':
        # Unscored physical preparation: current dining table supports the
        # filled centre of the current plate. Only robot initial pose changes.
        relative_for_station=Gf.Matrix4d(native_relative)
        relative_for_station.SetTranslateOnly(relative_for_station.ExtractTranslation()+Gf.Vec3d(*seat_shift))
        nominal=relative_for_station*body_transform(stage,HAND,joints,robot_world)
        finish=joints['idx01_body_joint1']
        start=finish+surface['support_top_m']+.0001-nominal.ExtractTranslation()[2]
        if not surface['lift_limits_m'][0]<=start<finish<=surface['lift_limits_m'][1]:
            raise ValueError('Table-supported preparation exceeds native lift range')
        dx=table_min[0]+.045-nominal.ExtractTranslation()[0]
        robot_world.SetTranslateOnly(robot_world.ExtractTranslation()+Gf.Vec3d(dx,0,0))
        robot.GetOrderedXformOps()[0].Set(robot_world)
        joints['idx01_body_joint1']=start
        table_support=dict(policy='arena.dining_place.table_supported.v1',path=task.DESTINATION,
            top_m=surface['support_top_m'],bounds_m=surface['world_bounds_m'],
            plate_bottom_clearance_m=.0001,plate_centre_inset_m=.045,
            start_lift_m=float(start),finish_lift_m=float(finish),lift_duration_s=2.5,
            note='Existing table support only during preparation; native PD lift, no object pose writes after seating')
    written={}
    for name,value in joints.items():
        prim=stage.GetPrimAtPath(task.ROBOT+'/joints/'+name)
        linear=prim.IsA(UsdPhysics.PrismaticJoint)
        native=value if linear else math.degrees(value)
        lo,hi=[prim.GetAttribute('physics:'+s+'Limit').Get() for s in ('lower','upper')]
        if not lo-1e-5 <= native <= hi+1e-5:
            raise ValueError('Joint outside native limits '+name)
        axis='linear' if linear else 'angular'
        state=PhysxSchema.JointStateAPI.Apply(prim,axis)
        state.CreatePositionAttr(native);state.CreateVelocityAttr(0)
        if prim.HasAPI(UsdPhysics.DriveAPI,axis):
            drive=UsdPhysics.DriveAPI(prim,axis)
            drive.CreateTargetPositionAttr(native);drive.CreateTargetVelocityAttr(0)
        written[name]=dict(value=value,unit='m' if linear else 'rad')
    hand=body_transform(stage,HAND,joints,robot_world)
    old=Gf.Matrix4d(*[v for row in manifest['historical_grasp']['plate_to_hand_row_matrix'] for v in row])
    az=Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0,1,0),manifest['historical_grasp']['grip_azimuth_deg']))
    # Explicit axis conversion only; current source geometry is never rotated
    # internally or scaled. New local +Z corresponds to legacy plate local +Y.
    axis=Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1,0,0),-90))
    relative=axis*az*old if calibration == 'legacy' else native_relative
    relative.SetTranslateOnly(relative.ExtractTranslation()+Gf.Vec3d(*seat_shift))
    plate_world=relative*hand
    def set_pose(path,matrix):
        prim=stage.GetPrimAtPath(path)
        if not prim:raise ValueError('Missing task body '+path)
        prim.SetActive(True)
        xf=UsdGeom.Xformable(prim);xf.ClearXformOpOrder();xf.AddTransformOp().Set(matrix);xf.SetResetXformStack(True)
    set_pose(PLATE_ROOT,plate_world)
    # Current composed beef-box surface and placement; retain its native mass.
    box_relative=Gf.Matrix4d().SetTranslate(Gf.Vec3d(.0219,0,.02607983112))
    set_pose(BOX,box_relative*plate_world)
    if staged_reset:
        # Empty-hand robot warmup, then one pre-episode dynamic-object reset.
        # Park on a clear part of the existing dining tabletop, not in mid-air.
        park=Gf.Matrix4d().SetTranslate(Gf.Vec3d((table_min[0]+table_max[0])/2,
            table_min[1]+.60,surface['support_top_m']+.001))
        set_pose(PLATE_ROOT,park)
        set_pose(BOX,box_relative*park)
    for name in ('MicrowaveFoodbox','MicrowaveSaladbox','MicrowavePayload'):
        p=stage.GetPrimAtPath('/World/Assets/'+name)
        if p:p.SetActive(False)
    for path in (PLATE,BOX):
        p=stage.GetPrimAtPath(path)
        body=UsdPhysics.RigidBodyAPI(p)
        if not body or body.GetKinematicEnabledAttr().Get() is True:
            raise ValueError('Expected dynamic current target '+path)
        body.CreateVelocityAttr(Gf.Vec3f(0));body.CreateAngularVelocityAttr(Gf.Vec3f(0))
    for p in Usd.PrimRange(stage.GetPrimAtPath(task.ROBOT)):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            body=UsdPhysics.RigidBodyAPI(p);body.CreateVelocityAttr(Gf.Vec3f(0));body.CreateAngularVelocityAttr(Gf.Vec3f(0))
        for a in ('linear','angular'):
            if p.HasAPI(PhysxSchema.JointStateAPI,a):PhysxSchema.JointStateAPI(p,a).CreateVelocityAttr(0)
    report=dict(schema='arena.dining_place.initial.v1',task_id='10.2',physics_validated=False,
        source_uuid=manifest['baseline_uuid'],source_candidate=manifest['candidate_id'],
        bindings=dict(robot=task.ROBOT,plate=PLATE,payload=BOX,hand=HAND,destination=task.DESTINATION),
        robot_station=pose(robot_world),surface=surface,source_initial=source,
        joint_targets=written,hand_fk=pose(hand),plate=pose(plate_world),payload=pose(box_relative*plate_world),
        plate_to_hand=[list(relative[i]) for i in range(4)],
        plate_axis_conversion=('current Z to historical Y: Rx(-90deg)' if calibration=='legacy'
                               else 'current Z-up rim tangent calibrated against native hand X and +Z'),
        initial_gripper_aperture_rad=aperture,settle_gripper_target_rad=settle_aperture,seat_shift_hand_m=list(seat_shift),
        grasp_calibration=calibration,
        grasp_setup=grasp_setup,
        table_support=table_support,
        grasp_azimuth_deg=grasp_azimuth_deg,
        initial_wrist_ik=ik,
        staged_reset=staged_reset,robot_warmup_steps=120 if staged_reset else 0,
        payload_to_plate=[list(box_relative[i]) for i in range(4)],
        assets=dict(plate=str(assets/'black_plate_325mm_filled_recess_v12.usda'),
                    payload_contact=str(BEEF_ASSET_DIR/'beefbox_contact_standard243_v1.usda')),
        boundary=('One pre-episode object reset after empty-hand warmup; no pose writes during grasp validation or model execution'
                  if staged_reset else 'poses authored only before physics; native PD during settle; no attachment or runtime pose writes'),
        source_layer_sha256=source_hashes)
    stage.GetRootLayer().customLayerData={'arenaTask':'10.2','initialPlanJson':json.dumps(report,sort_keys=True),'physicsValidated':False}
    stage.GetRootLayer().Save()
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in source_hashes.items()):
        raise RuntimeError('Shared source layer changed')
    report['scene_sha256']=hashlib.sha256((out/'initial.usda').read_bytes()).hexdigest()
    (out/'initial.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--aperture',type=float,default=.065)
    p.add_argument('--settle-aperture',type=float,default=.025)
    p.add_argument('--calibration',choices=('native_rim','legacy'),default='native_rim')
    p.add_argument('--staged-reset',action='store_true')
    p.add_argument('--grasp-setup',choices=('legacy_staged','l2_closed_loop','table_supported_v1','l2_inhand_source_v1'),default='legacy_staged')
    p.add_argument('--grasp-azimuth-deg',type=float,default=0.)
    p.add_argument('--seat-shift',type=float,nargs=3,default=(0,0,0))
    a=p.parse_args();r=prepare(a.base,a.output,aperture=a.aperture,seat_shift=a.seat_shift,
                            settle_aperture=a.settle_aperture,calibration=a.calibration,staged_reset=a.staged_reset,
                            grasp_azimuth_deg=a.grasp_azimuth_deg,grasp_setup=a.grasp_setup)
    print(json.dumps({k:r[k] for k in ('robot_station','hand_fk','plate','payload','physics_validated')},indent=2))
