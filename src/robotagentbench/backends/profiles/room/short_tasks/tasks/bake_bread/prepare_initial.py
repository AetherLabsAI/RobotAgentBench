"""Approved fryer reset sampling, before physics; no score or drive edits.

Composition owns the four object poses. The public preparation hook owns the
source-range robot station. Both are seeded and checked in the actual room.
"""
import hashlib
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import numpy as np

HERE = Path(__file__).resolve().parent
POLICY = HERE / 'initial_policy.json'
ATOMIC_POSE = HERE / 'atomic_pose_source.json'
ATOMIC_POSE_SHA256 = '87d5ffb717ecd43df00860791335e5ad5b8f4378ab5f90cd6dba91c9bf2fbe4d'
ASSETS = ('air_fryer', 'paper_plate_task2', 'cake_plain_yellow', 'cake_frosted_white')


class NoFeasibleInitial(ValueError):
    pass


def policy():
    return json.loads(POLICY.read_text())


def robot_pose():
    """User-selected atomic basket-pull reset; copy joints, not its scene."""
    from short_tasks.tasks.robot_pose_alignment_v1.postures import ACTIVE_NAMES
    content=ATOMIC_POSE.read_bytes()
    if hashlib.sha256(content).hexdigest()!=ATOMIC_POSE_SHA256:
        raise ValueError('Atomic fryer source posture changed')
    source=json.loads(content)
    if source['candidate_benchmark_id']!='2.3' or source['source_recording_label']!='66':
        raise ValueError('Atomic basket-pull identity mismatch')
    q=source['robot_initialization']['positions']
    if set(q)!=set(ACTIVE_NAMES) or not all(math.isfinite(v) for v in q.values()):
        raise ValueError('Exactly twenty finite source joint coordinates required')
    if q!=source['robot_initialization']['hold_targets']:
        raise ValueError('Different source reset and hold targets require review')
    return dict(targets_rad=dict(q),posture_source='atomic_basket_pull:task66_initial_v1',
        source_episode=source['source_episode'],source_benchmark_id='2.3',
        source_manifest_sha256=ATOMIC_POSE_SHA256,complete_source_joint_tuple=True,
        joint_coordinates_independently_sampled=False,joint_coordinates_clipped=False,
        source_selection='user_selected_atomic_basket_pull_fixed_initial',
        active20_source_exact=True,native_physics_validated=False)


def posture_references():
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),POLICY,ATOMIC_POSE)}


def align_posture(stage,manifest,plan,receipt,*,seed):
    """Retain this task's explicit atomic posture at the final generic gate."""
    from short_tasks.tasks.robot_pose_alignment_v1.geometry import checker, NoFeasibleSourcePosture
    if plan['task_name']!='bake_bread':raise ValueError('Fryer-only posture hook')
    out=deepcopy(receipt);pose=robot_pose()
    base=[*out['position'][:2],out['yaw_rad']]
    failure=checker(stage,manifest,base)(pose)
    if failure:raise NoFeasibleSourcePosture('Atomic basket-pull initial rejected: '+failure)
    out['targets_rad']=pose['targets_rad']
    out['robot_posture_alignment']=dict(pose,base_xyyaw_preserved=base,
        station_ranges_changed=False,object_transforms_changed=False,
        geometry_rechecked_with_source_joints=True,robot_self_collision_validated=False)
    out['candidate']['targets_rad']=pose['targets_rad']
    out['candidate']['sampling']=pose['source_selection']
    out['posture_source']=pose['posture_source']
    out.setdefault('reference_hashes',{}).update(posture_references())
    return out


def object_samples(seed, count=10000):
    """Existing preview budgets/ranges; preserve rejected proposals as evidence."""
    p = policy()
    rng = random.Random(f'fryer-initial:{seed}')
    for attempt in range(count):
        def sample(section, prefix=''):
            q = p[section]
            return [rng.uniform(*q[prefix + k]) for k in
                    ('translation_x_m','translation_y_m','yaw_delta_deg')]
        yield dict(attempt=attempt, fryer=sample('fryer_randomization'),
            plate=sample('plate_randomization'),
            cakes=[sample('cake_randomization','per_cake_') for _ in range(2)],
            target_slot=random.Random(f'fryer-cake-slots:{seed}').choice(
                p['cake_randomization']['target_slot_choices']))


def transform(stage, path):
    from pxr import UsdGeom
    return UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path))


def set_world(stage, path, matrix):
    from pxr import UsdGeom
    p = stage.GetPrimAtPath(path)
    if not p:
        raise ValueError('Missing initial object: ' + path)
    parent = UsdGeom.XformCache().GetParentToWorldTransform(p)
    UsdGeom.Xformable(p).MakeMatrixXform().Set(matrix * parent.GetInverse())


def delta(center, translation, yaw):
    from pxr import Gf
    c = Gf.Vec3d(*map(float, center))
    return (Gf.Matrix4d(1).SetTranslate(-c) *
            Gf.Matrix4d(1).SetRotate(Gf.Rotation(Gf.Vec3d(0,0,1), float(yaw))) *
            Gf.Matrix4d(1).SetTranslate(c + Gf.Vec3d(*map(float,translation))))


def support_top_at(stage, support, xy):
    """Highest horizontal face under the plate centre; same geometry tolerances.

    Full footprint coverage and foreign obstacles are still checked separately.
    No polygon extraction dependency is needed by the reset worker.
    """
    from pxr import Usd, UsdGeom
    from openend.tasks.scene_requirements.surface_registry import _mesh_triangles
    from openend.tasks.scene_requirements.station_scene_checks import EPS
    heights=[];cache=UsdGeom.XformCache()
    for prim in Usd.PrimRange(stage.GetPrimAtPath(support)):
        image=UsdGeom.Imageable(prim)
        if image and image.ComputeVisibility()=='invisible':continue
        triangles=_mesh_triangles(prim,cache)
        if triangles is None:continue
        for t in triangles:
            if np.ptp(t[:,2])>1e-5:continue
            a=np.column_stack((t[1,:2]-t[0,:2],t[2,:2]-t[0,:2]))
            if abs(np.linalg.det(a))<=1e-12:continue
            u,v=np.linalg.solve(a,np.asarray(xy)-t[0,:2])
            if u>=-EPS and v>=-EPS and u+v<=1+EPS:
                heights.append(float(t[:,2].max()))
    if not heights:raise ValueError('No horizontal support below plate centre')
    return max(heights)


def compose(stage, objects, placement, layout, *, seed, variant):
    """Only the fresh composed episode can invoke this reset hook.

    Align the small plate and both cakes together to the measured worktop.
    This is an approved reset-only transform, not a physical-asset override.
    """
    from pxr import Gf, UsdGeom
    from short_tasks.agent_probe_v1.fryer_cake_slots import finalize_slots
    from openend.tasks.scene_requirements.surface_registry import collision_vertices, check_station_footprints, surface_set
    rows = {}
    for asset in ASSETS:
        matching = [o for o in objects if o['id'] == asset]
        if len(matching) != 1:
            raise ValueError('Unique fryer object required: ' + asset)
        rows[asset] = matching[0]
    if variant not in ('sugar','no_sugar'):
        raise ValueError('Explicit cake variant required')
    paths = {k:v['path'] for k,v in rows.items()}
    unaligned = {k:transform(stage,v) for k,v in paths.items()}
    surfaces=surface_set(layout['surfaces'],placement['support'])
    heights=[]
    for surface in surfaces:
        try:
            heights.append((support_top_at(stage,surface['path'],
                np.asarray(unaligned['paper_plate_task2'].ExtractTranslation())[:2]),surface['path']))
        except ValueError as exc:
            if str(exc)!='No horizontal support below plate centre':raise
    if not heights:
        raise NoFeasibleInitial('Approved fryer object proposals exhausted: no registered support below plate')
    top,plate_support=max(heights)
    plate_bottom = float(collision_vertices(stage, paths['paper_plate_task2'])[:,2].min())
    dz = top - plate_bottom
    for asset in ASSETS[1:]:
        set_world(stage,paths[asset],unaligned[asset]*Gf.Matrix4d(1).SetTranslate(Gf.Vec3d(0,0,dz)))
    originals = {k:transform(stage,v) for k,v in paths.items()}
    front = np.asarray(placement['front'],float)
    front /= np.linalg.norm(front)
    left = np.array([-front[1], front[0]])
    # Workspace coordinates have forward +X and left +Y, independent of layout.
    def translation(x,y):
        xy = x*front+y*left
        return [float(xy[0]),float(xy[1]),0.]
    plate = paths['paper_plate_task2']
    pcenter = np.asarray(originals['paper_plate_task2'].ExtractTranslation())
    base_slots = []
    for asset in ASSETS[2:]:
        origin = np.asarray(originals[asset].ExtractTranslation())
        offset = origin - pcenter
        base_slots.append([float(offset[:2]@front),float(offset[:2]@left),float(origin[2])])
    # Scene assembly already contains the historical second-slot shift. Undo
    # it only in the input representation; finalize_slots reapplies it once.
    legacy = policy()['cake_randomization']['legacy_second_slot_offset_workspace_xyz_m']
    base_slots[1] = [a-b for a,b in zip(base_slots[1],legacy)]
    failures = {}
    for proposal in object_samples(seed):
        for asset,path in paths.items():
            set_world(stage,path,originals[asset])
        f,p = proposal['fryer'],proposal['plate']
        fc = originals['air_fryer'].ExtractTranslation()
        set_world(stage,paths['air_fryer'],originals['air_fryer']*delta(fc,translation(*f[:2]),f[2]))
        pd = delta(pcenter,translation(*p[:2]),p[2])
        set_world(stage,plate,originals['paper_plate_task2']*pd)
        new_pc = np.asarray(transform(stage,plate).ExtractTranslation())
        angle=math.radians(p[2]);co,si=math.cos(angle),math.sin(angle)
        sampled=[]
        for b,j in zip(base_slots,proposal['cakes']):
            sampled.append(dict(position=[co*b[0]-si*b[1]+j[0],si*b[0]+co*b[1]+j[1],b[2]],
                                yaw_deg=p[2]+j[2]))
        try:
            slots=finalize_slots(dict(stage='sampled_before_legacy_slot_offset',
                plate_center_xyz=[0.,0.,new_pc[2]],slots=sampled),seed=seed,target_slot=proposal['target_slot'])
        except ValueError as exc:
            failures[str(exc)]=failures.get(str(exc),0)+1
            continue
        target_asset=ASSETS[3] if variant=='sugar' else ASSETS[2]
        for asset in ASSETS[2:]:
            s=slots['target' if asset==target_asset else 'distractor']
            old=np.asarray(originals[asset].ExtractTranslation())
            new=new_pc.copy();new[:2]+=s['position'][0]*front+s['position'][1]*left;new[2]=s['position'][2]
            set_world(stage,paths[asset],originals[asset]*delta(old,new-old,s['yaw_deg']))
        points={k:collision_vertices(stage,v) for k,v in paths.items()}
        boxes={k:np.array([v.min(0),v.max(0)]) for k,v in points.items()}
        a,b=boxes['air_fryer'],boxes['paper_plate_task2']
        gap=float(np.linalg.norm(np.maximum(np.maximum(a[0,:2]-b[1,:2],b[0,:2]-a[1,:2]),0)))
        edge=np.asarray(placement['edge_point'])
        edge_gap=float(edge@front-(points['air_fryer'][:,:2]@front).max())
        reason=None
        if gap < policy()['plate_randomization']['minimum_planar_aabb_gap_to_whole_fryer_m']:
            reason='plate_fryer_gap'
        elif not policy()['fryer_randomization']['countertop_front_edge_gap_m'][0] <= edge_gap <= policy()['fryer_randomization']['countertop_front_edge_gap_m'][1]:
            reason='fryer_front_edge_gap'
        if reason:
            failures[reason]=failures.get(reason,0)+1;continue
        group=dict(id='fryer',support=placement['support'])
        # Only objects resting on the worktop enter the support containment
        # test; cake-on-plate containment is the original finalize_slots gate.
        gate=check_station_footprints(stage,layout,group,[points['air_fryer'],points['paper_plate_task2']])
        if not gate['passed']:
            reason=str(gate['reason']);failures[reason]=failures.get(reason,0)+1;continue
        # Preserve source all_assets physical attributes: only wrapper transforms
        # changed, not their children, collision settings or mass.
        for row in rows.values():row['initial_transform_preserved']=False
        return dict(schema='short_tasks.fryer_initial.v1',seed=seed,variant=variant,
            proposal=proposal,slots=slots,plate_fryer_gap_m=gap,front_edge_gap_m=edge_gap,
            rejections=failures,paths=paths,object_world_matrices={k:np.asarray(transform(stage,v)).tolist() for k,v in paths.items()},
            original_world_matrices={k:np.asarray(v).tolist() for k,v in unaligned.items()},
            whole_fryer_moved_as_one=True,legacy_slot_offset_applied_once=True,
            support_alignment=dict(support=plate_support,station_anchor=placement['support'],top_m=top,
                previous_plate_bottom_m=plate_bottom,translation_z_m=dz,
                moved_together=list(ASSETS[1:]),relative_support_poses_preserved=True),
            physics_parameters_changed=False,
            native_physics_validated=False,support_check=gate)
    for asset,path in paths.items():set_world(stage,path,unaligned[asset])
    raise NoFeasibleInitial('Approved fryer object proposals exhausted: '+json.dumps(failures))


def choose(stage, manifest, *, seed, count=200):
    from pxr import Gf
    from short_tasks.tasks.robot_pose_alignment_v1.geometry import checker
    receipt=manifest.get('fryer_initialization')
    if not receipt or receipt.get('seed')!=seed:
        raise ValueError('Fryer composition must apply its task-local reset first')
    path=receipt['paths']['air_fryer']+'/Asset'
    matrix=transform(stage,path);origin=np.asarray(matrix.ExtractTranslation())
    # Source-measured bearing from the previous accepted fryer preview.
    direction=np.asarray(matrix.TransformDir(Gf.Vec3d(.8533573851891723,-.05060100460763106,0)))[:2]
    direction/=np.linalg.norm(direction)
    p=policy();rng=random.Random(f'fryer-station:{seed}');rejected=[]
    pose=robot_pose()
    for index in range(count):
        distance=rng.uniform(*p['horizontal_distance_m']['inclusive_range'])
        heading=rng.uniform(*p['heading']['absolute_error_deg_inclusive_range'])
        xy=origin[:2]+direction*distance
        yaw=math.atan2(origin[1]-xy[1],origin[0]-xy[0])+math.radians(heading)
        base=[float(xy[0]),float(xy[1]),yaw]
        failure=checker(stage,manifest,base)(pose)
        if failure:
            rejected.append(dict(index=index,reason=failure));continue
        refs=posture_references()
        return dict(position=[*base[:2],0.],yaw_rad=yaw,targets_rad=pose['targets_rad'],
            candidate=dict(sampling=pose['source_selection'],initial_base_xyyaw=base),
            method='approved_fryer_relative_station_actual_room_fk',
            robot_distance_m=distance,robot_heading_error_deg=heading,
            target=dict(paths=[path],source='fryer',reference_files=list(refs)),
            object_initialization=receipt,reference_hashes=refs,rejections=rejected,
            robot_posture_alignment=pose,
            robot_room_geometry_checked=True,robot_self_collision_validated=False,native_physics_validated=False)
    raise NoFeasibleInitial('Fryer station candidates exhausted: '+json.dumps(rejected))


def apply_target(stage, receipt):
    # Object randomization already happened in composition, before its checks.
    if receipt.get('object_initialization',{}).get('legacy_slot_offset_applied_once') is not True:
        raise ValueError('Missing finalized fryer initial slots')


def refresh_manifest(stage, manifest, receipt):
    manifest['fryer_station_initialization']=receipt
