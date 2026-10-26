"""Actual all_assets geometry to canonical support-pivot sampling; pose-only layers."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom, Gf
from .geometry_kernel import sample_geometry, TOOLS

HERE=Path(__file__).resolve().parent
HOLDER='/World/RoomEpisode/CutleryHolder/Asset/Object'
ROOTS={name:'/World/RoomEpisode/Cutlery_'+name for name in TOOLS}
# Column matrices from scene_initial.usda, relative to its Holder.
CANONICAL_TRANSLATIONS=((-0.0874572,-0.0068081,0.13721943316211704),
                        (-0.0006918,-0.0068081,0.12267706212749485),
                        (0.0862805,-0.0068081,0.12267706212749485))

def geometry(stage):
    cache=UsdGeom.XformCache()
    bounds=UsdGeom.BBoxCache(0,['default','render'],False,True)
    walls_cache=UsdGeom.BBoxCache(0,['default','guide'],False,True)
    holder=stage.GetPrimAtPath(HOLDER)
    if not holder:raise ValueError('Missing actual holder')
    roots,points,corners,cavities={},{},{},[]
    for index,name in enumerate(TOOLS):
        prim=stage.GetPrimAtPath(ROOTS[name]+'/Asset/Object')
        if not prim:raise ValueError('Missing actual utensil')
        root=np.eye(4)
        root[:3,:3]=[[1,0,0],[0,0,1],[0,-1,0]]
        root[:3,3]=CANONICAL_TRANSLATIONS[index]
        inv=np.linalg.inv(np.array(cache.GetLocalToWorldTransform(prim)))
        chunks=[]
        for mesh in Usd.PrimRange(prim,Usd.TraverseInstanceProxies()):
            if not mesh.IsA(UsdGeom.Mesh):continue
            if UsdGeom.Imageable(mesh).ComputePurpose() not in ('default','render'):continue
            v=np.array(mesh.GetAttribute('points').Get(),dtype=float)
            transform=np.array(cache.GetLocalToWorldTransform(mesh))@inv
            local=(np.c_[v,np.ones(len(v))]@transform)[:,:3]
            chunks.append(local@root[:3,:3].T+root[:3,3])
        if not chunks:raise ValueError('No utensil mesh')
        roots[name]=root
        points[name]=np.concatenate(chunks)
        # Build the canonical root-frame visual AABB. Asking BBoxCache for the
        # already-tilted authored Object can align in world axes and inflate it
        # when transformed back, which is not the source canonical proxy.
        local=(points[name]-root[:3,3])@root[:3,:3]
        lo,hi=local.min(0),local.max(0)
        corners[name]=np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
        walls={}
        for side in ('Left','Right','Front','Back'):
            p=stage.GetPrimAtPath(HOLDER+f'/Collision/Slot_{index}_{side}')
            if not p:raise ValueError('Missing cavity wall')
            r=walls_cache.ComputeRelativeBound(p,holder).ComputeAlignedRange()
            walls[side]=(np.array(r.GetMin()),np.array(r.GetMax()))
        cavities.append(dict(index=index,
            inner_lo_xy_m=np.array([walls['Left'][1][0],walls['Front'][1][1]]),
            inner_hi_xy_m=np.array([walls['Right'][0][0],walls['Back'][0][1]]),
            rim_z_m=float(min(walls[s][1][2] for s in walls))))
    return dict(roots=roots,points=points,root_bound_corners=corners,cavities=cavities,
        scene_path=stage.GetRootLayer().realPath,
        scene_sha256=hashlib.sha256(Path(stage.GetRootLayer().realPath).read_bytes()).hexdigest())

def verify_fixture(g):
    fixture=json.loads((HERE/'source_geometry_fixture.json').read_text())
    replay=sample_geometry(g,fixture['policy'])
    errors={n:float(np.max(np.abs(np.array(replay['tools'][n]['root_relative'])-
                  np.array(fixture['geometry']['tools'][n]['root_relative'])))) for n in TOOLS}
    if max(errors.values())>1e-7:
        raise ValueError('Actual assets differ from canonical source geometry: '+str(errors))
    return errors

def sample(stage,*,seed,family_index):
    if type(seed) is not int or not 1<=seed<=4294967295:raise ValueError('Positive uint32 seed required')
    families=json.loads((HERE/'source_pose_families.json').read_text())['families']
    if type(family_index) is not int or not 0<=family_index<len(families):
        raise ValueError('Explicit valid source family required')
    g=geometry(stage)
    errors=verify_fixture(g)
    policy=copy.deepcopy(json.loads((HERE/'source_geometry_fixture.json').read_text())['policy'])
    spec=families[family_index]['spec']
    policy['pose_seed']=int.from_bytes(hashlib.sha256(f'cutlery-pose-v1:{seed}:{family_index}'.encode()).digest()[:8],'little')
    policy['reachable_region']={
        'schema':'ziche.meal_cutlery50.accepted_template_neighborhood.v1',
        'anchor_id':spec['template_id'],
        'tools':copy.deepcopy(spec['tools']),
        'halfwidths':copy.deepcopy(spec['halfwidths'])}
    result=sample_geometry(g,policy)
    result.update(source_family_index=family_index,seed=seed,source_replay_matrix_errors=errors,
        native_physics_validated=False)
    return result

def select_family(seed):
    if type(seed) is not int or not 1<=seed<=4294967295:
        raise ValueError('Positive uint32 seed required')
    digest=hashlib.sha256(f'cutlery-equal-family-v1:{seed}'.encode()).digest()
    return int(np.random.default_rng(int.from_bytes(digest,'little')).integers(0,3))

def prepare_manifest(manifest, output):
    from . import environment_initial as environment
    value=copy.deepcopy(manifest)
    environment_path=Path(output).resolve().with_name('environment.usda')
    from short_tasks.tasks.robot_pose_alignment_v1.temporary_outputs import permits
    if environment_path.exists() or not (environment_path.is_relative_to(HERE) or permits(environment_path)):
        raise ValueError('Fresh task-owned environment layer required')
    environment_path.parent.mkdir(parents=True,exist_ok=True)
    original=Usd.Stage.Open(value['scene'])
    layer=Usd.Stage.CreateNew(str(environment_path))
    layer.GetRootLayer().subLayerPaths=[value['scene']]
    UsdGeom.SetStageMetersPerUnit(layer,UsdGeom.GetStageMetersPerUnit(original))
    UsdGeom.SetStageUpAxis(layer,UsdGeom.GetStageUpAxis(original))
    environment_receipt=environment.apply(layer,value,value['seed'])
    environment_receipt['geometry']=environment.validate(layer,value)
    layer.GetRootLayer().Save()
    environment_receipt.update(scene=str(environment_path),base_scene=value['scene'])
    value['cutlery_environment_initialization']=environment_receipt
    value['supports']['tray']['origin']=environment_receipt['actual_tray_origin_xy']
    value['scene']=str(environment_path)
    family=select_family(value['seed'])
    receipt=write_layer(value['scene'],output,seed=value['seed'],family_index=family)
    stage=Usd.Stage.Open(receipt['scene']);cache=UsdGeom.XformCache()
    from .audit_holder_cavities import points as visual_points
    for obj in value['objects']:
        if obj['path'] not in (*ROOTS.values(),environment.TRAY,environment.HOLDER_ROOT):continue
        v=visual_points(stage,obj['path'],np.eye(4),cache)
        lo,hi=v.min(0),v.max(0)
        obj['visible_bounds_world_m']=[lo.tolist(),hi.tolist()]
        obj['polygon']=[[lo[0],lo[1]],[hi[0],lo[1]],[hi[0],hi[1]],[lo[0],hi[1]]]
    value['scene']=receipt['scene'];value['scene_sha256']=receipt['scene_sha256']
    environment_receipt['final_geometry']=environment.validate(stage,value)
    receipt['family_selection']='equal_probability_three_source_neighborhoods_v1'
    value['cutlery_object_initialization']=receipt
    value['native_physics_validated']=False
    # Earlier composition audits describe the pre-randomization base scene.
    receipt['prior_composition_audit_scope']='base_scene_before_utensil_pose_overrides'
    refs=[Path(__file__),HERE/'geometry_kernel.py',HERE/'source_pose_families.json',
        HERE/'source_geometry_fixture.json',Path(receipt['scene']),environment_path,
        Path(environment.__file__),environment.RANGES]
    value['source_hashes'].update({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in refs})
    return value

def write_layer(scene,output,*,seed,family_index):
    scene,output=Path(scene).resolve(strict=True),Path(output).resolve()
    from short_tasks.tasks.robot_pose_alignment_v1.temporary_outputs import permits
    if output.exists() or not (output.is_relative_to(HERE) or permits(output)):
        raise ValueError('Fresh task-owned output required')
    original=Usd.Stage.Open(str(scene));receipt=sample(original,seed=seed,family_index=family_index)
    output.parent.mkdir(parents=True,exist_ok=True)
    layer=Usd.Stage.CreateNew(str(output))
    layer.GetRootLayer().subLayerPaths=[str(scene)]
    UsdGeom.SetStageMetersPerUnit(layer,UsdGeom.GetStageMetersPerUnit(original))
    UsdGeom.SetStageUpAxis(layer,UsdGeom.GetStageUpAxis(original))
    cache=UsdGeom.XformCache()
    world=lambda p:np.array(cache.GetLocalToWorldTransform(original.GetPrimAtPath(p))).T
    h=world(HOLDER)
    changes={}
    for name in TOOLS:
        path=ROOTS[name];obj=path+'/Asset/Object'
        desired_object=h@np.array(receipt['tools'][name]['root_relative'])
        desired_root=desired_object@np.linalg.inv(world(obj))@world(path)
        parent=str(original.GetPrimAtPath(path).GetParent().GetPath())
        local=np.linalg.inv(world(parent))@desired_root
        prim=layer.OverridePrim(path);x=UsdGeom.Xformable(prim)
        x.ClearXformOpOrder()
        x.AddTransformOp(opSuffix='cutleryInitial').Set(Gf.Matrix4d(local.T.tolist()))
        changes[path]=local.tolist()
    layer.GetRootLayer().Save()
    receipt.update(scene=str(output),base_scene=str(scene),pose_only_overrides=changes,
        scene_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    return receipt
