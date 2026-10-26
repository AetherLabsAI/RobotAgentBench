"""Scene-owned L1 four-slot initialization using current shared box assets.

Only invoked before simulation. No dock, velocity writes, gravity/collision
switches, door drive changes, or old L1 reset imports. Static placement is not
a native settling certificate. Coordinates are in the Y-up right-door frame.
"""
import hashlib
import json
import math
import random
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
API = HERE.parents[1] / 'api-protocol'
DOOR = '/World/Background/Shell/RefrigeratorRecovered/right_door'
OWNER = 'unified_scene_fridge_boxes_v1'
SLOTS = {'low_out': (.113, 'low'), 'low_in': (.304, 'low'),
         'up_out': (.100, 'up'), 'up_in': (.300, 'up')}
RAW = {'low': [-.617, .326, .344, .628], 'up': [-.536, .332, .411, .658]}
Y = {'low': .815, 'up': 1.168}
ASSETS = {
    'beef': ('beef_box', 'beef_box_standard243_v1/beefbox.usda', '/World/Assets/MicrowaveBeefbox'),
    'salad': ('salad_box', 'fridge_food_boxes_v1/saladbox.usda', '/World/Assets/SaladBoxR/Asset'),
    'food': ('leftover_box', 'fridge_food_boxes_v1/foodbox.usda', '/World/Assets/BinFoodR/Asset'),
}


def mul(a, b):
    w,x,y,z=a; v,i,j,k=b
    return [w*v-x*i-y*j-z*k,w*i+x*v+y*k-z*j,w*j-x*k+y*v+z*i,w*k+x*j-y*i+z*v]


def axis(i, deg):
    h=math.radians(deg)/2; q=[math.cos(h),0.,0.,0.]; q[i+1]=math.sin(h); return q


def normalize(q):
    n=math.sqrt(sum(v*v for v in q)); return [v/n for v in q]


@lru_cache(maxsize=2)
def lean(level):
    q=RAW[level]; lo,hi=-40.,60.
    for _ in range(60):
        mid=(lo+hi)/2; w,x,y,z=normalize(mul(axis(0,-mid),q))
        angle=math.degrees(math.asin(min(1.,abs(2*(x*y+z*w)))))
        if angle>14.: lo=mid
        else: hi=mid
    return normalize(mul(axis(0,-(lo+hi)/2),q))


@lru_cache(maxsize=2)
def straight(level):
    lo,hi=-60.,60.
    for _ in range(60):
        mid=(lo+hi)/2; w,x,y,z=normalize(mul(axis(1,mid),lean(level)))
        ax=1-2*(y*y+z*z); az=2*(x*z-y*w)
        if (-ax if az>0 else ax)>0: lo=mid
        else: hi=mid
    return (lo+hi)/2


def sample(seed, lengths):
    if type(seed) is not int: raise ValueError('An explicit integer room seed is required')
    rng=random.Random(seed); slots=rng.sample(list(SLOTS),3); rows=[]
    for key,slot in zip(ASSETS,slots):
        x,level=SLOTS[slot]; length=float(lengths[key])
        if not math.isfinite(length) or length<=0: raise ValueError('Invalid current asset length')
        x=round(x+rng.uniform(-.03,.03),3)
        z=round(.354-length/2-.005-.010+rng.uniform(-.006,.006),3)
        yaw=round(straight(level)+rng.uniform(0,20),1)
        pitch=round(rng.uniform(-20,20),1); roll=round(rng.uniform(-12,12),1)
        q=mul(axis(2,roll*.25),lean(level));q=mul(axis(0,pitch*.30),q);q=normalize(mul(axis(1,yaw),q))
        rows.append(dict(key=key,slot=slot,door_local_position_m=[x,Y[level],z],
                         door_local_quaternion_wxyz=q,yaw_deg=yaw,
                         tilt_jitter_deg=pitch*.30,roll_jitter_deg=roll*.25,
                         current_asset_length_m=length))
    return rows


def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def bind_materials(stage):
    """Restore original world-scoped bindings lost by subtree references."""
    from pxr import Usd,UsdPhysics
    from arena_api import food_box_assets
    food_box_assets.verify_files()
    for key,(_,rel,path) in ASSETS.items():
        if key=='beef' or not stage.GetPrimAtPath(path):continue
        asset=Usd.Stage.Open(str(API/'resources/shared_assets'/rel));root=asset.GetDefaultPrim()
        for p in Usd.PrimRange(root):
            reln=p.GetRelationship('material:binding:physics')
            targets=reln.GetTargets() if reln else []
            if not targets:continue
            for target in targets:
                actual=stage.GetPrimAtPath(target)
                if not actual or not actual.HasAPI(UsdPhysics.MaterialAPI):
                    raise ValueError('Existing shared room material missing '+str(target))
            suffix=str(p.GetPath())[len(str(root.GetPath())):]
            stage.GetPrimAtPath(path+suffix).CreateRelationship('material:binding:physics').SetTargets(targets)


def apply(stage, *, seed):
    from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf
    from arena_api import beef_asset, food_box_assets
    beef_asset.verify_files(); food_box_assets.verify_files()
    door=stage.GetPrimAtPath(DOOR)
    if not door or not door.IsActive(): raise ValueError('Missing current room right door')
    sources={}; lengths={}
    for key,(_,rel,path) in ASSETS.items():
        source=API/'resources/shared_assets'/rel
        native=Usd.Stage.Open(str(source));root=native.GetDefaultPrim()
        if not root or not root.HasAPI(UsdPhysics.RigidBodyAPI): raise ValueError('Missing source rigid body')
        bounds=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render']).ComputeWorldBound(root).ComputeAlignedRange()
        # Source geometry is baked in metres; do not apply old L1 instance scaling.
        lengths[key]=float(bounds.GetSize()[0]);sources[key]=(source,native)
        old=stage.GetPrimAtPath(path)
        if old and old.GetCustomDataByKey('roomInitializationOwner')!=OWNER:
            raise ValueError('Target path already occupied: '+path)
    cache=UsdGeom.XformCache();door_world=cache.GetLocalToWorldTransform(door)
    if abs(door_world.GetDeterminant()-1)>1e-6: raise ValueError('Door frame must be rigid, scale=1')
    rows=[]
    for row in sample(seed,lengths):
        key=row['key']; asset_id,_,path=ASSETS[key]; source,native=sources[key]
        p=stage.DefinePrim(path,'Xform');p.GetReferences().SetReferences([Sdf.Reference(str(source))])
        p.SetCustomDataByKey('roomAssetId',asset_id)
        p.SetCustomDataByKey('roomInitializationOwner',OWNER)
        q=row['door_local_quaternion_wxyz']
        local=Gf.Matrix4d().SetRotate(Gf.Quatd(q[0],Gf.Vec3d(*q[1:])))
        local.SetTranslateOnly(Gf.Vec3d(*row['door_local_position_m']))
        parent=UsdGeom.XformCache().GetLocalToWorldTransform(p.GetParent())
        xf=UsdGeom.Xformable(p);xf.MakeMatrixXform().Set(local*door_world*parent.GetInverse())
        # References remain dynamic and independent of the door hierarchy.
        assert p.HasAPI(UsdPhysics.RigidBodyAPI)
        assert p.GetAttribute('physics:kinematicEnabled').Get() is not True
        assert p.GetAttribute('physxRigidBody:disableGravity').Get() is not True
        got=UsdGeom.XformCache().GetLocalToWorldTransform(p)*door_world.GetInverse()
        error=max(abs(got[i][j]-local[i][j]) for i in range(4) for j in range(4))
        assert error<1e-8
        row.update(id=asset_id,path=path,entry=str(source),source_sha256=digest(source),
                   seed=seed,initialization_owner=OWNER,door_path=DOOR,
                   door_relative_matrix_error=error,instance_scale=1,
                   physics_parameters_unchanged=True,initial_state='sampled_requires_native_settling',
                   native_physics_validated=False)
        rows.append(row)
    bind_materials(stage)
    return rows


def extend_manifest(manifest, *, seed):
    """Add a layer atop the latest scene, with a backup and lost-update guard."""
    from pxr import Usd,UsdGeom
    manifest=Path(manifest).resolve(strict=True)
    if not manifest.is_relative_to(HERE): raise ValueError('Output outside task directory')
    before=manifest.read_bytes();data=json.loads(before);source=Path(data['scene']).resolve(strict=True)
    if data.get('fridge_box_initialization'): raise ValueError('Already initialized; generate a new episode to resample')
    out=manifest.with_name('room_fridge_boxes_v1.usda');backup=manifest.with_name('composition_before_fridge_boxes_v1.json')
    if out.exists() or backup.exists(): raise ValueError('Refusing to overwrite prior output')
    source_hash=digest(source);stage=Usd.Stage.CreateNew(str(out));stage.GetRootLayer().subLayerPaths=[str(source)]
    UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    rows=apply(stage,seed=seed);stage.GetRootLayer().Save()
    assert digest(source)==source_hash
    data['scene']=str(out);data['objects'].extend(rows)
    data['all_added_assets_from_all_assets']=False
    data['fridge_box_asset_source']='existing_api_resources/shared_assets_standard_packages'
    data['fridge_box_initialization']=dict(owner=OWNER,seed=seed,box_count=3,available_slots=4,
        source_scene=str(source),source_scene_sha256=source_hash,
        native_settling_validated=False,runtime_docking=False,physics_parameter_overrides=False)
    if manifest.read_bytes()!=before: raise RuntimeError('Manifest changed concurrently; additive layer retained, manifest untouched')
    with backup.open('xb') as f: f.write(before)
    temporary=manifest.with_name('composition_fridge_boxes_v1.tmp')
    with temporary.open('x') as f: json.dump(data,f,ensure_ascii=False,indent=2)
    temporary.replace(manifest)
    print(json.dumps(dict(manifest=str(manifest),scene=str(out),seed=seed,boxes=rows)),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    a=p.parse_args();extend_manifest(a.manifest,seed=a.seed)
