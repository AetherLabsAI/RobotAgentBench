"""Unified initialization of the already-approved microwave plate and stand.

Copies only these two subtrees; never applies L1 robot/room resets to a room.
"""
import hashlib
import json
from pathlib import Path
from pxr import Gf, Sdf, Usd, UsdGeom
from evaluation.tasks.microwave_take.initial_state import (PLATE,STAND,MICROWAVE,
    PLATE_LAYER,PLATE_IN_MICROWAVE,STAND_IN_MICROWAVE,verify_plate_asset)
from arena_api.l1_scene import ROOT as API, BASE

HERE=Path(__file__).resolve().parent
PACK=HERE/'plate_stand_source_v2.usda'
RECEIPT=HERE/'support_source_receipt_v2.json'
OWNER='unified_scene_task9_support_v1'


def prepare_source():
    verify_plate_asset()
    if PACK.exists():return
    layer=Sdf.Layer.CreateAnonymous('approved_support.usda')
    layer.subLayerPaths=[str(PLATE_LAYER),str(API/'resources/house_scene_l1/approved_microwave.usda'),str(BASE)]
    source=Usd.Stage.OpenMasked(layer,Usd.StagePopulationMask([PLATE,STAND,'/World/PhysicsMaterials','/World/MicrowaveStandLooks']))
    flat=source.Flatten();result=Usd.Stage.CreateNew(str(PACK))
    UsdGeom.SetStageMetersPerUnit(result,1);UsdGeom.SetStageUpAxis(result,UsdGeom.Tokens.z)
    for path in (PLATE,STAND,'/World/MicrowaveStandLooks'):
        p=source.GetPrimAtPath(path)
        if not p or not p.IsActive():raise ValueError('Missing approved support '+path)
        Sdf.CreatePrimInLayer(result.GetRootLayer(),path)
        Sdf.CopySpec(flat,path,result.GetRootLayer(),path)
    result.GetRootLayer().Save()
    receipt=dict(source_layers={l.realPath:hashlib.sha256(Path(l.realPath).read_bytes()).hexdigest() for l in source.GetUsedLayers() if l.realPath},
                 copied_subtrees=[PLATE,STAND],pack_sha256=hashlib.sha256(PACK.read_bytes()).hexdigest(),
                 physical_parameters_unchanged=True)
    RECEIPT.write_text(json.dumps(receipt,indent=2)+'\n')


def apply(stage):
    if not PACK.is_file():raise ValueError('Prepare task-owned support source first')
    digest=json.loads(RECEIPT.read_text())['pack_sha256']
    if hashlib.sha256(PACK.read_bytes()).hexdigest()!=digest:raise ValueError('Support source changed')
    mw=stage.GetPrimAtPath(MICROWAVE)
    if not mw or not mw.IsActive():raise ValueError('Room microwave missing')
    cache=UsdGeom.XformCache();world=cache.GetLocalToWorldTransform(mw);rows=[]
    looks='/World/MicrowaveStandLooks'
    if not stage.GetPrimAtPath(looks):
        stage.DefinePrim(looks,'Scope').GetReferences().AddReference(str(PACK),looks)
    for path,pose,kind in [(PLATE,PLATE_IN_MICROWAVE,'microwave_support_plate'),(STAND,STAND_IN_MICROWAVE,'microwave_plate_stand')]:
        prim=stage.GetPrimAtPath(path)
        if prim and prim.GetCustomDataByKey('roomInitializationOwner')!=OWNER:raise ValueError('Occupied support path '+path)
        p=stage.DefinePrim(path,'Xform');p.GetReferences().SetReferences([Sdf.Reference(str(PACK),path)])
        rel=Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0,0,1),pose.yaw_deg))
        rel.SetTranslateOnly(Gf.Vec3d(*pose.xyz_m))
        parent=UsdGeom.XformCache().GetLocalToWorldTransform(p.GetParent())
        UsdGeom.Xformable(p).MakeMatrixXform().Set(rel*world*parent.GetInverse())
        p.SetCustomDataByKey('roomInitializationOwner',OWNER);p.SetCustomDataByKey('roomAssetId',kind)
        bounds=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render']).ComputeWorldBound(p).ComputeAlignedRange()
        rows.append(dict(id=kind,path=path,entry=str(PACK),initialization_owner=OWNER,
            visible_bounds_world_m=[list(bounds.GetMin()),list(bounds.GetMax())],
            physical_parameters_unchanged=True,native_physics_validated=False))
    # Rebind the original global appearance relationship across the subtree
    # reference boundary. No material values or physical configuration change.
    source=Usd.Stage.Open(str(PACK))
    for p in Usd.PrimRange(source.GetPrimAtPath(STAND)):
        rel=p.GetRelationship('material:binding')
        if rel and rel.GetTargets():stage.GetPrimAtPath(p.GetPath()).CreateRelationship('material:binding').SetTargets(rel.GetTargets())
    from initialize_room_fridge_boxes_v1 import bind_materials
    bind_materials(stage)
    return rows


def extend(manifest):
    manifest=Path(manifest).resolve(strict=True)
    if not manifest.is_relative_to(HERE.parent):raise ValueError('Outside unified task directory')
    before=manifest.read_bytes();data=json.loads(before)
    out=manifest.with_name('room_task9_support_v1.usda');backup=manifest.with_name('composition_before_task9_support_v1.json')
    if out.exists() or backup.exists():raise ValueError('Refusing overwrite')
    s=Usd.Stage.CreateNew(str(out));s.GetRootLayer().subLayerPaths=[data['scene']]
    # USD stage metadata does not inherit from sublayers. Preserve the unified
    # room's authored metre/Z-up contract on this new root, without scaling.
    UsdGeom.SetStageMetersPerUnit(s,1);UsdGeom.SetStageUpAxis(s,UsdGeom.Tokens.z)
    rows=apply(s);s.GetRootLayer().Save();data['scene']=str(out);data['objects'].extend(rows)
    data['task9_support_initialization']=dict(owner=OWNER,plate=PLATE,stand=STAND,physics_modified=False)
    if manifest.read_bytes()!=before:raise RuntimeError('Concurrent manifest change')
    with backup.open('xb') as f:f.write(before)
    tmp=manifest.with_suffix('.task9.tmp')
    with tmp.open('x') as f:json.dump(data,f,indent=2,ensure_ascii=False)
    tmp.replace(manifest)
    print(json.dumps(dict(manifest=str(manifest),scene=str(out),added=[r['id'] for r in rows])),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--manifest',action='append',default=[]);a=p.parse_args()
    prepare_source()
    for m in a.manifest:extend(m)
