"""Add-only accessory import, preserving source geometry/physics and old assets.

CPU composition checks only. Cloth remains explicitly static/source-only.
Build in a unique work directory; --publish adds validated entries to all_assets.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import io
import zipfile
from datetime import datetime, timezone
import numpy as np
from pxr import Ar, Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

API=Path('/home/lingjun/robot_agent_arena/api-protocol')
LIB=API/'all_assets'
LAYOUT=Path('/mnt/agent-bench-data/lingjun/releases/living50-candidate-20260919/layouts/ID_01_LIVING_STRAIGHT.usdc')
BUILTINS={'OmniPBR.mdl','OmniGlass.mdl','OmniPBR_Opacity.mdl','OmniSurface.mdl'}
PATH_RECOVERY={
 '/task8/cloth_input/textures/dirty_tablecloth.png':(
  '/mnt/agent-bench-data/lingjun/archive/task8_clear/16_task8-server-parallel-20260915/cloth_input/textures/dirty_tablecloth.png',
  '4647737780d8a58e1cf7791641c80862de4a844ef8b84859c5b512ba9714d717')}
def sha(raw): return hashlib.sha256(raw).hexdigest()
def dump(p,obj): p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def raw(p):
 if Ar.IsPackageRelativePath(p):
  package,inner=Ar.SplitPackageRelativePathOuter(p)
  with zipfile.ZipFile(io.BytesIO(raw(package))) as archive:
   return archive.read(inner)
 return Path(p).read_bytes()
def norm(layer):
 clone=Sdf.Layer.CreateAnonymous('norm.usda');clone.TransferContent(layer);clone.comment=''
 UsdUtils.ModifyAssetPaths(clone,lambda p:'__ASSET__' if p else p)
 return sha(clone.ExportToString().encode())
def links(prop): return prop.GetTargets() if isinstance(prop,Usd.Relationship) else prop.GetConnections()
def xform(p,m):
 for a in list(p.GetAttributes()):
  if a.GetName().startswith('xformOp:') or a.GetName()=='xformOpOrder':p.RemoveProperty(a.GetName())
 UsdGeom.Xformable(p).AddTransformOp().Set(m)
def check(entry,root,expected):
 stage=Usd.Stage.Open(str(entry));assert stage and stage.GetDefaultPrim(),entry
 assert norm(stage.Flatten(False))==expected,entry
 layers,assets,missing=UsdUtils.ComputeAllDependencies(str(entry))
 assert set(map(str,missing))<=BUILTINS,(entry,list(missing))
 for p in [*(l.realPath for l in layers),*assets]:
  assert Path(str(p)).resolve().is_relative_to(root.resolve()),p
 for p in stage.Traverse():
  for prop in p.GetProperties():
   for t in links(prop):assert stage.GetObjectAtPath(t),(entry,str(t))
 meshes=[p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
 assert meshes,entry
 test=Usd.Stage.CreateInMemory();test.DefinePrim('/Placed','Xform').GetReferences().AddReference(str(entry))
 assert len(meshes)==sum(p.IsA(UsdGeom.Mesh) for p in test.Traverse())
 return dict(meshes=len(meshes),mesh_vertices=sum(len(UsdGeom.Mesh(p).GetPointsAttr().Get() or []) for p in meshes),
  rigid_bodies=sum(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in stage.Traverse()),
  colliders=sum(p.HasAPI(UsdPhysics.CollisionAPI) and p.GetAttribute('physics:collisionEnabled').Get()!=False for p in stage.Traverse()),
  dependencies_local=True,relationship_targets_resolved=True,independent_reference_passed=True,
  normalized_composed_layer_sha256=expected,native_physics_run=False)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--work',required=True,type=Path);parser.add_argument('--publish',action='store_true');args=parser.parse_args()
 args.work.mkdir(parents=True,exist_ok=False)
 stage_dir=args.work/'staging';(stage_dir/'_shared').mkdir(parents=True)
 registry=json.loads((LIB/'registry.json').read_text())
 before={str(p.relative_to(LIB)):sha(p.read_bytes()) for p in LIB.rglob('*') if p.is_file()}
 # Use the existing Task2.1 initializer to recover its plate/support assembly,
 # not an unconfigured visual plate. This writes ONLY to the unique work area.
 sys.path.insert(0,str(API))
 from evaluation.tasks.air_fryer.initial_state import package,write_station
 from evaluation.tasks.air_fryer.specs import spec
 manifest=package('2.1','ruleyellowinner-42032')
 station=args.work/'source_task2_1_station.usda';write_station(station,manifest,spec('2.1'))
 cakebase=API/'evaluation/tasks/air_fryer/resources/producer_assets_20260922/cake_wide125_v1'
 cloth=API/'evaluation/tasks/tabletop_pick/book_cloth/source_16_seed1000000/initial_scene.usdc'
 selections=[
  ('cake_plain_yellow','黄色原味蛋糕',cakebase/'plain_cake_a.usda',['/Cake'],'source_rigid_body_unvalidated'),
  ('cake_frosted_white','白色糖霜蛋糕',cakebase/'frosted_cake.usda',['/Cake'],'source_rigid_body_unvalidated'),
  ('paper_plate_task2','Task2 纸盘（含原任务隐形承托）',station,['/World/Task66_67/DisplayPlatePlacement','/World/Task66_67/SourcePlateSupport'],'source_static_support_not_free_rigid_plate'),
  ('tablecloth_dirty','脏桌布（原任务静态初态）',cloth,['/World/Assets/Task69Items/DirtyTablecloth'],'static_source_only_deformable_physics_not_configured'),
  ('meal_tray_large','大餐托盘（含把手和碰撞体）',LAYOUT,['/World/FoodTray'],'source_rigid_body_unvalidated'),
  ('cutlery_holder','餐具盒',LAYOUT,['/World/Cutlery/Holder'],'source_rigid_body_unvalidated'),
  ('meal_fork','餐叉',LAYOUT,['/World/Cutlery/fork_1'],'source_rigid_body_unvalidated'),
  ('meal_knife','餐刀',LAYOUT,['/World/Cutlery/knife_0'],'source_rigid_body_unvalidated'),
 ]
 assert not {s[0] for s in selections}&{o['id'] for o in registry['objects']}
 assert all(not (LIB/s[0]).exists() for s in selections)
 source_hashes={};dependencies={};records=[];cache={}
 def store(data,suffix):
  name=sha(data)+suffix;p=stage_dir/'_shared'/name
  if p.exists():assert p.read_bytes()==data
  else:p.write_bytes(data)
  return name
 for slug,label,scene,roots,status in selections:
  print('EXTRACT',slug,flush=True)
  if scene not in cache:
   source=Usd.Stage.Open(str(scene));assert source,scene
   cache[scene]=(source,source.Flatten(False))
   for layer in source.GetUsedLayers():
    if layer.realPath:source_hashes[layer.realPath]=sha(raw(layer.realPath))
  source,flat=cache[scene]
  out=Usd.Stage.CreateInMemory();UsdGeom.SetStageMetersPerUnit(out,1);UsdGeom.SetStageUpAxis(out,'Z')
  root=UsdGeom.Xform.Define(out,'/Asset');out.SetDefaultPrim(root.GetPrim());layer=out.GetRootLayer()
  origin=UsdGeom.Xformable(source.GetPrimAtPath(roots[0])).ComputeLocalToWorldTransform(0).ExtractTranslation()
  delta=Gf.Matrix4d(1).SetTranslate(-origin);mappings=[]
  for i,path in enumerate(roots):
   dest='/Asset/Object' if i==0 else '/Asset/Support'
   assert source.GetPrimAtPath(path),path
   assert Sdf.CopySpec(flat,path,layer,dest)
   xform(out.GetPrimAtPath(dest),UsdGeom.Xformable(source.GetPrimAtPath(path)).ComputeLocalToWorldTransform(0)*delta)
   mappings.append((path,dest))
  # Material-only external relationship closure; never pull in robot/scene.
  for _ in range(12):
   extras={t.GetPrimPath() for p in out.Traverse() for prop in p.GetProperties() for t in links(prop)
    if not any(t.HasPrefix(Sdf.Path(a)) or t.HasPrefix(Sdf.Path(b)) for a,b in mappings)}
   if not extras:break
   for ext in sorted(extras):
    prim=source.GetPrimAtPath(ext);assert prim,(slug,str(ext))
    assert prim.GetTypeName() in ('Material','Shader','NodeGraph'),(slug,str(ext),prim.GetTypeName())
    dest='/Asset/Materials/M_'+sha(str(ext).encode())[:12]
    UsdGeom.Scope.Define(out,'/Asset/Materials');assert Sdf.CopySpec(flat,ext,layer,dest)
    mappings.append((str(ext),dest))
  else:raise RuntimeError('Material closure failed')
  for p in out.Traverse():
   for prop in p.GetProperties():
    old=links(prop);new=[]
    for t in old:
     for a,b in sorted(mappings,key=lambda m:-len(m[0])):
      if t.HasPrefix(Sdf.Path(a)):t=t.ReplacePrefix(Sdf.Path(a),Sdf.Path(b));break
     assert t.HasPrefix(Sdf.Path('/Asset')),str(t)
     new.append(t)
    if old:(prop.SetTargets if isinstance(prop,Usd.Relationship) else prop.SetConnections)(new)
  vertices=0;worst=0.;attributes=0
  for a,b in mappings[:len(roots)]:
   for p in Usd.PrimRange(source.GetPrimAtPath(a)):
    q=out.GetPrimAtPath(p.GetPath().ReplacePrefix(Sdf.Path(a),Sdf.Path(b)));assert q
    assert set(p.GetAppliedSchemas())==set(q.GetAppliedSchemas())
    for attr in p.GetAttributes():
     name=attr.GetName()
     if name.startswith('xformOp:') or name=='xformOpOrder' or links(attr):continue
     if attr.GetTypeName() in (Sdf.ValueTypeNames.Asset,Sdf.ValueTypeNames.AssetArray):continue
     assert str(attr.Get())==str(q.GetAttribute(name).Get()),(slug,str(p.GetPath()),name)
     attributes+=1
    if p.IsA(UsdGeom.Xformable):
     m=np.asarray(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(0)*delta)
     n=np.asarray(UsdGeom.Xformable(q).ComputeLocalToWorldTransform(0))
     assert np.allclose(m,n,atol=1e-9,rtol=0),(slug,str(p.GetPath()))
     if p.IsA(UsdGeom.Mesh):
      v=np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get(),float)
      if len(v):
       h=np.c_[v,np.ones(len(v))];worst=max(worst,float(np.max(abs(h@m-h@n))));vertices+=len(v)
  expected=norm(out.Flatten(False))
  def remap(value):
   if not value or value in BUILTINS:return value
   if value in PATH_RECOVERY:
    resolved,digest=PATH_RECOVERY[value];assert sha(raw(resolved))==digest
   else:resolved=str(Ar.GetResolver().Resolve(value))
   assert resolved,(slug,value)
   data=raw(resolved);inner=Ar.SplitPackageRelativePathInner(resolved)[1] if Ar.IsPackageRelativePath(resolved) else resolved
   suffix=Path(inner).suffix.lower();assert suffix not in ('.usd','.usda','.usdc','.usdz'),value
   name=store(data,suffix);dependencies[resolved]=dict(sha256=sha(data),bytes=len(data),payload=name)
   return './'+name
  UsdUtils.ModifyAssetPaths(layer,remap)
  payload=store(layer.ExportToString().encode(),'.usda');objdir=stage_dir/slug;objdir.mkdir()
  (objdir/'asset.usda').write_text('#usda 1.0\n(\n defaultPrim = "Asset"\n metersPerUnit = 1\n upAxis = "Z"\n subLayers = [@../_shared/'+payload+'@]\n)\n')
  checks=check(objdir/'asset.usda',stage_dir,expected)
  bounds=UsdGeom.BBoxCache(0,['default','render','proxy']).ComputeWorldBound(out.GetDefaultPrim()).ComputeAlignedRange()
  record=dict(id=slug,name=label,entry=slug+'/asset.usda',revision='room_accessories_20261004_v1',
   source_scene=str(scene),source_prims=roots,behaviors=[],physics_status=status,
   normalization=dict(removed_world_translation_m=list(origin),scale_and_orientation_preserved=True),
   bounds_m=dict(min=list(bounds.GetMin()),max=list(bounds.GetMax()),size=list(bounds.GetSize())),
   validation=dict(**checks,source_mesh_vertices_checked=vertices,max_vertex_world_error_m=worst,
    unchanged_composed_attributes_checked=attributes,source_parameters_unchanged=True),
   limitations=['Static extraction only; native physics and robot interaction not validated.'])
  dump(objdir/'manifest.json',record);records.append(record)
  print('VALIDATED',slug,checks,flush=True)
 assert all(sha(raw(p))==h for p,h in source_hashes.items())
 report=dict(schema='arena.all_assets.room_accessories_import.v1',records=records,source_layers_sha256=source_hashes,
  dependencies=dependencies,recovered_source_paths=PATH_RECOVERY,source_task2_episode=manifest,existing_objects=len(registry['objects']),
  added_objects=len(records),source_files_unchanged=True,physics_parameters_changed=False,native_physics_run=False,
  published=False,work_directory=str(args.work))
 dump(args.work/'report.json',report)
 if not args.publish:return
 lock=open(LIB/'.room_accessories_import.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert all(sha((LIB/p).read_bytes())==h for p,h in before.items()),'Library changed during build'
 assert not (LIB/'IMPORT_ROOM_ACCESSORIES_V1.json').exists()
 backups=args.work/'metadata_before';backups.mkdir()
 metadata=['registry.json','SOURCE_MAP.json','VALIDATION.json','BUILD_STATUS.json','README.txt']
 for name in metadata:shutil.copy2(LIB/name,backups/name)
 for p in (stage_dir/'_shared').iterdir():
  dest=LIB/'_shared'/p.name
  if dest.exists():assert dest.read_bytes()==p.read_bytes()
  else:shutil.copy2(p,dest)
 for r in records:
  assert not (LIB/r['id']).exists();shutil.copytree(stage_dir/r['id'],LIB/r['id'])
 for r in records:check(LIB/r['entry'],LIB,r['validation']['normalized_composed_layer_sha256'])
 assert all(sha((LIB/p).read_bytes())==h for p,h in before.items())
 report.update(published=True,published_utc=datetime.now(timezone.utc).isoformat(),existing_files_byte_identical=True,total_objects=len(registry['objects'])+len(records))
 dump(LIB/'IMPORT_ROOM_ACCESSORIES_V1.json',report)
 def atomic(name,value):
  p=LIB/('.'+name+'.room_accessories_tmp');assert not p.exists();dump(p,value);os.replace(p,LIB/name)
 srcmap=json.loads((LIB/'SOURCE_MAP.json').read_text());srcmap['room_accessories_v1']={'report':'IMPORT_ROOM_ACCESSORIES_V1.json','records':records}
 atomic('SOURCE_MAP.json',srcmap)
 val=json.loads((LIB/'VALIDATION.json').read_text());val['entries'].extend(dict(entry=r['entry'],**r['validation']) for r in records)
 val['room_accessories_v1']={'report':'IMPORT_ROOM_ACCESSORIES_V1.json','cpu_checks_passed':True,'native_physics_run':False};atomic('VALIDATION.json',val)
 status=json.loads((LIB/'BUILD_STATUS.json').read_text());status.update(objects=report['total_objects'],entries=report['total_objects'],room_accessories_report='IMPORT_ROOM_ACCESSORIES_V1.json',
  shared_payload_files=len(list((LIB/'_shared').iterdir())),shared_payload_bytes=sum(p.stat().st_size for p in (LIB/'_shared').iterdir()))
 atomic('BUILD_STATUS.json',status)
 registry['objects'].extend(records);registry['updated_utc']=report['published_utc']
 registry['room_accessories_v1']={'report':'IMPORT_ROOM_ACCESSORIES_V1.json','objects_added':len(records)}
 registry['limitations'].append('Room accessories: source geometry/material/physics preserved and CPU composition checked; cloth has no deformable configuration, paper plate retains static support; no native acceptance.')
 atomic('registry.json',registry)
 text='房间配件增补 v1（2026-10-04）\n新增：黄/白蛋糕、含原承托的纸盘、脏桌布、大餐托盘、餐具盒、餐叉、餐刀。\n旧资产字节不变；仅静态合成/属性/顶点/依赖检查通过。桌布为静态来源，无布料物理；纸盘为原静态承托组合。无原生物理验收。详见 IMPORT_ROOM_ACCESSORIES_V1.json。\n\n'
 p=LIB/'.README.txt.room_accessories_tmp';p.write_text(text+(LIB/'README.txt').read_text());os.replace(p,LIB/'README.txt')
 assert all(sha((LIB/p).read_bytes())==h for p,h in before.items() if p not in metadata)
 dump(args.work/'report.json',report)
 if os.getuid()==0:
  owner=(LIB.stat().st_uid,LIB.stat().st_gid)
  for p in [args.work,*args.work.rglob('*')]:os.chown(p,*owner)
  for p in LIB.rglob('*'):
   rel=str(p.relative_to(LIB))
   if (p.is_file() and (rel not in before or rel in metadata)) or (p.is_dir() and p.name in {r['id'] for r in records}):os.chown(p,*owner)
 print('PUBLISHED',json.dumps({'new':len(records),'total':report['total_objects'],'old_assets_unchanged':True}),flush=True)

if __name__=='__main__':main()
