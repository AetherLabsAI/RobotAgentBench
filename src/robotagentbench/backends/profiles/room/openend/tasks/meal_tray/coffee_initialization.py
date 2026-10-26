"""Add the existing 100-bean initialization to a fresh unified room.

Reuses the approved source loader and bag-local transforms. No physics values,
old asset files, or existing room transforms are changed. Never run at handoff.
"""
import importlib.util
from pathlib import Path
from pxr import Usd, UsdGeom, UsdPhysics, Gf


def compose(stage,rows,root,source_hashes,sha):
    machines=[r for r in rows if r['id']=='coffee_machine']
    if not machines:return [],None
    if len(machines)!=1:raise ValueError('One coffee preparation assembly required')
    module=Path(root)/'layout_task_activation_v1/task_scene_registration_v1/complete_coffee_initialization_v1.py'
    spec=importlib.util.spec_from_file_location('approved_coffee_initialization',module)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    from evaluation.tasks.coffee_machine.openended_composite_v1.scene_binding import resolve,ReadOnlySceneBinding
    manifest=dict(objects=rows,supports={'coffee':{'path':machines[0]['support_path']}})
    resolved=resolve(stage,manifest)
    if stage.GetPrimAtPath(old.BEANS) or 'beans' in resolved['roles']:
        raise ValueError('Duplicate bean initialization forbidden')
    if any(p.GetName().startswith('Bean_') for p in stage.Traverse()):
        raise ValueError('Unbound existing beans require explicit ownership')
    repo=Path(root)/'api-protocol'
    source_path,source,cluster_path,variants=old.load_source(repo)
    entries={v:repo/'all_assets'/v/'asset.usda' for v in set(variants)}
    for p in (module,source_path,cluster_path,*entries.values()):source_hashes[str(p)]=sha(p)
    poses={str(p.GetPath()):old.world(stage,str(p.GetPath())) for p in stage.Traverse() if p.IsA(UsdGeom.Xformable)}
    bag=old.world(stage,resolved['roles']['bag'])
    if abs(abs(bag.GetDeterminant())-1)>1e-8:raise ValueError('Scaled bag unsupported')
    source_bag=old.matrix(source['world']['bag'])
    UsdGeom.Xform.Define(stage,old.BEANS)
    parent_inverse=old.world(stage,old.BEANS).GetInverse()
    assets={v:Usd.Stage.Open(str(p)) for v,p in entries.items()}
    error=0.
    for i,(flat,variant) in enumerate(zip(source['beans'],variants)):
        path=f'{old.BEANS}/Bean_{i:03d}'
        prim=stage.DefinePrim(path,'Xform');prim.GetReferences().AddReference(str(entries[variant]))
        relative=old.matrix(flat)*source_bag.GetInverse()
        UsdGeom.Xformable(prim).MakeMatrixXform().Set(relative*bag*parent_inverse)
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.GetAttribute('physics:kinematicEnabled').Get() is True:
            raise ValueError('Original dynamic bean required')
        if old.fingerprint(stage,path)!=old.fingerprint(assets[variant],'/Asset'):
            raise ValueError('Bean geometry or physical parameters changed')
        got=old.world(stage,path)*bag.GetInverse()
        error=max(error,max(abs(got[r][c]-relative[r][c]) for r in range(4) for c in range(4)))
    if error>=1e-10:raise ValueError('Bean bag-local initial pose not preserved')
    if not all(Gf.IsClose(old.world(stage,p),m,1e-12) for p,m in poses.items()):
        raise ValueError('Existing room transform changed')
    row=dict(id='coffee_beans',path=old.BEANS,count=100,entries=sorted(map(str,entries.values())),
             task_id=machines[0]['task_id'],role='coffee_preparation_contents',initialization_owner='unified_scene')
    binding=ReadOnlySceneBinding(stage,dict(manifest,objects=rows+[row]))
    snapshot=binding.geometry_snapshot()
    if snapshot['beans_bound']!=100 or snapshot['beans_geometrically_in_bag']!=100:
        raise ValueError('Bean count or bag containment failed')
    binding.task_binding('scene_initialization_check',machines[0]['task_id'])
    receipt=dict(owner='unified_scene',source_api='1-2',source_binding=str(source_path),
        source_binding_sha256=sha(source_path),bean_count=100,
        bean_variant_counts={v:variants.count(v) for v in sorted(set(variants))},
        max_bag_local_transform_error=error,static_geometry=snapshot,
        existing_object_poses_unchanged=True,physics_parameters_unchanged=True,
        liquid_mode='visual_cpu_v1',gpu_fluid_solver=False,native_physics_verified=False)
    return [row],receipt
