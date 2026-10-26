"""Unified scene-owned microwave initial assembly; never used during stepping."""
import hashlib,json
from pathlib import Path
import numpy as np
from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,Gf,Sdf
from .prepare_microwave_mechanism import ROOT,physical_records
HERE=Path(__file__).resolve().parent

def compose(stage,requests,rows,project,source_hashes,sha,*,payload_from_fridge=False):
    requests=[r for r in requests if r['type']=='microwave_box_to_meal_tray']
    if not requests:return []
    task,=requests
    pack=HERE/'rotary_pack_v1.usda';receipt=json.loads((HERE/'rotary_pack_v1.json').read_text())
    if sha(pack)!=receipt['pack_sha256']:raise ValueError('Rotary source changed')
    if stage.GetPrimAtPath(ROOT+'/StartKnob'):raise ValueError('Mechanism already present')
    c=UsdGeom.XformCache();root=c.GetLocalToWorldTransform(stage.GetPrimAtPath(ROOT))
    body=c.GetLocalToWorldTransform(stage.GetPrimAtPath(ROOT+'/body'))*root.GetInverse()
    linear=np.asarray(root)[:3,:3]
    if not np.allclose(linear@linear.T,receipt['root_gram'],atol=1e-7,rtol=0) or not np.allclose(body,receipt['body_relative'],atol=1e-7,rtol=0):
        raise ValueError('Microwave mounting calibration mismatch')
    material,_=UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(ROOT+'/body/visual_shell')).ComputeBoundMaterial()
    if not material:raise ValueError('Original microwave shell material required')
    before=physical_records(stage);src=Sdf.Layer.FindOrOpen(str(pack))
    for suffix in receipt['parts']:
        path=ROOT+suffix
        Sdf.CreatePrimInLayer(stage.GetRootLayer(),Sdf.Path(path).GetParentPath())
        if not Sdf.CopySpec(src,path,stage.GetRootLayer(),path):raise ValueError('Mechanism copy failed')
    for suffix in ('/body/visual_shell','/StartKnob/VisualAndCollision'):
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(ROOT+suffix)).Bind(material)
    after=physical_records(stage)
    allowed={ROOT+s for s in ('/StartKnob','/StartKnob/VisualAndCollision','/StartKnobJoint')}
    if {p for p in set(before)|set(after) if before.get(p)!=after.get(p)}!=allowed:
        raise ValueError('Existing room physics unexpectedly changed')
    source_hashes[str(pack)]=sha(pack)
    if payload_from_fridge:
        # Only install the existing mechanism. The unified fridge recipe owns
        # the single original payload AND the existing oven plate/stand.
        return []
    from fridge_to_microwave_v1.support import apply as supports
    added=supports(stage)
    from evaluation.tasks.microwave_take import initial_state as old
    variant=task['payload_variant'];api=Path(project)/'api-protocol'
    if variant=='beef_box':
        entry=api/'resources/shared_assets/beef_box_standard243_v1/beefbox.usda'
        path=old.PAYLOAD;offset=old.PAYLOAD_IN_PLATE;scale=1.
    elif variant=='leftover_box':
        entry=api/'assets/house_scene_l1/demo_sim/kitchen_task_assets/dining/foodbox_burger_v1/foodbox_task108_physics_v2.usda'
        path=old.LEFTOVER_PAYLOAD;offset=old.LEFTOVER_IN_PLATE;scale=.94
        # Task 10.4's exact existing asset, NOT the differently sized fridge box.
        expected=old.LEFTOVER_ASSET_SHA256
        for p,key in ((entry,'payload_native_physics'),(entry.with_name('foodbox.usda'),'payload_native_mesh'),
                      (entry.parent/'textures/Image_0.png','payload_native_texture')):
            if p.is_file() and sha(p)!=expected[key]:raise ValueError('Task 10.4 source changed '+str(p))
    else:raise ValueError('Unknown microwave payload')
    if any(r['id']==variant for r in rows):raise ValueError('Ambiguous box ownership')
    existing=stage.GetPrimAtPath(path)
    if existing and existing.IsActive():raise ValueError('Occupied box path')
    p=stage.DefinePrim(path,'Xform');p.SetActive(True)
    p.GetReferences().SetReferences([Sdf.Reference(str(entry))])
    # Preserve the old payload's approved 0.94 instance scale (beef is baked 1).
    rel=Gf.Matrix4d().SetScale(scale)
    rel.SetTranslateOnly(Gf.Vec3d(*offset.xyz_m))
    cache=UsdGeom.XformCache();plate=cache.GetLocalToWorldTransform(stage.GetPrimAtPath(old.PLATE)).RemoveScaleShear()
    parent=cache.GetLocalToWorldTransform(p.GetParent())
    UsdGeom.Xformable(p).MakeMatrixXform().Set(rel*plate*parent.GetInverse())
    if not p.HasAPI(UsdPhysics.RigidBodyAPI) or p.GetAttribute('physics:kinematicEnabled').Get() is True:
        raise ValueError('Original dynamic box required')
    p.SetCustomDataByKey('roomAssetId',variant)
    added.append(dict(id=variant,path=path,entry=str(entry),source='microwave',
        initialization_owner='unified_microwave_meal_v1',task_id=task['id'],
        physical_parameters_unchanged=True,native_physics_validated=False))
    for row in added:
        source_hashes[row['entry']]=sha(row['entry'])
    source_hashes[str(pack)]=sha(pack)
    return added
