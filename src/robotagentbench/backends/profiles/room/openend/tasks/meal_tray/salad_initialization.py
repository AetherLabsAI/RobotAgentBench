"""Select the existing room salad recipe; no unrelated payload is instantiated.

The original three-slot sampler is read unchanged so seed/pose/asset contracts
stay identical. Only the requested salad is authored in the episode layer.
"""
def compose(stage,data,rows,source_hashes,sha):
    requests=[t for t in data['requirements'].get('meal_task_requirements',[]) if t['type']=='salad_to_meal_tray']
    if not requests:return []
    task,=requests
    from pxr import Usd,UsdGeom,UsdPhysics,Gf,Sdf
    from replace_room_refrigerator_v1 import apply as replace_fridge
    from initialize_room_fridge_boxes_v1 import API,ASSETS,DOOR,OWNER,sample
    from arena_api import food_box_assets,beef_asset
    food_box_assets.verify_files();beef_asset.verify_files()
    result=[]
    if not stage.GetPrimAtPath(DOOR):
        fridge=replace_fridge(stage);result.append(fridge)
        source_hashes[fridge['entry']]=sha(fridge['entry'])
    lengths={}
    for key,(_,rel,_) in ASSETS.items():
        asset=Usd.Stage.Open(str(API/'resources/shared_assets'/rel))
        bounds=UsdGeom.BBoxCache(0,['default','render']).ComputeWorldBound(asset.GetDefaultPrim()).ComputeAlignedRange()
        lengths[key]=float(bounds.GetSize()[0])
    row,=[r for r in sample(data['request']['seed'],lengths) if r['key']=='salad']
    asset_id,rel,path=ASSETS['salad'];entry=API/'resources/shared_assets'/rel
    if stage.GetPrimAtPath(path):raise ValueError('Salad instance already present')
    door_world=UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(DOOR))
    if abs(door_world.GetDeterminant()-1)>1e-6:raise ValueError('Nonrigid fridge door')
    p=stage.DefinePrim(path,'Xform');p.GetReferences().SetReferences([Sdf.Reference(str(entry))])
    p.SetCustomDataByKey('roomAssetId',asset_id);p.SetCustomDataByKey('roomInitializationOwner',OWNER)
    q=row['door_local_quaternion_wxyz']
    local=Gf.Matrix4d().SetRotate(Gf.Quatd(q[0],Gf.Vec3d(*q[1:])))
    local.SetTranslateOnly(Gf.Vec3d(*row['door_local_position_m']))
    parent=UsdGeom.XformCache().GetLocalToWorldTransform(p.GetParent())
    UsdGeom.Xformable(p).MakeMatrixXform().Set(local*door_world*parent.GetInverse())
    asset=Usd.Stage.Open(str(entry));root=asset.GetDefaultPrim()
    for child in Usd.PrimRange(root):
        reln=child.GetRelationship('material:binding:physics');targets=reln.GetTargets() if reln else []
        if not targets:continue
        for target in targets:
            actual=stage.GetPrimAtPath(target)
            if not actual or not actual.HasAPI(UsdPhysics.MaterialAPI):raise ValueError('Missing existing material '+str(target))
        suffix=str(child.GetPath())[len(str(root.GetPath())):]
        stage.GetPrimAtPath(path+suffix).CreateRelationship('material:binding:physics').SetTargets(targets)
    if not p.HasAPI(UsdPhysics.RigidBodyAPI) or p.GetAttribute('physics:kinematicEnabled').Get() is True:
        raise ValueError('Salad must remain original dynamic body')
    row.update(id=asset_id,path=path,entry=str(entry),task_id=task['id'],source='refrigerator',
        source_sha256=sha(entry),seed=data['request']['seed'],initialization_owner=OWNER,
        door_path=DOOR,instance_scale=1,physics_parameters_unchanged=True,native_physics_validated=False)
    source_hashes[str(entry)]=sha(entry);result.append(row)
    return result
