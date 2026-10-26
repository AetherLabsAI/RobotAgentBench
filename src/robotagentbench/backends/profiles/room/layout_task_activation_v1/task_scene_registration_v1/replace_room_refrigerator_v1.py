"""Reference all_assets fridge at the existing furniture pose; never edit base layouts."""
import json, hashlib
from pathlib import Path
from pxr import Usd,UsdGeom,UsdPhysics,Gf

LIB=Path('/home/lingjun/robot_agent_arena/api-protocol/all_assets')
OLD='/World/Background/Shell/KitchenRefrigerator'
NEW='/World/Background/Shell/RefrigeratorRecovered'

def apply(stage):
    UsdGeom.SetStageMetersPerUnit(stage,1.0)
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    existing=stage.GetPrimAtPath(NEW)
    if existing and existing.IsActive():
        if existing.GetCustomDataByKey('roomAssetId')!='refrigerator':
            raise ValueError('An unowned recovered fridge already exists')
        return dict(id='refrigerator',path=NEW,entry=str(LIB/'refrigerator/asset.usda'),
                    helper_runtime='arena_api.l1_fridge.Controller',helper_required_before_physics=True)
    old=stage.GetPrimAtPath(OLD)
    if not old or not old.IsActive():raise ValueError('Missing layout refrigerator anchor')
    cache=UsdGeom.XformCache();old_world=cache.GetLocalToWorldTransform(old)
    parent_world=cache.GetLocalToWorldTransform(old.GetParent())
    stage.OverridePrim(OLD).SetActive(False)
    prim=stage.DefinePrim(NEW,'Xform')
    prim.GetReferences().AddReference(str(LIB/'refrigerator/asset.usda'))
    # The layout anchor already includes the original Y-up -> Z-up rotation.
    # Override xformOpOrder, not multiply another 90-degree root rotation.
    UsdGeom.Xformable(prim).MakeMatrixXform().Set(old_world*parent_world.GetInverse())
    prim.SetCustomDataByKey('roomAssetId','refrigerator')
    prim.SetCustomDataByKey('arenaHelperRequired','arena_api.l1_fridge.Controller')
    got=UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    assert max(abs(got[i][j]-old_world[i][j]) for i in range(4) for j in range(4))<1e-9
    for side in ('left','right'):
        joint=stage.GetPrimAtPath(NEW+'/'+side+'_door_hinge')
        assert joint and joint.IsA(UsdPhysics.RevoluteJoint)
        for name in ('physics:body0','physics:body1'):
            targets=joint.GetRelationship(name).GetTargets()
            assert len(targets)==1 and str(targets[0]).startswith(NEW+'/') and stage.GetPrimAtPath(targets[0])
    bounds=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render']).ComputeWorldBound(prim).ComputeAlignedRange()
    return dict(id='refrigerator',path=NEW,entry=str(LIB/'refrigerator/asset.usda'),
        visible_bounds_world_m=[list(bounds.GetMin()),list(bounds.GetMax())],
        replaced=OLD,world_pose_preserved=True,helper_runtime='arena_api.l1_fridge.Controller',
        helper_required_before_physics=True,native_physics_validated=False)

def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args()
    root=args.root;out=root/'refrigerator_layouts_v1';out.mkdir(exist_ok=False)
    bases=Path('/mnt/agent-bench-data/lingjun/releases/living50-candidate-20260919/layouts')
    index={}
    for source in sorted(bases.glob('ID_*_LIVING_STRAIGHT.usdc')):
        target=out/(source.stem+'.usda')
        stage=Usd.Stage.CreateNew(str(target));stage.GetRootLayer().subLayerPaths=[str(source)]
        stage.SetPopulationMask(Usd.StagePopulationMask([OLD,NEW]))
        receipt=apply(stage);stage.GetRootLayer().Save()
        index[source.name]=dict(scene=str(target),source=str(source),replacement=receipt)
        print('REPLACED',source.name,flush=True)
    assert len(index)==50,len(index)
    (out/'index.json').write_text(json.dumps(index,indent=2))
    # Current fully initialized room manifests now point to additive layers.
    receipts=[]
    for manifest in sorted((root/'coffee_initialized_rooms_v1').glob('layout*/composition.json')):
        data=json.loads(manifest.read_text());source=Path(data['scene'])
        backup=manifest.with_name('composition_before_refrigerator_v1.json')
        assert not backup.exists();backup.write_bytes(manifest.read_bytes())
        target=manifest.with_name('room_refrigerator_v1.usda')
        assert not target.exists()
        stage=Usd.Stage.CreateNew(str(target));stage.GetRootLayer().subLayerPaths=[str(source)]
        stage.SetPopulationMask(Usd.StagePopulationMask([OLD,NEW]))
        receipt=apply(stage);stage.GetRootLayer().Save()
        data['scene']=str(target);data['objects'].append(receipt)
        data['refrigerator_replacement']=receipt
        manifest.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
        receipts.append(dict(manifest=str(manifest),source=str(source),scene=str(target)))
    (out/'current_rooms.json').write_text(json.dumps(receipts,indent=2))
    print('COMPLETE',len(index),len(receipts),flush=True)

if __name__=='__main__':main()
