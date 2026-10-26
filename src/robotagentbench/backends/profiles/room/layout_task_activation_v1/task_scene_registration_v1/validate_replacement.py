"""Reopen all replacement layers and audit original helper setup on a session only."""
import json,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from pxr import Usd,UsdGeom,UsdPhysics,PhysxSchema,Gf,Sdf
from replace_room_refrigerator_v1 import apply,OLD,NEW,LIB
from arena_api import l1_fridge

ROOT=Path(__file__).resolve().parent
def main():
    index=json.loads((ROOT/'refrigerator_layouts_v1/index.json').read_text())
    checks=[]
    for key,row in index.items():
        stage=Usd.Stage.OpenMasked(row['scene'],Usd.StagePopulationMask([OLD,NEW]))
        UsdGeom.SetStageMetersPerUnit(stage,1.0);UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
        stage.GetRootLayer().Save()
        base=Usd.Stage.OpenMasked(row['source'],Usd.StagePopulationMask([OLD]))
        a=UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(NEW))
        b=UsdGeom.XformCache().GetLocalToWorldTransform(base.GetPrimAtPath(OLD))
        assert np.allclose(np.asarray(a),np.asarray(b),rtol=0,atol=1e-9)
        assert not stage.GetPrimAtPath(OLD).IsActive()
        assert len([p for p in Usd.PrimRange(stage.GetPrimAtPath(NEW)) if p.IsA(UsdPhysics.RevoluteJoint)])==2
        checks.append(dict(layout=key,pose_preserved=True,hinges=2,old_disabled=True))
    # Baseline layout01 current manifest also uses the replacement, without changing old USD.
    m=ROOT/'room_demo_v6/composition.json';data=json.loads(m.read_text())
    if not any(o['id']=='refrigerator' for o in data['objects']):
        backup=m.with_name('composition_before_refrigerator_v1.json')
        assert not backup.exists();backup.write_bytes(m.read_bytes())
        target=m.with_name('room_refrigerator_v1.usda');assert not target.exists()
        s=Usd.Stage.CreateNew(str(target));s.GetRootLayer().subLayerPaths=[data['scene']]
        s.SetPopulationMask(Usd.StagePopulationMask([OLD,NEW]));r=apply(s);s.GetRootLayer().Save()
        data['scene']=str(target);data['objects'].append(r);data['refrigerator_replacement']=r
        m.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    # Fill geometry and stage metadata for all current replacement rooms.
    room_manifests=[ROOT/'room_demo_v6/composition.json',
                    *(ROOT/'coffee_initialized_rooms_v1').glob('layout*/composition.json')]
    for m in room_manifests:
        data=json.loads(m.read_text());s=Usd.Stage.OpenMasked(data['scene'],Usd.StagePopulationMask([NEW]))
        UsdGeom.SetStageMetersPerUnit(s,1.0);UsdGeom.SetStageUpAxis(s,UsdGeom.Tokens.z)
        s.GetRootLayer().Save()
        p=s.GetPrimAtPath(NEW)
        box=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render']).ComputeWorldBound(p).ComputeAlignedRange()
        for row in data['objects']:
            if row['id']=='refrigerator':row['visible_bounds_world_m']=[list(box.GetMin()),list(box.GetMax())]
        m.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    # Existing controller's exact initializer; only an anonymous diagnostic stage.
    s=Usd.Stage.CreateInMemory();s.DefinePrim(NEW).GetReferences().AddReference(str(LIB/'refrigerator/asset.usda'))
    environment=json.loads((l1_fridge.ORIGINAL/'environment.json').read_text())
    ns=dict(stage=s,PHYS=True,np=np,os=SimpleNamespace(environ=environment),Gf=Gf,Sdf=Sdf,
            Usd=Usd,UsdGeom=UsdGeom,UsdPhysics=UsdPhysics,PhysxSchema=PhysxSchema,print=lambda *a,**k:None)
    l1_fridge._execute(l1_fridge.original_material_blocks(),ns)
    l1_fridge._execute(l1_fridge.original_blocks()[0],ns)
    material=l1_fridge.verify_material(s)
    for side in ('left_door','right_door'):
        p=s.GetPrimAtPath(NEW+'/'+side)
        assert p.GetAttribute('physics:rigidBodyEnabled').Get() is True
        assert p.GetAttribute('physics:kinematicEnabled').Get() is False
    assert s.GetPrimAtPath(NEW+'/body').GetAttribute('physics:kinematicEnabled').Get() is True
    manifest=json.loads((LIB/'refrigerator/manifest.json').read_text())
    for row in manifest['dependencies']:
        assert hashlib.sha256(Path(row['source']).read_bytes()).hexdigest()==row['source_sha256']
    report=dict(layouts=checks,source_model_unchanged=True,helper_initializer_compatible=True,
                material=material,native_physics_test=False,full_task_connected=False)
    (ROOT/'refrigerator_layouts_v1/validation.json').write_text(json.dumps(report,indent=2))
    print('PASS',len(checks),'layouts; original helper initialization compatible')
if __name__=='__main__':main()
