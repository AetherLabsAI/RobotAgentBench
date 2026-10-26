"""Read-only USD inventory evidence, not a native physics test."""
import json,sys
from pathlib import Path
from pxr import Usd,UsdPhysics,UsdGeom
folder=Path(sys.argv[1]);manifest=json.loads((folder/'composition.json').read_text());stage=Usd.Stage.Open(manifest['scene'])
records=[]
for row in manifest['objects']:
    if not row.get('task_id'):continue
    prim=stage.GetPrimAtPath(row['path'])
    assert prim and prim.IsActive(),row['path']
    bodies=[]
    for p in Usd.PrimRange(prim):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            b=UsdPhysics.RigidBodyAPI(p);bodies.append(dict(path=str(p.GetPath()),enabled=b.GetRigidBodyEnabledAttr().Get(),kinematic=b.GetKinematicEnabledAttr().Get()))
    records.append(dict(asset=row['id'],task=row['task_id'],path=row['path'],rigid_bodies=bodies))
expected=sum(len(t['object_paths']) for t in manifest['tasks'])
assert len(records)==expected
assert UsdGeom.GetStageMetersPerUnit(stage)==1 and UsdGeom.GetStageUpAxis(stage)=='Z'
print(json.dumps(dict(layout=manifest['layout_id'],tasks=len(manifest['tasks']),target_objects=len(records),objects=records,native_physics_test=False),ensure_ascii=False))
