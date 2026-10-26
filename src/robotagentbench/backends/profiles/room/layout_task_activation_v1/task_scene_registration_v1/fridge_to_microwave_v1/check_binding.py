"""Read-only CPU USD preflight for current initialized rooms, both variants."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pxr import Sdf,Usd,UsdGeom,UsdPhysics
from fridge_to_microwave_v1 import support
from fridge_to_microwave_v1.scene import Geometry,pose_matrix,PLATE_BODY

result=[]
for name in ('room_demo_v6','coffee_initialized_rooms_v1/layout03','coffee_initialized_rooms_v1/layout04'):
    m=json.loads((ROOT/name/'composition.json').read_text())
    layer=Sdf.Layer.CreateAnonymous();layer.subLayerPaths=[m['scene']];s=Usd.Stage.Open(layer)
    UsdGeom.SetStageMetersPerUnit(s,1);UsdGeom.SetStageUpAxis(s,UsdGeom.Tokens.z)
    if not s.GetPrimAtPath(PLATE_BODY):support.apply(s)
    for variant in ('beef_box','leftover_box'):
        target=next(r['path'] for r in m['objects'] if r['id']==variant)
        geometry=Geometry(s,target,variant);cache=UsdGeom.XformCache()
        poses={r:pose_matrix(cache.GetLocalToWorldTransform(s.GetPrimAtPath(p))) for r,p in [('payload',target),('plate',PLATE_BODY)]}
        contents=geometry.contents(poses)
        assert contents['plate']['inside'],contents
        for p in s.Traverse():
            if str(p.GetPath()).startswith(('/World/Assets/MicrowaveTakePlate','/World/Assets/MicrowavePlateStand')):
                for rel in p.GetRelationships():
                    for destination in rel.GetTargets():
                        assert s.GetPrimAtPath(destination.GetPrimPath()),str(destination)
        result.append(dict(room=name,variant=variant,target=target,contents=contents))
receipt=dict(status='passed',bindings=result)
(ROOT/'fridge_to_microwave_v1/binding_check.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
