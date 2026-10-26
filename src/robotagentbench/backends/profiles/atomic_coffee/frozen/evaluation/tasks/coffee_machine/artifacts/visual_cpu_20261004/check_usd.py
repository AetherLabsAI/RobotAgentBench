from pathlib import Path
import json
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Gf
from evaluation.tasks.coffee_machine.visual_coffee import VisualCoffee
class Geometry:
 floor_y=.005;rim_y=.09;opening_radius=.037
 heights=np.array([.005,.09]);inner=np.array([.025,.037])
s=Usd.Stage.CreateInMemory();v=VisualCoffee(s,'/World/CoffeeLiquid',Geometry(),(0,0,0))
v.sync_cup((0,0,0),(2**-.5,2**-.5,0,0))
m=Gf.Matrix4d().SetTranslate(Gf.Vec3d(0,0,.15))
for _ in range(2400):v.update(1/240,True,m)
assert v.count==8400 and v.in_cup>8000, v.counts()
for _ in range(48):v.update(1/240,False,m)
assert v.in_cup==8400 and not v.pending
assert v.stream.GetVisibilityAttr().Get()=='invisible'
before=v.surface_op.Get().ExtractTranslation()
v.sync_cup((.3,.2,.1),(2**-.5,2**-.5,0,0))
after=v.surface_op.Get().ExtractTranslation()
assert (after-before-Gf.Vec3d(.3,.2,.1)).GetLength()<1e-8
for p in s.Traverse():
 assert not p.HasAPI(UsdPhysics.RigidBodyAPI) and not p.HasAPI(UsdPhysics.CollisionAPI)
 assert 'Particle' not in str(p.GetTypeName()) and not any('physx' in a.GetName().lower() for a in p.GetAttributes())
v.sync_cup((3,2,.1),(2**-.5,2**-.5,0,0))
for _ in range(120):v.update(1/240,True,m)
for _ in range(48):v.update(1/240,False,m)
assert v.missed>0
out=Path('evaluation/tasks/coffee_machine/artifacts/visual_cpu_20261004')
s.GetRootLayer().Export(str(out/'animation_test.usda'))
(out/'usd_validation.json').write_text(json.dumps(dict(passed=True,no_physics_prims=True,flow_units_per_second=840,stop_drains_stream=True,cup_motion_followed=True,missed_cup_detected=True,render_verified=False),indent=2))
print('USD animation checks passed')
