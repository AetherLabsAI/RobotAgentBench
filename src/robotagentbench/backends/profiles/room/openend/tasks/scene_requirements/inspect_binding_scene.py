import sys
from pxr import Usd,UsdPhysics,UsdGeom
stage=Usd.Stage.Open(sys.argv[1])
for p in stage.Traverse():
    if 'floor' in str(p.GetPath()).lower():
        print(str(p.GetPath()),p.GetTypeName(),p.HasAPI(UsdPhysics.CollisionAPI),p.GetAttribute('physics:collisionEnabled').Get())
for path in ('/World/Assets/CoffeeTable','/World/Assets/CoffeeTable/black_box_1'):
    m=UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path))
    print(path,'matrix',m,'scale',[m.GetRow3(i).GetLength() for i in range(3)])
