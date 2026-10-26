import sys
from pxr import Usd,UsdGeom
s=Usd.Stage.Open(sys.argv[1]);c=UsdGeom.BBoxCache(0,['default','render'])
for p in s.Traverse():
    if p.IsA(UsdGeom.Boundable) and ('Background' in str(p.GetPath()) or 'Shell' in str(p.GetPath())):
        r=c.ComputeWorldBound(p).ComputeAlignedRange()
        print(str(p.GetPath()),list(r.GetMin()),list(r.GetMax()))
