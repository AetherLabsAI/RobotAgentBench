"""Read the current stage BEFORE adding the approved arm-only filters."""
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics

def value(x):
    if x is None or isinstance(x,(str,int,float,bool)): return x
    if hasattr(x,'GetReal'): return [float(x.GetReal()),*map(float,x.GetImaginary())]
    return [value(y) for y in x]

def from_stage(stage):
    root=stage.GetPrimAtPath('/World/Robot')
    if not root: raise ValueError('Robot missing for collision guard')
    cache=UsdGeom.XformCache()
    result=dict(shapes=[],joints=[],filters=[])
    for p in Usd.PrimRange(root,Usd.TraverseInstanceProxies()):
        for target in p.GetRelationship('physics:filteredPairs').GetTargets():
            result['filters'].append([str(p.GetPath()),str(target)])
        if p.IsA(UsdPhysics.Joint):
            attrs={a.GetName():value(a.Get()) for a in p.GetAttributes() if a.GetName().startswith('physics:')}
            result['joints'].append(dict(name=p.GetName(),type=p.GetTypeName(),attrs=attrs,
                body0=list(map(str,p.GetRelationship('physics:body0').GetTargets())),
                body1=list(map(str,p.GetRelationship('physics:body1').GetTargets()))))
        if not p.HasAPI(UsdPhysics.CollisionAPI) or p.GetAttribute('physics:collisionEnabled').Get() is False:
            continue
        body=p
        while body and not body.HasAPI(UsdPhysics.RigidBodyAPI): body=body.GetParent()
        if not body: raise ValueError('Robot collider without body: '+str(p.GetPath()))
        relative=cache.GetLocalToWorldTransform(p)*cache.GetLocalToWorldTransform(body).GetInverse()
        row=dict(path=str(p.GetPath()),body=str(body.GetPath()),type=p.GetTypeName(),
                 local=np.asarray(relative,dtype=float).T.tolist())
        # Faces are not needed: existing convexHull/enclosing-hull construction
        # uses the original vertex coordinates without changing the USD asset.
        for name in ('size','radius','height','axis','points','physics:approximation'):
            attr=p.GetAttribute(name)
            if attr and attr.Get() is not None: row[name]=value(attr.Get())
        result['shapes'].append(row)
    return result
