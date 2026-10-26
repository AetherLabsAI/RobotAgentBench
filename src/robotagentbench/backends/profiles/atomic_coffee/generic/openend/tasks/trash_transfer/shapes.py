"""Read-only primitive sampling, matching trash_disposal.bin_geometry's 48 ring points."""
import math


def cylinder_vertices(radius, height, axis):
    if axis not in ('X','Y','Z') or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in (radius,height)):
        raise ValueError('Finite positive authored cylinder dimensions and axis required')
    points=[]
    for k in range(48):
        a=2*math.pi*k/48;u,v=radius*math.cos(a),radius*math.sin(a)
        for z in (-height/2,height/2):
            points.append({'X':(z,u,v),'Y':(u,z,v),'Z':(u,v,z)}[axis])
    return points


def shape_points(stage, root):
    from pxr import Usd,UsdGeom,UsdPhysics,Gf
    import numpy as np
    result=[]
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if not prim.IsA(UsdGeom.Boundable):continue
        image=UsdGeom.Imageable(prim)
        colliding=prim.HasAPI(UsdPhysics.CollisionAPI) and prim.GetAttribute('physics:collisionEnabled').Get() is not False
        visible=image.ComputeVisibility()!='invisible' and image.ComputePurpose() in ('default','render')
        if not (colliding or visible):continue
        if prim.IsA(UsdGeom.Mesh):points=UsdGeom.Mesh(prim).GetPointsAttr().Get()
        elif prim.IsA(UsdGeom.Cube):
            h=float(UsdGeom.Cube(prim).GetSizeAttr().Get())/2
            points=[(x,y,z) for x in (-h,h) for y in (-h,h) for z in (-h,h)]
        elif prim.IsA(UsdGeom.Cylinder):
            c=UsdGeom.Cylinder(prim)
            points=cylinder_vertices(float(c.GetRadiusAttr().Get()),float(c.GetHeightAttr().Get()),str(c.GetAxisAttr().Get() or 'Z'))
        else:raise ValueError('Unsupported active garbage shape: '+str(prim.GetPath()))
        m=UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        result.append((prim,np.array([list(m.Transform(Gf.Vec3d(*p))) for p in points]),colliding,visible))
    if not result:raise ValueError('Empty garbage geometry')
    return result
