"""Read-only local collision surface cache. No world AABB distance shortcut."""
import numpy as np


DISTANCE_CAP_M = .4


def scoring_distance(points,surface,cap=DISTANCE_CAP_M):
    """Exact inside 40 cm, capped farther away; conservative triangle pruning.

    A triangle's AABB is only a lower-bound cull, never a scored surface.
    The cap is beyond all scoring thresholds, not an additional award boundary.
    """
    best=cap
    low,high=surface['aabb_min'],surface['aabb_max']
    for p in np.asarray(points):
        delta=np.maximum(np.maximum(low-p,p-high),0.)
        selected=np.einsum('ij,ij->i',delta,delta)<=best*best
        if selected.any():best=min(best,point_surface_distance([p],surface['triangles'][selected]))
    if surface['capsules']:
        best=min(best,point_surface_distance(points,dict(triangles=[],capsules=surface['capsules'])))
    return best


def point_surface_distance(points, triangles):
    """Exact point-to-triangle distance; vectorized, including degenerate faces."""
    if isinstance(triangles,dict):
        distances=[]
        if len(triangles['triangles']):
            distances.append(point_surface_distance(points,triangles['triangles']))
        for a,b,radius in triangles['capsules']:
            a,b=np.asarray(a),np.asarray(b);v=b-a;vv=float(v@v)
            for p in np.asarray(points):
                u=float(np.clip((p-a)@v/vv,0,1)) if vv else 0.
                distances.append(max(0.,float(np.linalg.norm(p-a-u*v))-radius))
        if not distances:raise ValueError('Empty collision surfaces')
        return min(distances)
    points=np.asarray(points,dtype=float);t=np.asarray(triangles,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or t.ndim!=3 or t.shape[1:]!=(3,3):
        raise ValueError('Expected points and triangle surfaces')
    if not len(points) or not len(t) or not np.isfinite(points).all() or not np.isfinite(t).all():
        raise ValueError('Finite nonempty geometry required')
    best=float('inf')
    for p in points:
        a,b,c=t[:,0],t[:,1],t[:,2];ab=b-a;ac=c-a;ap=p-a
        n=np.cross(ab,ac);nn=np.einsum('ij,ij->i',n,n)
        safe=np.where(nn>1e-24,nn,1.)
        signed=np.einsum('ij,ij->i',ap,n)
        q=p-(signed/safe)[:,None]*n
        inside=np.ones(len(t),dtype=bool)
        edge_best=np.full(len(t),np.inf)
        for x,y in ((a,b),(b,c),(c,a)):
            edge=y-x;den=np.einsum('ij,ij->i',edge,edge)
            u=np.clip(np.einsum('ij,ij->i',p-x,edge)/np.where(den>0,den,1),0,1)
            delta=p-x-u[:,None]*edge
            edge_best=np.minimum(edge_best,np.einsum('ij,ij->i',delta,delta))
            inside &= np.einsum('ij,ij->i',np.cross(edge,q-x),n)>=-1e-14
        d=np.where(inside&(nn>1e-24),signed*signed/safe,edge_best)
        best=min(best,float(np.sqrt(max(0.,d.min()))))
    return best


def collision_surfaces(stage, roots):
    """Map actual body paths to local triangle arrays, with USD scale baked in."""
    from pxr import Usd,UsdGeom,UsdPhysics,Gf
    from evaluation.tasks.find_object.isaac_evidence import collider_points
    from scipy.spatial import ConvexHull
    cache=UsdGeom.XformCache();surfaces={};capsules={};seen=set()
    for root in roots:
        prim=stage.GetPrimAtPath(root)
        if not prim or not prim.IsActive():raise ValueError('Missing interaction geometry '+root)
        for p in Usd.PrimRange(prim,Usd.TraverseInstanceProxies()):
            if not p.HasAPI(UsdPhysics.CollisionAPI) or p.GetAttribute('physics:collisionEnabled').Get() is False:continue
            if str(p.GetPath()) in seen:continue
            seen.add(str(p.GetPath()))
            body=p
            while body and not body.IsPseudoRoot() and not body.HasAPI(UsdPhysics.RigidBodyAPI):body=body.GetParent()
            if not body or body.IsPseudoRoot():raise ValueError('Interaction collider has no native body '+str(p.GetPath()))
            world=cache.GetLocalToWorldTransform(body);rot=Gf.Matrix4d(world);rot.Orthonormalize()
            frame=Gf.Matrix4d(rot.ExtractRotationMatrix(),world.ExtractTranslation())
            transform=cache.GetLocalToWorldTransform(p)*frame.GetInverse()
            key=str(body.GetPath())
            if p.IsA(UsdGeom.Capsule) or p.IsA(UsdGeom.Sphere):
                shape=UsdGeom.Capsule(p) if p.IsA(UsdGeom.Capsule) else UsdGeom.Sphere(p)
                basis=np.asarray([transform.TransformDir(Gf.Vec3d(*(1 if i==j else 0 for j in range(3)))) for i in range(3)])
                scales=np.linalg.norm(basis,axis=1)
                if not np.allclose(basis@basis.T,np.eye(3)*scales[0]**2,rtol=1e-6,atol=1e-10):
                    raise ValueError('Nonuniform analytic collision scale '+str(p.GetPath()))
                endpoint=np.zeros(3)
                if p.IsA(UsdGeom.Capsule):
                    endpoint['XYZ'.index(shape.GetAxisAttr().Get())]=shape.GetHeightAttr().Get()/2
                a=np.asarray(transform.Transform(Gf.Vec3d(*(-endpoint))))
                b=np.asarray(transform.Transform(Gf.Vec3d(*endpoint)))
                capsules.setdefault(key,[]).append((a,b,float(shape.GetRadiusAttr().Get()*scales[0])))
                continue
            vertices=collider_points(p,transform)
            if p.IsA(UsdGeom.Mesh):
                mesh=UsdGeom.Mesh(p);idx=list(mesh.GetFaceVertexIndicesAttr().Get());counts=mesh.GetFaceVertexCountsAttr().Get()
                faces=[];offset=0
                for count in counts:
                    face=idx[offset:offset+count];offset+=count
                    faces.extend((face[0],face[i],face[i+1]) for i in range(1,count-1))
                if not faces:raise ValueError('Empty collision mesh')
                triangles=vertices[np.asarray(faces)]
            else:
                # Cube exact; cylinder retains the existing 48-segment geometry.
                triangles=vertices[ConvexHull(vertices).simplices]
            surfaces.setdefault(key,[]).append(triangles)
    if not surfaces and not capsules:raise ValueError('No enabled interaction collision surface')
    result={}
    for k in surfaces.keys()|capsules.keys():
        triangles=np.concatenate(surfaces[k]) if k in surfaces else np.empty((0,3,3))
        result[k]=dict(triangles=triangles,capsules=capsules.get(k,[]),
                       aabb_min=triangles.min(axis=1),aabb_max=triangles.max(axis=1))
    return result
