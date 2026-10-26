"""Measured scene geometry; no scene edits and no assumed tabletop rectangle."""
import numpy as np
from scipy.spatial import ConvexHull
from openend.tasks.scene_requirements.source_geometry import floor_paths


def hull(points):
    p=np.unique(np.asarray(points,dtype=float),axis=0)
    return tuple(tuple(float(x) for x in v) for v in p[ConvexHull(p).vertices])


def meshes(stage,root):
    from pxr import Usd,UsdGeom,UsdPhysics
    from evaluation.tasks.tabletop_pick.placement import mesh_points
    p=stage.GetPrimAtPath(root)
    if not p or not p.IsActive():raise ValueError('Missing scene root '+root)
    result=[]
    for prim in Usd.PrimRange(p):
        if not prim.IsA(UsdGeom.Mesh):continue
        visible=(UsdGeom.Imageable(prim).ComputeVisibility()!='invisible' and
                 UsdGeom.Imageable(prim).ComputePurpose() in ('default','render'))
        colliding=prim.HasAPI(UsdPhysics.CollisionAPI) and prim.GetAttribute('physics:collisionEnabled').Get() is not False
        if visible or colliding:result.append((prim,mesh_points(stage,prim),bool(colliding),bool(visible)))
    if not result:raise ValueError('No measured mesh geometry '+root)
    return result


def top_surface(stage,root,*,convex_required=True):
    from pxr import UsdGeom
    from evaluation.tasks.small_table_pick.current_table_reset import polygon_area
    rows=meshes(stage,root)
    z=max(float(p[:,2].max()) for _,p,_,v in rows if v)
    faces=[];paths=set()
    for prim,points,colliding,visible in rows:
        if not visible:continue
        m=UsdGeom.Mesh(prim);idx=m.GetFaceVertexIndicesAttr().Get();offset=0
        holes=set(m.GetHoleIndicesAttr().Get() or [])
        for i,n in enumerate(m.GetFaceVertexCountsAttr().Get()):
            face=points[list(idx[offset:offset+n])];offset+=n
            if n<3 or i in holes or not np.all(np.abs(face[:,2]-z)<1e-6):continue
            if polygon_area(face[:,:2].tolist())<=1e-10:continue
            faces.append(face[:,:2].tolist());paths.add(str(prim.GetPath()))
    if not faces:raise ValueError('No planar top '+root)
    polygon=hull([v for f in faces for v in f])
    # Refuse disconnected/holed/overlapping tops instead of silently filling them.
    area=sum(polygon_area(f) for f in faces)
    covered=bool(np.isclose(area,polygon_area(list(polygon)),rtol=1e-5,atol=1e-8))
    if convex_required and not covered:
        raise ValueError('Top is not a single covered convex surface '+root)
    return dict(z=z,polygon=polygon,mesh_paths=sorted(paths),convex_coverage_verified=covered)


class SceneGeometry:
    def __init__(self,stage,manifest):
        from pxr import Usd,UsdGeom,UsdPhysics,Gf
        from evaluation.tasks.shelf_pick.geometry import rotation
        if UsdGeom.GetStageUpAxis(stage)!='Z' or UsdGeom.GetStageMetersPerUnit(stage)!=1:raise ValueError('Expected metres Z-up')
        rows=[o for o in manifest['objects'] if o['id']=='remote_tv']
        if len(rows)!=1:raise ValueError('Exactly one scene-owned remote required')
        self.target=rows[0]['path'];self.entry=rows[0]['entry']
        self.source=None;self.destination='/World/Assets/TvTable'
        root=stage.GetPrimAtPath(self.target)
        if not root:raise ValueError('Missing remote')
        bodies=[p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.RigidBodyAPI) and p.GetAttribute('physics:rigidBodyEnabled').Get() is not False]
        if len(bodies)!=1 or bodies[0].GetAttribute('physics:kinematicEnabled').Get() is True:raise ValueError('One dynamic remote body required')
        self.body=str(bodies[0].GetPath())
        target_meshes=meshes(stage,self.target)
        if not any(c for _,_,c,_ in target_meshes):raise ValueError('Remote has no collider')
        matrix=UsdGeom.XformCache().GetLocalToWorldTransform(bodies[0]);q=matrix.RemoveScaleShear().ExtractRotationQuat()
        self.initial_pose=(tuple(matrix.ExtractTranslation()),(q.GetReal(),*q.GetImaginary()))
        points=np.vstack([p for _,p,_,_ in target_meshes])
        self.local=rotation(self.initial_pose[1]).inv().apply(points-np.asarray(self.initial_pose[0]))
        self.top=top_surface(stage,self.destination);self.source_top=None
        self.floors=[]
        for path in floor_paths(stage):
            p=stage.GetPrimAtPath(path)
            if p and any(c.HasAPI(UsdPhysics.CollisionAPI) and c.GetAttribute('physics:collisionEnabled').Get() is not False for c in Usd.PrimRange(p)):self.floors.append(path)
        if len(self.floors)!=1:raise ValueError('Missing/duplicate physical floor')
        for path in (self.destination,):
            if not any(c for _,_,c,_ in meshes(stage,path)):raise ValueError('Missing furniture colliders '+path)
        self.furniture=(self.destination,'/World/Assets/Tv')
        self.fixed={p:UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(p))
                    for p in (self.destination,*self.floors,*self.furniture) if stage.GetPrimAtPath(p)}
        self.exclusions=[]
        # Actual other object geometry (including TV) conservatively excludes its XY projection.
        candidates={o['path'] for o in manifest['objects'] if o['path']!=self.target}
        if stage.GetPrimAtPath('/World/Assets/Tv'):candidates.add('/World/Assets/Tv')
        from .geometry import overlap
        cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render'],useExtentsHint=False)
        for path in sorted(candidates):
            prim=stage.GetPrimAtPath(path)
            if not prim:raise ValueError('Missing inventory object '+path)
            box=cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if box.IsEmpty():raise ValueError('Empty inventory object '+path)
            lo,hi=box.GetMin(),box.GetMax()
            coarse=((lo[0],lo[1]),(hi[0],lo[1]),(hi[0],hi[1]),(lo[0],hi[1]))
            if hi[2]<self.top['z'] or not overlap(coarse,self.top['polygon']):continue
            ps=np.vstack([p for _,p,_,_ in meshes(stage,path)])
            poly=hull(ps[:,:2])
            if ps[:,2].max()>=self.top['z'] and overlap(poly,self.top['polygon']):self.exclusions.append(poly)

    def footprint(self,pose):
        from evaluation.tasks.shelf_pick.geometry import rotation
        return hull((rotation(pose.quaternion_wxyz).apply(self.local)+pose.position_m)[:,:2])

    def validate_static(self,stage):
        from pxr import UsdGeom
        for p,m in self.fixed.items():
            if UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(p))!=m:raise RuntimeError('Bound furniture moved '+p)
