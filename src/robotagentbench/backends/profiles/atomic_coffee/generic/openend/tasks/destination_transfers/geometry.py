"""Read-only destination geometry. No physical asset or pose modifications.

Basket: exactly the old 14-4 origin-in-zone predicate, transformed with the
actual basket. It is NOT a watertight cavity/whole-object certificate. The mesh
has multiple openings; never label an AABB or an invalid winding test success.
Upward contact on the actual basket mesh (not the riser) is required separately.
"""
from functools import lru_cache
from pathlib import Path
import numpy as np
from openend.tasks.remote_transfer.native_geometry import meshes,hull,top_surface
from evaluation.tasks.shelf_pick.geometry import rotation
from .registration import BASKET,TABLE

def target_geometry(self,stage,manifest,asset):
    from pxr import Usd,UsdGeom,UsdPhysics
    if UsdGeom.GetStageUpAxis(stage)!='Z' or UsdGeom.GetStageMetersPerUnit(stage)!=1:
        raise ValueError('Metres Z-up required')
    scenes=[p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if len(scenes)!=1 or scenes[0].GetAttribute('physxScene:enableGPUDynamics').Get() is not False:
        raise ValueError('Existing CPU physics scene required')
    rows=[o for o in manifest['objects'] if o['id']==asset]
    if len(rows)!=1:raise ValueError('Exactly one scene-owned '+asset+' required')
    self.target=rows[0]['path'];self.source=None;self.source_top=None;self.exclusions=[]
    bodies=[p for p in Usd.PrimRange(stage.GetPrimAtPath(self.target)) if p.HasAPI(UsdPhysics.RigidBodyAPI)
            and p.GetAttribute('physics:rigidBodyEnabled').Get() is not False]
    if len(bodies)!=1 or bodies[0].GetAttribute('physics:kinematicEnabled').Get() is True:
        raise ValueError('One dynamic target required')
    self.body=str(bodies[0].GetPath());cache=UsdGeom.XformCache()
    m=cache.GetLocalToWorldTransform(bodies[0]);q=m.RemoveScaleShear().ExtractRotationQuat()
    self.initial_pose=(tuple(m.ExtractTranslation()),(q.GetReal(),*q.GetImaginary()))
    data=meshes(stage,self.target)
    if not any(c for _,_,c,_ in data):raise ValueError('Target has no collider')
    self.local=rotation(self.initial_pose[1]).inv().apply(np.vstack([v for _,v,c,visible in data if c or visible])-self.initial_pose[0])
    from openend.tasks.scene_requirements.source_geometry import floor_paths
    self.floors=floor_paths(stage)
    if not self.floors:raise ValueError('Physical layout floor required')
    self.fixed={};self.fixed_points={}

class FixedTarget:
    def footprint(self,pose):return hull(self.points(pose)[:,:2])
    def points(self,pose):return rotation(pose.quaternion_wxyz).apply(self.local)+pose.position_m
    def freeze(self,stage,root):
        from pxr import Usd,UsdGeom,UsdPhysics
        for p in Usd.PrimRange(stage.GetPrimAtPath(root)):
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                # A table may contain its dynamic drawer, outside the top reader.
                continue
            if p.IsA(UsdGeom.Mesh):
                self.fixed[str(p.GetPath())]=UsdGeom.XformCache().GetLocalToWorldTransform(p)
                self.fixed_points[str(p.GetPath())]=tuple(tuple(x) for x in UsdGeom.Mesh(p).GetPointsAttr().Get())
        self.fixed[root]=UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(root))
    def validate_static(self,stage):
        from openend.tasks.meal_tray.nonmotion_closeout_v1.static_cache import for_stage
        cache=for_stage(stage)
        for p,m in self.fixed.items():
            if cache.matrix(p)!=m:raise RuntimeError('Destination geometry moved '+p)
        for p,v in self.fixed_points.items():
            if cache.mesh_points(p)!=v:raise RuntimeError('Destination mesh changed '+p)

@lru_cache(maxsize=1)
def canonical_basket():
    import json
    from pxr import Gf
    # Exact audited legacy geometry, extracted once; no old room or external
    # scene layers need to be loaded during a task.
    record=json.loads(Path(__file__).with_name('canonical_basket_v1.json').read_text())
    if record['schema']!='openend.canonical_basket_geometry.v1' or record['basket_path']!=BASKET:
        raise ValueError('Invalid canonical basket reference')
    matrix=np.asarray(record['reference_matrix'],dtype=float)
    local=np.asarray(record['local_points_homogeneous'],dtype=float)
    if (matrix.shape!=(4,4) or local.ndim!=2 or local.shape[1]!=4
            or not np.isfinite(matrix).all() or not np.isfinite(local).all()):
        raise ValueError('Invalid canonical basket geometry')
    return (Gf.Matrix4d(*matrix.reshape(-1).tolist()), local,
            tuple(record['face_vertex_indices']), record['collider_suffix'])

class BasketGeometry(FixedTarget):
    def __init__(self,stage,manifest,asset):
        from pxr import Usd,UsdGeom,UsdPhysics
        target_geometry(self,stage,manifest,asset)
        self.destination=BASKET;self.cabinet=BASKET+'/riser'
        reference,local,indices,suffix=canonical_basket()
        p=stage.GetPrimAtPath(BASKET);matrix=UsdGeom.XformCache().GetLocalToWorldTransform(p)
        if not p or not p.IsActive() or not stage.GetPrimAtPath(self.cabinet):raise ValueError('Missing actual basket/riser')
        if any(q.HasAPI(UsdPhysics.RigidBodyAPI) or any('Deformable' in str(s) for s in q.GetAppliedSchemas()) for q in Usd.PrimRange(p)):
            raise ValueError('Basket must remain fixed CPU infrastructure')
        self.contact_destination=BASKET+suffix
        mesh=stage.GetPrimAtPath(self.contact_destination)
        data=next((v for q,v,c,_ in meshes(stage,BASKET) if c and str(q.GetPath())==self.contact_destination),None)
        if data is None or mesh.GetAttribute('physics:approximation').Get()!='none':raise ValueError('Original concave basket collider required')
        actual=np.c_[data,np.ones(len(data))]@np.asarray(matrix.GetInverse())
        if (actual.shape!=local.shape or not np.allclose(actual,local,atol=1e-6,rtol=0)
            or tuple(UsdGeom.Mesh(mesh).GetFaceVertexIndicesAttr().Get())!=indices):
            raise ValueError('Basket differs from audited legacy 14-4 geometry')
        # Row-vector transforms: current world -> basket -> canonical world.
        self.to_canonical=np.asarray(matrix.GetInverse()*reference)
        from evaluation.tasks.tidy_sofa.specs import spec
        old=spec('14-4');self.zone=old.zone_xy;self.zone_top=old.zone_top_z
        self.top=dict(semantics='legacy_14_4_origin_zone_plus_actual_basket_mesh_support',polygon=hull(data[:,:2]),whole_object_containment_verified=False)
        self.furniture=(self.cabinet,self.destination);self.freeze(stage,BASKET)
    def insertion_evidence(self,pose):
        p=np.asarray([*pose.position_m,1.])@self.to_canonical
        cx,cy,hx,hy=self.zone
        return bool(abs(p[0]-cx)<=hx and abs(p[1]-cy)<=hy and p[2]<self.zone_top+.10),False

def polygon_contains(polygon,points):
    poly=np.asarray(polygon,dtype=float);p=np.asarray(points,dtype=float)
    edge=np.roll(poly,-1,axis=0)-poly
    cross=edge[None,:,0]*(p[:,None,1]-poly[None,:,1])-edge[None,:,1]*(p[:,None,0]-poly[None,:,0])
    return bool(np.all(cross>=-1e-9) or np.all(cross<=1e-9))

class TrayGeometry(FixedTarget):
    def __init__(self,stage,manifest):
        from pxr import UsdGeom
        target_geometry(self,stage,manifest,'meal_tray_large')
        self.destination=TABLE;self.cabinet=TABLE
        from openend.tasks.coffee_table_workspaces.binding import bind
        region,=[r for r in bind(stage,manifest) if r.id=='table_objects']
        self.region=region.polygon;self.top=top_surface(stage,TABLE)
        # Some historical rectangles extend a few mm past the actual mesh.
        # Keep the approved rectangle unchanged and require the target itself
        # to fit BOTH it and the measured tabletop. Never expand the table.
        from openend.tasks.remote_transfer.geometry import overlap
        if not overlap(self.top['polygon'],self.region):raise ValueError('Tray region misses actual tabletop')
        self.contact_roots=tuple(self.top['mesh_paths'])
        if not self.contact_roots:raise ValueError('No measured tabletop')
        self.furniture=(TABLE,)
        # Do not freeze the drawer; it is another task's movable object.
        cache=UsdGeom.XformCache()
        for path in (TABLE,*self.contact_roots):
            self.fixed[path]=cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
        for path in self.contact_roots:
            self.fixed_points[path]=tuple(tuple(x) for x in UsdGeom.Mesh(stage.GetPrimAtPath(path)).GetPointsAttr().Get())
    def insertion_evidence(self,pose):
        footprint=self.footprint(pose)
        return polygon_contains(self.region,footprint) and polygon_contains(self.top['polygon'],footprint),False
