"""Task-local geometry for explicit drawer/table combinations; read-only."""
from types import SimpleNamespace
import numpy as np
from evaluation.tasks.shelf_pick.geometry import rotation
from openend.tasks.remote_transfer.native_geometry import hull, top_surface
from openend.tasks.scene_requirements.source_geometry import floor_paths

DESTINATIONS={
 'bag':('/World/Assets/Task5PackStation/ToteSpawn/Tote','/World/Assets/Task5PackStation/Collider'),
 'surface':('/World/Assets/SmallCoffeeTable','/World/Assets/SmallCoffeeTable'),
 'hang':('/World/Assets/CoatRack','/World/Assets/CoatRack'),
 'trash':('/World/Assets/TrashBinRed','/World/Assets/TrashBinRed'),
 'drawer':('/World/Assets/CoffeeTable/black_box_1','/World/Assets/CoffeeTable'),
}

def shape_points(stage,root):
    from pxr import Usd,UsdGeom,UsdPhysics,Gf
    result=[]
    p=stage.GetPrimAtPath(root)
    if not p:raise ValueError('Missing geometry '+root)
    for prim in Usd.PrimRange(p):
        if not prim.IsA(UsdGeom.Boundable):continue
        image=UsdGeom.Imageable(prim)
        c=prim.HasAPI(UsdPhysics.CollisionAPI) and prim.GetAttribute('physics:collisionEnabled').Get() is not False
        v=image.ComputeVisibility()!='invisible' and image.ComputePurpose() in ('default','render')
        if not (c or v):continue
        if prim.IsA(UsdGeom.Mesh):points=UsdGeom.Mesh(prim).GetPointsAttr().Get()
        elif prim.IsA(UsdGeom.Cube):
            h=float(UsdGeom.Cube(prim).GetSizeAttr().Get())/2
            points=[(x,y,z) for x in (-h,h) for y in (-h,h) for z in (-h,h)]
        else:raise ValueError('Unsupported target geometry '+str(prim.GetPath()))
        m=UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        result.append((prim,np.array([list(m.Transform(Gf.Vec3d(*x))) for x in points]),c,v))
    if not result:raise ValueError('Empty geometry '+root)
    return result

def under(path,root):return path==root or path.startswith(root+'/')

def _source_object_roots(manifest,target,mode):
    """Separate declared support objects from furniture identity candidates.

    Raw measured supports remain unchanged in initial checks and scoring.
    A genuine remaining furniture contact is still required by the resolver.
    """
    targets=[o for o in manifest['objects'] if o['path']==target]
    if len(targets)!=1:raise ValueError('Exact scene-owned target required')
    source=targets[0].get('source')
    if not isinstance(source,str) or not source:return frozenset()
    background={o['path'] for o in manifest['objects']
        if o['path']!=target and o.get('role')=='background'
        and o.get('task_role')=='background' and o.get('scoring_target') is False
        and o.get('source')==source}
    # Original build_living/build_shelf declare these fixed CoffeeTable-relative
    # anchors as workspace_support, not additional furniture or task targets.
    workspace={o['path'] for o in manifest['objects']
        if source=='coffee_table.objects' and o['path']!=target
        and o.get('role')=='workspace_support'
        and o.get('id') in ('tablecloth_dirty','japanese_tray')}
    selected={o['path'] for o in manifest['objects']
        if o['path']!=target and o.get('source')==source
        and isinstance(o.get('task_id'),str) and o['task_id'].strip()
        and isinstance(o.get('instance_id'),str) and o['instance_id'].strip()}
    return frozenset(background|workspace|selected)

class Geometry:
    def __init__(self,stage,manifest,asset,mode,target_path=None):
        from pxr import Usd,UsdGeom,UsdPhysics
        self.stage=stage;self.mode=mode
        if UsdGeom.GetStageUpAxis(stage)!='Z' or UsdGeom.GetStageMetersPerUnit(stage)!=1:raise ValueError('Metre Z-up required')
        scenes=[p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
        if len(scenes)!=1 or scenes[0].GetAttribute('physxScene:enableGPUDynamics').Get() is not False:raise ValueError('Existing CPU scene required')
        rows=[o for o in manifest['objects'] if o['id']==asset and (target_path is None or o['path']==target_path)
              and not (asset=='book_red' and o.get('task_role')=='cloth_clear_target')]
        if len(rows)!=1:raise ValueError('Exact scene-owned object required '+asset)
        self.target=rows[0]['path'];self.source=None;self.source_top=None
        self._source_object_roots=_source_object_roots(manifest,self.target,mode)
        self.destination,self.cabinet=(('/World/Assets/CoffeeTable','/World/Assets/CoffeeTable')
            if mode=='surface' else DESTINATIONS[mode])
        self.top=None;self.exclusions=[];self.extra_bodies={}
        bodies=[p for p in Usd.PrimRange(stage.GetPrimAtPath(self.target)) if p.HasAPI(UsdPhysics.RigidBodyAPI) and p.GetAttribute('physics:rigidBodyEnabled').Get() is not False]
        if len(bodies)!=1 or bodies[0].GetAttribute('physics:kinematicEnabled').Get() is True:raise ValueError('One dynamic target body required '+self.target)
        self.body=str(bodies[0].GetPath())
        m=UsdGeom.XformCache().GetLocalToWorldTransform(bodies[0]);q=m.RemoveScaleShear().ExtractRotationQuat()
        self.initial_pose=(tuple(m.ExtractTranslation()),(q.GetReal(),*q.GetImaginary()))
        self.parts=[];self.cube_parts=set();visible=[]
        for prim,points,c,v in shape_points(stage,self.target):
            local=rotation(self.initial_pose[1]).inv().apply(points-self.initial_pose[0])
            if c:
                self.parts.append((str(prim.GetPath()),local))
                if prim.IsA(UsdGeom.Cube):self.cube_parts.add(str(prim.GetPath()))
            if v:visible.append(local)
        if not self.parts:raise ValueError('Missing target collision geometry')
        self.local=np.vstack([p for _,p in self.parts]);self.visible=np.vstack(visible) if visible else self.local
        self.floors=floor_paths(stage)
        if len(self.floors)!=1:raise ValueError('Unique floor required')
        for path in (self.destination,self.cabinet):
            prim=stage.GetPrimAtPath(path)
            if not prim or not any(p.HasAPI(UsdPhysics.CollisionAPI) and p.GetAttribute('physics:collisionEnabled').Get() is not False for p in Usd.PrimRange(prim)):raise ValueError('Missing destination colliders '+path)
        self.fixed={p:UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(p)) for p in (*self.floors,self.cabinet)}
        if mode=='bag':
            from evaluation.tasks.basket_place.cavity_geometry import OpenCavity
            sim=stage.GetPrimAtPath(self.destination+'/Simulation')
            if not sim or sim.GetAttribute('physics:approximation').Get()!='none':raise ValueError('Existing fixed triangle bag required')
            if any(p.HasAPI(UsdPhysics.RigidBodyAPI) or any('Deformable' in s for s in p.GetAppliedSchemas()) for p in Usd.PrimRange(stage.GetPrimAtPath(self.destination))):raise ValueError('CPU static bag only')
            mesh=UsdGeom.Mesh(sim)
            if any(n!=3 for n in mesh.GetFaceVertexCountsAttr().Get()):raise ValueError('Triangulated bag required')
            self.bag_prim=sim;self.bag_authored=tuple(map(tuple,mesh.GetPointsAttr().Get()))
            self.bag_points=shape_points(stage,str(sim.GetPath()))[0][1]
            self.cavity=OpenCavity(np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3),len(self.bag_points))
            self.fixed[self.destination]=UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(self.destination))
            self.fixed[str(sim.GetPath())]=UsdGeom.XformCache().GetLocalToWorldTransform(sim)
        elif mode=='surface':
            # Keep the actual union of top faces, not a filled convex hull over holes.
            from openend.tasks.remote_transfer.native_geometry import meshes
            from evaluation.tasks.small_table_pick.current_table_reset import polygon_area
            rows=meshes(stage,self.destination)
            z=max(float(p[:,2].max()) for _,p,_,v in rows if v)
            faces=[]
            for prim,points,colliding,visible in rows:
                if not visible:continue
                mesh=UsdGeom.Mesh(prim);indices=mesh.GetFaceVertexIndicesAttr().Get();offset=0
                holes=set(mesh.GetHoleIndicesAttr().Get() or [])
                for i,n in enumerate(mesh.GetFaceVertexCountsAttr().Get()):
                    face=points[list(indices[offset:offset+n])];offset+=n
                    if n<3 or i in holes or not np.all(np.abs(face[:,2]-z)<1e-6):continue
                    if polygon_area(face[:,:2].tolist())>1e-10:faces.append(face[:,:2].tolist())
            if not faces:raise ValueError('No measured tabletop faces')
            self.top=dict(z=z,faces=faces,policy='actual_planar_face_union_not_convex_envelope')
            from openend.tasks.coffee_table_workspaces.binding import bind
            matches=[r for r in bind(stage,manifest) if r.id=='table_objects']
            if len(matches)!=1:raise ValueError('Unique approved coffee-table object region required')
            self.legal_region=matches[0].polygon
        elif mode=='hang':
            from evaluation.tasks.tidy_sofa.hang_hat.process import ALLOWED_HOOKS
            for b in ALLOWED_HOOKS:
                for part in ('Shaft','Tip'):
                    p=stage.GetPrimAtPath(self.destination+'/'+b+part)
                    if not p or not p.HasAPI(UsdPhysics.CollisionAPI) or p.GetAttribute('physics:collisionEnabled').Get() is False:raise ValueError('Missing real hook '+b+part)
        elif mode=='trash':
            from evaluation.tasks.trash_disposal.bin_geometry import SceneGeometry
            # The old reader takes paths via its definition; no global constants changed.
            self.bin=SceneGeometry(stage,SimpleNamespace(target=self.target,destination=self.destination))
            # Imported assets may put the actual rigid body below the instance root.
            # Express the old full-visible/collider hull in the measured BODY frame.
            frame=UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(self.target)).RemoveScaleShear()
            q=frame.ExtractRotationQuat()
            points=rotation((q.GetReal(),*q.GetImaginary())).apply(self.bin.target_vertices)+np.asarray(frame.ExtractTranslation())
            self.bin.target_vertices=rotation(self.initial_pose[1]).inv().apply(points-self.initial_pose[0])
        else:
            from evaluation.tasks.tidy_sofa.specs import DRAWER_JOINT
            p=stage.GetPrimAtPath(self.destination);j=stage.GetPrimAtPath(DRAWER_JOINT)
            if p.GetAttribute('physics:kinematicEnabled').Get() is not False or not j or not j.IsA(UsdPhysics.PrismaticJoint):raise ValueError('Scene lacks dynamic drawer and existing slide: '+DRAWER_JOINT)
            import json
            calibration=p.GetCustomDataByKey('openend:drawerCalibration')
            if not calibration:raise ValueError('Drawer scene-frame calibration required before native scoring')
            self.drawer=json.loads(calibration)
            self.drawer_rotation=np.asarray(self.drawer['source_to_layout_matrix'])[:3,:3]
            if not np.allclose(self.drawer_rotation@self.drawer_rotation.T,np.eye(3),atol=1e-6):raise ValueError('Nonrigid drawer calibration')
            self.extra_bodies[self.destination]=self.destination

    def source_furniture(self,supports):
        # Resolve actual contacts to their owning scene furniture, not a source whitelist.
        from pxr import UsdPhysics
        result=set()
        for support in supports:
            if under(support,'/World/Assets/SeatOverlay'):result.add('/World/Assets/Sofa');continue
            p=self.stage.GetPrimAtPath(support)
            if not p:raise ValueError('Unknown measured support '+support)
            if any(under(support,root) for root in self._source_object_roots):continue
            while p.GetParent() and str(p.GetParent().GetPath()) not in ('/World','/World/Assets','/World/RoomEpisode'):
                p=p.GetParent()
            result.add(str(p.GetPath()))
        if len(result)!=1:raise ValueError('Ambiguous source furniture '+str(result))
        return next(iter(result))

    def world(self,pose,points=None):return rotation(pose.quaternion_wxyz).apply(self.local if points is None else points)+pose.position_m
    def footprint(self,pose):return hull(self.world(pose)[:,:2])
    def height(self,pose,reference):
        if reference=='rigid_origin':return float(pose.position_m[2])
        p=self.world(pose,self.visible)
        return float((p[:,2].min()+p[:,2].max())/2)
    def insertion_evidence(self,pose):
        from openend.tasks.meal_tray.nonmotion_closeout_v1.static_cache import exact_pose_result
        return exact_pose_result(self,pose,self._uncached_insertion_evidence)

    def _uncached_insertion_evidence(self,pose):
        from evaluation.tasks.basket_place.geometric_placement import insertion
        world=self.world(pose)
        if np.any(world.max(0)<self.bag_points.min(0)) or np.any(self.bag_points.max(0)<world.min(0)):return False,False
        points=[np.asarray(pose.position_m),world[world[:,2].argmin()]]
        names=['target_origin','lowest_authored_collider_vertex']
        for name,local in self.parts:
            points.append(self.world(pose,local).mean(0));names.append(name+':centre')
            if name in self.cube_parts:
                for i,p in enumerate(self.world(pose,local)):points.append(p);names.append(name+':corner'+str(i))
        diagnostic=self.cavity.inspect(self.bag_points,points)
        if any(v is None for v in diagnostic['witness_inside']):return False,True
        return insertion(dict(cavity=diagnostic,witness_world_positions=points,witness_names=names))

    def placement(self,pose):
        in_zone=False;rim=False;inside=False
        if self.mode=='bag':in_zone,rim=self.insertion_evidence(pose)
        elif self.mode=='surface':
            from openend.tasks.remote_transfer.geometry import inside as point_inside,polygon
            # Approved table object region excludes the cloth and wooden tray.
            footprint=self.footprint(pose)
            in_zone=(all(point_inside(p,polygon(self.legal_region)) for p in footprint)
                and all(any(point_inside(p,polygon(face)) for face in self.top['faces']) for p in footprint))
        elif self.mode=='trash':
            inside=self.bin.measure(dict(target=self.target,target_xyz_m=pose.position_m,target_wxyz=pose.quaternion_wxyz))
        return bool(in_zone),bool(rim),bool(inside)

    def drawer_measure(self,pose,drawer_pose):
        from evaluation.tasks.tidy_sofa.isaac_reader import DRAWER_INTERIOR_LOCAL
        # Same canonical-axis interior predicate as the old component, carried by the drawer.
        rel=self.drawer_rotation@(np.asarray(pose.position_m)-drawer_pose[:3])
        (x0,x1),(y0,y1),(z0,z1)=DRAWER_INTERIOR_LOCAL
        inside=bool(x0<=rel[0]<=x1 and y0<=rel[1]<=y1 and z0-.005<=rel[2]<=z1-.01)
        travel=float(np.dot(np.asarray(drawer_pose[:3])-self.drawer['initial_origin_m'],self.drawer_rotation[0]))+self.drawer['initial_lip_m']
        return inside,travel

    def validate_static(self,stage):
        from openend.tasks.meal_tray.nonmotion_closeout_v1.static_cache import for_stage
        cache=for_stage(stage)
        for path,m in self.fixed.items():
            if cache.matrix(path)!=m:raise RuntimeError('Bound furniture moved '+path)
        if self.mode=='bag' and cache.mesh_points(str(self.bag_prim.GetPath()))!=self.bag_authored:raise RuntimeError('Fixed bag geometry changed')
