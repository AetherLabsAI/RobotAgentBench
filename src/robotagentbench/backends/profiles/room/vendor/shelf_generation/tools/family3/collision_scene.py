#!/usr/bin/env python3
"""Read-only Family-3 geometry reconstruction. No simulator or GPU required.

Historical capture_scope=target_robot does not establish actual scene membership.
The default reconstruction retains every authored active object and activates the
canonical 22 stocked objects, without the historical neighbor-drop workaround.
This is explicitly a reconstructed full scene, not proof of historical physics.
"""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "work/family3_repair/collision/python_deps"))
import numpy as np
import trimesh
import fcl
from pxr import Usd, UsdGeom, UsdPhysics, Gf
from scipy.spatial.transform import Rotation

SCENE = ROOT / "runtime/house/demo_sim/scene_room.usda"
ROBOT = Path(
    "/mnt/agent-bench-data/lingjun/workspace/agent_bench_0807/arena_scene/v1_2/"
    "source/geniesim/assets/robot/G1_omnipicker/robot.usda"
)
STOCK = ([f"/World/Assets/ShelfItems/{x}" for x in
          "cacao cayman_jacked coke_c3 diet_coke gatorade_m gatorade_t good2grow grapefruit_can pring_g rb_box red_bull_pink sprite_box sprite_c1 sprite_c2 starbucks".split()]
         + [f"/World/Assets/Tier2Items/{x}" for x in
            "chomps_r honeymaid oreo_gf pring_bbq skit_littles smores_k zbar_t".split()])

def matrix(g):
    return np.asarray(g, float).T

def quat_matrix(q):
    q = np.asarray(q, float)
    return Rotation.from_quat(q[[1,2,3,0]]).as_matrix()

def pose(xyz, quat):
    m = np.eye(4); m[:3, :3] = quat_matrix(quat); m[:3, 3] = xyz
    return m

def points(v, m):
    return np.asarray(v) @ m[:3,:3].T + m[:3,3]

def rigid_part(m):
    u, _, vt = np.linalg.svd(m[:3,:3]); r = u @ vt
    if np.linalg.det(r) < 0:
        u[:,-1] *= -1; r = u @ vt
    out = np.eye(4); out[:3,:3] = r; out[:3,3] = m[:3,3]
    return out

def traversal(root):
    return Usd.PrimRange(root, Usd.TraverseInstanceProxies())

def shape_mesh(p):
    """Triangulate authored meshes; tessellate native analytic visual shapes."""
    if p.IsA(UsdGeom.Mesh):
        g = UsdGeom.Mesh(p)
        v = np.asarray(g.GetPointsAttr().Get(), float)
        if v.ndim != 2 or not len(v): return None
        counts = g.GetFaceVertexCountsAttr().Get(); ids = g.GetFaceVertexIndicesAttr().Get()
        faces=[]; k=0
        for n in counts:
            for j in range(1,n-1): faces.append((ids[k],ids[k+j],ids[k+j+1]))
            k+=n
        if not faces: return None
        mesh=trimesh.Trimesh(v,faces,process=False)
        if g.GetOrientationAttr().Get() == "leftHanded": mesh.faces=mesh.faces[:,::-1]
        return mesh
    if p.IsA(UsdGeom.Cube): return trimesh.creation.box(extents=[float(UsdGeom.Cube(p).GetSizeAttr().Get())]*3)
    if p.IsA(UsdGeom.Sphere): return trimesh.creation.icosphere(subdivisions=3,radius=float(UsdGeom.Sphere(p).GetRadiusAttr().Get()))
    if p.IsA(UsdGeom.Cylinder) or p.IsA(UsdGeom.Capsule):
        g=UsdGeom.Cylinder(p) if p.IsA(UsdGeom.Cylinder) else UsdGeom.Capsule(p)
        kw=dict(radius=float(g.GetRadiusAttr().Get()),height=float(g.GetHeightAttr().Get()))
        mesh=trimesh.creation.cylinder(**kw,sections=48) if p.IsA(UsdGeom.Cylinder) else trimesh.creation.capsule(**kw,count=[16,32])
        axis=str(g.GetAxisAttr().Get())
        if axis=='X': mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[0,1,0]))
        if axis=='Y': mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[1,0,0]))
        return mesh
    return None

class Body:
    def __init__(self,name,mesh,transform=None,metadata=None):
        # USD commonly duplicates vertices at UV/normal seams (even a box has24).
        # Welding equal coordinates changes topology only, preserving the surface.
        mesh.merge_vertices(digits_vertex=12)
        self.name=name; self.mesh=mesh; self.transform=np.eye(4) if transform is None else transform
        self.metadata=metadata or {}; self.v=np.asarray(mesh.vertices); self.f=np.asarray(mesh.faces,dtype=np.int32)
        self.authored_transform=self.transform.copy()
        # After no triangle crossing, each connected source component has one
        # inside/outside classification against a closed destination surface.
        # One vertex per component is complete for this containment case and
        # avoids 32 repeated rays against large furniture meshes.
        labels=trimesh.graph.connected_component_labels(mesh.edges_unique,node_count=len(self.v))
        self.component_vertex_ids=np.unique(labels,return_index=True)[1]
        self.vertex_components=labels; self._solid_components=None
        self.model=fcl.BVHModel(); self.model.beginModel(len(self.v),len(self.f))
        self.model.addSubModel(self.v,self.f); self.model.endModel()
        self.obj=fcl.CollisionObject(self.model,fcl.Transform(self.transform[:3,:3],self.transform[:3,3]))
        self.corners=trimesh.bounds.corners(mesh.bounds); self.update(self.transform)
    @property
    def solid_components(self):
        if self._solid_components is None:
            if self.mesh.is_watertight:
                self._solid_components=[(None,self.mesh)]
            else:
                self._solid_components=[]
                face_labels=self.vertex_components[self.f[:,0]]
                for label in np.unique(face_labels):
                    sub=trimesh.Trimesh(self.v,self.f[face_labels==label],process=False)
                    sub.remove_unreferenced_vertices()
                    if sub.is_watertight:self._solid_components.append((int(label),sub))
        return self._solid_components
    def update(self,m):
        self.transform=m; self.obj.setTransform(fcl.Transform(m[:3,:3],m[:3,3]))
        v=points(self.corners,m); self.lo=v.min(0); self.hi=v.max(0)
    def info(self):
        return dict(name=self.name,vertices=len(self.v),triangles=len(self.f),watertight=bool(self.mesh.is_watertight),
                    bounds_world_m=[self.lo.tolist(),self.hi.tolist()],**self.metadata)

def mesh_parts(root,cache,reference=None,visual=True):
    inv=np.eye(4) if reference is None else np.linalg.inv(reference)
    out=[]
    for p in traversal(root):
        path=str(p.GetPath()); im=UsdGeom.Imageable(p)
        if visual and ("/collisions/" in path.lower() or "collision" in p.GetName().lower() or "proxy" in p.GetName().lower()): continue
        if im:
            if visual and (im.ComputeVisibility()=="invisible" or im.ComputePurpose() in ("proxy","guide")): continue
        mesh=shape_mesh(p)
        if mesh is None: continue
        mesh.apply_transform(inv @ matrix(cache.GetLocalToWorldTransform(p)))
        out.append((p,mesh))
    return out

def rigid_ancestor(prim):
    """Find an enabled physical rigid root; disabled rigid APIs are static."""
    while prim and not prim.IsPseudoRoot():
        if (prim.HasAPI(UsdPhysics.RigidBodyAPI)
                and UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is not False):
            return prim
        prim=prim.GetParent()
    return None

def collider_mesh(prim):
    """Exact mesh/cube or conservative analytic/cooked collision envelope.

    Unlike shape_mesh (visual tessellation), every analytic face encloses the
    corresponding true support surface. Suitable for native self/world and
    dynamic proxies too. Convex decomposition is enclosed by its full hull;
    intersections with either conservative envelope need that limitation kept.
    """
    mesh=shape_mesh(prim)
    if mesh is None:raise ValueError('Enabled static collider has unsupported or empty geometry: '+str(prim.GetPath()))
    approximation=(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
                   if prim.IsA(UsdGeom.Mesh) else 'analytic')
    if prim.IsA(UsdGeom.Mesh):
        approximation=approximation or 'none'
        if approximation in ('convexHull','convexDecomposition'):
            mesh=mesh.convex_hull
        elif approximation!='none':
            raise ValueError('Enabled static collider cooking is not covered: '+str(prim.GetPath())+' '+str(approximation))
        method='exact_authored_triangles' if approximation=='none' else 'conservative_convex_cooking_envelope'
    elif prim.IsA(UsdGeom.Cube):
        method='exact_cube_boundary'
    else:
        # shape_mesh tessellates inside the analytic surface. Enclose the
        # actual sphere/cylinder/capsule instead of leaving facet-sized gaps.
        mesh=mesh.convex_hull; normals=mesh.face_normals
        distances=np.einsum('ij,ij->i',normals,mesh.triangles[:,0])
        if prim.IsA(UsdGeom.Sphere):
            support=np.full(len(normals),float(UsdGeom.Sphere(prim).GetRadiusAttr().Get()))
        elif prim.IsA(UsdGeom.Cylinder) or prim.IsA(UsdGeom.Capsule):
            geom=UsdGeom.Cylinder(prim) if prim.IsA(UsdGeom.Cylinder) else UsdGeom.Capsule(prim)
            axis='XYZ'.index(str(geom.GetAxisAttr().Get()));axial=np.abs(normals[:,axis])
            radial=np.sqrt(np.maximum(0.,1.-axial**2));radius=float(geom.GetRadiusAttr().Get());height=float(geom.GetHeightAttr().Get())
            support=radius*(radial if prim.IsA(UsdGeom.Cylinder) else np.ones(len(normals)))+height/2*axial
        else:raise ValueError('Unsupported static analytic collider: '+str(prim.GetPath()))
        if np.any(distances<=0):raise ValueError('Invalid analytic collider envelope: '+str(prim.GetPath()))
        mesh.vertices*=max(1.,float(np.max(support/distances)))
        method='conservative_circumscribed_analytic_envelope'
    return mesh,method,approximation

# Preserve the public name and existing static geometry exactly.
static_collision_mesh=collider_mesh

def _unique_body_names(bodies):
    names=[b.name for b in bodies]
    if len(names)!=len(set(names)):raise ValueError('Duplicate geometry names would alias the collision pair cache')

def include_dynamic_collision_geometry(scene,recorded_paths):
    """Retain visuals and append every enabled non-robot rigid-body collider.

    Each appended mesh is in its recorded owner's rigid frame, with authored
    scale retained in its vertices. No same-owner visual/proxy pair is tested.
    """
    if len(recorded_paths)!=len(set(recorded_paths)):raise ValueError('Duplicate recorded rigid-body paths')
    rows=[];missing=[];cache=scene.cache
    # AllPrims retains undefined prims for failure reporting; instance-proxy
    # traversal supplies composed colliders which AllPrims silently skips.
    prims={str(p.GetPath()):p for p in Usd.PrimRange(scene.stage.GetPseudoRoot(),Usd.PrimAllPrimsPredicate)}
    prims.update({str(p.GetPath()):p for p in traversal(scene.stage.GetPseudoRoot())})
    for prim in prims.values():
        path=str(prim.GetPath())
        if path.startswith('/World/Robot') or not prim.IsActive() or not prim.IsLoaded() or prim.IsAbstract():continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():continue
        owner=rigid_ancestor(prim)
        if owner is None:continue
        owner_path=str(owner.GetPath())
        if not prim.IsDefined() or owner_path not in recorded_paths:
            missing.append(dict(collider=path,owner=owner_path,is_defined=prim.IsDefined()));continue
        mesh,method,approximation=collider_mesh(prim)
        owner_world=rigid_part(matrix(cache.GetLocalToWorldTransform(owner)))
        mesh.apply_transform(np.linalg.inv(owner_world)@matrix(cache.GetLocalToWorldTransform(prim)))
        metadata=dict(owner=owner_path,role='dynamic_collision',geometry_kind='dynamic_physical_collider',
                      geometry_type=str(prim.GetTypeName()),source_collider_prim=path,
                      collision_geometry_method=method,collision_approximation=approximation)
        body=Body(path+'#dynamic_physical_collider',mesh,owner_world,metadata)
        scene.bodies.append(body)
        rows.append(dict(prim_path=path,geometry_body=body.name,owner=owner_path,method=method,
                         approximation=approximation,geometry_type=str(prim.GetTypeName()),triangles=len(body.f)))
    _unique_body_names(scene.bodies)
    return dict(schema='family3.dynamic_collider_coverage.v1',complete=not missing,
                represented_dynamic_collider_count=len(rows),recorded_owner_count=len({r['owner'] for r in rows}),
                unresolved_enabled_dynamic_colliders=missing,colliders=rows,
                limitations=['Analytic surfaces and convex decomposition use conservative envelopes; exact cooked PhysX triangles are not available offline.'])

def attach_native_robot_colliders(robot,stage,recording,metadata):
    """Add all metadata-declared native shapes without replacing visible parts."""
    required=('episode_uid','robot_collision_profile_complete','robot_collision_enabled_paths','robot_collision_enabled_path_count')
    if any(k not in metadata for k in required) or 'episode_uid' not in recording:
        raise ValueError('Incomplete native robot collision metadata')
    if str(metadata['episode_uid'])!=str(recording['episode_uid']):raise ValueError('Native collision metadata episode UID mismatch')
    if not bool(metadata['robot_collision_profile_complete']):raise ValueError('Recorded native collision profile is incomplete')
    # Validate literal arrays before using their owners; no FK fallback in this mode.
    literal=RecordedRobotLinks.from_recording(robot,recording,len(recording['robot_joint_position_rad_or_m']))
    if literal is None:raise ValueError('Native robot/world requires literal recorded link poses')
    enabled=list(map(str,metadata['robot_collision_enabled_paths']))
    if len(enabled)!=len(set(enabled)) or len(enabled)!=int(metadata['robot_collision_enabled_path_count']):
        raise ValueError('Native enabled collider paths are duplicated or disagree with recorded count')
    root=stage.GetPrimAtPath('/World/Robot')
    if not root:raise ValueError('Actual snapshot has no /World/Robot')
    actual={str(p.GetPath()) for p in traversal(root) if p.HasAPI(UsdPhysics.CollisionAPI)
            and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()}
    mismatch=dict(missing=sorted(set(enabled)-actual),extra=sorted(actual-set(enabled)))
    # An instance-proxy omission is retained as a failed same-profile gate,
    # while all actual shapes are still checked. Never call metadata complete
    # merely because its own count agrees with its truncated path list.
    if mismatch['missing'] or any(not stage.GetPrimAtPath(p).IsInstanceProxy() for p in mismatch['extra']):
        raise ValueError('Snapshot native enabled colliders differ from recorded profile: '+str(mismatch))
    links=list(map(str,recording['robot_link_paths']));cache=UsdGeom.XformCache()
    canonical_links={j['a'] for j in robot.joints}|{j['b'] for j in robot.joints}
    visible_count=len(robot.bodies);rows=[]
    for path in sorted(actual):
        prim=stage.GetPrimAtPath(path)
        if not prim or not prim.IsDefined() or not prim.IsActive() or not prim.IsLoaded():raise ValueError('Missing defined active native collider '+path)
        owner=prim
        while owner and str(owner.GetPath()) not in links:owner=owner.GetParent()
        if not owner:raise ValueError('Native collider has no recorded owner '+path)
        captured=str(owner.GetPath())
        candidates=[link for link in canonical_links if captured in (link,'/World/Robot'+link[len(robot.root):])]
        if len(candidates)!=1:raise ValueError('Native collider owner has no unique canonical link '+captured)
        link=candidates[0];mesh,method,approximation=collider_mesh(prim)
        mesh.apply_transform(np.linalg.inv(matrix(cache.GetLocalToWorldTransform(owner)))@matrix(cache.GetLocalToWorldTransform(prim)))
        camera='/Cam/' in path or path.endswith('/Cam')
        body=Body(path+'#native_robot_collider',mesh,metadata=dict(link=link,owner=captured,
                  role='camera' if camera else 'robot',geometry_kind='native_robot_collider',geometry_type=str(prim.GetTypeName()),
                  source_collider_prim=path,collision_geometry_method=method,collision_approximation=approximation))
        robot.bodies.append(body)
        instance=prim
        while instance and not instance.IsInstance():instance=instance.GetParent()
        rows.append(dict(prim_path=path,geometry_body=body.name,owner=captured,link=link,camera=camera,
                         recorded_in_metadata=path in enabled,is_instance_proxy=prim.IsInstanceProxy(),
                         instance_root=str(instance.GetPath()) if instance else None,
                         geometry_type=str(prim.GetTypeName()),method=method,approximation=approximation,triangles=len(body.f)))
    _unique_body_names(robot.bodies)
    return dict(schema='family3.native_robot_world_coverage.v1',complete=True,episode_uid=str(recording['episode_uid']),
                visible_geometry_body_count=visible_count,native_collider_count=len(rows),
                recorded_native_collider_count=len(enabled),same_profile_gate_passed=not any(mismatch.values()),
                metadata_mismatch=mismatch,profile_status='REVIEW' if any(mismatch.values()) else 'MATCH',
                native_camera_collider_count=sum(r['camera'] for r in rows),combined_geometry_body_count=len(robot.bodies),
                colliders=rows,limitations=['Native mesh triangles/cubes are exact for uncooked meshes. Analytic surfaces and convex decomposition use conservative envelopes; this world sweep does not check robot self collision.'])

def _shelf_support_metadata(stage,prim,world_mesh,cache):
    """Recognize the supplied shelf slabs by identity and actual board top.

    This only preserves the existing depth-reviewed support category. It does
    not waive a contact or certify a penetration depth.
    """
    path=str(prim.GetPath())
    shelf='/World/Assets/ShelvingUnit'
    candidates={'/World/Assets/Tier2Overlay','/World/Assets/Tier3Overlay',
                shelf+'/BoardSupports/Bottom',shelf+'/BoardSupports/Top'}
    if path not in candidates or not prim.IsA(UsdGeom.Cube):return {}
    board_path=shelf+'/o/Meshes/Sketchfab_model/root/GLTF_SceneRootNode/shelf_boards_0/Object_4/Object_0'
    board=stage.GetPrimAtPath(board_path)
    if not board or not board.IsDefined() or not board.IsActive():return {}
    mesh=shape_mesh(board)
    if mesh is None:return {}
    mesh.apply_transform(matrix(cache.GetLocalToWorldTransform(board)))
    triangles=mesh.triangles;top=float(world_mesh.bounds[1,2])
    selected=triangles[(np.ptp(triangles[:,:,2],axis=1)<1e-6)&(np.abs(triangles[:,:,2].mean(1)-top)<1e-5)]
    if not len(selected):return {}
    visible=selected.reshape(-1,3);bounds=world_mesh.bounds
    if np.any(visible[:,:2].min(0)<bounds[0,:2]-1e-7) or np.any(visible[:,:2].max(0)>bounds[1,:2]+1e-7):return {}
    return dict(shelf_board_support=True,source_support_visual_prim=board_path,support_top_world_m=top)

def include_static_collision_geometry(scene):
    """Append enabled static colliders omitted by the visual scene view.

    Existing native visual meshes are retained byte-for-byte. Exact mesh/cube
    colliders already represented by a visible body are annotated, not copied.
    Hidden/guide/collision-named geometry and non-equivalent cooked/analytic
    shapes get separate physical bodies. Every considered prim is recorded.
    """
    existing={b.name:b for b in scene.bodies};rows=[];added=[];undefined=[]
    for prim in Usd.PrimRange(scene.stage.GetPseudoRoot(),Usd.PrimAllPrimsPredicate):
        path=str(prim.GetPath())
        if path.startswith('/World/Robot') or not prim.IsActive() or not prim.IsLoaded() or prim.IsAbstract():continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():continue
        if rigid_ancestor(prim):continue
        if not prim.IsDefined():
            undefined.append(path);continue
        mesh,method,approximation=static_collision_mesh(prim)
        mesh.apply_transform(matrix(scene.cache.GetLocalToWorldTransform(prim)))
        metadata=dict(enabled_static_collider=True,source_collider_prim=path,collision_geometry_method=method,
                      collision_approximation=approximation,**_shelf_support_metadata(scene.stage,prim,mesh,scene.cache))
        exact=method in ('exact_authored_triangles','exact_cube_boundary')
        if path in existing and exact:
            body=existing[path];body.metadata.update(metadata);representation='retained_exact_visible_geometry'
        else:
            name=path if path not in existing else path+'#physical_collider'
            body=Body(name,mesh,metadata=dict(role='static_collision',geometry_type=str(prim.GetTypeName()),**metadata))
            scene.bodies.append(body);added.append(name);representation='added_static_collision_geometry'
        rows.append(dict(prim_path=path,geometry_body=body.name,representation=representation,
                         is_defined=prim.IsDefined(),is_active=prim.IsActive(),is_loaded=prim.IsLoaded(),
                         visibility=str(UsdGeom.Imageable(prim).ComputeVisibility()),purpose=str(UsdGeom.Imageable(prim).ComputePurpose()),
                         method=method,approximation=approximation,triangles=len(body.f),
                         shelf_board_support=bool(body.metadata.get('shelf_board_support'))))
    return dict(schema='family3.static_collider_coverage.v1',complete=not undefined,
                enabled_defined_static_collider_count=len(rows),represented_static_collider_count=len(rows),
                added_static_geometry_count=len(added),added_static_geometry=added,
                undefined_active_enabled_static_colliders=undefined,colliders=rows,
                limitations=['Static analytic surfaces and convex decomposition use conservative envelopes; exact cooked PhysX triangles are not available offline.',
                             'This inventory does not replace dynamic-object proxies, native robot collision geometry, or actual trajectory checks.'])

def is_shelf_support_body(body):
    return bool(body.metadata.get('shelf_board_support') or ('/ShelvingUnit/' in body.name and 'shelf_boards_' in body.name))

class Scene:
    def __init__(self,path=SCENE,stock=True,stage=None,include_static_colliders=False):
        self.path=Path(path); self.stage=stage if stage is not None else Usd.Stage.Open(str(path)); self.stage.SetEditTarget(self.stage.GetSessionLayer())
        self.activated=[]; self.missing=[]
        if stock:
            for path in STOCK:
                p=self.stage.GetPrimAtPath(path)
                if not p: self.missing.append(path); continue
                if not p.IsActive(): p.SetActive(True); self.activated.append(path)
        self.cache=UsdGeom.XformCache(); self.bodies=[]; self.objects={}
        # Keep mesh components separate to avoid making open concatenated furniture
        # into an artificial solid. All active background/furniture remains present.
        for p in traversal(self.stage.GetPseudoRoot()):
            path=str(p.GetPath())
            if path.startswith("/World/Robot") or path.startswith("/World/physics") or path.startswith("/Render"): continue
            im=UsdGeom.Imageable(p)
            if not im: continue
            if im.ComputeVisibility()=="invisible" or im.ComputePurpose() in ("proxy","guide"): continue
            if "collision" in path.lower() or "/collisions/" in path.lower(): continue
            mesh=shape_mesh(p)
            if mesh is None: continue
            owner=next((s for s in STOCK if path==s or path.startswith(s+'/')),None)
            w=matrix(self.cache.GetLocalToWorldTransform(p)); mesh.apply_transform(w)
            body=Body(path,mesh,metadata=dict(owner=owner,authored_collision=bool(p.HasAPI(UsdPhysics.CollisionAPI)),geometry_type=str(p.GetTypeName())))
            self.bodies.append(body)
        for name in STOCK:
            p=self.stage.GetPrimAtPath(name)
            if not p: continue
            w=rigid_part(matrix(self.cache.GetLocalToWorldTransform(p)))
            ms=mesh_parts(p,self.cache,reference=w)
            if ms: self.objects[name]=Body(name,trimesh.util.concatenate([m for _,m in ms]),w,dict(role="stock_object"))
        # Replace stock mesh pieces by complete individual object triangle meshes.
        self.bodies=[b for b in self.bodies if not b.metadata.get('owner')]+list(self.objects.values())
        self.static_collider_coverage=include_static_collision_geometry(self) if include_static_colliders else None

class Robot:
    def __init__(self,path=ROBOT):
        self.path=Path(path); self.stage=Usd.Stage.Open(str(path)); self.root=str(self.stage.GetDefaultPrim().GetPath()); cache=UsdGeom.XformCache()
        self.joints=[]; self.bodies=[]; self.unresolved=[]
        for p in traversal(self.stage.GetDefaultPrim()):
            if not p.IsA(UsdPhysics.Joint) or p.IsA(UsdPhysics.SphericalJoint): continue
            j=UsdPhysics.Joint(p); a=j.GetBody0Rel().GetTargets(); b=j.GetBody1Rel().GetTargets()
            if not a or not b: continue
            def local(side):
                x=p.GetAttribute(f'physics:localPos{side}').Get() or Gf.Vec3f(0)
                q=p.GetAttribute(f'physics:localRot{side}').Get() or Gf.Quatf(1)
                return pose(np.array(x),[q.GetReal(),*q.GetImaginary()])
            axis=p.GetAttribute('physics:axis').Get() or 'Z'
            self.joints.append(dict(name=p.GetName(),a=str(a[0]),b=str(b[0]),l0=local(0),l1inv=np.linalg.inv(local(1)),kind=str(p.GetTypeName()),axis="XYZ".index(str(axis))))
        links={j['a'] for j in self.joints}|{j['b'] for j in self.joints}
        for name in sorted(links):
            p=self.stage.GetPrimAtPath(name); w=matrix(cache.GetLocalToWorldTransform(p)); ms=mesh_parts(p,cache,reference=w)
            if not ms: self.unresolved.append(name); continue
            cams=[m for p,m in ms if '/Cam/' in str(p.GetPath()) or str(p.GetPath()).endswith('/Cam')]
            rest=[m for p,m in ms if not ('/Cam/' in str(p.GetPath()) or str(p.GetPath()).endswith('/Cam'))]
            for suffix,parts in (('',rest),('/camera_housing',cams)):
                if parts: self.bodies.append(Body(name+suffix,trimesh.util.concatenate(parts),w,dict(link=name,role='camera' if suffix else 'robot')))
    def fk(self,qnames,qvalues,base):
        q=dict(zip(qnames,qvalues)); ws={self.root+'/base_link':base}; pending=self.joints.copy()
        while pending:
            next_pending=[]
            for j in pending:
                if j['a'] not in ws: next_pending.append(j); continue
                m=np.eye(4); value=float(q.get(j['name'],0))
                if 'Revolute' in j['kind']:
                    rv=np.zeros(3); rv[j['axis']]=value; m[:3,:3]=Rotation.from_rotvec(rv).as_matrix()
                elif 'Prismatic' in j['kind']: m[j['axis'],3]=value
                ws[j['b']]=ws[j['a']] @ j['l0'] @ m @ j['l1inv']
            if len(next_pending)==len(pending): raise ValueError('Unresolved USD joint graph')
            pending=next_pending
        for b in self.bodies: b.update(ws[b.metadata['link']])
        return ws

class RecordedRobotLinks:
    """Validated literal link poses, independent of FK and visual body names.

    Robot meshes, including camera housings, are already expressed in their
    metadata link frame. Apply that recorded link transform exactly once.
    """
    keys=('robot_link_paths','robot_link_xyz_m','robot_link_wxyz')

    @classmethod
    def from_recording(cls,robot,recording,n_frames):
        present=[key in recording for key in cls.keys]
        if not any(present):return None
        if not all(present):raise ValueError('Incomplete literal robot link fields; refusing silent FK fallback')
        return cls(robot,recording,n_frames)

    def __init__(self,robot,recording,n_frames):
        self.robot=robot;paths=list(map(str,recording['robot_link_paths']))
        if len(paths)!=len(set(paths)):raise ValueError('Duplicate recorded robot link paths')
        self.xyz=np.asarray(recording['robot_link_xyz_m'],float)
        self.quat=np.asarray(recording['robot_link_wxyz'],float)
        if self.xyz.shape!=(n_frames,len(paths),3) or self.quat.shape!=(n_frames,len(paths),4):
            raise ValueError('Literal robot link pose shape/frame count mismatch')
        if not np.all(np.isfinite(self.xyz)) or not np.all(np.isfinite(self.quat)) or np.any(np.linalg.norm(self.quat,axis=2)<1e-8):
            raise ValueError('Invalid literal robot link positions or quaternions')
        self.indices={};rows=[]
        for link in sorted({b.metadata['link'] for b in robot.bodies}):
            if not link.startswith(robot.root+'/'):raise ValueError('Robot mesh has invalid owner link '+link)
            relative=link[len(robot.root):]
            exact=[p for p in (link,'/World/Robot'+relative) if p in paths]
            candidates=list(dict.fromkeys(exact)) if exact else [p for p in paths if p.endswith(relative)]
            if len(candidates)!=1:raise ValueError('Missing or ambiguous recorded owner link '+link+': '+str(candidates))
            captured=candidates[0];self.indices[link]=paths.index(captured)
            rows.append(dict(mesh_owner_link=link,recorded_link=captured,recorded_index=self.indices[link],
                             mapping='exact_or_canonical_root' if exact else 'unique_relative_link_path'))
        self.used_indices=sorted(set(self.indices.values()))
        self.receipt=dict(source='recorded_physical_link_poses',recorded_link_count=len(paths),
                          mapped_owner_count=len(self.indices),geometry_body_count=len(robot.bodies),mapping=rows,
                          interpolation='linear world translation and shortest-arc SO(3); each native recorded tick is retained')

    def motion(self,frame,next_frame):
        ids=self.used_indices
        translation=float(np.linalg.norm(self.xyz[next_frame,ids]-self.xyz[frame,ids],axis=1).max())
        r0=Rotation.from_quat(self.quat[frame,ids][:,[1,2,3,0]])
        r1=Rotation.from_quat(self.quat[next_frame,ids][:,[1,2,3,0]])
        angle=float((r0.inv()*r1).magnitude().max())
        return translation,angle

    def world(self,frame,next_frame,alpha):
        result={}
        for link,index in self.indices.items():
            r0=Rotation.from_quat(self.quat[frame,index][[1,2,3,0]])
            r1=Rotation.from_quat(self.quat[next_frame,index][[1,2,3,0]])
            transform=np.eye(4)
            transform[:3,:3]=(r0*Rotation.from_rotvec(alpha*(r0.inv()*r1).as_rotvec())).as_matrix()
            transform[:3,3]=(1-alpha)*self.xyz[frame,index]+alpha*self.xyz[next_frame,index]
            result[link]=transform
        return result

    def update(self,frame,next_frame,alpha):
        world=self.world(frame,next_frame,alpha)
        for body in self.robot.bodies:body.update(world[body.metadata['link']])
        return world

def broad_overlap(a,b,tol=0.):
    return bool(np.all(np.minimum(a.hi,b.hi)-np.maximum(a.lo,b.lo)>=-tol))

def solid_contains(mesh,p):
    """Preserve closed-mesh ray parity while culling disconnected shells.

    A compound shelf can have a large shared bounding box around empty space.
    Rays against that whole mesh repeat work for every source component. Ray
    parity over disconnected closed shells is exactly their XOR, including
    nested shells; this is not a convex hull or an arbitrary vertex sample.
    """
    if len(mesh.faces)<2000:return mesh.contains(p)
    key='closed_containment_shells'
    parts=mesh._cache[key]
    if parts is None:
        parts=list(mesh.split(only_watertight=False))
        if not all(part.is_watertight for part in parts):
            raise ValueError('Closed containment mesh split has an open shell')
        mesh._cache[key]=parts
    if len(parts)<=1:return mesh.contains(p)
    inside=np.zeros(len(p),dtype=bool)
    for part in parts:
        mask=np.all((p>part.bounds[0])&(p<part.bounds[1]),axis=1)
        if np.any(mask):inside[mask]^=part.contains(p[mask])
    return inside

def contact(a,b,containment=False):
    """Exact triangle intersection; FCL contact depth is NOT a solid penetration depth."""
    if not broad_overlap(a,b): return None
    r=fcl.CollisionResult(); fcl.collide(a.obj,b.obj,fcl.CollisionRequest(num_max_contacts=1,enable_contact=True),r)
    if not r.is_collision:
        if containment:
            # No surfaces cross. Test one vertex of EVERY connected source
            # component, not an arbitrary sampled subset of up to32vertices.
            for src,dst in ((a,b),(b,a)):
                ids=src.component_vertex_ids
                p=points(points(src.v[ids],src.transform),np.linalg.inv(dst.transform))
                for component,solid in dst.solid_components:
                    mask=np.all((p>solid.bounds[0])&(p<solid.bounds[1]),axis=1)
                    if not np.any(mask):continue
                    tested=p[mask];inside=solid_contains(solid,tested)
                    if np.any(inside):
                        witness=points(tested[np.where(inside)[0][:1]],dst.transform)[0]
                        return dict(position_world_m=witness.tolist(),triangle_a=None,triangle_b=None,
                                    solid_containment_witness=True,solid_destination=dst.name,
                                    solid_destination_component=component)
        return None
    c=r.contacts[0]
    return dict(position_world_m=np.asarray(c.pos).tolist(),triangle_a=int(c.b1),triangle_b=int(c.b2))

def burial(a,b,max_points=512):
    """Bidirectional deterministic sampled-vertex depth in watertight mesh solids.

    A lower bound on maximum vertex burial, not a global separation distance.
    Nonwatertight components are explicitly unmeasured, never repaired silently.
    """
    best=0.; tested=0; valid=[]; coverage={}
    for src,dst in ((a,b),(b,a)):
        ids=np.linspace(0,len(src.v)-1,min(max_points,len(src.v)),dtype=int)
        p=points(points(src.v[ids],src.transform),np.linalg.inv(dst.transform))
        coverage[dst.name]=dict(whole_mesh_watertight=bool(dst.mesh.is_watertight),
            closed_components=len(dst.solid_components),open_components=(0 if dst.mesh.is_watertight else len(dst.component_vertex_ids)-len(dst.solid_components)))
        for component,solid in dst.solid_components:
            mask=np.all((p>=solid.bounds[0])&(p<=solid.bounds[1]),axis=1); selected=p[mask]
            if not len(selected):continue
            ds=trimesh.proximity.signed_distance(solid,selected)
            best=max(best,float(np.max(ds)));tested+=len(selected);valid.append(dst.name+(f'#component{component}' if component is not None else ''))
    return dict(sampled_burial_lower_bound_mm=best*1000 if tested else None,points_tested=tested,
                solid_destinations=valid,solid_destination_coverage=coverage,
                measurement_status='SAMPLED_LOWER_BOUND_ONLY' if tested else 'NO_SOLID_DEPTH_MEASUREMENT')
