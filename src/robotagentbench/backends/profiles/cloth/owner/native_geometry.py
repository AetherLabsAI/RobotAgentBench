"""Read actual cooked pad geometry without changing physical settings."""
import numpy as np


def cooked_pad_geometry(stage):
    from pxr import Usd,UsdGeom,UsdPhysics,UsdUtils,PhysicsSchemaTools
    from omni.physx import get_physx_cooking_interface
    stage_id=UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    cache=UsdGeom.XformCache();result={}
    for hand in ('l','r'):
        for side in ('inner','outer'):
            path=f'/World/Robot/gripper_{hand}_{side}_link4'
            body=stage.GetPrimAtPath(path)
            if not body:raise ValueError('Required source pad missing')
            rows=[]
            for prim in Usd.PrimRange(body):
                if not prim.IsA(UsdGeom.Mesh)or not prim.HasAPI(UsdPhysics.CollisionAPI):continue
                if prim.GetAttribute('physics:collisionEnabled').Get()is False:continue
                relative=np.asarray(cache.GetLocalToWorldTransform(prim)*cache.GetLocalToWorldTransform(body).GetInverse())
                row=dict(collider=str(prim.GetPath()),convexes=[])
                def received(status,convexes):
                    row['native_cooking_status']=str(status)
                    for convex in convexes:
                        v=np.asarray([[p.x,p.y,p.z]for p in convex.vertices],float)
                        v=(np.column_stack((v,np.ones(len(v))))@relative)[:,:3]
                        row['convexes'].append(dict(vertices_body_m=v.tolist(),indices=list(convex.indices)))
                get_physx_cooking_interface().request_convex_collision_representation(
                    stage_id=stage_id,collision_prim_id=PhysicsSchemaTools.sdfPathToInt(str(prim.GetPath())),
                    run_asynchronously=False,on_result=received)
                if not row['convexes']:raise RuntimeError('Native pad cooking representation unavailable')
                rows.append(row)
            if not rows:raise RuntimeError('Source pad has no active mesh collider')
            result[path]=rows
    return result


def table_geometry(stage):
    from pxr import UsdGeom,UsdPhysics,Gf
    path='/World/Assets/CoffeeTable/Mesh';prim=stage.GetPrimAtPath(path)
    if not prim or not prim.IsA(UsdGeom.Mesh)or not prim.HasAPI(UsdPhysics.CollisionAPI):raise RuntimeError('Exact physical table missing')
    if UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()!='convexHull':raise ValueError('Expected approved source table collider')
    transform=UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    q=np.asarray([transform.Transform(Gf.Vec3d(*p))for p in UsdGeom.Mesh(prim).GetPointsAttr().Get()])
    return dict(path=path,world_vertices_m=q.tolist(),top_z_m=float(q[:,2].max()),
                source='actual_source_convex_hull_input_not_visible_extent')


def cooked_table_geometry(stage):
    from pxr import UsdGeom,UsdUtils,PhysicsSchemaTools
    from omni.physx import get_physx_cooking_interface
    path='/World/Assets/CoffeeTable/Mesh';prim=stage.GetPrimAtPath(path)
    transform=UsdGeom.XformCache().GetLocalToWorldTransform(prim);rows=[]
    from pxr import Gf
    def received(status,convexes):
        for convex in convexes:
            rows.append([list(transform.Transform(Gf.Vec3d(p.x,p.y,p.z)))for p in convex.vertices])
    get_physx_cooking_interface().request_convex_collision_representation(
        stage_id=UsdUtils.StageCache.Get().GetId(stage).ToLongInt(),
        collision_prim_id=PhysicsSchemaTools.sdfPathToInt(path),run_asynchronously=False,on_result=received)
    if not rows:raise RuntimeError('Native table convex representation missing')
    return rows


def floor_top_triangles(stage):
    from pxr import Usd,UsdGeom,UsdPhysics,Gf
    root=stage.GetPrimAtPath('/World/Background/Shell/Floor')
    if not root:raise RuntimeError('Original physical floor binding missing')
    triangles=[];cache=UsdGeom.XformCache()
    for p in Usd.PrimRange(root):
        if not p.HasAPI(UsdPhysics.CollisionAPI)or p.GetAttribute('physics:collisionEnabled').Get()is False:continue
        transform=cache.GetLocalToWorldTransform(p)
        if p.IsA(UsdGeom.Mesh):
            mesh=UsdGeom.Mesh(p);local_points=mesh.GetPointsAttr().Get();faces=np.asarray(mesh.GetFaceVertexIndicesAttr().Get());counts=mesh.GetFaceVertexCountsAttr().Get()
        elif p.IsA(UsdGeom.Cube):
            from itertools import product
            half=float(UsdGeom.Cube(p).GetSizeAttr().Get())/2
            local_points=np.array(list(product((-half,half),repeat=3)))
            faces=np.array([0,1,3,2,4,6,7,5,0,4,5,1,2,3,7,6,0,2,6,4,1,5,7,3]);counts=[4]*6
        else:raise RuntimeError('Floor shape requires its native schema-specific reader')
        points=np.asarray([transform.Transform(Gf.Vec3d(*v))for v in local_points]);offset=0;top=points[:,2].max()
        for count in counts:
            vertices=points[faces[offset:offset+count]];offset+=count
            if np.all(abs(vertices[:,2]-top)<1e-9):
                triangles.extend([vertices[[0,k,k+1]]for k in range(1,count-1)])
    if not triangles:raise RuntimeError('Measured physical floor top faces missing')
    return np.asarray(triangles)
