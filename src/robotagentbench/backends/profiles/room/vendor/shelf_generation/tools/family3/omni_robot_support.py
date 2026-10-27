"""Deterministic initial base support registration, independent of random draws."""
import copy
import numpy as np
from pxr import Usd,UsdGeom,UsdPhysics
from collision_scene import collider_mesh,shape_mesh,traversal,matrix,points,pose
from shapely.geometry import MultiPoint,Polygon
from shapely.ops import unary_union

FLOOR='/World/Background/Shell/Floor'

def registered_robot(robot,scene):
    floor=scene.GetPrimAtPath(FLOOR)
    if not floor or not floor.HasAPI(UsdPhysics.CollisionAPI):
        raise ValueError('Robot support requires the actual collision-enabled floor')
    cache=UsdGeom.XformCache();fm=shape_mesh(floor)
    if fm is None:raise ValueError('Missing floor surface')
    fm.apply_transform(matrix(cache.GetLocalToWorldTransform(floor)))
    z=np.asarray(fm.vertices)[:,2]
    if np.ptp(z)>1e-9:raise ValueError('Planar base registration requires a horizontal floor')
    floor_z=float(z.mean())
    result=copy.deepcopy(robot)
    source_xyz=np.array(robot.get('support_registration',{}).get('unregistered_base_xyz_m',robot['base_xyz_m']),float)
    base=pose(source_xyz,robot['base_wxyz'])
    asset=Usd.Stage.Open(robot['asset']);root=asset.GetDefaultPrim();ac=UsdGeom.XformCache()
    root_inv=np.linalg.inv(matrix(ac.GetLocalToWorldTransform(root)))
    base_prim=asset.GetPrimAtPath(str(root.GetPath())+'/base_link')
    arrays=[];paths=[]
    for prim in traversal(base_prim):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
        mesh,method,approximation=collider_mesh(prim)
        arrays.append(points(mesh.vertices,base@root_inv@matrix(ac.GetLocalToWorldTransform(prim))))
        paths.append(dict(path=str(prim.GetPath()),method=method,approximation=approximation))
    if not arrays:raise ValueError('No original native base geometry')
    vertices=np.concatenate(arrays);bottom=float(vertices[:,2].min())
    support=unary_union([Polygon(np.asarray(fm.vertices)[face,:2]) for face in fm.faces])
    footprint=MultiPoint(vertices[:,:2]).convex_hull
    if not support.covers(footprint):raise ValueError('Native base footprint leaves floor surface')
    dz=floor_z-bottom;xyz=source_xyz.copy();xyz[2]+=dz;result['base_xyz_m']=xyz.tolist()
    result['support_registration']=dict(schema='family3.initial_base_support.v1',method='Exact full native base boundary tangent to the actual horizontal floor before physics; common deterministic registration, no Z random draw',
        floor_path=FLOOR,floor_z_m=floor_z,unregistered_base_xyz_m=source_xyz.tolist(),native_bottom_before_m=bottom,
        fixed_z_offset_m=dz,native_bottom_after_m=bottom+dz,geometry=paths,
        shared_robot_geometry_modified=False,physics_attributes_modified=False,task_base_motion_allowed=False)
    return result
