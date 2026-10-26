"""Source-observed tray/holder jitter in the actual support frame, pose only."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom, Gf
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union
from .audit_holder_cavities import points
from openend.tasks.scene_requirements.station_scene_checks import check, intersects

HERE = Path(__file__).resolve().parent
TRAY = '/World/RoomEpisode/MealTray'
HOLDER_ROOT = '/World/RoomEpisode/CutleryHolder'
TOOLS = tuple('/World/RoomEpisode/Cutlery_' + n for n in ('knife_0', 'fork_1', 'fork_2'))
RANGES = HERE / 'dataset_initial_evidence.json'


class NoFeasibleEnvironment(ValueError):
    pass


def sample(seed):
    if type(seed) is not int or not 1 <= seed <= 4294967295:
        raise ValueError('Positive uint32 seed required')
    ranges = json.loads(RANGES.read_text())['audit']['observed_ranges']
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(
        f'cutlery-environment-observed-v1:{seed}'.encode()).digest(), 'little'))
    return {name: {axis: float(rng.uniform(*ranges[name + '_jitter.' + axis]))
                  for axis in ('dx_m', 'dy_m', 'yaw_deg')} for name in ('tray', 'holder')}


def shifted(world, inward, jitter):
    """Column matrix: source-frame translation and world-Z yaw about own origin."""
    left = np.array([-inward[1], inward[0]])
    a = math.radians(jitter['yaw_deg']); c, s = math.cos(a), math.sin(a)
    rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    out = world.copy(); out[:3, :3] = rotation @ world[:3, :3]
    out[:2, 3] += inward*jitter['dx_m'] + left*jitter['dy_m']
    return out


def apply(stage, manifest, seed):
    """Caller chooses a task-owned layer or anonymous session edit target."""
    cache = UsdGeom.XformCache()
    world = lambda p: np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(p))).T
    front = np.asarray(manifest['supports']['tray']['front'], dtype=float)
    if front.shape != (2,) or not np.isfinite(front).all() or np.linalg.norm(front) < 1e-8:
        raise ValueError('Actual support front required')
    inward = -front / np.linalg.norm(front)
    draws = sample(seed)
    old_tray, old_holder = world(TRAY), world(HOLDER_ROOT)
    moved = {TRAY: shifted(old_tray, inward, draws['tray']),
             HOLDER_ROOT: shifted(old_holder, inward, draws['holder'])}
    holder_delta = moved[HOLDER_ROOT] @ np.linalg.inv(old_holder)
    for path in TOOLS:
        if not stage.GetPrimAtPath(path): raise ValueError('Missing expected utensil: '+path)
        moved[path] = holder_delta @ world(path)
    for path, matrix in moved.items():
        parent = str(stage.GetPrimAtPath(path).GetParent().GetPath())
        local = np.linalg.inv(world(parent)) @ matrix
        xf = UsdGeom.Xformable(stage.OverridePrim(path)); xf.ClearXformOpOrder()
        xf.AddTransformOp(opSuffix='cutleryEnvironment').Set(Gf.Matrix4d(local.T.tolist()))
    return dict(seed=seed, draws=draws, sampling='continuous_observed_environment_ranges_v1',
        frame='support_inward_x_left_y_world_z',
        nominal_station_origin_xy=old_tray[:2, 3].tolist(),
        actual_tray_origin_xy=moved[TRAY][:2, 3].tolist(),
        nominal_holder_world=old_holder.tolist(),
        pose_only_world_overrides={p: m.tolist() for p,m in moved.items()},
        robot_not_moved=True, height_unchanged=True, native_physics_validated=False)


def surface_polygon(stage, support):
    """Actual horizontal top triangles, not the broad furniture AABB."""
    prim = stage.GetPrimAtPath(support['path']); cache = UsdGeom.XformCache()
    if not prim: raise ValueError('Missing support')
    top = float(support['actual_top_m']); polygons = []; measured=[]
    for mesh in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if not mesh.IsActive() or not (mesh.IsA(UsdGeom.Mesh) or mesh.IsA(UsdGeom.Cube)): continue
        im = UsdGeom.Imageable(mesh)
        if im.ComputeVisibility() == 'invisible' or im.ComputePurpose() not in ('default','render'): continue
        if mesh.IsA(UsdGeom.Cube):
            h=UsdGeom.Cube(mesh).GetSizeAttr().Get()/2
            v=np.array(list(itertools.product((-h,h),repeat=3)))
            faces=[[0,1,3,2],[4,6,7,5],[0,4,5,1],[2,3,7,6],[0,2,6,4],[1,5,7,3]]
        else:
            v = np.asarray(mesh.GetAttribute('points').Get(), dtype=float)
            ids = list(mesh.GetAttribute('faceVertexIndices').Get()); offset = 0; faces=[]
            for count in mesh.GetAttribute('faceVertexCounts').Get():
                faces.append(ids[offset:offset+count]);offset+=count
        v = (np.c_[v,np.ones(len(v))] @ np.array(cache.GetLocalToWorldTransform(mesh)))[:,:3]
        measured.append((str(mesh.GetPath()),float(v[:,2].min()),float(v[:,2].max())))
        for face in faces:
            for i in range(1,len(face)-1):
                t = v[[face[0],face[i],face[i+1]]]
                if np.max(np.abs(t[:,2]-top)) <= 1e-5:
                    p = Polygon(t[:,:2])
                    if p.is_valid and p.area > 1e-12: polygons.append(p)
    if not polygons: raise ValueError('No measured horizontal support top triangles: '+repr((top,measured)))
    return unary_union(polygons)


def validate(stage, manifest):
    cache = UsdGeom.XformCache()
    vertices = {p: points(stage,p,np.eye(4),cache) for p in (TRAY,HOLDER_ROOT,*TOOLS)}
    bounds = {p: np.array([v.min(0),v.max(0)]) for p,v in vertices.items()}
    support = manifest['supports']['tray']; polygon = surface_polygon(stage,support)
    for path in (TRAY,HOLDER_ROOT):
        hull = MultiPoint(vertices[path][:,:2]).convex_hull
        if not polygon.buffer(1e-7).covers(hull):
            raise NoFeasibleEnvironment('Randomized assembly outside actual tabletop: '+path)
    for path in (HOLDER_ROOT,*TOOLS):
        if intersects(bounds[TRAY],bounds[path]):
            raise NoFeasibleEnvironment('Randomized tray overlaps holder/tool: '+path)
    group = dict(id='randomized_cutlery',members=list(vertices),support=support['path'],
                 internal_supports=['/World/Robot'])
    clearance = check(stage,[group],bounds.__getitem__)
    if not clearance['passed']:
        raise NoFeasibleEnvironment('Randomized assembly room overlap: '+str(clearance['failures'][:3]))
    return dict(passed=True,top_surface_area_m2=polygon.area,station_clearance=clearance,
        target_bounds={p:b.tolist() for p,b in bounds.items()},
        robot_excluded_here='separate source-posture FK gate follows',
        native_physics_validated=False)
