"""Shared CoffeeTable workspaces; coordinates follow the table across layouts."""
from __future__ import annotations
import itertools
import json
from pathlib import Path

TABLE = '/World/Assets/CoffeeTable'
LINE = TABLE + '/WorkspaceDivider'


def configuration():
    return json.loads(Path(__file__).with_name('episode_distractor_policy_v1.json').read_text())['coffee_table_semantic_refresh']


def ensure_partition(stage):
    """Keep workspace coordinates as metadata, with no visible dividing line."""
    from pxr import Gf, Sdf, UsdGeom
    cfg = configuration()
    table = stage.GetPrimAtPath(TABLE)
    if not table:
        raise ValueError('CoffeeTable is missing')
    matrix = UsdGeom.Xformable(table).ComputeLocalToWorldTransform(0)
    old = stage.GetPrimAtPath(LINE)
    attr = old.GetAttribute('workspace:canonicalTableTransform') if old else None
    canonical = attr.Get() if attr and attr.HasAuthoredValueOpinion() else matrix
    # Replace an earlier painted mesh with a non-renderable metadata scope.
    stage.RemovePrim(LINE)
    marker = stage.DefinePrim(LINE, "Scope")
    marker.CreateAttribute('workspace:canonicalTableTransform', Sdf.ValueTypeNames.Matrix4d).Set(canonical)
    return {'prim_path':LINE, 'visible_line':False, 'split_axis':'canonical_world_x',
            'split_x_m':cfg['split_x_m'],'cleanup_side':'toward_sofa','tablecloth_side':'away_from_sofa',
            'collision_authored':False}



def validate_cleanup_objects(stage, paths, *, clearance_m=0.003):
    """Require complete initial visual footprints in the sofa half, including rotated layouts."""
    import numpy as np
    from pxr import Usd, UsdGeom
    line = stage.GetPrimAtPath(LINE)
    if not line:
        raise ValueError('CoffeeTable workspace metadata missing; deploy the current House baseline')
    canonical = line.GetAttribute('workspace:canonicalTableTransform').Get()
    table = stage.GetPrimAtPath(TABLE)
    inverse = UsdGeom.Xformable(table).ComputeLocalToWorldTransform(0).GetInverse()
    x0,x1,y0,y1 = configuration()['sofa_side_bounds_xy_m']
    checked=[]
    for path in paths:
        prim=stage.GetPrimAtPath(path)
        if not prim or not prim.IsActive():
            raise ValueError(f'cleanup object missing or inactive: {path}')
        minima=[]
        maxima=[]
        for child in Usd.PrimRange(prim):
            if not child.IsA(UsdGeom.Boundable):
                continue
            imageable=UsdGeom.Imageable(child)
            if imageable.ComputeVisibility() == 'invisible' or imageable.ComputePurpose() == 'guide':
                continue
            transform=UsdGeom.Xformable(child).ComputeLocalToWorldTransform(0)*inverse*canonical
            if child.IsA(UsdGeom.PointBased):
                local=UsdGeom.PointBased(child).GetPointsAttr().Get() or []
            else:
                extent=UsdGeom.Boundable.ComputeExtentFromPlugins(UsdGeom.Boundable(child),0)
                local=list(itertools.product(*zip(*extent))) if extent else []
            if len(local):
                matrix=np.asarray(transform)
                points=np.asarray(local,dtype=float) @ matrix[:3,:3] + matrix[3,:3]
                minima.append(points.min(axis=0)); maxima.append(points.max(axis=0))
        if not minima:
            raise ValueError(f'cleanup object has empty visible geometry: {path}')
        low=np.min(minima,axis=0); high=np.max(maxima,axis=0)
        bounds=[float(low[0]),float(high[0]),float(low[1]),float(high[1])]
        if not (bounds[0]>=x0+clearance_m-1e-6 and bounds[1]<=x1-clearance_m+1e-6
                and bounds[2]>=y0+clearance_m-1e-6 and bounds[3]<=y1-clearance_m+1e-6):
            raise ValueError(f'cleanup object crosses sofa-side workspace boundary: {path}: {bounds}')
        checked.append({'prim_path':path,'canonical_bounds_xy_m':bounds})
    return {'workspace':'sofa_side','bounds_xy_m':[x0,x1,y0,y1], 'clearance_m':clearance_m,
            'checked_objects':checked,'passed':True,'physics_rollout_performed':False}
