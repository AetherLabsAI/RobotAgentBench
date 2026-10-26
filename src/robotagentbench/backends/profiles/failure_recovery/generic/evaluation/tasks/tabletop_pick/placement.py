"""Task-local, geometry-bounded tabletop sampling; no reachability claim."""


def mesh_points(stage, prim):
    import numpy as np
    from pxr import Gf, UsdGeom
    xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
    points = UsdGeom.Mesh(prim).GetPointsAttr().Get(0)
    if points is None or not len(points):
        raise ValueError('Missing mesh points: '+str(prim.GetPath()))
    return np.asarray([list(xf.Transform(Gf.Vec3d(*p))) for p in points])


def visible_bounds(stage, path):
    import numpy as np
    from pxr import Usd, UsdGeom
    chunks = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(path)):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable.ComputeVisibility(0) == UsdGeom.Tokens.invisible:
            continue
        if imageable.ComputePurpose() not in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render):
            continue
        chunks.append(mesh_points(stage, prim))
    if not chunks:
        raise ValueError('No visible mesh geometry: '+path)
    points = np.concatenate(chunks)
    return np.array([points.min(axis=0), points.max(axis=0)])


def tabletop_rectangle(stage, mesh_path):
    """Use actual flat top faces, not drawers or an authored extent box.

    This task supports the current horizontal rectangular table only. Fail if
    the top-face projection ceases to cover that rectangle (e.g. another asset).
    """
    import numpy as np
    from pxr import UsdGeom
    prim = stage.GetPrimAtPath(mesh_path)
    if not prim or not prim.IsA(UsdGeom.Mesh):
        raise ValueError('Missing table surface mesh')
    points = mesh_points(stage, prim)
    mesh = UsdGeom.Mesh(prim)
    counts = mesh.GetFaceVertexCountsAttr().Get(0)
    indices = mesh.GetFaceVertexIndicesAttr().Get(0)
    holes = set(mesh.GetHoleIndicesAttr().Get(0) or [])
    top, area, offset = [], 0., 0
    z = float(points[:, 2].max())
    for face, count in enumerate(counts):
        vertices = points[list(indices[offset:offset+count])]
        offset += count
        if face in holes or count < 3 or not np.all(np.abs(vertices[:, 2]-z) < 1e-6):
            continue
        xy = vertices[:, :2]
        area += abs(float(np.sum(xy[:, 0]*np.roll(xy[:, 1], -1)
                                 -xy[:, 1]*np.roll(xy[:, 0], -1))))/2
        top.append(vertices)
    if not top:
        raise ValueError('No horizontal flat tabletop')
    vertices = np.concatenate(top)
    lo, hi = vertices[:, :2].min(axis=0), vertices[:, :2].max(axis=0)
    rectangle_area = float(np.prod(hi-lo))
    if rectangle_area <= 0 or not np.isclose(area, rectangle_area, rtol=1e-5, atol=1e-8):
        raise ValueError('Flat tabletop is not a fully covered world-XY rectangle')
    return dict(mesh=mesh_path, xy_min=lo.tolist(), xy_max=hi.tolist(), z_m=z)


def approved_points(kind, surface):
    """Reference points defining the user-accepted region, not discrete placements."""
    import json
    from pathlib import Path
    import numpy as np
    data = json.loads((Path(__file__).parent/'resources/approved_sampling_points.json').read_text())
    if kind not in data['variants']:
        raise ValueError('No approved placement grid for '+str(kind))
    variant = data['variants'][kind]
    for name in ('xy_min', 'xy_max', 'z_m'):
        if not np.allclose(surface[name], variant['surface'][name], rtol=0, atol=1e-7):
            raise ValueError('Table no longer matches the approved world-space sampling grid')
    return variant['centers_xy_m'], data['report_sha256']


def region_hull(points):
    """Counterclockwise outer polygon; no expansion beyond reference points."""
    import numpy as np
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or not np.all(np.isfinite(p)):
        raise ValueError('Invalid region reference points')
    p = sorted(set(map(tuple, p)))
    def cross(a,b,c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    chains=[]
    for order in (p,p[::-1]):
        chain=[]
        for xy in order:
            while len(chain)>1 and cross(chain[-2],chain[-1],xy)<=0:
                chain.pop()
            chain.append(xy)
        chains.append(chain[:-1])
    hull=np.asarray(chains[0]+chains[1])
    if len(hull)<3:
        raise ValueError('Sampling region has no area')
    return hull


def inside_region(xy, polygon):
    import numpy as np
    polygon=np.asarray(polygon)
    edge=np.roll(polygon,-1,axis=0)-polygon
    relative=np.asarray(xy)-polygon
    return bool(np.all(edge[:,0]*relative[:,1]-edge[:,1]*relative[:,0]>=-1e-12))


def sample_placement(seed, surface, bounds, robot_xy, *, candidates=None):
    import numpy as np
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Invalid object randomization seed')
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)) or np.any(bounds[1] <= bounds[0]):
        raise ValueError('Invalid target bounds')
    half = (bounds[1, :2]-bounds[0, :2])/2
    lo = np.asarray(surface['xy_min'])+half+.005
    hi = np.asarray(surface['xy_max'])-half-.005
    robot_xy = np.asarray(robot_xy, dtype=float)
    if (robot_xy.shape != (2,) or not np.all(np.isfinite(robot_xy))
            or not np.all(np.isfinite([*lo, *hi, surface['z_m']])) or np.any(hi <= lo)):
        raise ValueError('No valid tabletop placement region')
    # Separate stream: adding object randomization never changes robot draws.
    rng = np.random.default_rng([seed, 0x5441424C])
    legal_lo, legal_hi = lo.copy(), hi.copy()
    points = None
    if candidates is not None:
        points = np.asarray(candidates, dtype=float)
        if (points.ndim != 2 or points.shape[1] != 2 or not len(points)
                or not np.all(np.isfinite(points))
                or np.any(points < lo-1e-9) or np.any(points > hi+1e-9)):
            raise ValueError('Approved grid is invalid or does not fit current target footprint')
        lo, hi = points.min(axis=0), points.max(axis=0)
        if np.any(hi <= lo):
            raise ValueError('Approved grid must span both axes')
    polygon = None if points is None else region_hull(points)
    mode = np.clip(robot_xy, lo, hi)
    near = bool(rng.random() < .5)
    # Condition each continuous distribution on the accepted polygon.
    # No nearest-grid snap; the branch stays fixed across geometric proposals.
    for proposals in range(1,10001):
        xy = rng.triangular(lo, mode, hi) if near else rng.uniform(lo, hi)
        if polygon is None or inside_region(xy,polygon):
            break
    else:
        raise ValueError('Could not sample the continuous approved region')
    delta = np.r_[xy-(bounds[0, :2]+bounds[1, :2])/2,
                  surface['z_m']+.003-bounds[0, 2]]
    return dict(distribution=('half_uniform_half_robot_biased_triangular' if points is None else
                              'half_uniform_half_robot_biased_triangular_conditioned_on_polygon'),
                branch='robot_biased' if near else 'uniform',
                rng_seed=[seed, 0x5441424C], edge_margin_m=.005,
                bottom_clearance_before_settle_m=.003, surface=surface,
                center_xy_region_m=[lo.tolist(), hi.tolist()],
                legal_table_center_region_m=[legal_lo.tolist(), legal_hi.tolist()],
                region_polygon_xy_m=None if polygon is None else polygon.tolist(),
                region_reference_point_count=None if points is None else len(points),
                geometric_proposal_count=proposals, continuous_xy=True,
                robot_xy_m=robot_xy.tolist(), triangular_mode_xy_m=mode.tolist(),
                center_xy_m=xy.tolist(), translation_m=delta.tolist(),
                before_bbox_m=bounds.tolist(), after_bbox_m=(bounds+delta).tolist())
