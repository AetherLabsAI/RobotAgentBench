"""Source-derived planar station mapping; no scene writes or feasibility claim.

The caller must obtain the actual shelf front midpoint and inward heading from
its furniture-local geometry, not the rotated world AABB or product positions.
Existing floor/collider checks must accept a pose before it becomes a launch.
"""
import json
import math
import hashlib
import itertools
from pathlib import Path


def world_pose(front_midpoint_xy, inward_heading_rad, perturbation):
    """Map explicit (lateral metres, inward metres, yaw degrees) to world XY/yaw.

    Perturbations are supplied by the owner's seeded random stream. This helper
    neither substitutes a nearest-grid pose nor expands the approved ranges.
    """
    policy = json.loads(Path(__file__).with_name('initial_station_policy.json').read_text())
    if len(front_midpoint_xy) != 2 or len(perturbation) != 3:
        raise ValueError('Expected shelf front XY and three perturbation values')
    x, y = map(float, front_midpoint_xy)
    heading = float(inward_heading_rad)
    dx, dy, da = map(float, perturbation)
    if not all(math.isfinite(v) for v in (x, y, heading, dx, dy, da)):
        raise ValueError('Non-finite shelf frame or perturbation')
    ranges = policy['source_randomization_ranges']
    for value, name in zip((dx, dy, da), (
            'lateral_delta_uniform_m', 'depth_delta_uniform_m', 'heading_delta_uniform_deg')):
        lo, hi = ranges[name]
        if not lo <= value <= hi:
            raise ValueError('Outside approved station range: ' + name)
    nominal = policy['nominal']
    lateral = nominal['lateral_m'] + dx
    inward = -nominal['front_edge_standoff_m'] + dy
    c, s = math.cos(heading), math.sin(heading)
    # Rightward lateral and inward axes, matching source +X / +Y at 90 deg.
    px, py = x + s*lateral + c*inward, y - c*lateral + s*inward
    yaw = heading + math.radians(nominal['heading_relative_inward_deg'] + da)
    yaw = (yaw + math.pi) % (2*math.pi) - math.pi
    return dict(position_xy=[px, py], yaw_rad=yaw, policy=policy['policy'],
                local_perturbation=[dx, dy, da],
                front_edge_standoff_m=-inward, lateral_m=lateral,
                collision_checked=False, physics_certified=False)


def shelf_frame(stage, manifest):
    """Read the actual canonical shelf wrapper, including layout repair offsets."""
    import numpy as np
    from pxr import Usd, UsdGeom
    paths = {r['path'] for r in manifest['objects'] if r['id'] == 'snack_shelf'}
    if len(paths) != 1:
        raise ValueError('Exactly one scene-owned snack shelf required')
    path, = paths
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsActive() or not stage.GetPrimAtPath(path + '/Asset'):
        raise ValueError('Canonical shelf wrapper missing: ' + path)
    cache = UsdGeom.XformCache()
    matrix = np.asarray(cache.GetLocalToWorldTransform(prim))
    a = math.atan2(matrix[0, 1], matrix[0, 0])
    expected = np.array([[math.cos(a), math.sin(a), 0],
                         [-math.sin(a), math.cos(a), 0], [0, 0, 1]])
    if not np.isfinite(matrix).all() or not np.allclose(matrix[:3, :3], expected, atol=1e-6, rtol=0):
        raise ValueError('Shelf wrapper must be upright and rigid, not scaled/tilted')
    # Transform source geometry directly into the wrapper frame. Aggregated
    # USD bounds can expand at rotated descendants; unrotating a world AABB
    # cannot recover the actual front edge. Stock objects are separate siblings.
    inverse = np.linalg.inv(matrix)
    points = []
    for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if not child.IsActive() or not child.IsA(UsdGeom.Boundable):
            continue
        if UsdGeom.Imageable(child).ComputePurpose() not in ('default', 'render', 'proxy'):
            continue
        if child.IsA(UsdGeom.Mesh):
            vertices = np.asarray(UsdGeom.Mesh(child).GetPointsAttr().Get(), dtype=float)
        else:
            extent = UsdGeom.Boundable(child).ComputeExtent(Usd.TimeCode.Default())
            if extent is None or len(extent) != 2:
                raise ValueError('Missing shelf geometry extent: ' + str(child.GetPath()))
            vertices = np.array(list(itertools.product(*zip(*extent))))
        if vertices.size:
            tf = np.asarray(cache.GetLocalToWorldTransform(child)) @ inverse
            points.append((np.c_[vertices, np.ones(len(vertices))] @ tf)[:, :3])
    if not points:
        raise ValueError('No canonical shelf geometry')
    points = np.concatenate(points)
    lo, hi = points.min(0), points.max(0)
    if not np.isfinite([lo, hi]).all() or np.any(hi <= lo):
        raise ValueError('Invalid shelf-local bounds')
    local_front = np.array([(lo[0]+hi[0])/2, lo[1], 0., 1.])
    front = local_front @ matrix
    return dict(front_midpoint_xy=front[:2].tolist(), inward_heading_rad=a+math.pi/2,
                shelf_path=path, local_bounds=[lo.tolist(), hi.tolist()],
                world_transform=matrix.tolist())


def sample_candidates(frame, seed, count=256):
    """Bounded seeded rejection-search proposals, never sorted by closeness.

    Count is a CPU search budget, not a widened pose or acceptance threshold.
    Exhaustion must be reported as such, not as proof that no solution exists.
    """
    import numpy as np
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError('Explicit integer episode seed required')
    policy = json.loads(Path(__file__).with_name('initial_station_policy.json').read_text())
    ranges = policy['source_randomization_ranges']
    limits = [ranges[k] for k in ('lateral_delta_uniform_m', 'depth_delta_uniform_m', 'heading_delta_uniform_deg')]
    key = hashlib.sha256(f"{policy['policy']}:{seed}".encode()).digest()
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(key[:16], 'little')))
    for index, delta in enumerate(rng.uniform(np.asarray(limits)[:, 0], np.asarray(limits)[:, 1], size=(count, 3))):
        pose = world_pose(frame['front_midpoint_xy'], frame['inward_heading_rad'], delta)
        yield dict(pose, candidate_index=index, search_budget=count, seed=seed)
