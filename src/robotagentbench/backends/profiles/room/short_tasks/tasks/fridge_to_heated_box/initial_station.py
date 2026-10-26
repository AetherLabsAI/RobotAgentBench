"""Approved source-range station in the refrigerator's Y-up asset frame.

Source: 50 published fridge_microwave layouts, completed.jsonl -> job.json;
fridge_world_from_source @ fridge_source_frame, inverse mapped initial_base.
Distances refer to the asset origin, NOT the door face. Joint pose is unchanged.
"""
import math
import random
from pathlib import Path

LATERAL_M = (-0.28774942028337236, -0.2503012563752356)
FORWARD_M = (1.1012209509895827, 1.1408997624224195)
HEADING_DEG = (-36.90461661174604, -32.96842395266826)
POLICY = 'fridge_source50_station_range_v1'


class NoFeasibleInitial(ValueError):
    pass


def world_pose(frame, lateral, forward, heading):
    for value, limits in zip((lateral, forward, heading), (LATERAL_M, FORWARD_M, HEADING_DEG)):
        if not math.isfinite(value) or not limits[0] <= value <= limits[1]:
            raise ValueError('Outside approved refrigerator station range')
    origin, right, front = (frame[k] for k in ('origin_xy', 'lateral_axis_xy', 'front_axis_xy'))
    xy = [origin[i] + right[i]*lateral + front[i]*forward for i in range(2)]
    yaw = math.atan2(-front[1], -front[0]) + math.radians(heading)
    yaw = (yaw + math.pi) % (2*math.pi) - math.pi
    return dict(position_xy=xy, yaw_rad=yaw, policy=POLICY,
                local_lateral_m=lateral, local_forward_m=forward,
                heading_relative_inward_deg=heading, collision_checked=False,
                physics_certified=False)


def fridge_frame(stage, manifest):
    import numpy as np
    from pxr import UsdGeom
    rows = [r for r in manifest['objects'] if r.get('id') == 'refrigerator']
    if len(rows) != 1:
        raise ValueError('Unique scene-owned refrigerator required')
    path = rows[0]['path']; prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsActive():
        raise ValueError('Missing refrigerator frame')
    m = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim), dtype=float)
    r = m[:3,:3]
    if (not np.isfinite(m).all() or not np.allclose(r@r.T, np.eye(3), atol=1e-6, rtol=0)
            or abs(np.linalg.det(r)-1) > 1e-6 or not np.allclose(r[1], [0,0,1], atol=1e-6, rtol=0)):
        raise ValueError('Expected rigid Y-up refrigerator asset frame; no rescaling/tilt fallback')
    return dict(fridge_path=path, origin_xy=m[3,:2].tolist(),
                lateral_axis_xy=r[0,:2].tolist(), front_axis_xy=r[2,:2].tolist(),
                world_transform=m.tolist(), reference_files=[str(Path(__file__))])


def sample_candidates(frame, seed, count=256):
    if type(seed) is not int:
        raise ValueError('Explicit integer room seed required')
    rng = random.Random(f'{POLICY}:{seed}')
    for index in range(count):
        values = [rng.uniform(*limits) for limits in (LATERAL_M, FORWARD_M, HEADING_DEG)]
        yield dict(world_pose(frame, *values), candidate_index=index, seed=seed, search_budget=count)


def transformed_collider_bounds(robot_boxes, origin, rotation, position):
    """Retain every authored/ready collider; don't fill gaps between links."""
    import itertools
    import numpy as np
    result = []
    for box in robot_boxes:
        points = (np.array(list(itertools.product(*zip(*box)))) - origin) @ rotation + position
        result.append(np.array([points.min(0), points.max(0)]))
    return result


def collider_hits(boxes, obstacle_box, triangles):
    from openend.tasks.scene_requirements.station_scene_checks import intersects, triangles_hit_box
    return any(intersects(box, obstacle_box) and
               (triangles is None or triangles_hit_box(triangles, box)) for box in boxes)
