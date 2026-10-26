"""Initial task-facing pose search. Read-only; not an IK/reachability proof.

Reuse spawn_v2's 10 cm search grid, actual colliders and measured folded-ready
envelope. The calibrated sector range bounds the search, not a new stand-off.
Only the launch's pre-physics root pose changes; no scene/asset writes.
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

class NoFeasibleTaskFront(ValueError):
    """Only exhausted geometric candidates, not malformed inputs or assets."""


def unit(vector):
    a = np.asarray(vector, dtype=float)[:2]
    if not np.all(np.isfinite(a)) or np.linalg.norm(a) < 1e-8:
        raise ValueError('Missing horizontal interaction front')
    return a / np.linalg.norm(a)


def starting_target(manifest, task_name):
    """Scene roles, never the destination or a coincidental asset name."""
    rows = manifest['tasks']
    if not rows:
        raise ValueError('No bound task targets')
    if task_name == 'fridge_to_heated_box':
        paths = [o['path'] for o in manifest['objects'] if o['id'] == 'refrigerator']
        if len(paths) != 1:
            raise ValueError('Unique starting refrigerator required')
        return dict(paths=paths, source='refrigerator')
    if task_name == 'retrieve_from_drawer':
        if not manifest.get('openend_drawer'):
            raise ValueError('Drawer mechanism required')
        return dict(paths=['/World/Assets/CoffeeTable/black_box_1'], source='coffee_table.drawer')
    sources = {r.get('source') for r in rows}
    if len(sources) != 1:
        raise ValueError('Ambiguous initial interaction sources')
    paths = []
    for row in rows:
        # Group tasks explicitly own all their object_paths. Other tasks can
        # include a destination in that list, so use target_path only.
        items = row['object_paths'] if row['type'] in ('trash_transfer', 'cloth_clear') else [row['target_path']]
        for path in items:
            if path not in paths:
                paths.append(path)
    return dict(paths=paths, source=next(iter(sources)))


def target_geometry(stage, manifest, task_name):
    from pxr import UsdGeom
    spec = starting_target(manifest, task_name)
    cache = UsdGeom.XformCache()
    bbox = UsdGeom.BBoxCache(0, ['default', 'render', 'proxy'], False, True)

    def box(path):
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsActive():
            raise ValueError('Missing initial target ' + path)
        r = bbox.ComputeWorldBound(prim).ComputeAlignedRange()
        b = np.array([r.GetMin(), r.GetMax()])
        if not np.isfinite(b).all() or np.any(b[0] > b[1]):
            raise ValueError('Invalid initial target bounds ' + path)
        return b

    boxes = [box(p) for p in spec['paths']]
    target = np.mean([(b[0, :2] + b[1, :2]) / 2 for b in boxes], axis=0)
    source = spec['source']
    references = []
    if task_name in ('prepare_cutlery', 'carry_meal_tray'):
        support = manifest['supports']['tray']
        front = unit(support['front'])
        origin = np.asarray(support['origin'])
        evidence = 'scene_declared_tray_support_front'
    elif source == 'refrigerator':
        path = spec['paths'][0]
        # Both closed doors, not the box inside or the microwave destination.
        doors = [box(path + '/' + name) for name in ('left_door', 'right_door')]
        origin = np.mean([(b[0, :2] + b[1, :2]) / 2 for b in doors], axis=0)
        front = unit(origin - target)
        target = origin.copy()
        evidence = 'closed_door_centres_outward_from_fridge_body'
    else:
        key = {'sofa': 'sofa', 'small_table': 'side', 'side_table': 'side',
               'snack_shelf': 'shelf'}.get(source)
        if source and source.startswith('coffee_table.'):
            key = 'table'
        if key is None:
            raise ValueError('No verified starting support direction: ' + str(source))
        reference = ROOT / 'layout_task_activation_v1/calibration_drafts_layout01.json'
        entries = json.loads(reference.read_text())['entries']
        row, = [r for r in entries if r['id'] == key]
        # Reuse only published station directions, never draft stand-offs,
        # design-proposal headings or draft activation/radius values.
        if row['front_source'] != 'published_station_direction':
            raise ValueError('Unverified station heading')
        anchor = row['target']['anchor']
        prim = stage.GetPrimAtPath(anchor)
        if not prim:
            raise ValueError('Source furniture frame missing: ' + anchor)
        previous = np.asarray(row['target']['world_transform'])
        current = np.asarray(cache.GetLocalToWorldTransform(prim))
        radians = math.radians(row['front_deg'])
        front = unit((np.array([math.cos(radians), math.sin(radians), 0., 0.])
                      @ np.linalg.inv(previous) @ current)[:2])
        origin = target.copy()
        evidence = 'published_station_direction_transformed_with_actual_furniture'
        references.append(str(reference))
    return dict(**spec, target_xy=target.tolist(), front=front.tolist(),
                front_origin=origin.tolist(), direction_source=evidence,
                reference_files=references)


def candidates(target, front, origin, polygons, radius, half_angle):
    """Task-facing candidates only; no room-safe fallback behind the target."""
    target, front, origin = map(np.asarray, (target, front, origin))
    pts = np.asarray([v for polygon in polygons for v in polygon])
    lo, hi = pts.min(0), pts.max(0)
    choices = [target + front * d for d in np.arange(.1, radius + 1e-9, .1)]
    choices.extend(np.array(x) for x in itertools.product(
        np.arange(max(lo[0], target[0]-radius), min(hi[0], target[0]+radius), .1),
        np.arange(max(lo[1], target[1]-radius), min(hi[1], target[1]+radius), .1)))
    result = []
    for xy in choices:
        delta = xy - target
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-8 or distance > radius or np.dot(xy-origin, front) <= 0:
            continue
        if np.dot(delta / distance, front) < math.cos(math.radians(half_angle)) - 1e-9:
            continue
        yaw = math.atan2(-delta[1], -delta[0])
        result.append((distance, xy, yaw))
    return sorted(result, key=lambda r: (r[0], float(r[1][0]), float(r[1][1])))


def choose(stage, manifest, task_name, sensor, *, candidate_rows=None, target=None):
    from pxr import Usd, UsdGeom, UsdPhysics
    from openend.tasks.scene_requirements.source_geometry import floor_paths
    from openend.tasks.scene_requirements.station_scene_checks import intersects, triangles_hit_box
    from openend.tasks.layout_repair_v1.ready_envelope import load, bounds
    from openend.tasks.layout_repair_v1.spawn_v2 import inside
    target = target_geometry(stage, manifest, task_name) if target is None else target
    radius, half = float(sensor['range']), float(sensor['halfAngle'])
    if not (math.isfinite(radius) and radius > 0 and 0 < half <= 90):
        raise ValueError('Existing calibrated forward sector required')
    robot = '/World/Robot'
    cache = UsdGeom.XformCache()
    matrix = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(robot)))
    origin = matrix[3, :3]
    yaw0 = math.atan2(matrix[0, 1], matrix[0, 0])
    expected = np.array([[math.cos(yaw0), math.sin(yaw0), 0],
                         [-math.sin(yaw0), math.cos(yaw0), 0], [0, 0, 1]])
    if not np.allclose(matrix[:3, :3], expected, atol=1e-6, rtol=0) or abs(origin[2]) > 1e-6:
        raise ValueError('Unscaled upright robot root at z=0 required')
    floors = floor_paths(stage)
    polygons = manifest.get('overview', {}).get('floors', [])
    if not floors or not polygons:
        raise ValueError('Measured physical floor required')
    bbox = UsdGeom.BBoxCache(0, ['default', 'render', 'proxy', 'guide'], False, True)
    parts, robot_boxes = [], []
    ready = load(stage)
    if candidate_rows is not None:
        from openend.tasks.layout_repair_v1.ready_envelope import REFERENCE
        ready_targets=json.loads((REFERENCE/'navigation_result.json').read_text())['navigation_posture']['targets_rad']
    for p in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not p.IsActive() or not p.HasAPI(UsdPhysics.CollisionAPI) or p.GetAttribute('physics:collisionEnabled').Get() is False:
            continue
        path = str(p.GetPath())
        r = bbox.ComputeWorldBound(p).ComputeAlignedRange()
        box = np.array([r.GetMin(), r.GetMax()], dtype=float)
        if not np.isfinite(box).all() or np.any(box[0] > box[1]):
            raise ValueError('Invalid collider bounds ' + path)
        if path.startswith(robot + '/'):
            # Explicit folded-ready proposals are restored before simulation.
            # The asset's unrelated authored arm pose is not part of that reset.
            # Keep the old two-pose envelope for the unchanged generic search.
            if candidate_rows is None:robot_boxes.append(box)
            robot_boxes.append(bounds(stage, p, box, ready, cache))
            continue
        if any(path == f or path.startswith(f + '/') for f in floors):
            continue
        triangles = None
        if p.IsA(UsdGeom.Mesh):
            vs = p.GetAttribute('points').Get()
            indices = list(p.GetAttribute('faceVertexIndices').Get() or [])
            faces, offset = [], 0
            for n in p.GetAttribute('faceVertexCounts').Get() or []:
                ids = indices[offset:offset+n]
                offset += n
                faces.extend((ids[0], ids[i], ids[i+1]) for i in range(1, n-1))
            if vs is not None and len(vs) and faces:
                world = (np.c_[np.asarray(vs), np.ones(len(vs))] @ np.asarray(cache.GetLocalToWorldTransform(p)))[:, :3]
                triangles = world[np.asarray(faces, dtype=int)]
        parts.append((path, box, triangles))
    if not robot_boxes:
        raise ValueError('No robot collider geometry')
    # Rotate each conservative measured collider box, then form a single
    # enclosing body envelope, retaining the old search's conservative policy.
    corners = np.concatenate([np.array(list(itertools.product(*zip(*b)))) for b in robot_boxes]) - origin
    rejections = []
    if candidate_rows is None:
        proposals=((distance,xy,yaw,None) for distance,xy,yaw in
            candidates(target['target_xy'], target['front'], target['front_origin'], polygons, radius, half))
    else:
        proposals=((float(np.linalg.norm(np.asarray(row['initial_base_xyyaw'][:2])-target['target_xy'])),
            np.asarray(row['initial_base_xyyaw'][:2]),float(row['initial_base_xyyaw'][2]),row)
            for row in candidate_rows)
    for distance, xy, yaw, proposal in proposals:
        if proposal is not None and proposal.get('targets_rad') != ready_targets:
            raise ValueError('Explicit candidates must use the measured folded-ready joint targets')
        a = yaw - yaw0
        rotation = np.array([[math.cos(a), math.sin(a), 0], [-math.sin(a), math.cos(a), 0], [0, 0, 1]])
        points = corners @ rotation + [*xy, 0.]
        box = np.array([points.min(0), points.max(0)])
        perimeter = [(x, y) for x in np.linspace(box[0, 0], box[1, 0], 11) for y in (box[0, 1], box[1, 1])]
        perimeter += [(x, y) for y in np.linspace(box[0, 1], box[1, 1], 11) for x in (box[0, 0], box[1, 0])]
        if not all(any(inside(p, polygon) for polygon in polygons) for p in perimeter):
            continue
        checks=[box]
        if proposal is not None:
            # Empty space between separate links is not robot collision shape.
            # Each bound is still conservative and covers all measured frames.
            groups=points.reshape(-1,8,3)
            checks=[np.array([v.min(0),v.max(0)]) for v in groups]
        hit = next((path for path, b, tri in parts if any(
            intersects(part,b) and (tri is None or triangles_hit_box(tri,part))
            for part in checks)),None)
        if hit:
            if hit not in rejections and len(rejections) < 10:
                rejections.append(hit)
            continue
        return dict(position=[*map(float, xy), 0.], yaw_rad=yaw, target=target, candidate=proposal,
                    target_distance_m=distance, robot_envelope=box.tolist(),
                    original_position=origin.tolist(), original_rejections=rejections,
                    obstacle_colliders=len(parts), method='task_front_yaw_aware_measured_envelope',
                    collision_shape_policy=('individual_measured_ready_colliders'
                        if proposal is not None else 'combined_authored_and_ready_envelope'),
                    search_range_source='existing_sector_calibration_not_IK_reach_proof',
                    physics_certified=False, manipulation_reachability_verified=False)
    raise NoFeasibleTaskFront('No collision-free task-front pose within calibrated range: ' +
                     json.dumps(dict(target=target, rejections=rejections)))
