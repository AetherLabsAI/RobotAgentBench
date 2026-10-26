"""Read-only measured initialization and palm-path admission.

Conservative swept boxes reject collisions rather than moving fixtures or
changing any shape. Native current robot/target checks remain necessary at
activation; this report alone is never a physical preview acceptance.
"""
import numpy as np
from .push_motion import proposal, palm_fit


def collision_vertices(stage, body_path):
    return np.concatenate(collision_shapes(stage, body_path))


def collision_shapes(stage, body_path, *, names=None):
    from pxr import UsdGeom
    from evaluation.tasks.coffee_machine.isaac_evidence import body_colliders
    tf = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(body_path)), dtype=float)
    chunks = []
    for name, (points, sphere) in body_colliders(stage, body_path).items():
        if names is not None and name not in names:
            continue
        if points is not None:
            chunks.append(points)
        if sphere is not None:
            from itertools import product
            center, radius = sphere
            chunks.append(np.array(list(product((-radius, radius), repeat=3))) + center)
    return [local @ tf[:3, :3] + tf[3, :3] for local in chunks]


def main_case_shapes(stage, body_path, *, include_back=False):
    """Original lower side walls, optionally Back; never hopper/head/tray."""
    names = {'/Collision/Left', '/Collision/Right'}
    if include_back:
        names.add('/Collision/Back')
    shapes = collision_shapes(stage, body_path, names=names)
    if len(shapes) != len(names):
        raise ValueError('Required original lower machine case walls missing')
    return shapes


def static_collision_vertices(stage, path):
    """Read an authored static collider WITH its existing USD scale.

    Countertop_17 is a scaled static cube, not an unscaled rigid-body frame.
    Applying the original rigid-object inspector to it would wrongly reject
    the unchanged counter. No scaling opinion is authored here.
    """
    from pxr import Usd, UsdGeom, UsdPhysics
    from evaluation.tasks.coffee_machine.isaac_evidence import collider_points
    root = stage.GetPrimAtPath(path)
    if not root:
        raise ValueError('Missing static support: ' + path)
    cache = UsdGeom.XformCache()
    chunks = []
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.CollisionAPI) and prim.GetAttribute('physics:collisionEnabled').Get() is not False:
            points, sphere = collider_points(prim, cache.GetLocalToWorldTransform(prim))
            if points is not None:
                chunks.append(points)
            if sphere is not None:
                from itertools import product
                center, radius = sphere
                chunks.append(np.array(list(product((-radius, radius), repeat=3))) + center)
    if not chunks:
        raise ValueError('No enabled support collision geometry: ' + path)
    return np.concatenate(chunks)


def bag_mouth_local(stage):
    from evaluation.tasks.coffee_machine.specs import BAG
    from evaluation.tasks.coffee_machine.isaac_evidence import body_colliders
    points = np.concatenate([p for p, _ in body_colliders(stage, BAG).values() if p is not None])
    return np.array([(points[:, 0].min() + points[:, 0].max()) / 2,
                     points[:, 1].max(), (points[:, 2].min() + points[:, 2].max()) / 2])


def box(points):
    return np.array([points.min(0), points.max(0)])


def intersect(a, b):
    return bool(np.all(a[0] < b[1]) and np.all(b[0] < a[1]))


def swept(a, delta):
    return np.array([np.minimum(a[0], a[0] + delta), np.maximum(a[1], a[1] + delta)])


def inspect_routes(stage, seed):
    from pxr import UsdGeom
    from evaluation.tasks.coffee_machine.specs import MACHINE, MACHINE_BODY, CUP, BAG, COUNTER
    from .open_palm import OpenPalmHand
    tf = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(MACHINE)), dtype=float).T
    sample = proposal(seed, tf)
    asset = OpenPalmHand()
    machine_shapes = collision_shapes(stage, MACHINE_BODY)
    vertices = np.concatenate(machine_shapes)
    machine = box(vertices)
    counter = box(static_collision_vertices(stage, COUNTER))
    # Bag can be lifted later, so its initial intersection is a recorded runtime
    # dependency, not grounds for teleporting it or forcing a different seed.
    bag = box(collision_vertices(stage, BAG))
    cup = box(collision_vertices(stage, CUP))
    routes = []
    points, _ = asset.sample(0.)
    for sign in (-1, 1):
        side = dict(sample, sign=sign, side='left' if sign < 0 else 'right')
        original_axis = np.asarray(sample['direction_world_xy_unit']) * sample['sign']
        side['direction_world_xy_unit'] = (sign * original_axis).tolist()
        direction = np.asarray(side['direction_world_xy_unit'])
        delta = direction * sample['distance_m']
        fit = palm_fit(asset, vertices, direction, avoid_boxes=[cup, counter], machine_shapes_world=machine_shapes,
                       contact_shapes_world=main_case_shapes(stage, MACHINE_BODY), actor_travel_world=delta)
        machine_route = swept(machine, delta)
        hand = points @ np.asarray(fit['orientation']).T + np.asarray(fit['wrist_position_m'])
        shapes = [swept(box(hand[ids]), delta) for ids in asset.collision_groups().values()]
        reasons = []
        if not (np.all(machine_route[0, :2] >= counter[0, :2]) and np.all(machine_route[1, :2] <= counter[1, :2])):
            reasons.append('machine_sweep_outside_counter')
        if any(intersect(shape, counter) for shape in shapes):
            reasons.append('hand_crosses_counter')
        if any(intersect(shape, cup) for shape in shapes):
            reasons.append('hand_crosses_cup')
        if not fit['palm_is_leading_shape']:
            reasons.append('finger_would_contact_before_palm')
        routes.append(dict(proposal=side, palm_fit=fit, static_admitted=not reasons, reasons=reasons,
                           machine_swept_box_m=machine_route.tolist(),
                           bag_initially_in_machine_sweep=intersect(machine_route, bag),
                           robot_native_route_verified=False, full_fixture_route_verified=False,
                           physical_push_verified=False))
    selected = next(row for row in routes if row['proposal']['sign'] == sample['sign'])
    return dict(selected_side=sample['side'], selected_route_static_admitted=selected['static_admitted'],
                planned_distance_m=sample['distance_m'], routes=routes, resampled=False,
                scope='counter/cup and full open anatomical hand; other fixtures/native robot pending',
                runtime_admission_required=True)
