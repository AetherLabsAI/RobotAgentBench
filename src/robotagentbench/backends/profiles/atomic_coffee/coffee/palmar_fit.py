"""Fit an actual +Z palmar surface to a finite machine side panel.

The source rest thumb/fingertips can project beyond the palm in +Z. They may
still clear the FINITE panel at its edge/top. Do not rotate to the back of the
hand just to make the entire hand a globally leading infinite support plane.
The task-local actor supplies its open pushing pose; source mesh/rest data,
scale and physics are unchanged. Convex overlap is a geometric admission proxy,
not native evidence.
"""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog


def clipped_band(shapes, lower, upper):
    result = []
    for shape in shapes:
        shape = np.asarray(shape, dtype=float)
        result.extend(shape[(shape[:, 2] >= lower) & (shape[:, 2] <= upper)])
        for height in (lower, upper):
            for a in shape[shape[:, 2] < height]:
                for b in shape[shape[:, 2] > height]:
                    result.append(a + (height - a[2]) / (b[2] - a[2]) * (b - a))
    if not result:
        raise ValueError('No actual collision side panel at hand height')
    return np.asarray(result)


def positive_overlap(a, b):
    if not (np.all(a.min(0) < b.max(0)) and np.all(b.min(0) < a.max(0))):
        return False
    ea, eb = ConvexHull(a).equations, ConvexHull(b).equations
    eq = np.concatenate([ea, eb])
    # Largest common interior-ball radius. Zero means geometric touching;
    # 1e-9 is numerical computation tolerance, NOT a physical collision margin.
    result = linprog([0., 0., 0., -1.], A_ub=np.column_stack([eq[:, :3], np.ones(len(eq))]),
                     b_ub=-eq[:, 3], bounds=[(None, None)] * 4, method='highs')
    if not result.success:
        raise ValueError('Could not validate finite side-panel overlap')
    return bool(result.x[3] > 1e-9)


def first_palmar_translation(palm, shapes, direction):
    """Exact convex finite-palm contact along the push direction, not AABB gap."""
    hand = ConvexHull(palm).equations
    results = []
    for shape in shapes:
        target = ConvexHull(shape).equations
        a = np.vstack([np.column_stack([hand[:, :3], -(hand[:, :3] @ direction)]),
                       np.column_stack([target[:, :3], np.zeros(len(target))])])
        result = linprog([0., 0., 0., 1.], A_ub=a,
                         b_ub=-np.r_[hand[:, 3], target[:, 3]],
                         bounds=[(None, None)] * 3 + [(0., None)], method='highs')
        if result.success:
            results.append(float(result.x[3]))
    if not results:
        raise ValueError('Finite palm cannot touch the actual case along this push axis')
    return min(results)


def fit_open_palm(asset, vertices, direction, *, avoid_boxes, machine_shapes_world,
                  contact_shapes_world=None, actor_travel_world=None):
    vertices = np.asarray(vertices, dtype=float)
    direction = np.asarray(direction, dtype=float)
    if direction.shape != (3,) or direction[2] != 0 or not np.isclose(np.linalg.norm(direction), 1.):
        raise ValueError('Measured horizontal machine direction required')
    shapes = [np.asarray(p, dtype=float) for p in (machine_shapes_world or [vertices])]
    if contact_shapes_world is None:
        raise ValueError('Explicit original lower case side walls required; no hopper fallback')
    contact_shapes = [np.asarray(p, dtype=float) for p in contact_shapes_world]
    travel = np.zeros(3) if actor_travel_world is None else np.asarray(actor_travel_world, dtype=float)
    vertices = np.concatenate(contact_shapes)
    points, _ = asset.sample(0.)
    groups = asset.collision_groups()
    palm = points[groups['hand.R']]
    # Same source palmar skin identification as fit_static_grip.py, not guessed
    # from whichever hull face happens to be largest or easiest to collide.
    central = palm[(palm[:, 1] > .075) & (palm[:, 1] < .111)
                   & (palm[:, 2] > 0) & (np.abs(palm[:, 0]) < .026)]
    if len(central) < 3:
        raise ValueError('Original model palmar skin patch missing')
    errors = []
    # Candidate wrist rolls come from the actual palm's X/Z hull faces.
    # Rotation about the source finger axis keeps fingers horizontal; never
    # turn the dorsal surface toward the case or deform source finger bones.
    hull = ConvexHull(palm[:, [0, 2]])
    angles = sorted({0., *(-float(np.arctan2(e[0], e[1]))
                          for e in hull.equations if e[1] > 0)}, key=abs)
    for angle in angles:
        for reverse_fingers in (False, True):
            try:
                return _fit_at_tilt(asset, vertices, direction, shapes, contact_shapes, avoid_boxes,
                                    points, groups, palm, central, angle, travel, reverse_fingers)
            except ValueError as error:
                errors.append(str(error))
    raise ValueError('No palmar fit at original/geometrically cleared wrist poses: ' + repr(errors))


def _fit_at_tilt(asset, vertices, direction, shapes, contact_shapes, avoid_boxes, points, groups, palm, central, angle, travel,
                 reverse_fingers):
    from scipy.spatial.transform import Rotation
    local_rotation = Rotation.from_rotvec([0., angle, 0.]).as_matrix()
    rotated_palm = palm @ local_rotation.T
    rotated_anchor = (central @ local_rotation.T).mean(0)
    rotated_anchor[2] = rotated_palm[:, 2].max()
    anchor = local_rotation.T @ rotated_anchor
    # User-confirmed horizontal palm: local +Y fingers along the countertop,
    # local -X thumb side up, local +Z palmar surface toward the side panel.
    up = np.array([0., 0., 1.])
    tangent = np.cross(up, direction)
    if reverse_fingers:
        tangent = -tangent
    orientation = np.column_stack([np.cross(tangent, direction), tangent, direction]) @ local_rotation
    relative = (points - anchor) @ orientation.T
    patch_relative = (central - anchor) @ orientation.T
    lower = float(vertices[:, 2].min() - patch_relative[:, 2].min())
    upper = float(vertices[:, 2].max() - patch_relative[:, 2].max())
    if lower > upper:
        raise ValueError('Original open palm patch cannot fit the lower case')
    # Rest thumb remains outside the top of the finite panel. This is an actor
    # placement derived from source anatomy; no thumb joint pose is retuned.
    thumb_ids = np.unique(np.concatenate([ids for name, ids in groups.items() if name.startswith('thumb.')]))
    thumb_clear_height = float(vertices[:, 2].max() - relative[thumb_ids, 2].min())
    heights = tuple(dict.fromkeys(((lower + upper) / 2., lower, upper,
                                  max(lower, thumb_clear_height))))
    errors = []
    for height in heights:
        try:
            return _fit_horizontal_at_height(asset, vertices, direction, shapes, points, groups,
                orientation, relative, patch_relative, anchor, tangent, up, lower, upper, angle, height,
                contact_shapes, avoid_boxes, travel)
        except ValueError as error:
            errors.append(str(error))
    raise ValueError('Horizontal palm cannot clear the finite panel: ' + repr(errors))


def _fit_horizontal_at_height(asset, vertices, direction, shapes, points, groups,
    orientation, relative, patch_relative, anchor, tangent, up, lower, upper, angle, height,
    contact_shapes, avoid_boxes, travel):
    if not lower <= height <= upper:
        raise ValueError('Palm patch above or below lower case side wall')
    panel = clipped_band(contact_shapes, height + relative[:, 2].min(), height + relative[:, 2].max())
    plane = float((panel @ direction).min())
    panel_lo, panel_hi = float((panel @ tangent).min()), float((panel @ tangent).max())
    patch_t = patch_relative @ tangent
    lo, hi = panel_lo - float(patch_t.min()), panel_hi - float(patch_t.max())
    if lo > hi:
        raise ValueError('Palm patch wider than the measured side panel')
    candidates = []
    # Whole patch first; if the rest thumb projects forward, placing the PALM
    # centre at a real panel edge lets the thumb stay outside that finite side.
    # No thumb joint/mesh/collision is altered. Native palm contact is required
    # in the later physics check; this is not assumed from geometric touching.
    contact_positions = [lo, hi, (lo + hi) / 2, panel_lo, panel_hi]
    # Include exact finite anatomical-edge alignments. Five generic panel
    # positions can miss a valid rear-side pose by only a few millimetres.
    from itertools import product
    for ids in groups.values():
        span = relative[ids] @ tangent
        contact_positions.extend((panel_lo - span.min(), panel_lo - span.max(),
                                  panel_hi - span.min(), panel_hi - span.max()))
        for obstacle in avoid_boxes:
            corners = np.array(list(product(*zip(obstacle[0], obstacle[1]))))
            o = corners @ tangent
            swept_min = span.min() + min(0., travel @ tangent)
            swept_max = span.max() + max(0., travel @ tangent)
            contact_positions.extend((float(o.min() - swept_max), float(o.max() - swept_min)))
    for contact_t in dict.fromkeys(contact_positions):
        center = contact_t * tangent + plane * direction + height * up
        wrist = center - orientation @ anchor
        hand = points @ orientation.T + wrist
        # The frontmost palm vertex can lie outside the panel edge. Compute
        # contact of the actual finite palm hull, rather than assume zero gap.
        try:
            shift = first_palmar_translation(hand[groups['hand.R']], contact_shapes, direction)
        except ValueError:
            continue
        wrist += shift * direction
        hand += shift * direction
        collided = []
        for name, ids in groups.items():
            part = hand[ids]
            if any(positive_overlap(part, shape) for shape in shapes):
                collided.append(name)
            for obstacle in avoid_boxes:
                lo = np.minimum(part.min(0), part.min(0) + travel)
                hi = np.maximum(part.max(0), part.max(0) + travel)
                if np.all(lo < obstacle[1]) and np.all(obstacle[0] < hi):
                    collided.append(name + ':protected_cup')
        candidates.append(dict(contact_t_m=contact_t, intersecting_anatomical_groups=collided))
        if not collided:
            return dict(wrist_position_m=wrist.tolist(), orientation=orientation.tolist(),
                        palm_gap_m=0., palm_normal_local=[0., 0., 1.],
                        palmar_face_required=True, palmar_alignment_cosine=float(np.cos(angle)),
                        wrist_clearance_tilt_deg=float(np.degrees(angle)),
                        finite_panel_admitted=True, palm_is_leading_shape=True,
                        leading_scope='finite panel at actual palm contact; not infinite-plane whole-hand support',
                        source_pose=asset.pose_identity, feasible_contact_height_m=[lower, upper],
                        actor_thumb_splay_deg=asset.thumb_splay_deg,
                        fingers_horizontal=True, finite_palmar_contact_shift_m=shift,
                        contact_region='lower_main_case_side_wall', hopper_contact_allowed=False,
                        case_height_bounds_m=[float(vertices[:, 2].min()), float(vertices[:, 2].max())],
                        side_plane_source='height_clipped_existing_collision_shapes',
                        native_contact_verified=False, fingers_closure=0.,
                        contact_candidates=candidates)
    raise ValueError('No finite-panel palmar fit without thumb/finger penetration: ' + repr(candidates))
