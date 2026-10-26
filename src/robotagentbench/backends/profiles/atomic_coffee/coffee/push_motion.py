"""Open-palm trajectory only; the coffee machine is never commanded.

Runtime checks left/right then front/back in the same eligible frame at the
approved 10 cm upper bound. All blocked is an explicit error, never a delay.
Seeded proposals remain for placeholder geometry only.
"""
import math
import random
import numpy as np
from .moving_frame import rigid_matrix, mouth_above_opening

DISTANCE_M = (0.05, 0.10)
PUSH_DURATION_S = 1.0


def away_from_mouth_proposal(seed, machine_matrix, mouth_world, hopper, *, require_clear=True):
    """Latest user definition: 10 cm, side selected from the LIVE bag mouth.

    Never use future trajectory targets to choose direction. The recorded
    source pour window is an offline acceptance test, not privileged control.
    This returns the preferred direction; route_candidates adds the explicitly
    user-approved opposite-side fallback without suppressing collision checks.
    """
    matrix = rigid_matrix(machine_matrix)
    mouth = np.asarray(mouth_world, dtype=float)
    if mouth.shape != (3,) or not np.isfinite(mouth).all():
        raise ValueError('Finite current bag mouth required')
    if not mouth_above_opening(mouth, matrix, hopper):
        raise ValueError('Push trigger requires current mouth above actual opening')
    up = matrix[:3, 1]
    rim = matrix[:3, :3] @ np.array([0., hopper.rim_y, hopper.center_z]) + matrix[:3, 3]
    drop = float(up @ (mouth - rim) / up[2])
    local = (mouth - [0., 0., drop] - matrix[:3, 3]) @ matrix[:3, :3]
    base = proposal(seed, matrix)
    sign = -1 if local[0] > 0 else 1 if local[0] < 0 else base['sign']
    axis = matrix[:3, 0].copy()
    axis[2] = 0.
    axis /= np.linalg.norm(axis)
    direction = sign * axis
    target = matrix.copy()
    target[:3, 3] += DISTANCE_M[1] * direction
    clears = not mouth_above_opening(mouth, target, hopper)
    if require_clear and not clears:
        raise ValueError('Approved 10 cm cannot clear the current mouth; no distance expansion permitted')
    return dict(base, side='left' if sign < 0 else 'right', sign=sign,
                direction_world_xy_unit=direction.tolist(), distance_m=DISTANCE_M[1],
                direction_source='current_native_bag_mouth_projection',
                mouth_projection_local_x_m=float(local[0]), current_mouth_world_m=mouth.tolist(),
                distance_policy='user_confirmed_10cm_upper_bound',
                predicted_current_mouth_outside_after_push=clears, future_source_actions_used=False)


def route_candidates(seed, machine_matrix, mouth_world, hopper):
    """Four fixed-distance options, in one frame, without future actions."""
    preferred = away_from_mouth_proposal(seed, machine_matrix, mouth_world, hopper, require_clear=False)
    preferred['machine_axis'] = 'lateral'
    opposite = dict(preferred, sign=-preferred['sign'],
                    side='right' if preferred['side'] == 'left' else 'left',
                    direction_world_xy_unit=(-np.asarray(preferred['direction_world_xy_unit'])).tolist(),
                    direction_source='opposite_side_after_preferred_route_blocked')
    target = rigid_matrix(machine_matrix).copy()
    target[:3, 3] += opposite['distance_m'] * np.asarray(opposite['direction_world_xy_unit'])
    # An opposite-side push need not clear this exact fixed mouth position.
    # Record the prediction honestly; the real subsequent pour is verified later.
    opposite['predicted_current_mouth_outside_after_push'] = not mouth_above_opening(mouth_world, target, hopper)
    matrix = rigid_matrix(machine_matrix)
    axis = matrix[:3, 2].copy()
    axis[2] = 0.
    if np.linalg.norm(axis) == 0:
        raise ValueError('Measured machine front/back axis must have horizontal projection')
    axis /= np.linalg.norm(axis)
    up = matrix[:3, 1]
    rim = matrix[:3, :3] @ [0., hopper.rim_y, hopper.center_z] + matrix[:3, 3]
    mouth = np.asarray(mouth_world, dtype=float)
    projected = mouth - [0., 0., float(up @ (mouth - rim) / up[2])]
    local = (projected - matrix[:3, 3]) @ matrix[:3, :3]
    # Prefer the depth direction away from the CURRENT mouth (not future rows).
    sign = -1 if local[2] >= hopper.center_z else 1
    depth = []
    for value in (sign, -sign):
        direction = value * axis
        target = matrix.copy()
        target[:3, 3] += preferred['distance_m'] * direction
        depth.append(dict(preferred, side='front' if value < 0 else 'back', sign=value,
            machine_axis='depth', direction_world_xy_unit=direction.tolist(),
            direction_source='front_back_after_lateral_routes_blocked',
            predicted_current_mouth_outside_after_push=not mouth_above_opening(mouth_world, target, hopper)))
    return preferred, opposite, *depth


class NoPushRoute(RuntimeError):
    """Intervention infeasible now; stop explicitly, never wait for a late push."""
    def __init__(self, attempts):
        import json
        self.attempts = attempts
        super().__init__('All push directions blocked; late retry disabled: ' + json.dumps(attempts))


def select_admitted_direction(candidates, admit):
    """Find an admitted direction now; fail explicitly if none is available."""
    attempts = []
    for candidate in candidates:
        try:
            clear, evidence = admit(candidate)
        except ValueError as error:
            clear, evidence = False, dict(reason=str(error))
        attempts.append(dict(side=candidate['side'], admitted=clear, evidence=evidence))
        if clear:
            return True, dict(evidence, direction_attempts=attempts,
                              opposite_side_fallback_used=len(attempts) == 2,
                              front_back_fallback_used=len(attempts) > 2,
                              selection_attempt_count=len(attempts))
    raise NoPushRoute(attempts)


def proposal(seed, machine_matrix):
    if type(seed) is not int or seed < 0:
        raise ValueError('Explicit nonnegative integer seed required')
    matrix = rigid_matrix(machine_matrix)
    axis = matrix[:3, 0].copy()
    axis[2] = 0.
    length = float(np.linalg.norm(axis))
    if length == 0:
        raise ValueError('Machine left/right axis must be horizontal')
    axis /= length
    rng = random.Random(seed)
    sign = rng.choice((-1, 1))
    distance = rng.uniform(*DISTANCE_M)
    return dict(seed=seed, side='left' if sign < 0 else 'right', sign=sign,
                direction_world_xy_unit=(sign * axis).tolist(), distance_m=distance,
                push_duration_s=PUSH_DURATION_S, resampled=False,
                machine_pose_commands=False, native_push_verified=False)


def palm_fit(asset, machine_vertices_world, direction, *, avoid_boxes=(), machine_shapes_world=None,
             contact_shapes_world=None, actor_travel_world=None):
    """Fit an open palm against the measured side support plane.

    All anatomical points begin outside the machine's outer convex envelope.
    If protruding fingers would hit first, report it instead of deforming the
    asset or claiming a palm contact. Native contact admission is still needed.
    """
    if getattr(asset, 'pose_identity', None) == 'source_cropped_rig_rest_open_palmar_face':
        from .palmar_fit import fit_open_palm
        return fit_open_palm(asset, machine_vertices_world, direction,
                             avoid_boxes=avoid_boxes, machine_shapes_world=machine_shapes_world,
                             contact_shapes_world=contact_shapes_world, actor_travel_world=actor_travel_world)
    vertices = np.asarray(machine_vertices_world, dtype=float)
    direction = np.asarray(direction, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError('Measured machine collision vertices required')
    if direction.shape != (3,) or not np.isfinite(direction).all() or direction[2] != 0:
        raise ValueError('Horizontal palm direction required')
    if not math.isclose(float(np.linalg.norm(direction)), 1., abs_tol=1e-9):
        raise ValueError('Unit palm direction required')
    points, _ = asset.sample(0.)
    points = np.asarray(points, dtype=float)
    palm = points[asset.collision_groups()['hand.R']]
    # This baked asset is rotated for a side bottle grip: local +Z is NOT
    # necessarily the exposed palm face. Read the actual palm hull normals.
    from scipy.spatial import ConvexHull
    hull = ConvexHull(palm)
    palmar = getattr(asset, 'palmar_normal_local', None)
    options = []
    for triangle, equation in zip(hull.simplices, hull.equations):
        normal = equation[:3]
        if palmar is not None and float(normal @ palmar) <= 0:
            continue
        lead = float(np.max(points @ normal))
        palm_lead = float(np.max(palm @ normal))
        if lead - palm_lead > 1e-9:
            continue
        vertices_face = palm[triangle]
        area = float(np.linalg.norm(np.cross(vertices_face[1] - vertices_face[0],
                                            vertices_face[2] - vertices_face[0])) / 2.)
        options.append((area, normal.copy(), vertices_face.copy()))
    if not options:
        raise ValueError('No exposed palm face in the unchanged open hand pose')
    _, palm_normal, face = max(options, key=lambda item:
        ((float(item[1] @ palmar), item[0]) if palmar is not None else (item[0],)))
    # Keep the anatomical long hand axis upward, using measured geometry rather
    # than assuming the source bake's axes. This is a wrist pose, not rig tuning.
    _, _, axes = np.linalg.svd(points - points.mean(0), full_matrices=False)
    longitudinal = axes[0] - float(axes[0] @ palm_normal) * palm_normal
    if np.linalg.norm(longitudinal) == 0:
        raise ValueError('Cannot orient palm without a measured long hand axis')
    longitudinal /= np.linalg.norm(longitudinal)
    if longitudinal @ [1., 0., 0.] < 0:
        longitudinal = -longitudinal
    local_basis = np.column_stack((longitudinal, np.cross(palm_normal, longitudinal), palm_normal))
    world_basis = np.column_stack(([0., 0., 1.], np.cross(direction, [0., 0., 1.]), direction))
    orientation = world_basis @ local_basis.T
    side_plane = float(np.min(vertices @ direction))
    # Centre the selected leading palm patch on the machine's side face.
    anchor = face.mean(0)
    lead = float(np.max(points @ palm_normal))
    anchor += (lead - float(anchor @ palm_normal)) * palm_normal
    center = (vertices.min(0) + vertices.max(0)) / 2
    center += (side_plane - float(center @ direction)) * direction
    # Select the midpoint of the geometrically feasible side-panel height,
    # above any protected cup. This is actor placement, not target alteration.
    relative = (points - anchor) @ orientation.T
    face_relative = (face - anchor) @ orientation.T
    lower = float(vertices[:, 2].min() - face_relative[:, 2].min())
    upper = float(vertices[:, 2].max() - face_relative[:, 2].max())
    for obstacle in avoid_boxes:
        obstacle = np.asarray(obstacle, dtype=float)
        if obstacle.shape != (2, 3) or not np.isfinite(obstacle).all():
            raise ValueError('Measured protected fixture bounds required')
        lower = max(lower, float(obstacle[1, 2] - relative[:, 2].min()))
    if lower > upper:
        raise ValueError('No exposed side-panel palm placement clear of protected cup')
    center[2] = (lower + upper) / 2.
    if machine_shapes_world is not None:
        # Feet/tray can protrude farther sideways than the upper case. Locate
        # the collision silhouette in the actual HAND height band, otherwise
        # part of the 5–10 cm stroke is spent traversing empty air. Clip each
        # existing convex collider by the band's two horizontal planes; all
        # pairwise vertex segments include every actual hull-edge crossing.
        band = center[2] + np.array([relative[:, 2].min(), relative[:, 2].max()])
        clipped = []
        for shape in machine_shapes_world:
            shape = np.asarray(shape, dtype=float)
            clipped.extend(shape[(shape[:, 2] >= band[0]) & (shape[:, 2] <= band[1])])
            for height in band:
                below = shape[shape[:, 2] < height]
                above = shape[shape[:, 2] > height]
                for a in below:
                    for b in above:
                        clipped.append(a + (height - a[2]) / (b[2] - a[2]) * (b - a))
        if not clipped:
            raise ValueError('No actual case collision surface at the palm height')
        actual_plane = float(np.min(np.asarray(clipped) @ direction))
        center += (actual_plane - float(center @ direction)) * direction
    wrist = center - orientation @ anchor
    leading_palm = float(np.max(palm @ palm_normal))
    return dict(wrist_position_m=wrist.tolist(), orientation=orientation.tolist(),
                palm_gap_m=lead - leading_palm,
                palm_normal_local=palm_normal.tolist(),
                palmar_face_required=palmar is not None,
                palmar_alignment_cosine=(float(palm_normal @ palmar) if palmar is not None else None),
                feasible_contact_height_m=[lower, upper],
                side_plane_source=('height_clipped_existing_collision_shapes' if machine_shapes_world is not None
                                   else 'whole_machine_outer_envelope'),
                palm_is_leading_shape=math.isclose(lead, leading_palm, abs_tol=1e-9),
                native_contact_verified=False, fingers_closure=0.)


class PalmPushMotion:
    def __init__(self, contact_wrist, push_proposal):
        self.contact = np.asarray(contact_wrist, dtype=float)
        self.direction = np.asarray(push_proposal['direction_world_xy_unit'], dtype=float)
        self.distance = push_proposal['distance_m']
        if self.contact.shape != (3,) or not np.isfinite(self.contact).all():
            raise ValueError('Finite fitted wrist required')
        if not DISTANCE_M[0] <= self.distance <= DISTANCE_M[1]:
            raise ValueError('Travel outside approved range')
        if (self.direction.shape != (3,) or not np.isfinite(self.direction).all()
                or self.direction[2] != 0
                or not math.isclose(float(np.linalg.norm(self.direction)), 1., abs_tol=1e-9)):
            raise ValueError('Unit horizontal push direction required')
        if push_proposal['push_duration_s'] != PUSH_DURATION_S:
            raise ValueError('Use the approved 1 second push')

    def sample(self, elapsed_s):
        if isinstance(elapsed_s, bool) or not isinstance(elapsed_s, (int, float)) or not math.isfinite(elapsed_s) or elapsed_s < 0:
            raise ValueError('Finite nonnegative elapsed time required')
        fraction = min(elapsed_s / PUSH_DURATION_S, 1.)
        # Smooth start/stop; does not make the machine follow this trajectory.
        travel = fraction * fraction * (3. - 2. * fraction) * self.distance
        return dict(phase='push' if elapsed_s < PUSH_DURATION_S else 'push_complete',
                    wrist_position_m=(self.contact + travel * self.direction).tolist(),
                    closure=0., machine_pose_commands=False)
