"""Task-owned initialization reservation; never a new asset/scoring target.

All geometry must use the same shelf-local frame (or the axis-aligned probe's
world frame). A reservation freezes an already checked route. It does not
certify physical grasping, and never searches for another slot at trigger time.
"""
from copy import deepcopy
import math

if __package__:
    from .motion import horizontal_relocation_allowed
    from .distance_bands import checked_rule, allows, validate_reservation_rule
else:  # Historical task-local diagnostic imports.
    from motion import horizontal_relocation_allowed
    from distance_bands import checked_rule, allows, validate_reservation_rule

SCHEMA = 'arena.human_relocation_reservation.v1'


def _bounds(value):
    box = tuple(tuple(float(x) for x in row) for row in value)
    if (len(box) != 2 or any(len(row) != 3 for row in box)
            or not all(math.isfinite(x) for row in box for x in row)
            or any(box[0][i] >= box[1][i] for i in range(3))):
        raise ValueError('Finite positive 3D bounds required')
    return box


def _overlap(a, b):
    return all(a[0][i] < b[1][i] and b[0][i] < a[1][i] for i in range(3))


def slot_is_clear(reservation, obstacle_bounds):
    """Used by background placement and again before one-shot activation."""
    if reservation.get('schema') != SCHEMA:
        raise ValueError('Missing initialization reservation')
    slot = _bounds(reservation['reserved_bounds'])
    source = _bounds(reservation['initial_target_bounds'])
    start = [(source[0][i] + source[1][i])/2 for i in (0, 1)]
    end = [(slot[0][i] + slot[1][i])/2 for i in (0, 1)]
    if not horizontal_relocation_allowed(start, end):
        raise ValueError('Horizontal relocation must exceed 10 cm; regenerate this episode')
    validate_reservation_rule(reservation)
    return not any(_overlap(slot, _bounds(b)) for b in obstacle_bounds)


def freeze_reservation(*, target_path, shelf_level_path, seed, shelf_bounds,
                       target_bounds, destination_xy, obstacle_bounds,
                       waypoints, route_evidence, distance_rule=None):
    """Call ONLY during initialization, after transport + exit admission.

    Keeping the target's Z interval reserves its full volume at the same
    support height, rather than an un-sized center marker. No target moves.
    """
    if not target_path or not shelf_level_path or type(seed) is not int:
        raise ValueError('Exact target/shelf identities and seed required')
    shelf, target = _bounds(shelf_bounds), _bounds(target_bounds)
    if len(destination_xy) != 2 or not all(math.isfinite(x) for x in destination_xy):
        raise ValueError('Finite destination XY required')
    center = [(target[0][i] + target[1][i]) / 2 for i in (0, 1)]
    if not horizontal_relocation_allowed(center, destination_xy):
        raise ValueError('Horizontal relocation must exceed 10 cm')
    distance_rule = checked_rule(distance_rule)
    if not allows(distance_rule, center, destination_xy):
        raise ValueError('Horizontal relocation is outside selected distance band')
    delta = [destination_xy[i] - center[i] for i in (0, 1)] + [0.]
    slot = tuple(tuple(row[i] + delta[i] for i in range(3)) for row in target)
    if any(slot[0][i] < shelf[0][i] - 1e-12 or slot[1][i] > shelf[1][i] + 1e-12 for i in (0, 1)):
        raise ValueError('Whole target footprint must fit its original shelf')
    if _overlap(slot, target):
        raise ValueError('Reservation must be separate from initial target')
    points = tuple(tuple(float(x) for x in p) for p in waypoints)
    if (len(points) != 6 or any(len(p) != 3 for p in points)
            or not all(math.isfinite(x) for p in points for x in p)):
        raise ValueError('Six fixed motion waypoints required')
    if any(abs((points[4][i]-points[1][i])-delta[i]) > 1e-9 for i in range(3)):
        raise ValueError('Reserved slot and route destination disagree')
    if route_evidence.get('transport_clear') is not True or route_evidence.get('retreat_clear') is not True:
        raise ValueError('Transport and retreat admission both required')
    value = dict(schema=SCHEMA, role='human_relocation_destination',
                 target_path=target_path, shelf_level_path=shelf_level_path, seed=seed,
                 reserved_bounds=slot, destination_xy=tuple(destination_xy),
                 initial_target_bounds=target, waypoints=points,
                 relocation_distance=deepcopy(distance_rule),
                 route_evidence=deepcopy(route_evidence),
                 scoring_target=False, select_at='initialization', runtime_replanning=False,
                 physical_validation=False)
    if not slot_is_clear(value, obstacle_bounds):
        raise ValueError('Reserved destination occupied')
    return value


def with_reservation(manifest, reservation):
    """Store separately from objects/tasks; don't accidentally award points."""
    if reservation.get('schema') != SCHEMA:
        raise ValueError('Invalid reservation schema')
    if manifest.get('environment_reservations'):
        raise ValueError('Refusing to replace an existing episode reservation')
    result = deepcopy(manifest)
    result['environment_reservations'] = [deepcopy(reservation)]
    return result
