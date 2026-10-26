"""User-selected horizontal relocation bands; never silently downgrade."""
from copy import deepcopy
import math

if __package__:
    from .motion import horizontal_relocation_allowed
else:
    from motion import horizontal_relocation_allowed

PRESETS_CM = {'short': (10., 20.), 'medium': (20., 30.), 'long': (30., 40.)}


def resolve_selection(value=None):
    if value is None:
        return dict(mode='default', min_cm=10., max_cm=None)
    if isinstance(value, str):
        if value not in PRESETS_CM:
            raise ValueError('relocation_distance must be short, medium, long, or a min_cm/max_cm object')
        low, high = PRESETS_CM[value]
        return dict(mode=value, min_cm=low, max_cm=high)
    if not isinstance(value, dict) or set(value) != {'min_cm', 'max_cm'}:
        raise ValueError('Custom relocation_distance requires only min_cm and max_cm')
    low, high = value['min_cm'], value['max_cm']
    if (any(type(v) not in (int, float) or not math.isfinite(v) for v in (low, high))
            or not 10 <= low < high):
        raise ValueError('Custom relocation range must satisfy 10 <= min_cm < max_cm, finite centimetres')
    return dict(mode='custom', min_cm=float(low), max_cm=float(high))


def checked_rule(rule=None):
    if rule is None:
        return resolve_selection()
    if not isinstance(rule, dict) or set(rule) != {'mode', 'min_cm', 'max_cm'}:
        raise ValueError('Invalid frozen relocation distance rule')
    mode = rule['mode']
    if mode == 'default': expected = resolve_selection()
    elif mode == 'custom': expected = resolve_selection({k:rule[k] for k in ('min_cm','max_cm')})
    else: expected = resolve_selection(mode)
    if rule != expected:
        raise ValueError('Relocation preset bounds were changed')
    return deepcopy(expected)


def allows(rule, source, destination):
    rule = checked_rule(rule)
    if not horizontal_relocation_allowed(source, destination): return False
    distance = math.dist(source[:2], destination[:2])
    low = rule['min_cm']/100.
    # Same roundoff handling as the global >10 cm constraint.
    if distance < low or math.isclose(distance, low, rel_tol=0., abs_tol=1e-12): return False
    high = rule['max_cm']
    return high is None or distance <= high/100. or math.isclose(distance, high/100., rel_tol=0., abs_tol=1e-12)


def validate_reservation_rule(reservation, expected=None):
    if not isinstance(reservation, dict) or not {'initial_target_bounds','reserved_bounds'} <= set(reservation):
        raise ValueError('Complete relocation endpoint geometry required')
    rule = checked_rule(reservation.get('relocation_distance'))
    if expected is not None and rule != checked_rule(expected):
        raise ValueError('Room relocation distance differs from requested band; regenerate this episode')
    if __package__:
        from .reservation import _bounds
    else:
        from reservation import _bounds
    bounds = _bounds(reservation['initial_target_bounds'])
    start = [(bounds[0][i]+bounds[1][i])/2 for i in (0,1)]
    end_bounds = _bounds(reservation['reserved_bounds'])
    end = [(end_bounds[0][i]+end_bounds[1][i])/2 for i in (0,1)]
    if not allows(rule, start, end):
        raise ValueError('Frozen relocation is outside its selected distance band; regenerate this episode')
    return rule


def require_manifest_band(plan, manifest):
    expected = checked_rule(plan.get('relocation_distance'))
    reservations = manifest.get('environment_reservations', [])
    if not reservations:
        if expected['mode'] != 'default':
            raise ValueError('Selected distance band requires a generated reservation')
        return
    if len(reservations) != 1:
        raise ValueError('Exactly one relocation reservation required')
    validate_reservation_rule(reservations[0], expected)
