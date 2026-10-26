"""Approved task95 INITIAL-state sampler, independent of simulator/motion code.

No environment overrides, arm noise, trajectory generation, docking or physics
setters. Returned poses are candidates, NOT accepted physical reset states.
L1 math and RNG order are preserved for the same layout RNG and initpert seed.
"""
import math
import random
from functools import lru_cache

SCHEMA = 'arena.refrigerator_take.initial_randomization.v1'
SLOTS = {
    'low_out': {'x': .113, 'level': 'low'},
    'low_in': {'x': .304, 'level': 'low'},
    'up_out': {'x': .100, 'level': 'up'},
    'up_in': {'x': .300, 'level': 'up'},
}
HALF_END = {'beef': .1215, 'salad': .0963, 'food': .1176}
LEAN_Y = {'low': .815, 'up': 1.168}
LEAN_RAW = {'low': [-.617, .326, .344, .628],
            'up': [-.536, .332, .411, .658]}
BASE_XY_YAW = (2.865, .928, 53.)


def qmul(a, b):
    return [a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],
            a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
            a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],
            a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]]


def qaxis(axis, degrees):
    h = math.radians(degrees)/2
    q = [math.cos(h), 0., 0., 0.]
    q[1+axis] = math.sin(h)
    return q


def lean_deg(q):
    n = math.sqrt(sum(x*x for x in q))
    w, x, y, z = [v/n for v in q]
    return math.degrees(math.asin(min(1., abs(2*(x*y+z*w)))))


@lru_cache(maxsize=2)
def lean_template(level):
    # Exact L1 _lean_to bisection, confirmed default LEAN_TARGET=14 degrees.
    q = LEAN_RAW[level]
    lo, hi = -40., 60.
    for _ in range(60):
        mid = (lo+hi)/2
        if lean_deg(qmul(qaxis(0, -mid), q)) > 14.:
            lo = mid
        else:
            hi = mid
    return tuple(qmul(qaxis(0, -(lo+hi)/2), q))


@lru_cache(maxsize=2)
def yaw_straight(level):
    def tip_x(yaw):
        q = qmul(qaxis(1, yaw), lean_template(level))
        n = math.sqrt(sum(v*v for v in q))
        w, x, y, z = [v/n for v in q]
        ax = (1-2*(y*y+z*z), 2*(x*y+z*w), 2*(x*z-y*w))
        return -ax[0] if ax[2] > 0 else ax[0]
    lo, hi = -60., 60.
    for _ in range(60):
        mid = (lo+hi)/2
        if tip_x(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2


def lean_lq(level, yaw, pitch, roll):
    q = qmul(qaxis(2, roll*.25), lean_template(level))
    q = qmul(qaxis(0, pitch*.30), q)
    return qmul(qaxis(1, yaw), q)


def sample_layout(rng):
    """Same draw order/rounding as L1 make_layout; three distinct slots."""
    picked = rng.sample(list(SLOTS), 3)
    result = {}
    for kind, slot in zip(('beef', 'salad', 'food'), picked):
        spec = SLOTS[slot]
        z0 = .354-HALF_END[kind]-.005-.010
        result[kind] = dict(
            slot=slot,
            x=round(spec['x']+rng.uniform(-.03, .03), 3),
            z=round(z0+rng.uniform(-.006, .006), 3),
            yaw=round(yaw_straight(spec['level'])+rng.uniform(0., 20.), 1),
            pitch=round(rng.uniform(-20., 20.), 1),
            roll=round(rng.uniform(-12., 12.), 1))
    return result


def build_plan(seed):
    import numpy as np
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    layout = sample_layout(random.Random(seed))
    # Separate L1 robot RNG; do not add the later persistent arm-offset draws.
    rng = np.random.RandomState(seed)
    dx, dy = float(rng.uniform(-.05, .05)), float(rng.uniform(-.05, .05))
    dyaw = float(rng.uniform(-5., 5.))
    boxes = {}
    for kind, entry in layout.items():
        level = SLOTS[entry['slot']]['level']
        q = lean_lq(level, entry['yaw'], entry['pitch'], entry['roll'])
        norm = math.sqrt(sum(x*x for x in q))
        boxes[kind] = dict(
            **entry, level=level,
            door_local_position_m=[entry['x'], LEAN_Y[level], entry['z']],
            door_local_quaternion_wxyz=[v/norm for v in q],
            source_compartment='shelf4' if level == 'up' else 'bin34')
    return dict(
        schema=SCHEMA, seed=seed, boxes=boxes,
        robot=dict(baseline_xy_yaw_m_deg=list(BASE_XY_YAW),
                   delta_xy_yaw_m_deg=[dx, dy, dyaw],
                   world_xy_yaw_m_deg=[BASE_XY_YAW[0]+dx, BASE_XY_YAW[1]+dy,
                                       BASE_XY_YAW[2]+dyaw]),
        frame='closed right-door local: Y up, metres; robot world Z-up',
        refrigerator_pose_changed=False, arm_noise=False, motion_retargeting=False,
        physical_initial_state_accepted=False,
        source_policy='L1 task95 defaults approved 2026-09-14',
        seed_contract='API seed independently initializes Python layout RNG and NumPy robot RandomState; not the legacy batch attempt-number mapping')


def fixed_plan():
    """Audited L1 placement_layout.json; no random draws or inferred grasp pose."""
    import json
    from .initial_state import TASK_DIR
    snapshot = json.loads((TASK_DIR/'sources/l1_snapshot.json').read_text())
    source = snapshot['reports']['placement_layout.json']
    boxes = {}
    for kind, entry in source['data'].items():
        level = SLOTS[entry['slot']]['level']
        q = lean_lq(level, entry['yaw'], entry['pitch'], entry['roll'])
        norm = math.sqrt(sum(x*x for x in q))
        boxes[kind] = dict(**entry, level=level,
            door_local_position_m=[entry['x'], LEAN_Y[level], entry['z']],
            door_local_quaternion_wxyz=[v/norm for v in q],
            source_compartment='shelf4' if level == 'up' else 'bin34')
    if set(boxes) != {'beef', 'salad', 'food'} or len({b['slot'] for b in boxes.values()}) != 3:
        raise ValueError('Incomplete L1 fixed three-box layout')
    return dict(schema=SCHEMA, seed=None, boxes=boxes,
        robot=dict(baseline_xy_yaw_m_deg=list(BASE_XY_YAW),
                   delta_xy_yaw_m_deg=[0.,0.,0.], world_xy_yaw_m_deg=list(BASE_XY_YAW)),
        frame='closed right-door local: Y up, metres; robot world Z-up',
        refrigerator_pose_changed=False, arm_noise=False, motion_retargeting=False,
        physical_initial_state_accepted=False,
        source_policy='L1 placement_layout.json fixed candidate',
        source_layout_sha256=source['sha256'], randomization_enabled=False)
