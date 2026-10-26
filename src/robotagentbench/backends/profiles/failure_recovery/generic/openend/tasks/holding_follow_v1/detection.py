"""Read-only position following. No contact fabrication or physical attachment.

An unbroken segment can take arbitrarily longer than two seconds to reach
10 cm. Displacement is from its fixed start, never accumulated path length.
The start-relative position anchor is NOT a rolling window (no slow drift).
"""
from collections import deque
from dataclasses import dataclass
import math

HANDS = {h: '/World/Robot/gripper_' + h + '_base_link' for h in ('l', 'r')}


@dataclass(frozen=True)
class DetectionConfig:
    grasp_window_s: float = 2.
    relative_position_tolerance_m: float = .03
    minimum_displacement_m: float = .10
    # Existing unrelated support/initial-state thresholds are unchanged.
    support_face_tolerance_m: float = .003
    initial_history_s: float = .25
    initial_position_span_m: float = .002
    contact_release_debounce_s: float = 0.


def _vector(value, length):
    v = tuple(float(x) for x in value)
    if len(v) != length or not all(math.isfinite(x) for x in v):
        raise ValueError('Finite native position/quaternion required')
    return v


def _relative(obj, hand):
    """Object origin in the current hand frame; object rotation is irrelevant."""
    p = _vector(obj['position_m'], 3)
    hp = _vector(hand['position_m'], 3)
    q = _vector(hand['quaternion_wxyz'], 4)
    norm = math.sqrt(sum(x*x for x in q))
    if norm < 1e-12:
        raise ValueError('Zero hand quaternion')
    w, x, y, z = (v/norm for v in q)
    dx, dy, dz = (a-b for a, b in zip(p, hp))
    # Transpose of the hand's rotation matrix, applied to world displacement.
    rel = ((1-2*(y*y+z*z))*dx + 2*(x*y+w*z)*dy + 2*(x*z-w*y)*dz,
           2*(x*y-w*z)*dx + (1-2*(x*x+z*z))*dy + 2*(y*z+w*x)*dz,
           2*(x*z+w*y)*dx + 2*(y*z-w*x)*dy + (1-2*(x*x+y*y))*dz)
    return p, hp, rel


class GraspDetector:
    """Compatibility protocol for Openend readers only; bilateral is ignored.

`history[key].clear()` remains an explicit dormancy/reset hook. Each deque
contains only the segment anchor and its confirmed flag, so long slow carries
do not grow memory. Invalid evidence is rejected before any state is changed.
Native readers already validate clock continuity; skipped >60 Hz intervals
also break every segment here, instead of counting unobserved time as holding.
"""
    def __init__(self, objects, config=None):
        self.config = DetectionConfig()
        if config is not None and config != self.config:
            # Legacy configs are not an implicit override of the approved rule.
            raise ValueError('Openend position-follow DetectionConfig required')
        objects = tuple(objects)
        if len(objects) != len(set(objects)) or not objects:
            raise ValueError('Distinct measured objects required')
        self.history = {(obj, h): deque(maxlen=1) for obj in objects for h in HANDS}
        self.last_time = None

    def reset(self):
        for history in self.history.values():
            history.clear()
        self.last_time = None

    def observe(self, time_s, poses, bilateral=None):
        if not math.isfinite(time_s) or (self.last_time is not None and time_s <= self.last_time):
            raise ValueError('Stale grasp evidence')
        measured = {key: _relative(poses[key[0]], poses[HANDS[key[1]]]) for key in self.history}
        gap = self.last_time is not None and time_s-self.last_time > 1/60 + 1e-6
        result = {obj: False for obj, _ in self.history}
        details = {}
        for (obj, hand), history in self.history.items():
            p, hp, rel = measured[obj, hand]
            if gap or (history and math.dist(rel, history[0][3]) > self.config.relative_position_tolerance_m + 1e-12):
                history.clear()
            if not history:
                history.append((time_s, p, hp, rel, False))
            start, p0, hp0, rel0, was_confirmed = history[0]
            span = time_s-start
            od, hd = math.dist(p, p0), math.dist(hp, hp0)
            confirmed = was_confirmed or (span+1e-9 >= self.config.grasp_window_s
                and min(od, hd)+1e-12 >= self.config.minimum_displacement_m)
            history[0] = (start, p0, hp0, rel0, bool(confirmed))
            result[obj] |= bool(confirmed)
            details[obj+':'+hand] = dict(confirmed=bool(confirmed),
                following_span_s=span, object_displacement_m=od, hand_displacement_m=hd,
                relative_position_error_m=math.dist(rel, rel0),
                predicate='position_follow_v1', bilateral_required=False,
                relative_angle_checked=False)
        self.last_time = time_s
        return result, details
