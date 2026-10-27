"""Audited model-side projection. Never changes the owner contract or physics.

User approved limit clipping, base deadband, and straight/turn exclusivity.
All limits and acceleration bounds come from the current owner's handshake.
Two adapter-only deadbands: 0.001 m/s and 0.005 rad/s (inclusive).
Call commit only for acknowledged executed rows; rejection leaves state intact.
"""
import math
from copy import deepcopy

VERSION = 'model_action_projection.v1'
DEADBAND = (0.001, 0.005)


class ActionAdapter:
    def __init__(self, contract):
        self.contract = deepcopy(contract)
        self.order = contract['action_order']
        self.limits = contract['joint_limits']
        base = contract['base']
        if (len(self.order) != 22 or self.order[20:] != base['fields']
                or base['simultaneous_translation_rotation'] is not False):
            raise ValueError('Unsupported action contract')
        self.maximum = tuple(base['max_abs'])
        self.delta = tuple(x * base['control_dt_s'] for x in base['max_acceleration'])
        if any(not math.isfinite(x) or x <= 0 for x in (*self.maximum, *self.delta)):
            raise ValueError('Invalid base limits')
        for name in self.order[:20]:
            lo, hi = self.limits[name]
            if not all(math.isfinite(x) for x in (lo, hi)) or lo > hi:
                raise ValueError('Invalid joint limits')
        self.previous = [0., 0.]
        self.pending = None

    def project(self, actions):
        if self.pending is not None:
            raise RuntimeError('Previous projection not acknowledged')
        if not isinstance(actions, list) or not 1 <= len(actions) <= self.contract['max_tool_steps']:
            raise ValueError('Invalid action chunk length')
        # Validate the ENTIRE raw chunk before storing a pending projection.
        for row in actions:
            if (not isinstance(row, list) or len(row) != 22
                    or any(isinstance(x, bool) or not isinstance(x, (float, int))
                           or not math.isfinite(x) for x in row)):
                raise ValueError('Expected finite 22D numeric rows')
        previous, result, events = self.previous[:], [], []
        for index, raw in enumerate(actions):
            row = list(map(float, raw))
            reasons = [[] for _ in row]
            for i, name in enumerate(self.order[:20]):
                lo, hi = self.limits[name]
                row[i] = max(lo, min(hi, row[i]))
                if row[i] != raw[i]:
                    reasons[i].append('joint_limit')
            desired = row[20:]
            for axis in (0, 1):
                value = desired[axis]
                desired[axis] = max(-self.maximum[axis], min(self.maximum[axis], value))
                if value != desired[axis]:
                    reasons[20+axis].append('base_speed_limit')
                if abs(desired[axis]) <= DEADBAND[axis]:
                    if desired[axis] != 0:
                        reasons[20+axis].append('base_deadband')
                    desired[axis] = 0.
            if all(desired):
                # Compare dimensionless fractions, not m/s against rad/s.
                # On an exact tie retain prior mode; otherwise prefer straight.
                strengths = [abs(desired[a])/self.maximum[a] for a in (0, 1)]
                winner = 0 if strengths[0] >= strengths[1] else 1
                if strengths[0] == strengths[1] and previous[1] != 0:
                    winner = 1
                desired[1-winner] = 0.
                reasons[21-winner].append('base_mutual_exclusion')
            prior_mode = next((i for i in (0, 1) if previous[i] != 0), None)
            next_mode = next((i for i in (0, 1) if desired[i] != 0), None)
            if prior_mode is not None and next_mode is not None and prior_mode != next_mode:
                reasons[20+next_mode].append('base_stop_before_switch')
                desired = [0., 0.]
            for axis in (0, 1):
                step = max(-self.delta[axis], min(self.delta[axis], desired[axis]-previous[axis]))
                value = previous[axis] + step
                if abs(desired[axis]-previous[axis]) <= self.delta[axis]:
                    value = desired[axis]  # exact zero on the final braking step
                if value != desired[axis]:
                    reasons[20+axis].append('base_acceleration_limit')
                row[20+axis] = value
            previous = row[20:]
            for i in range(22):
                if row[i] != raw[i] or reasons[i]:
                    events.append(dict(row=index, channel=self.order[i], raw=raw[i],
                                       applied=row[i], reasons=reasons[i]))
            result.append(row)
        self.pending = deepcopy(result)
        return result, dict(version=VERSION, deadband=list(DEADBAND), events=events,
                            raw=deepcopy(actions), applied=deepcopy(result))

    def commit(self, executed):
        if self.pending is None or type(executed) is not int or not 0 <= executed <= len(self.pending):
            raise ValueError('Invalid execution acknowledgement')
        if executed:
            self.previous = self.pending[executed-1][20:]
        self.pending = None

