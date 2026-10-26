"""Approved 1.5-second rear-side, constant-height human intervention.

Only the human wrist/closure is commanded. The bag still moves through native
contact and the existing opposed-contact-gated temporary constraint.
"""
import math
from short_task_robustness.tasks.stock_wooden_tray_interference.motion import (
    MotionSample, interpolate, smooth)

DURATION_S = 1.5


class RelocationMotion:
    def __init__(self, entry, grasp, raised, destination_raised, destination, exit_pose):
        self.points = tuple(tuple(float(v) for v in p) for p in
                            (entry, grasp, raised, destination_raised, destination, exit_pose))
        if any(len(p) != 3 or not all(math.isfinite(v) for v in p) for p in self.points):
            raise ValueError('Finite admitted 3D waypoints required')
        if self.points[1] == self.points[4]:
            raise ValueError('Relocation needs a different destination')
        if self.points[0] != self.points[1]:
            raise ValueError('Direct appearance requires entry at the open grip')
        if self.points[2] != self.points[1] or self.points[3] != self.points[4]:
            raise ValueError('Direct horizontal motion has no lift/lower waypoints')
        if any(abs(p[2] - self.points[1][2]) > 1e-12 for p in self.points):
            raise ValueError('Direct horizontal motion must retain wrist height')
        self.duration_s = DURATION_S

    def sample(self, elapsed):
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError('Invalid elapsed simulation time')
        t = elapsed / self.duration_s
        entry, grasp, _, _, destination, exit_pose = self.points
        if t < .3:
            return MotionSample('approach', grasp, smooth(t/.3))
        if t < .7:
            return MotionSample('carry', interpolate(grasp, destination, (t-.3)/.4), 1.)
        if t < .8:
            return MotionSample('release', destination, 1-smooth((t-.7)/.1))
        if t < 1:
            return MotionSample('retreat', interpolate(destination, exit_pose, (t-.8)/.2), 0.)
        return MotionSample('done', exit_pose, 0.)
