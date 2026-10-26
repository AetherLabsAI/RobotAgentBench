"""Read-only geometry of recorded observations; no IK or action generation."""
import math
import numpy as np


def rotation_wxyz(value):
    q = np.asarray(value, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("Invalid wxyz quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def measured_angle(evidence, initial):
    a = np.array(initial["bodies"]["right_door"]["quaternion_wxyz"])
    b = np.array(evidence["bodies"]["right_door"]["quaternion_wxyz"])
    cosine = abs(float(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))))
    return math.degrees(2*math.acos(min(1.0, max(0.0, cosine))))
