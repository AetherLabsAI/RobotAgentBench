"""Read-only live machine coordinates, independent of USD pose synchronization.

Matrices here use column vectors. The body-to-root transform is measured once
from the unchanged asset, then native body poses determine the moving root.
No scene pose, rigid-body velocity, force, or joint setter is used.
"""
import math
import numpy as np
from scipy.spatial.transform import Rotation


def rigid_matrix(value):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('Finite 4x4 rigid transform required')
    if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6, rtol=0)
            or not math.isclose(float(np.linalg.det(matrix[:3, :3])), 1., abs_tol=1e-6)):
        raise ValueError('Unscaled proper rigid transform required')
    return matrix.copy()


def pose_matrix(position, quaternion_wxyz):
    position = np.asarray(position, dtype=float)
    q = np.asarray(quaternion_wxyz, dtype=float)
    if position.shape != (3,) or q.shape != (4,) or not np.isfinite(position).all() or not np.isfinite(q).all():
        raise ValueError('Finite native body pose required')
    if not math.isclose(float(np.linalg.norm(q)), 1., abs_tol=1e-5):
        raise ValueError('Unit native quaternion required')
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    matrix[:3, 3] = position
    return matrix


class MovingMachineFrame:
    def __init__(self, body_in_root):
        self.body_in_root = rigid_matrix(body_in_root)
        self.body_inverse = np.linalg.inv(self.body_in_root)
        self.matrix = None
        self.time_s = None

    def update(self, position, quaternion_wxyz, time_s):
        if isinstance(time_s, bool) or not isinstance(time_s, (int, float)) or not math.isfinite(time_s):
            raise ValueError('Finite native simulation time required')
        if self.time_s is not None and time_s <= self.time_s:
            raise ValueError('Repeated or stale machine pose')
        self.matrix = rigid_matrix(pose_matrix(position, quaternion_wxyz) @ self.body_inverse)
        self.time_s = float(time_s)

    def current(self, time_s):
        if self.matrix is None or self.time_s != time_s:
            raise ValueError('Machine frame does not belong to this physical substep')
        return self.matrix.copy()

    def world(self, local, time_s):
        m = self.current(time_s)
        return np.asarray(local, dtype=float) @ m[:3, :3].T + m[:3, 3]

    def local(self, world, time_s):
        m = self.current(time_s)
        return (np.asarray(world, dtype=float) - m[:3, 3]) @ m[:3, :3]


def mouth_above_opening(mouth_world, machine_matrix, hopper):
    """Vertical downward ray from the bag mouth intersects the actual rim.

    Uses the source hopper's measured rounded opening without enlarging it.
    A mouth below the rim, or a toppled machine with no upward-facing opening,
    cannot trigger. Bag root proximity is deliberately not used.
    """
    m = rigid_matrix(machine_matrix)
    point = np.asarray(mouth_world, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError('Finite world-space bag mouth required')
    up = m[:3, 1]
    if up[2] <= 0:
        return False
    rim = m[:3, :3] @ np.array([0., hopper.rim_y, hopper.center_z]) + m[:3, 3]
    drop = float(np.dot(up, point - rim) / up[2])
    if drop <= 0:
        return False
    local = (point - np.array([0., 0., drop]) - m[:3, 3]) @ m[:3, :3]
    dx = max(abs(local[0]) - (hopper.half_width - hopper.corner_radius), 0.)
    dz = max(abs(local[2] - hopper.center_z) - (hopper.half_depth - hopper.corner_radius), 0.)
    return bool(dx * dx + dz * dz < hopper.corner_radius ** 2)
