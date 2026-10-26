"""Recorded initial-state evidence and audit-only mapping helpers.

Production candidate sampling is defined in range_initial_state.py. The intact
sample ordering below is retained only for reproducing source audit fixtures.
No stage, physics, scoring or runtime writes occur here.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).with_name("source_initial_samples.json")


def pose(value):
    a = np.asarray(value, dtype=float)
    if a.shape != (7,) or not np.isfinite(a).all():
        raise ValueError("Finite XYZ + XYZW body pose required")
    q = a[3:]
    if not np.isclose(np.linalg.norm(q), 1, atol=1e-5, rtol=0):
        raise ValueError("Unit source quaternion required")
    return a[:3], q / np.linalg.norm(q)


def yaw(quaternion):
    x, y, z, w = quaternion
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def upright_frame(value):
    p, q = pose(value)
    # PhysX readback has numerical roll/pitch noise. Do not use a tilted or
    # scaled furniture transform as though it were a planar layout change.
    x, y, z, w = q
    up = np.array([2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)])
    if not np.allclose(up, [0, 0, 1], atol=1e-3, rtol=0):
        raise ValueError("Drawer frame must be upright")
    return p, yaw(q)


def read_samples():
    data = json.loads(SOURCE.read_text())
    if data.get("schema") != "short_tasks.retrieve_from_drawer.initial_samples.v1":
        raise ValueError("Wrong source initial-state bank")
    samples = data["samples"]
    if len(samples) != 50 or len({s['episode_index'] for s in samples}) != 50:
        raise ValueError("Expected the audited 50 unique source episodes")
    for s in samples:
        if s["asset"] not in ("wallet", "glasses_case") or s["source_physics_passed"] is not True:
            raise ValueError("Unverified source identity")
        names, values = s["joint_names"], s["joint_positions_rad_or_m"]
        if len(names) != 34 or len(set(names)) != 34 or len(values) != 34:
            raise ValueError("Complete matching articulation initial readback required")
        if not all(type(v) in (int, float) and math.isfinite(v) for v in values):
            raise ValueError("Non-finite initial joint readback")
        upright_frame(s["drawer_xyz_xyzw"])
        upright_frame(s["robot_base_xyz_xyzw"])
        pose(s["target_xyz_xyzw"])
    return samples


def candidates(asset, seed):
    """Audit-only seeded permutation of recorded samples, not episode sampling.

    The owner may reject candidates on fit/collision grounds. Exhaustion must
    fail preparation rather than clamp a pose or widen a random range.
    """
    if asset not in ("wallet", "glasses_case"):
        raise ValueError("Drawer dataset has only wallet and glasses_case")
    if type(seed) is not int or not 1 <= seed <= 4294967295:
        raise ValueError("Valid integer seed required")
    samples = [s for s in read_samples() if s["asset"] == asset]
    key = lambda s: hashlib.sha256(f'drawer-short4-v1:{seed}:{s["source_id"]}'.encode()).digest()
    return sorted(samples, key=key)


def map_sample(sample, drawer_xyz_xyzw):
    """Preserve one sample's robot/target relation under layout translation/yaw.

    Drawer body frames must be rigid poses with scale removed. Target height
    follows the drawer height; the mobile base remains on its original floor
    plane, never lifted to compensate for furniture height changes.
    """
    old, old_yaw = upright_frame(sample["drawer_xyz_xyzw"])
    new, new_yaw = upright_frame(drawer_xyz_xyzw)
    delta = new_yaw-old_yaw
    c, s = math.cos(delta), math.sin(delta)
    r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    base, base_q = pose(sample["robot_base_xyz_xyzw"])
    target, target_q = pose(sample["target_xyz_xyzw"])
    mapped_base = new + r @ (base-old)
    mapped_base[2] = base[2]
    mapped_target = new + r @ (target-old)
    # Quaternion product: layout delta (Z rotation) * recorded target pose.
    z, w = math.sin(delta/2), math.cos(delta/2)
    x0, y0, z0, w0 = target_q
    target_q_new = [w*x0-z*y0, w*y0+z*x0, w*z0+z*w0, w*w0-z*z0]
    heading = (yaw(base_q)+delta+math.pi) % (2*math.pi)-math.pi
    return dict(schema="short_tasks.retrieve_from_drawer.mapped_initial.v1",
        asset=sample["asset"], source_episode=sample["episode_index"], source_id=sample["source_id"],
        initial_base_xyyaw=[float(mapped_base[0]), float(mapped_base[1]), heading],
        source_base_height_m=float(base[2]),
        target_xyz_xyzw=[*mapped_target.tolist(), *map(float, target_q_new)],
        targets_rad=dict(zip(sample["joint_names"], sample["joint_positions_rad_or_m"])),
        joint_units="rad except idx01_body_joint1 in metres",
        source_drawer_xyz_xyzw=deepcopy(sample["drawer_xyz_xyzw"]),
        destination_drawer_xyz_xyzw=list(map(float, drawer_xyz_xyzw)),
        source_files_sha256=deepcopy(sample["source_files_sha256"]),
        source_bank_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        source_drawer_gap_after_settle_m=sample["drawer_initial_gap_m"],
        sampling="one_intact_recorded_sample",
        target_fit_checked=False, robot_collision_checked=False,
        native_physics_validated=False, shared_preparer_connected=False)
