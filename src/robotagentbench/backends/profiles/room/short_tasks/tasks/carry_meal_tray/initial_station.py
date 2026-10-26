"""Approved fore/aft and yaw jitter; no independent lateral/joint jitter."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
POLICY = HERE / "initial_policy.json"

def sample_candidates(seed, delta, context, count=100):
    if type(seed) is not int or not 1 <= seed <= 4294967295:
        raise ValueError("Positive uint32 seed required")
    if type(count) is not int or count < 1:
        raise ValueError("Positive candidate count required")
    delta = np.asarray(delta, dtype=float)
    if (delta.shape != (4,4) or not np.isfinite(delta).all()
        or not np.allclose(delta[:,3], [0,0,0,1], atol=1e-8, rtol=0)
        or not np.allclose(delta[:3,:3] @ delta[:3,:3].T, np.eye(3), atol=1e-8, rtol=0)
        or not np.isclose(np.linalg.det(delta[:3,:3]), 1, atol=1e-8, rtol=0)
        or not np.allclose(delta[2,:3], [0,0,1], atol=1e-8, rtol=0)
        or abs(delta[3,2]) > 1e-8):
        raise ValueError("Upright rigid floor-preserving frame required")
    p = json.loads(POLICY.read_text())
    nominal, jitter = p["nominal"], p["approved_jitter"]
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(
        f"carry-meal-tray-initial-v1:{seed}".encode()).digest(), "big"))
    frame_yaw = math.atan2(delta[0,1], delta[0,0])
    for index in range(count):
        shift = float(rng.uniform(*jitter["forward_m"]))
        turn = float(rng.uniform(*jitter["yaw_deg"]))
        distance = nominal["outward_distance_m"] + shift
        left = nominal["left_offset_m"]
        base = np.array([-distance, left, 0, 1]) @ delta
        yaw = frame_yaw + math.radians(nominal["yaw_from_inward_deg"] + turn)
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))
        yield dict(candidate_index=index, sampling="approved_carry_tray_fore_aft_yaw_v1",
            initial_base_xyyaw=[float(base[0]), float(base[1]), yaw],
            outward_distance_m=distance, left_offset_m=left,
            forward_jitter_m=shift, yaw_jitter_deg=turn,
            yaw_from_inward_deg=nominal["yaw_from_inward_deg"]+turn,
            targets_rad=dict(p["targets_rad"]), joints_sampling="fixed_source_preparation")

def station_context(stage, manifest):
    from pxr import UsdGeom
    matches = [o for o in manifest["objects"] if o.get("id") == "meal_tray_large"]
    if len(matches) != 1:
        raise ValueError("Unique scene-owned large tray required")
    support = manifest["supports"]["tray"]
    path = matches[0]["path"]
    prim = stage.GetPrimAtPath(path)
    surface = stage.GetPrimAtPath(support["path"])
    if not prim or not prim.IsActive() or not surface or not surface.IsActive():
        raise ValueError("Actual tray and support must exist")
    # Use the actual rigid-body frame from source data, not its wrapper origin.
    body_path = path + "/Asset/Object"
    body = stage.GetPrimAtPath(body_path)
    if not body or not body.IsActive():
        raise ValueError("Expected large tray body frame missing")
    front = np.asarray(support["front"], dtype=float)
    if front.shape != (2,) or not np.isfinite(front).all() or np.linalg.norm(front) < 1e-8:
        raise ValueError("Actual support front required")
    front /= np.linalg.norm(front)
    inward = -front
    left = np.array([-inward[1], inward[0]])
    matrix = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(body))
    if not np.isfinite(matrix).all():
        raise ValueError("Invalid measured tray transform")
    delta = np.eye(4)
    delta[0,:2], delta[1,:2], delta[3,:2] = inward, left, matrix[3,:2]
    context = dict(tray_path=path, tray_body_path=body_path, support_path=support["path"],
        origin_xy=matrix[3,:2].tolist(), outward_front=front.tolist(),
        actual_surface_height_m=support["actual_top_m"],
        reference_files=[str(POLICY), str(HERE/"dataset_initial_evidence.json")])
    return delta, context, POLICY
