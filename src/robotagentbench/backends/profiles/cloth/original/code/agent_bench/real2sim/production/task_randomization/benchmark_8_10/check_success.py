"""Sheet 8.10: unfold the clean tablecloth onto CoffeeTable and release it.

The checker consumes actual mesh and clamp state, including growing episode
prefixes. It does not use producer success flags. Motion style, initial
perturbation, source compatibility and physical/visual quality have separate
checks; semantic success does not approve a dataset for release.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCHEMA = "real2sim.tablecloth_spread_score.v1"
TAIL_SECONDS = .5


def _validated_arrays(state):
    required = ("cloth_q", "clamp_owner", "faces", "uv", "time_s")
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError("Missing actual state arrays: " + ", ".join(missing))
    q = np.asarray(state["cloth_q"], dtype=float)
    owner = np.asarray(state["clamp_owner"])
    faces = np.asarray(state["faces"])
    uv = np.asarray(state["uv"], dtype=float)
    time = np.asarray(state["time_s"], dtype=float)
    if (time.ndim != 1 or len(time) < 2 or q.ndim != 3 or q.shape[-1] != 3
            or q.shape[0] != len(time) or q.shape[1] < 3 or owner.shape != q.shape[:2]):
        raise ValueError("Invalid cloth/time/ownership dimensions")
    if owner.dtype.kind not in "iu" or np.any(owner < -1):
        raise ValueError("Clamp ownership must be integer body indices or -1 for released")
    if (faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0
            or faces.dtype.kind not in "iu" or faces.min() < 0
            or faces.max() >= q.shape[1] or uv.shape != (q.shape[1], 2)):
        raise ValueError("Invalid cloth material topology")
    if (not np.isfinite(q).all() or not np.isfinite(uv).all()
            or not np.isfinite(time).all() or np.any(np.diff(time) <= 0)):
        raise ValueError("Nonfinite or nonmonotonic state evidence")
    if len(np.unique(np.sort(faces, axis=1), axis=0)) != len(faces):
        raise ValueError("Duplicate material faces")
    if len(np.unique(faces)) != q.shape[1]:
        raise ValueError("Every cloth particle must belong to the material mesh")
    rest = uv[faces]
    a, b = rest[:, 1] - rest[:, 0], rest[:, 2] - rest[:, 0]
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    if np.any(np.abs(cross) <= 1e-12) or not (np.all(cross > 0) or np.all(cross < 0)):
        raise ValueError("Degenerate or inconsistently wound material faces")
    rest_area = float(np.abs(cross).sum() / 2)
    if rest_area <= 1e-5:
        raise ValueError("Zero material rest area")
    return q, owner, faces, time, rest_area


def score_arrays(state, evidence):
    """Return semantic gates and measured diagnostics for a saved state prefix."""
    q, owner, faces, time, rest_area = _validated_arrays(state)
    bounds = np.asarray(evidence.get("destination_table_bounds_m"), dtype=float)
    if (bounds.shape != (2, 3) or not np.isfinite(bounds).all()
            or np.any(bounds[1] <= bounds[0])):
        raise ValueError("Missing measured destination table bounds")

    # Include the sample at/before the start of the final half-second. This
    # remains correct for irregular timestamps and never extrapolates duration.
    start = max(0, int(np.searchsorted(time, time[-1] - TAIL_SECONDS, side="right")) - 1)
    tail = q[start:]
    tri = tail[:, faces, :2]
    a, b = tri[:, :, 1] - tri[:, :, 0], tri[:, :, 2] - tri[:, :, 0]
    signed_area = np.abs((a[:, :, 0] * b[:, :, 1] - a[:, :, 1] * b[:, :, 0]).sum(axis=1)) / 2
    projection = signed_area / rest_area
    supported = (
        (np.abs(tail[:, :, 2] - bounds[1, 2]) <= .015)
        & (tail[:, :, 0] >= bounds[0, 0] - .003)
        & (tail[:, :, 0] <= bounds[1, 0] + .003)
        & (tail[:, :, 1] >= bounds[0, 1] - .003)
        & (tail[:, :, 1] <= bounds[1, 1] + .003)
    )
    fraction = supported.mean(axis=1)
    gates = {
        "clean_tablecloth_identity": evidence.get("target_identity") == "clean_tablecloth",
        "coffee_table_destination": evidence.get("destination_prim_path") == "/World/Assets/CoffeeTable",
        "sufficient_tail_evidence": bool(time[-1] - time[0] >= TAIL_SECONDS - 1e-6),
        "cloth_released_through_tail": bool(np.all(owner[start:] == -1)),
        "cloth_on_table_through_tail": bool(np.all(fraction >= .95)),
        "cloth_unfolded_through_tail": bool(np.all((projection >= .85) & (projection <= 1.10))),
    }
    velocity = np.linalg.norm(np.diff(tail, axis=0) / np.diff(time[start:])[:, None, None], axis=2)
    return {
        "schema": SCHEMA,
        "success": all(gates.values()),
        "gates": gates,
        "reasons": [key for key, passed in gates.items() if not passed],
        "metrics": {
            "tail_seconds": float(time[-1] - time[start]),
            "tail_start_frame": start,
            "minimum_supported_particle_fraction": float(fraction.min()),
            "minimum_signed_projected_area_over_rest": float(projection.min()),
            "maximum_signed_projected_area_over_rest": float(projection.max()),
            "material_rest_area_m2": rest_area,
        },
        "diagnostics": {
            "tail_particle_speed_max_m_s": float(velocity.max()),
            "grasp_model": evidence.get("grasp_model"),
            "independent_full_trajectory_collision_audit_required": True,
        },
    }


def score(directory):
    root = Path(directory)
    with np.load(root / "full_state.npz", allow_pickle=False) as archive:
        state = {key: archive[key] for key in archive.files}
    evidence = json.loads((root / "tablecloth_evidence.json").read_text())
    return score_arrays(state, evidence)


def check_success(episode_directory):
    """Registered saved-episode success entrypoint."""
    return score(episode_directory)


def check_state_success(state, evidence):
    """Use the same semantic gates on a growing live trajectory prefix."""
    return score_arrays(state, evidence)


if __name__ == "__main__":
    import sys
    print(json.dumps(score(sys.argv[1]), indent=2))
