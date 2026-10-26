"""Strict validation for one published place-metrics NPZ/JSON pair."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from check_success import (
    CURRENT_COFFEE_TABLE_PLACE_SUCCESS_TASKS,
    current_layout_trace_errors,
    validate_place_metrics_v3,
)


PLACE_METRICS_SCHEMA = "real2sim.place_metrics.v3"
IDENTITY_KEYS = (
    "schema", "task_id", "episode_uid", "run_id", "runtime_source_revision",
    "target_body_path", "destination_kind", "destination_body_path",
)


def _reject_json_constant(value):
    raise ValueError(f"non-standard JSON constant {value}")


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _scalar_text(metrics, key: str) -> str:
    raw = np.asarray(metrics[key])
    values = raw.reshape(-1)
    if values.size != 1 or raw.dtype.kind != "U":
        raise ValueError(f"metrics {key} must be one unicode-text scalar")
    value = str(values[0])
    if (not value.strip()
            or any(0xD800 <= ord(char) <= 0xDFFF
                   or ord(char) < 0x20
                   or ord(char) == 0x7F
                   for char in value)):
        raise ValueError(
            f"metrics {key} must contain printable Unicode scalar values")
    return value


def _json_integer(summary: dict, key: str) -> int:
    value = summary.get(key)
    if type(value) is not int:
        raise ValueError(f"metrics JSON {key} must be one integer")
    return value


def _json_number(summary: dict, key: str) -> float:
    value = summary.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"metrics JSON {key} must be one finite number")
    try:
        value = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(
            f"metrics JSON {key} must be one finite number") from exc
    if not math.isfinite(value):
        raise ValueError(f"metrics JSON {key} must be one finite number")
    return value


def validate_place_sidecar_pair(
    metrics_path,
    summary_path,
    *,
    expected_run_id=None,
    expected_task_id=None,
    expected_episode_uid=None,
    expected_target_body_path=None,
    expected_destination_kind=None,
    expected_destination_body_path=None,
    expected_destination_bottom_z=None,
    expected_zone=None,
) -> dict:
    """Validate structure, cross-file identity, and optional expected identity.

    Returns the normalized ``identity`` and ``zone``.  Raises ``ValueError`` on
    any mismatch so callers cannot silently publish or resume stale evidence.
    """
    metrics_path = Path(metrics_path)
    summary_path = Path(summary_path)
    if not metrics_path.is_file() or not summary_path.is_file():
        raise ValueError("place metrics NPZ/JSON pair is incomplete")

    try:
        with np.load(metrics_path, allow_pickle=False) as metrics:
            contract_errors = validate_place_metrics_v3(metrics)
            if contract_errors:
                raise ValueError(contract_errors[0])
            identity = {
                key: _scalar_text(metrics, key)
                for key in IDENTITY_KEYS
            }
            if (
                identity["task_id"]
                in CURRENT_COFFEE_TABLE_PLACE_SUCCESS_TASKS
            ):
                layout_errors = current_layout_trace_errors(
                    identity["task_id"], metrics)
                if layout_errors:
                    raise ValueError(
                        "place metrics current-layout mismatch: "
                        + "; ".join(layout_errors))
            zone = np.asarray(metrics["zone"], dtype=float).reshape(-1)
            rendered_frames = int(
                np.asarray(metrics["rendered_frames"]).reshape(-1)[0])
            fps = float(np.asarray(metrics["fps"]).reshape(-1)[0])
            contact_force_threshold_n = float(np.asarray(
                metrics["contact_force_threshold_n"]).reshape(-1)[0])
            hold_threshold = float(np.asarray(
                metrics["hold_threshold"]).reshape(-1)[0])
            destination_bottom_z = float(np.asarray(
                metrics["destination_bottom_z"]).reshape(-1)[0])
            release_t = int(
                np.asarray(metrics["release_t"]).reshape(-1)[0])
            contact_available = bool(
                np.asarray(metrics["contact_available"]).reshape(-1)[0])
            samples = len(np.asarray(metrics["obj_xyz"]))
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            f"invalid metrics NPZ: {type(exc).__name__}: {exc}") from exc

    try:
        summary = json.loads(
            summary_path.read_text(errors="strict"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except Exception as exc:
        raise ValueError(
            f"invalid metrics JSON: {type(exc).__name__}: {exc}") from exc
    if not isinstance(summary, dict):
        raise ValueError("metrics JSON must contain one object")

    for key, npz_value in identity.items():
        summary_value = summary.get(key)
        if not isinstance(summary_value, str) or summary_value != npz_value:
            raise ValueError(f"NPZ/JSON {key} mismatch")
    try:
        raw_zone = summary["zone"]
        if (not isinstance(raw_zone, list)
                or any(isinstance(value, bool)
                       or not isinstance(value, (int, float))
                       for value in raw_zone)):
            raise ValueError
        summary_zone = np.asarray(raw_zone, dtype=float).reshape(-1)
    except (KeyError, OverflowError, TypeError, ValueError) as exc:
        raise ValueError("invalid metrics JSON zone") from exc
    if (summary_zone.shape != (5,)
            or not np.isfinite(summary_zone).all()
            or not np.allclose(summary_zone, zone, atol=1e-9, rtol=0)):
        raise ValueError("NPZ/JSON zone mismatch")
    if _json_integer(summary, "release_t") != release_t:
        raise ValueError("NPZ/JSON release_t mismatch")
    if _json_integer(summary, "rendered_frames") != rendered_frames:
        raise ValueError("NPZ/JSON rendered_frames mismatch")
    if _json_integer(summary, "samples") != samples:
        raise ValueError("NPZ/JSON samples mismatch")
    if _json_number(summary, "fps") != fps:
        raise ValueError("NPZ/JSON fps mismatch")
    if (_json_number(summary, "contact_force_threshold_n")
            != contact_force_threshold_n):
        raise ValueError("NPZ/JSON contact_force_threshold_n mismatch")
    if _json_number(summary, "hold_threshold") != hold_threshold:
        raise ValueError("NPZ/JSON hold_threshold mismatch")
    if (_json_number(summary, "destination_bottom_z")
            != destination_bottom_z):
        raise ValueError("NPZ/JSON destination_bottom_z mismatch")
    summary_available = summary.get("contact_available")
    if (type(summary_available) is not bool
            or summary_available is not contact_available):
        raise ValueError("NPZ/JSON contact_available mismatch")

    expected = {
        "run_id": expected_run_id,
        "task_id": expected_task_id,
        "episode_uid": expected_episode_uid,
        "target_body_path": expected_target_body_path,
        "destination_kind": expected_destination_kind,
        "destination_body_path": expected_destination_body_path,
    }
    for key, value in expected.items():
        if value is not None and identity[key] != str(value):
            raise ValueError(
                f"metrics {key} mismatch: expected {str(value)!r}, "
                f"got {identity[key]!r}")
    if expected_zone is not None:
        try:
            wanted_zone = np.asarray(expected_zone, dtype=float).reshape(-1)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError("expected zone is invalid") from exc
        if (wanted_zone.shape != (5,)
                or not np.isfinite(wanted_zone).all()
                or not np.allclose(wanted_zone, zone, atol=1e-9, rtol=0)):
            raise ValueError("metrics zone does not match expected calibration")
    if expected_destination_bottom_z is not None:
        try:
            wanted_bottom = float(expected_destination_bottom_z)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(
                "expected destination_bottom_z is invalid") from exc
        if (not math.isfinite(wanted_bottom)
                or not math.isclose(
                    destination_bottom_z, wanted_bottom,
                    abs_tol=1e-9, rel_tol=0)):
            raise ValueError(
                "metrics destination_bottom_z does not match expected "
                "calibration")

    return {
        "identity": identity,
        "zone": [float(value) for value in zone],
        "rendered_frames": rendered_frames,
        "fps": fps,
        "contact_force_threshold_n": contact_force_threshold_n,
        "hold_threshold": hold_threshold,
        "destination_bottom_z": destination_bottom_z,
    }
