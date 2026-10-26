"""Derive one-object Family-7 actions from multi-object REAL episodes.

The reviewed REAL demonstrations continue after the first successful disposal
because they are end-to-end cleanup episodes.  Training samples in this
pipeline intentionally stop after that first object has landed, then return
the robot command smoothly to its frame-zero pose.  The source archive is
never modified.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np


FIRST_OBJECT_END_FRAME_EXCLUSIVE = 201
RETURN_HOME_FRAMES = 60
SINGLE_TARGET_FRAMES = (
    FIRST_OBJECT_END_FRAME_EXCLUSIVE + RETURN_HOME_FRAMES
)
MOTION_CHANNELS = ("arm", "waist", "head", "base_xyyaw")
GRIP_CHANNELS = ("lgrip", "rgrip")


def minimum_jerk_weights(frames: int = RETURN_HOME_FRAMES) -> np.ndarray:
    """Return endpoint-exclusive-to-inclusive minimum-jerk blend weights."""
    if frames < 2:
        raise ValueError("return-home motion needs at least two frames")
    t = np.arange(1, frames + 1, dtype=np.float64) / float(frames)
    return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5


def interpolate_home(
    start: np.ndarray,
    home: np.ndarray,
    *,
    frames: int = RETURN_HOME_FRAMES,
) -> np.ndarray:
    start_value = np.asarray(start)
    home_value = np.asarray(home)
    if start_value.shape != home_value.shape:
        raise ValueError(
            f"return endpoints differ: {start_value.shape} != "
            f"{home_value.shape}"
        )
    weights = minimum_jerk_weights(frames)
    flat_shape = (frames,) + (1,) * start_value.ndim
    result = (
        start_value.astype(np.float64)[None, ...]
        + weights.reshape(flat_shape)
        * (
            home_value.astype(np.float64)
            - start_value.astype(np.float64)
        )[None, ...]
    )
    result[-1] = home_value
    return result.astype(start_value.dtype, copy=False)


def derive_single_target_action(
    arrays: Mapping[str, np.ndarray],
    *,
    source_frames: int,
    first_object_end_frame_exclusive: int = (
        FIRST_OBJECT_END_FRAME_EXCLUSIVE
    ),
    return_home_frames: int = RETURN_HOME_FRAMES,
) -> dict[str, np.ndarray]:
    """Cut the E2E episode after object one and append a smooth home return."""
    required = {*MOTION_CHANNELS, *GRIP_CHANNELS, "fps"}
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"source action is missing channels: {missing}")
    if first_object_end_frame_exclusive >= source_frames:
        raise ValueError("first-object cutoff must precede source episode end")
    for name in (*MOTION_CHANNELS, *GRIP_CHANNELS):
        value = np.asarray(arrays[name])
        if value.ndim < 1 or value.shape[0] != source_frames:
            raise ValueError(
                f"source channel {name} has shape {value.shape}, expected "
                f"frame dimension {source_frames}"
            )

    end = first_object_end_frame_exclusive
    result: dict[str, np.ndarray] = {}
    for name, raw in arrays.items():
        value = np.asarray(raw)
        if value.ndim == 0 or value.shape[0] != source_frames:
            result[name] = value.copy()
            continue
        prefix = value[:end].copy()
        if name in MOTION_CHANNELS:
            recovery = interpolate_home(
                prefix[-1], value[0], frames=return_home_frames
            )
        elif name in GRIP_CHANNELS:
            if float(np.max(np.abs(prefix[-1] - value[0]))) > 1.0e-6:
                raise ValueError(
                    f"{name} must already be open before return-home begins"
                )
            recovery = np.repeat(
                value[0:1], return_home_frames, axis=0
            )
        else:
            recovery = np.repeat(
                prefix[-1:], return_home_frames, axis=0
            )
        result[name] = np.concatenate((prefix, recovery), axis=0)

    expected = end + return_home_frames
    if any(
        np.asarray(result[name]).shape[0] != expected
        for name in (*MOTION_CHANNELS, *GRIP_CHANNELS)
    ):
        raise RuntimeError("derived single-target action length changed")
    for name in MOTION_CHANNELS:
        if not np.array_equal(result[name][-1], np.asarray(arrays[name])[0]):
            raise RuntimeError(f"{name} did not return exactly to frame zero")
    for name in GRIP_CHANNELS:
        if not np.array_equal(result[name][-1], np.asarray(arrays[name])[0]):
            raise RuntimeError(f"{name} did not finish at its open home value")
    return result


def replace_arm_recovery(
    arm: np.ndarray,
    manipulated_prefix: np.ndarray,
    *,
    first_object_end_frame_exclusive: int = (
        FIRST_OBJECT_END_FRAME_EXCLUSIVE
    ),
    return_home_frames: int = RETURN_HOME_FRAMES,
) -> np.ndarray:
    """Install a retargeted manipulation prefix and recompute its home path."""
    source = np.asarray(arm)
    prefix = np.asarray(manipulated_prefix)
    if (
        source.shape[0] != first_object_end_frame_exclusive + return_home_frames
        or prefix.shape != source[:first_object_end_frame_exclusive].shape
    ):
        raise ValueError("retargeted arm prefix does not match action contract")
    result = source.copy()
    result[:first_object_end_frame_exclusive] = prefix
    result[first_object_end_frame_exclusive:] = interpolate_home(
        prefix[-1], source[0], frames=return_home_frames
    )
    if not np.array_equal(result[-1], source[0]):
        raise RuntimeError("retargeted arm recovery did not reach home")
    return result


__all__ = [
    "FIRST_OBJECT_END_FRAME_EXCLUSIVE",
    "MOTION_CHANNELS",
    "RETURN_HOME_FRAMES",
    "SINGLE_TARGET_FRAMES",
    "derive_single_target_action",
    "interpolate_home",
    "minimum_jerk_weights",
    "replace_arm_recovery",
]
