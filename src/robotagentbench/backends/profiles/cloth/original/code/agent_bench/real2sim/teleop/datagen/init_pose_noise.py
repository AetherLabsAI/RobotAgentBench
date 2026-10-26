# -*- coding: utf-8 -*-
"""Initial arm-pose noise: start each generated episode from a perturbed
standard pose instead of the frame-zero the operator happened to record.

Pure numpy (testable without USD).  The mechanism is the mirror image of the
G2 warp-weight profile: the G2 warp is zero at frame 0 and full from
``contact_start``; init-pose noise is FULL at frame 0 and decays smoothly to
zero well before ``contact_start``, so every object-relative frame — the
grasp itself, and the object pose ``align_grasp`` derives from the published
trajectory — is bit-identical to the seed.  Only the free-space approach
changes, which is exactly the part a policy must learn to be robust to.

Both arms receive noise: the passive arm's start pose is part of the visual
state too, and its delta decays on the same profile, so it returns to the
recorded hold pose before anything contact-relevant happens.
"""
from __future__ import annotations

import numpy as np

# Hard sanity cap: more than this is not "noise around the standard pose",
# it is a different pose that nobody FK-verified for self-collision.
MAX_NOISE_RAD_CAP = np.deg2rad(15.0)
# The decayed delta must be gone at least this many frames before contact:
# a still-active delta at contact_start would move the derived object pose.
MIN_SETTLE_FRAMES = 8
JERK_RATIO_MAX = 1.5    # same L2 motion gate as apply_retarget
# Relative gates explode on unnaturally smooth seeds (a synthetic linear ramp
# has ~zero third difference).  Third differences below this are far under
# actuator resolution at 20 fps and never a physical smoothness problem.
JERK_ABS_FLOOR_RAD = 1e-3


def sample_noise(gen_seed: int, max_rad: float, joints: int = 14) -> np.ndarray:
    """Deterministic per-joint start-pose delta for one generation seed.

    A dedicated stream key decorrelates this draw from every other consumer
    of the same gen_seed (ΔT sampling, clutter identity, ...).
    """
    if not np.isfinite(max_rad) or not 0.0 < max_rad <= MAX_NOISE_RAD_CAP:
        raise ValueError(
            f"max_rad must be in (0, {MAX_NOISE_RAD_CAP:.4f}] rad, "
            f"got {max_rad!r}")
    rng = np.random.default_rng([int(gen_seed), 0x1A17])
    return rng.uniform(-max_rad, max_rad, size=joints)


def decay_weights(frames: int, settle_frames: int) -> np.ndarray:
    """w[0]=1 → cosine decay → w[settle_frames:]=0.  Shape (frames,)."""
    if settle_frames < MIN_SETTLE_FRAMES:
        raise ValueError(
            f"settle_frames {settle_frames} < minimum {MIN_SETTLE_FRAMES}; "
            "the noise would still be active around contact")
    if frames <= settle_frames + 1:
        raise ValueError(
            f"trajectory has {frames} frames, too short to settle "
            f"{settle_frames}-frame init noise before the task begins")
    ramp = 0.5 * (1.0 + np.cos(
        np.pi * np.arange(settle_frames + 1) / settle_frames))
    weights = np.zeros(frames, dtype=float)
    weights[:settle_frames + 1] = ramp
    return weights


def effective_settle(settle_frames: int, contact_start: int | None) -> int:
    """Clamp the decay so it ends strictly before object-relative frames."""
    if contact_start is None:
        return int(settle_frames)
    clamped = min(int(settle_frames), int(contact_start) - 1)
    if clamped < MIN_SETTLE_FRAMES:
        raise ValueError(
            f"contact starts at frame {contact_start}: no room to settle "
            f"init-pose noise (need >= {MIN_SETTLE_FRAMES} free-space frames)")
    return clamped


def jerk_ratio(seed_arm: np.ndarray, noised_arm: np.ndarray) -> float:
    """Max third-difference ratio with an absolute floor on the denominator.

    Mirrors datagen.retarget.jerk_ratio, but a seed smoother than the floor
    is compared against the floor instead of its own near-zero jerk, so the
    gate measures physical roughness rather than division noise.
    """
    def peak(q):
        return float(np.abs(np.diff(q, n=3, axis=0)).max()) if len(q) > 3 else 0.0
    seed = peak(seed_arm)
    warped = peak(noised_arm)
    return warped / max(seed, JERK_ABS_FLOOR_RAD)


def apply_noise(arm: np.ndarray, gen_seed: int, max_rad: float,
                settle_frames: int, contact_start: int | None = None) -> dict:
    """Return the noised arm channel plus an audit record.

    ``arm`` is the (T, 14) command trajectory; grip/waist/head/base are the
    caller's responsibility to leave untouched.
    """
    arm = np.asarray(arm, dtype=float)
    if arm.ndim != 2 or arm.shape[1] != 14:
        raise ValueError(f"arm must be (T, 14), got {arm.shape}")
    settle = effective_settle(settle_frames, contact_start)
    delta = sample_noise(gen_seed, max_rad)
    weights = decay_weights(len(arm), settle)
    noised = arm + np.outer(weights, delta)
    ratio = jerk_ratio(arm, noised)
    if ratio > JERK_RATIO_MAX:
        raise ValueError(
            f"init-noised jerk is {ratio:.2f}x the seed's "
            f"(cap {JERK_RATIO_MAX}); reduce --max-deg or raise settle")
    return {
        "arm": noised,
        "delta_rad": delta,
        "settle_frames": settle,
        "jerk_ratio": ratio,
        "max_delta_deg": float(np.rad2deg(np.abs(delta).max())),
    }
