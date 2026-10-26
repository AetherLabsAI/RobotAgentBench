"""Structural Task 29 phase plan used by replay and scoring.

The plan describes the three cup cycles in source-frame coordinates.  Source
identity belongs to Git; runtime validation checks the numeric trajectory that
will actually be replayed instead of creating a second artifact identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


FPS = 20
FRAME_COUNT = 423
SETTLE_SAMPLE_COUNT = 40
SETTLE_POLICY = "hold_terminal_source_frame_fixed_samples"


@dataclass(frozen=True, slots=True)
class PhasePlan:
    ordinal: int
    cup_id: str
    source_start_frame: int
    source_stop_frame_exclusive: int
    close_frame: int
    strong_close_frame: int
    release_frame: int
    fully_open_frame: int

    @property
    def replay_sample_count(self) -> int:
        return self.source_stop_frame_exclusive - self.source_start_frame

    @property
    def settle_source_frame(self) -> int:
        return self.source_stop_frame_exclusive - 1

    @property
    def settle_sample_count(self) -> int:
        return SETTLE_SAMPLE_COUNT

    @property
    def total_sample_count(self) -> int:
        return self.replay_sample_count + self.settle_sample_count

    def source_frames(self) -> np.ndarray:
        replay = np.arange(
            self.source_start_frame,
            self.source_stop_frame_exclusive,
            dtype=np.int64,
        )
        settle = np.full(
            self.settle_sample_count,
            self.settle_source_frame,
            dtype=np.int64,
        )
        return np.concatenate((replay, settle))


PHASES = (
    PhasePlan(0, "cup0", 0, 195, 101, 102, 147, 148),
    PhasePlan(1, "cup1", 148, 288, 195, 196, 227, 227),
    PhasePlan(2, "cup2", 227, 423, 288, 290, 319, 319),
)


def phase_plan(ordinal: int) -> PhasePlan:
    if isinstance(ordinal, bool) or not isinstance(ordinal, int):
        raise ValueError("Task29 phase ordinal must be an integer")
    if not 0 <= ordinal < len(PHASES):
        raise ValueError("Task29 phase ordinal must be 0, 1, or 2")
    return PHASES[ordinal]


def _transition_frames(mask: np.ndarray, *, rising: bool) -> tuple[int, ...]:
    hits = (
        np.flatnonzero((~mask[:-1]) & mask[1:]) + 1
        if rising
        else np.flatnonzero(mask[:-1] & (~mask[1:])) + 1
    )
    return tuple(int(frame) for frame in hits)


def validate_source_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    """Validate the consumed commands and their three grasp cycles directly."""

    shapes = {
        "arm": (FRAME_COUNT, 14),
        "lgrip": (FRAME_COUNT,),
        "rgrip": (FRAME_COUNT,),
        "waist": (FRAME_COUNT, 2),
        "head": (FRAME_COUNT, 2),
    }
    for name, expected_shape in shapes.items():
        try:
            array = np.asarray(arrays[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Task29 source is missing numeric {name}") from exc
        if (
            array.shape != expected_shape
            or array.dtype.kind not in "fiu"
            or not np.isfinite(array).all()
        ):
            raise ValueError(
                f"Task29 {name} must be finite numeric {expected_shape}"
            )

    left = np.asarray(arrays["lgrip"], dtype=float)
    right = np.asarray(arrays["rgrip"], dtype=float)
    if np.any(left < 0.0) or np.any(left > 1.0):
        raise ValueError("Task29 left grip commands must stay in [0,1]")
    if not np.array_equal(right, np.zeros_like(right)):
        raise ValueError("Task29 inactive right grip must remain open")

    observed = (
        _transition_frames(left >= 0.5, rising=True),
        _transition_frames(left >= 0.9, rising=True),
        _transition_frames(left >= 0.5, rising=False),
        _transition_frames(left <= 0.0, rising=True),
    )
    expected = (
        tuple(plan.close_frame for plan in PHASES),
        tuple(plan.strong_close_frame for plan in PHASES),
        tuple(plan.release_frame for plan in PHASES),
        tuple(plan.fully_open_frame for plan in PHASES),
    )
    if observed != expected:
        raise ValueError(
            "Task29 grip events do not match the three-cup phase plan: "
            f"observed={observed}, expected={expected}"
        )


__all__ = [
    "FPS",
    "FRAME_COUNT",
    "PHASES",
    "SETTLE_POLICY",
    "PhasePlan",
    "phase_plan",
    "validate_source_arrays",
]
