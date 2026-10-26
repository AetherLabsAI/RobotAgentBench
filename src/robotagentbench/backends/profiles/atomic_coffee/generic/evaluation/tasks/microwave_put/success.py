"""Candidate final-state check; hold duration MUST be explicitly approved.

No production hold value is supplied, and no task is registered by this module.
Only the same progress snapshot from the current control frame may be used.
"""
from .process_rules import PutProgress, finite_number


def check_success(progress, *, stable_hold_s):
    if not isinstance(progress, PutProgress):
        raise ValueError("Current PutProgress required")
    if not finite_number(stable_hold_s) or stable_hold_s < 0:
        raise ValueError("An explicit nonnegative stable hold duration is required")
    stages = tuple(event.stage for event in progress.events)
    return (stages == ("opened", "placed_and_released", "closed_after_placement")
            and progress.final_candidate and progress.final_hold_s >= stable_hold_s)
