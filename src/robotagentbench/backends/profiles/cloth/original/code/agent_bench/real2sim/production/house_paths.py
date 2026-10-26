#!/usr/bin/env python3
"""One path model for the shared House source and runtime mirror."""

from __future__ import annotations

from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TELEOP = REPO / "real2sim" / "teleop"
if str(TELEOP) not in sys.path:
    sys.path.insert(0, str(TELEOP))

from machine_paths import DEMO_ROOT, HOUSE_SOURCE_ROOT  # noqa: E402


SOURCE_ROOT = Path(HOUSE_SOURCE_ROOT).resolve()
RUNTIME_ROOT = Path(DEMO_ROOT).resolve()
SOURCE_BASELINE = SOURCE_ROOT / "scene_room_baseline.usda"
RUNTIME_BASELINE = RUNTIME_ROOT / "scene_room_baseline.usda"
WORKING_SCENE = RUNTIME_ROOT / "scene_room.usda"


def describe() -> dict[str, str]:
    return {
        "source_root": str(SOURCE_ROOT),
        "source_baseline": str(SOURCE_BASELINE),
        "runtime_root": str(RUNTIME_ROOT),
        "runtime_baseline": str(RUNTIME_BASELINE),
        "working_scene": str(WORKING_SCENE),
        "stage_root": "/World",
    }
