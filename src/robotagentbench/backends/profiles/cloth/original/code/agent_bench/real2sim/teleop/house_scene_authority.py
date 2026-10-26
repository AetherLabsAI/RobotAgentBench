#!/usr/bin/env python3
"""Fail-closed selector for the one current House scene root.

The baseline is a restore source, task-specific roots are migration inputs, and
candidate/review stages are audit artifacts.  Runtime and preview entrypoints
must select ``scene_room.usda`` under their configured demo root.
"""
from __future__ import annotations

import fnmatch
from pathlib import Path


CURRENT_SCENE_NAME = "scene_room.usda"
RESTORE_ONLY_SCENE_NAME = "scene_room_baseline.usda"
REPO_DEMO_RELATIVE = Path("arena_scene/v1_2/demo_sim")
SUPERSEDED_PATTERNS = (
    "scene_livingroom_decor_candidate*.usda",
    "scene_room_*candidate*.usda",
    "scene_room_task*.usda",
    "scene_room_[A-Z]*.usda",
    "*_superseded*.usda",
)


class SceneAuthorityError(ValueError):
    """Raised when code tries to use a non-current House root."""


def canonical_scene(demo_root: Path | str) -> Path:
    return Path(demo_root).resolve() / CURRENT_SCENE_NAME


def repository_source_baseline(repo_root: Path | str) -> Path:
    """Return the only checked-in House scene source."""
    return (
        Path(repo_root).resolve()
        / REPO_DEMO_RELATIVE
        / RESTORE_ONLY_SCENE_NAME
    )


def classify_scene(scene_path: Path | str, demo_root: Path | str) -> str:
    scene = Path(scene_path).resolve()
    current = canonical_scene(demo_root)
    if scene == current:
        return "current_working_scene"
    if scene == current.with_name(RESTORE_ONLY_SCENE_NAME):
        return "restore_source_only"
    if any(fnmatch.fnmatch(scene.name, pattern) for pattern in SUPERSEDED_PATTERNS):
        return "superseded_or_migration_only"
    return "unregistered_scene_root"


def require_current_scene(
    scene_path: Path | str,
    demo_root: Path | str,
    *,
    must_exist: bool = True,
) -> Path:
    scene = Path(scene_path).resolve()
    role = classify_scene(scene, demo_root)
    if role != "current_working_scene":
        raise SceneAuthorityError(
            f"House runtime accepts only {canonical_scene(demo_root)}; "
            f"{scene} is {role}"
        )
    if must_exist and not scene.is_file():
        raise FileNotFoundError(f"current House scene is missing: {scene}")
    return scene
