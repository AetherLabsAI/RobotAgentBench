#!/usr/bin/env python3
"""Session-local, fail-closed full collision profile for the G1 robot.

The shared G1 asset deliberately authors self collision off and disables a
number of base, torso, head and arm colliders.  Reproduce runs must not edit
that shared asset.  This helper enables only real authored collision geometry,
turns articulation self-collision on, and installs the bounded robot-internal
filter set required by the official geometry's closed gripper linkage and
frame-zero overlapping shells.  Robot-to-world filtering is forbidden.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any


SCHEMA = "real2sim.full_robot_collision_profile.v2"
ROBOT_PATH = "/World/Robot"
OFFICIAL_ROBOT_ASSET_ID = "g1_official_40_collider"
EXPECTED_COLLISION_GEOMETRY_COUNT = 40
REQUIRED_REGIONS = (
    "base_and_torso",
    "head",
    "left_arm",
    "right_arm",
    "left_gripper",
    "right_gripper",
)

# These are robot-internal topology exclusions, never robot-to-world
# exclusions.  The torso/shoulder entries are corroborated by the supplied G1
# SRDF where names match; the gripper entries are the non-adjacent contact
# pairs measured at frame zero in the official omnipicker closed linkage.  All
# remaining non-adjacent robot pairs continue to self-collide.
FILTERED_BODY_PAIRS = tuple(
    tuple(f"{ROBOT_PATH}/{name}" for name in pair)
    for pair in (
        ("base_link", "body_link2"),
        ("base_link", "head_link2"),
        ("base_link", "arm_l_base_link"),
        ("base_link", "arm_l_link1"),
        ("base_link", "arm_l_link2"),
        ("base_link", "arm_r_base_link"),
        ("base_link", "arm_r_link1"),
        ("base_link", "arm_r_link2"),
        ("body_link2", "arm_l_base_link"),
        ("body_link2", "arm_l_link1"),
        ("body_link2", "arm_l_link2"),
        ("body_link2", "arm_r_base_link"),
        ("body_link2", "arm_r_link1"),
        ("body_link2", "arm_r_link2"),
        ("body_link2", "head_link2"),
        ("gripper_l_inner_link1", "gripper_l_inner_link2"),
        ("gripper_l_inner_link2", "gripper_l_inner_link3"),
        ("gripper_l_outer_link1", "gripper_l_outer_link2"),
        ("gripper_l_outer_link2", "gripper_l_outer_link3"),
        ("gripper_r_inner_link1", "gripper_r_inner_link2"),
        ("gripper_r_inner_link2", "gripper_r_inner_link3"),
        ("gripper_r_outer_link1", "gripper_r_outer_link2"),
        ("gripper_r_outer_link2", "gripper_r_outer_link3"),
    )
)


def collision_region(path: str) -> str:
    """Classify one canonical G1 collider path for coverage evidence."""
    value = str(path)
    if "/gripper_l_" in value:
        return "left_gripper"
    if "/gripper_r_" in value:
        return "right_gripper"
    if "/arm_l_" in value:
        return "left_arm"
    if "/arm_r_" in value:
        return "right_arm"
    if "/head_" in value:
        return "head"
    if any(
        token in value
        for token in ("/base_link", "/body_link", "/arm_base_link")
    ):
        return "base_and_torso"
    return "unclassified"


def build_profile_evidence(
    paths: Iterable[str], *, self_collision_enabled: bool,
    filtered_body_pairs: Iterable[Iterable[str]] = FILTERED_BODY_PAIRS,
    robot_world_filtered_pairs: Iterable[Iterable[str]] = (),
) -> dict[str, Any]:
    """Build deterministic evidence without requiring Isaac/PXR imports."""
    enabled_paths = tuple(sorted({str(path) for path in paths}))
    normalized_filtered_pairs = tuple(sorted(
        tuple(sorted(str(path) for path in pair))
        for pair in filtered_body_pairs
    ))
    normalized_robot_world_pairs = tuple(sorted(
        tuple(sorted(str(path) for path in pair))
        for pair in robot_world_filtered_pairs
    ))
    region_counts = Counter(collision_region(path) for path in enabled_paths)
    missing_regions = tuple(
        region for region in REQUIRED_REGIONS if region_counts[region] < 1
    )
    identity = {
        "schema": SCHEMA,
        "robot_path": ROBOT_PATH,
        "official_robot_asset_id": OFFICIAL_ROBOT_ASSET_ID,
        "expected_collision_geometry_count": (
            EXPECTED_COLLISION_GEOMETRY_COUNT
        ),
        "self_collision_enabled": bool(self_collision_enabled),
        "collision_enabled_paths": list(enabled_paths),
        "filtered_body_pairs": [list(pair) for pair in normalized_filtered_pairs],
        "robot_world_filtered_pairs": [
            list(pair) for pair in normalized_robot_world_pairs
        ],
        "required_regions": list(REQUIRED_REGIONS),
        "region_counts": {
            region: int(region_counts[region]) for region in REQUIRED_REGIONS
        },
        "unclassified_count": int(region_counts["unclassified"]),
        "missing_regions": list(missing_regions),
    }
    return {
        **identity,
        "collision_enabled_path_count": len(enabled_paths),
        "filtered_body_pair_count": len(normalized_filtered_pairs),
        "robot_world_filtered_pair_count": len(normalized_robot_world_pairs),
        "complete": bool(
            self_collision_enabled
            and len(enabled_paths) == EXPECTED_COLLISION_GEOMETRY_COUNT
            and not missing_regions
            and normalized_filtered_pairs
            == tuple(sorted(tuple(sorted(pair)) for pair in FILTERED_BODY_PAIRS))
            and not normalized_robot_world_pairs
        ),
        "profile_id": SCHEMA,
    }


def _robot(stage):
    robot = stage.GetPrimAtPath(ROBOT_PATH)
    if not robot or not robot.IsValid() or not robot.IsActive():
        raise RuntimeError(f"full robot collision profile requires {ROBOT_PATH}")
    return robot


def _collision_candidates(stage):
    from pxr import Usd, UsdPhysics

    candidates = []
    for prim in Usd.PrimRange.AllPrims(_robot(stage)):
        if not prim.IsActive():
            continue
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            candidates.append(prim)
    if not candidates:
        raise RuntimeError("G1 robot has no authored collision candidates")
    return candidates


def _filtered_pairs(stage):
    """Return robot-internal and forbidden robot/world filtered pairs."""
    internal = set()
    robot_world = set()
    for prim in stage.TraverseAll():
        if not prim.IsActive():
            continue
        relationship = prim.GetRelationship("physics:filteredPairs")
        if not relationship:
            continue
        source = str(prim.GetPath())
        source_is_robot = (
            source == ROBOT_PATH or source.startswith(ROBOT_PATH + "/")
        )
        for target_path in relationship.GetTargets():
            target = str(target_path)
            target_is_robot = (
                target == ROBOT_PATH or target.startswith(ROBOT_PATH + "/")
            )
            pair = tuple(sorted((source, target)))
            if source_is_robot and target_is_robot:
                internal.add(pair)
            elif source_is_robot != target_is_robot:
                robot_world.add(pair)
    return tuple(sorted(internal)), tuple(sorted(robot_world))


def _self_collision_enabled(stage) -> bool:
    value = _robot(stage).GetAttribute(
        "physxArticulation:enabledSelfCollisions"
    ).Get()
    return value is True


def audit_full_robot_collision_profile(
    stage, *, expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Read back all robot collider and self-collision opinions."""
    from pxr import UsdPhysics

    enabled_paths = []
    disabled_paths = []
    for prim in _collision_candidates(stage):
        collision = UsdPhysics.CollisionAPI(prim)
        enabled = (
            prim.HasAPI(UsdPhysics.CollisionAPI)
            and collision.GetCollisionEnabledAttr().Get() is True
        )
        path = str(prim.GetPath())
        (enabled_paths if enabled else disabled_paths).append(path)
    filtered_body_pairs, robot_world_filtered_pairs = _filtered_pairs(stage)
    evidence = build_profile_evidence(
        enabled_paths,
        self_collision_enabled=_self_collision_enabled(stage),
        filtered_body_pairs=filtered_body_pairs,
        robot_world_filtered_pairs=robot_world_filtered_pairs,
    )
    if disabled_paths:
        raise RuntimeError(
            "full robot collision profile has disabled colliders: "
            + ", ".join(sorted(disabled_paths))
        )
    if not evidence["complete"]:
        raise RuntimeError(
            "full robot collision profile is incomplete: "
            f"self_collision={evidence['self_collision_enabled']}, "
            f"missing_regions={evidence['missing_regions']}, "
            f"filtered_body_pairs={evidence['filtered_body_pair_count']}, "
            "robot_world_filtered_pairs="
            f"{evidence['robot_world_filtered_pair_count']}"
        )
    if expected is not None:
        if evidence != dict(expected):
            raise RuntimeError(
                "full robot collision profile changed across physics reset"
            )
    return evidence


def apply_full_robot_collision_profile(stage) -> dict[str, Any]:
    """Enable real G1 geometry plus bounded self collision in session state."""
    from pxr import Sdf, Usd, UsdPhysics

    session_layer = stage.GetSessionLayer()
    if session_layer is None:
        raise RuntimeError("USD stage has no session layer for collision profile")
    candidates = _collision_candidates(stage)
    with Usd.EditContext(stage, session_layer):
        for first, second in FILTERED_BODY_PAIRS:
            first_prim = stage.GetPrimAtPath(first)
            second_prim = stage.GetPrimAtPath(second)
            if (
                not first_prim or not first_prim.IsValid()
                or not second_prim or not second_prim.IsValid()
            ):
                raise RuntimeError(
                    f"full robot collision filter pair is missing: {first}, {second}"
                )
            UsdPhysics.FilteredPairsAPI.Apply(
                first_prim
            ).CreateFilteredPairsRel().AddTarget(second_prim.GetPath())
        for prim in candidates:
            UsdPhysics.CollisionAPI(
                prim
            ).CreateCollisionEnabledAttr().Set(True)
        robot = _robot(stage)
        self_collision = robot.GetAttribute(
            "physxArticulation:enabledSelfCollisions"
        )
        if not self_collision:
            self_collision = robot.CreateAttribute(
                "physxArticulation:enabledSelfCollisions",
                Sdf.ValueTypeNames.Bool,
            )
        self_collision.Set(True)
    return audit_full_robot_collision_profile(stage)
