#!/usr/bin/env python3
"""Deterministic, lamp-aware randomization for fixed living-room plants."""
from __future__ import annotations

import argparse
import json
import random
from collections.abc import Mapping
from typing import Any


POLICY_VERSION = "real2sim.livingroom_plant_randomization.v1"
PLAN_SCHEMA = "real2sim.livingroom_plant_appearance_plan.v1"

PLANT_VARIANTS = (
    "costa_white_ficus",
    "volume_small_rubber",
    "volume_tall_audrey",
    "dark_rhodea",
    "eda_boston_fern",
    "sand_strelitzia",
    "sand_cactus",
    "bishop_cereus",
    "costa_dark_areca",
    "dark_majesty_palm",
    "dark_base_livistona",
    "metal_dracaena",
    "glazed_schefflera",
    "bright_calathea",
    "bamboo_monstera",
    "hanging_pothos",
    "cross_alocasia",
    "dark_cross_palm",
    "cecilia_spider",
    "bishop_dracaena",
)
PLANT_SLOT_PATHS = {
    "TvSouthWestWindowLeft": (
        "/World/Assets/LivingRoomPlantStands/"
        "TvSouthWestWindowLeft/Plant"
    ),
    "TvNorth": "/World/Assets/LivingRoomPlantStands/TvNorth/Plant",
    "WestWindowRight": (
        "/World/Assets/LivingRoomPlantStands/WestWindowRight/Plant"
    ),
    "EastWindowLeft": (
        "/World/Assets/LivingRoomPlantStands/EastWindowLeft/Plant"
    ),
    "EastWindowRight": (
        "/World/Assets/LivingRoomPlantStands/EastWindowRight/Plant"
    ),
}
TV_PLANT_STAND_SLOT_PATHS = {
    slot: f"/World/Assets/LivingRoomPlantStands/{slot}"
    for slot in ("TvSouthWestWindowLeft", "TvNorth")
}
TV_LAMP_SLOT_PATHS = {
    slot: f"/World/Assets/LivingRoomLampReplacements/{slot}"
    for slot in TV_PLANT_STAND_SLOT_PATHS
}


def build_plan(*, task_id: str, gen_seed: int) -> dict[str, Any]:
    rng = random.Random(f"{POLICY_VERSION}\0{task_id}\0{gen_seed}")
    choices = rng.sample(list(PLANT_VARIANTS), len(PLANT_SLOT_PATHS))
    return {
        "schema": PLAN_SCHEMA,
        "policy_version": POLICY_VERSION,
        "task_id": str(task_id),
        "gen_seed": int(gen_seed),
        "selections": dict(zip(PLANT_SLOT_PATHS, choices, strict=True)),
        "invariants": {
            "appearance_only": True,
            "unique_within_episode": True,
            "plant_transforms_authored": False,
            "lamp_variant_selections_authored": False,
            "lamp_light_attributes_authored": False,
            "collision_authored": False,
        },
    }


def _set_plant_variant(stage, path: str, selection: str) -> str:
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        raise ValueError(f"plant prim is missing: {path}")
    render_only = prim.GetAttribute("real2sim:renderOnlyDecoration")
    if not render_only.IsValid() or render_only.Get() is not True:
        raise ValueError(f"plant is not declared render-only: {path}")
    variants = prim.GetVariantSet("plantAsset")
    if not variants.IsValid() or selection not in variants.GetVariantNames():
        raise ValueError(f"invalid plant selection {selection!r} at {path}")
    variants.SetVariantSelection(selection)
    return f"{path}.variantSelections:plantAsset={selection}"


def _lamp_selection(stage, path: str) -> str:
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        raise ValueError(f"lamp replacement prim is missing: {path}")
    variants = prim.GetVariantSet("lampAsset")
    if not variants.IsValid():
        raise ValueError(f"lampAsset variant set is missing: {path}")
    return variants.GetVariantSelection() or "none"


def apply_plan_to_stage(stage, plan: Mapping[str, Any]) -> dict[str, Any]:
    """Apply plants while treating all lamp selections and lights as read-only."""
    from pxr import UsdGeom

    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("living-room plant plan has the wrong schema")
    selections = plan.get("selections")
    if not isinstance(selections, Mapping) or set(selections) != set(
        PLANT_SLOT_PATHS
    ):
        raise ValueError("living-room plant plan does not cover all slots")
    if len(set(map(str, selections.values()))) != len(PLANT_SLOT_PATHS):
        raise ValueError("living-room plant selections must be unique")

    authored_variants = [
        _set_plant_variant(stage, path, str(selections[slot]))
        for slot, path in PLANT_SLOT_PATHS.items()
    ]
    visibility_authored = []
    observed_lamps = {}
    for slot, lamp_path in TV_LAMP_SLOT_PATHS.items():
        selection = _lamp_selection(stage, lamp_path)
        observed_lamps[slot] = selection
        stand_path = TV_PLANT_STAND_SLOT_PATHS[slot]
        stand = stage.GetPrimAtPath(stand_path)
        if not stand or not stand.IsValid():
            raise ValueError(f"plant stand prim is missing: {stand_path}")
        visibility = "invisible" if selection != "none" else "inherited"
        UsdGeom.Imageable(stand).CreateVisibilityAttr().Set(visibility)
        visibility_authored.append(f"{stand_path}.visibility={visibility}")

    return {
        "schema": "real2sim.livingroom_plant_appearance_applied.v1",
        "policy_version": plan.get("policy_version", POLICY_VERSION),
        "authored_plant_variants": authored_variants,
        "authored_plant_visibility": visibility_authored,
        "observed_lamp_variants_read_only": observed_lamps,
        "lamp_attributes_authored": [],
        "physics_effect": "none",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print a deterministic living-room plant appearance plan."
    )
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--gen-seed", required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(
        build_plan(task_id=args.task_id, gen_seed=args.gen_seed),
        indent=2,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
