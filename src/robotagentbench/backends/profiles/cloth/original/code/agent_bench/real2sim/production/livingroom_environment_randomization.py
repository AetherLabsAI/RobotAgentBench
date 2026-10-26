#!/usr/bin/env python3
"""Coordinated appearance-only randomization for the fixed House living room.

The policy changes shader inputs plus render-only artwork/plant variant
selections.  The doors, wallpaper, rug, floor and plant stands remain at their
single canonical transforms in every task; no selected decor owns collision.
"""
from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

try:
    from real2sim.production import livingroom_plant_randomization as plant_random
except ModuleNotFoundError:  # Direct execution with production/ on sys.path.
    import livingroom_plant_randomization as plant_random


POLICY_VERSION = "real2sim.livingroom_environment_randomization.v6"
PLAN_SCHEMA = "real2sim.livingroom_environment_appearance_plan.v6"

ARTWORK_VARIANTS = (
    "modern_canvas",
    "wall_frame_16a", "wall_frame_16b", "wall_frame_16c",
    "wall_frame_16d", "wall_frame_16e", "wall_frame_16f",
    "wall_frame_16g", "wall_frame_16h", "wall_frame_16i",
    "wall_frame_16j",
)
PLANT_VARIANTS = plant_random.PLANT_VARIANTS
PLANT_SLOT_PATHS = plant_random.PLANT_SLOT_PATHS
TV_PLANT_STAND_SLOT_PATHS = plant_random.TV_PLANT_STAND_SLOT_PATHS
TV_LAMP_SLOT_PATHS = plant_random.TV_LAMP_SLOT_PATHS
LAMP_REPLACEMENT_MODES = {
    "left": ("TvSouthWestWindowLeft",),
    "right": ("TvNorth",),
    "both": ("TvSouthWestWindowLeft", "TvNorth"),
}
LAMP_VARIANTS = tuple(f"floor_lamp_{index:02d}" for index in range(1, 16))

FURNITURE_STYLE_TO_ENVIRONMENT = {
    "canonical": "warm_sage_oak",
    "warm_oak_linen": "terracotta_linen",
    "cool_gray_black": "blue_gray_walnut",
    "light_scandi": "scandi_light",
    "dark_walnut": "dramatic_olive",
}

# Git-owned base-color textures.  The default and warm suite retain the
# canonical floor; the other three palettes use image-generated wood variants
# with visible grain instead of flattening the canonical texture with a strong
# RGB multiplier.
FLOOR_TEXTURE_ASSETS = {
    "warm_sage_oak": "usdz/floor_wood.png",
    "terracotta_linen": "usdz/floor_wood.png",
    "blue_gray_walnut": "usdz/floor_wood_cool_gray_black_v1.png",
    "scandi_light": "usdz/floor_wood_light_scandi_v1.png",
    "dramatic_olive": "usdz/floor_wood_dark_walnut_v1.png",
}

# Deliberately separated suites: all values within one row coordinate, while
# different rows give the four-variant renderer an immediately visible change.
STYLE_SUITES: dict[str, dict[str, Any]] = {
    "warm_sage_oak": {
        "wall_rgb": (0.84, 0.82, 0.74),
        "wallpaper_scale_rgb": (0.74, 0.86, 0.70),
        "floor_scale_rgb": (1.08, 0.91, 0.67),
        "rug_scale_rgb": (0.87, 0.82, 0.70),
        "rug_border_rgb": (0.085, 0.26, 0.24),
        "rug_accent_rgb": (0.68, 0.39, 0.13),
        "house_entry_door_scale_rgb": (0.78, 0.91, 0.77),
        "room_door_scale_rgb": (1.04, 1.01, 0.92),
        "pantry_texture_scale_rgb": (1.02, 0.93, 0.78),
        "pantry_solid_rgb": (0.45, 0.27, 0.13),
        "decor_stand_texture_scale_rgb": (0.92, 0.80, 0.63),
        "decor_stand_solid_rgb": (0.38, 0.245, 0.13),
    },
    "terracotta_linen": {
        "wall_rgb": (0.83, 0.72, 0.62),
        "wallpaper_scale_rgb": (0.96, 0.72, 0.58),
        "floor_scale_rgb": (0.92, 0.70, 0.49),
        "rug_scale_rgb": (0.76, 0.49, 0.36),
        "rug_border_rgb": (0.16, 0.19, 0.25),
        "rug_accent_rgb": (0.88, 0.76, 0.52),
        "house_entry_door_scale_rgb": (0.92, 0.66, 0.51),
        "room_door_scale_rgb": (0.96, 0.87, 0.75),
        "pantry_texture_scale_rgb": (1.03, 0.79, 0.59),
        "pantry_solid_rgb": (0.47, 0.23, 0.12),
        "decor_stand_texture_scale_rgb": (1.02, 0.73, 0.52),
        "decor_stand_solid_rgb": (0.46, 0.22, 0.12),
    },
    "blue_gray_walnut": {
        "wall_rgb": (0.58, 0.67, 0.73),
        "wallpaper_scale_rgb": (0.58, 0.72, 0.82),
        "floor_scale_rgb": (1.0, 1.0, 1.0),
        "rug_scale_rgb": (0.34, 0.45, 0.54),
        "rug_border_rgb": (0.035, 0.045, 0.060),
        "rug_accent_rgb": (0.76, 0.40, 0.19),
        "house_entry_door_scale_rgb": (0.36, 0.49, 0.61),
        "room_door_scale_rgb": (0.86, 0.89, 0.91),
        "pantry_texture_scale_rgb": (0.49, 0.33, 0.24),
        "pantry_solid_rgb": (0.20, 0.105, 0.055),
        "decor_stand_texture_scale_rgb": (0.51, 0.35, 0.26),
        "decor_stand_solid_rgb": (0.19, 0.105, 0.060),
    },
    "scandi_light": {
        "wall_rgb": (0.92, 0.90, 0.84),
        "wallpaper_scale_rgb": (0.94, 0.94, 0.89),
        "floor_scale_rgb": (1.0, 1.0, 1.0),
        "rug_scale_rgb": (0.95, 0.92, 0.84),
        "rug_border_rgb": (0.36, 0.39, 0.40),
        "rug_accent_rgb": (0.78, 0.58, 0.18),
        "house_entry_door_scale_rgb": (1.08, 1.07, 1.01),
        "room_door_scale_rgb": (1.08, 1.07, 1.01),
        "pantry_texture_scale_rgb": (1.16, 1.06, 0.87),
        "pantry_solid_rgb": (0.61, 0.43, 0.25),
        "decor_stand_texture_scale_rgb": (1.15, 1.04, 0.84),
        "decor_stand_solid_rgb": (0.61, 0.45, 0.28),
    },
    "dramatic_olive": {
        "wall_rgb": (0.45, 0.51, 0.39),
        "wallpaper_scale_rgb": (0.48, 0.57, 0.39),
        "floor_scale_rgb": (1.0, 1.0, 1.0),
        "rug_scale_rgb": (0.18, 0.25, 0.29),
        "rug_border_rgb": (0.72, 0.51, 0.16),
        "rug_accent_rgb": (0.88, 0.82, 0.67),
        "house_entry_door_scale_rgb": (0.31, 0.37, 0.28),
        "room_door_scale_rgb": (0.79, 0.76, 0.65),
        "pantry_texture_scale_rgb": (0.42, 0.30, 0.20),
        "pantry_solid_rgb": (0.17, 0.105, 0.060),
        "decor_stand_texture_scale_rgb": (0.43, 0.34, 0.22),
        "decor_stand_solid_rgb": (0.16, 0.105, 0.065),
    },
}


def _rng(task_id: str, gen_seed: int) -> random.Random:
    return random.Random(f"{POLICY_VERSION}\0house-environment\0{gen_seed}")


def build_plan(
    *,
    task_id: str,
    gen_seed: int,
    furniture_style: str | None = None,
    environment_style: str | None = None,
    decor_constraints: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    rng = _rng(str(task_id), int(gen_seed))
    if environment_style is not None:
        style_name = environment_style
    elif furniture_style in FURNITURE_STYLE_TO_ENVIRONMENT:
        style_name = FURNITURE_STYLE_TO_ENVIRONMENT[str(furniture_style)]
    else:
        names = sorted(STYLE_SUITES)
        style_name = names[rng.randrange(len(names))]
    if style_name not in STYLE_SUITES:
        raise ValueError(f"unknown living-room environment style {style_name!r}")
    style = STYLE_SUITES[style_name]
    constraints = dict(decor_constraints or {})

    def constrained_pool(
        key: str, registered: Sequence[str], *, minimum: int = 1,
    ) -> tuple[str, ...]:
        requested = tuple(map(str, constraints.get(key, registered)))
        if len(requested) < minimum or len(requested) != len(set(requested)):
            raise ValueError(
                f"decor constraint {key!r} must contain at least {minimum} "
                "unique entries"
            )
        unknown = sorted(set(requested) - set(registered))
        if unknown:
            raise ValueError(f"decor constraint {key!r} has unknown assets: {unknown}")
        return requested

    artworks = constrained_pool("artworks", ARTWORK_VARIANTS)
    plants = constrained_pool(
        "plants", PLANT_VARIANTS, minimum=len(PLANT_SLOT_PATHS))
    lamps = constrained_pool("lamps", LAMP_VARIANTS)
    plant_plan = plant_random.build_plan(task_id=str(task_id), gen_seed=gen_seed)
    plant_plan = dict(plant_plan)
    plant_plan["selections"] = dict(zip(
        PLANT_SLOT_PATHS,
        rng.sample(list(plants), len(PLANT_SLOT_PATHS)),
        strict=True,
    ))
    lamp_mode = tuple(LAMP_REPLACEMENT_MODES)[
        rng.randrange(len(LAMP_REPLACEMENT_MODES))
    ]
    lamp_variant = lamps[rng.randrange(len(lamps))]
    decor_selection = {
        "artwork": artworks[rng.randrange(len(artworks))],
        "plants": dict(plant_plan["selections"]),
        "lamp_replacement": {
            "mode": lamp_mode,
            "slots": list(LAMP_REPLACEMENT_MODES[lamp_mode]),
            "lamp": lamp_variant,
        },
    }
    return {
        "schema": PLAN_SCHEMA,
        "policy_version": POLICY_VERSION,
        "task_id": str(task_id),
        "gen_seed": int(gen_seed),
        "style": style_name,
        "floor_texture_asset": FLOOR_TEXTURE_ASSETS[style_name],
        "style_values": {
            key: [float(value) for value in values]
            for key, values in style.items()
        },
        "decor_constraints": {
            "enforced": bool(decor_constraints),
            "artworks": list(artworks),
            "plants": list(plants),
            "lamps": list(lamps),
        },
        "decor_selection": decor_selection,
        "targets": {
            "house_entry_door": "/World/Assets/HouseEntryDoor",
            "original_room_door": "/World/Assets/Door",
            "pantry_door": "/World/Assets/PantrySlidingDoor",
            "wallpaper": "/World/Assets/LivingRoomWallpaper",
            "rug": "/World/Assets/LivingRoomRug",
            "floor": "/World/Background/Shell/Floor",
            "fireplace_artwork": "/World/Assets/FireplaceArtwork",
            "plant_stands": "/World/Assets/LivingRoomPlantStands",
            "lamp_replacements": "/World/Assets/LivingRoomLampReplacements",
        },
        "invariants": {
            "appearance_only": True,
            "furniture_transforms_authored": False,
            "camera_authored": False,
            "collision_authored": False,
            "task_geometry_authored": False,
        },
    }


def _set_float3(stage, path: str, values: Sequence[float]) -> str:
    from pxr import Gf, Sdf

    prim_path, attribute_name = path.rsplit(".", 1)
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise ValueError(f"environment material prim is missing: {prim_path}")
    attribute = prim.GetAttribute(attribute_name)
    if not attribute.IsValid():
        attribute = prim.CreateAttribute(attribute_name, Sdf.ValueTypeNames.Color3f)
    attribute.Set(Gf.Vec3f(*values))
    return path


def _set_float4(stage, path: str, values: Sequence[float]) -> str:
    from pxr import Gf, Sdf

    prim_path, attribute_name = path.rsplit(".", 1)
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise ValueError(f"environment texture prim is missing: {prim_path}")
    attribute = prim.GetAttribute(attribute_name)
    if not attribute.IsValid():
        attribute = prim.CreateAttribute(attribute_name, Sdf.ValueTypeNames.Float4)
    attribute.Set(Gf.Vec4f(*values, 1.0))
    return path


def _set_asset(stage, path: str, value: str) -> str:
    from pxr import Sdf

    prim_path, attribute_name = path.rsplit(".", 1)
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise ValueError(f"environment texture prim is missing: {prim_path}")
    attribute = prim.GetAttribute(attribute_name)
    if not attribute.IsValid():
        attribute = prim.CreateAttribute(attribute_name, Sdf.ValueTypeNames.Asset)
    attribute.Set(Sdf.AssetPath("./" + value.lstrip("./")))
    return path


def _set_variant_selection(
    stage, path: str, variant_set_name: str, selection: str
) -> str:
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        raise ValueError(f"render-only decor prim is missing: {path}")
    variants = prim.GetVariantSets().GetVariantSet(variant_set_name)
    if not variants.IsValid() or selection not in variants.GetVariantNames():
        raise ValueError(
            f"invalid {variant_set_name} selection {selection!r} at {path}"
        )
    render_only = prim.GetAttribute("real2sim:renderOnlyDecoration")
    if not render_only.IsValid() or render_only.Get() is not True:
        raise ValueError(f"decor variant is not declared render-only: {path}")
    variants.SetVariantSelection(selection)
    return f"{path}.variantSelections:{variant_set_name}={selection}"


def _set_visibility(stage, path: str, value: str) -> str:
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        raise ValueError(f"render-only replacement prim is missing: {path}")
    render_only = prim.GetAttribute("real2sim:renderOnlyDecoration")
    if not render_only.IsValid() or render_only.Get() is not True:
        raise ValueError(f"replacement target is not render-only: {path}")
    UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(value)
    return f"{path}.visibility={value}"


def apply_plan_to_stage(stage, plan: Mapping[str, Any]) -> dict[str, Any]:
    """Apply a validated plan to an already-open canonical session stage."""
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("living-room environment plan has the wrong schema")
    values = plan["style_values"]
    floor_texture_asset = str(plan.get("floor_texture_asset", ""))
    if floor_texture_asset != FLOOR_TEXTURE_ASSETS.get(str(plan.get("style"))):
        raise ValueError("living-room floor texture does not match its style")
    authored = [
        _set_float3(
            stage,
            "/World/Background/Shell/WallMat/S.inputs:diffuseColor",
            values["wall_rgb"],
        ),
        _set_asset(
            stage,
            "/World/Background/Shell/FloorMat/T.inputs:file",
            floor_texture_asset,
        ),
        _set_float4(
            stage,
            "/World/Background/Shell/FloorMat/T.inputs:scale",
            values["floor_scale_rgb"],
        ),
        _set_float4(
            stage,
            "/World/LivingRoomDecorLooks/Wallpaper/Texture.inputs:scale",
            values["wallpaper_scale_rgb"],
        ),
        _set_float4(
            stage,
            "/World/LivingRoomDecorLooks/RugSurface/Texture.inputs:scale",
            values["rug_scale_rgb"],
        ),
        _set_float3(
            stage,
            "/World/LivingRoomDecorLooks/Charcoal/Surface.inputs:diffuseColor",
            values["rug_border_rgb"],
        ),
        _set_float3(
            stage,
            "/World/LivingRoomDecorLooks/MutedTeal/Surface.inputs:diffuseColor",
            values["rug_accent_rgb"],
        ),
        _set_float4(
            stage,
            "/World/Assets/HouseEntryDoor/Materials/Door/tex_base.inputs:scale",
            values["house_entry_door_scale_rgb"],
        ),
        _set_float4(
            stage,
            "/World/Assets/Door/Obj/Materials/Door/tex_base.inputs:scale",
            values["room_door_scale_rgb"],
        ),
        _set_float4(
            stage,
            "/World/LivingRoomDecorLooks/PantryWood/Texture.inputs:scale",
            values["pantry_texture_scale_rgb"],
        ),
        _set_float3(
            stage,
            "/World/LivingRoomDecorLooks/PantryWoodSolid/Surface.inputs:diffuseColor",
            values["pantry_solid_rgb"],
        ),
        _set_float4(
            stage,
            "/World/LivingRoomDecorLooks/DecorStandWood/Texture.inputs:scale",
            values["decor_stand_texture_scale_rgb"],
        ),
        _set_float3(
            stage,
            "/World/LivingRoomDecorLooks/DecorStandSolid/Surface.inputs:diffuseColor",
            values["decor_stand_solid_rgb"],
        ),
    ]
    decor_selection = plan["decor_selection"]
    variant_authored = [
        _set_variant_selection(
            stage,
            "/World/Assets/FireplaceArtwork",
            "artworkAsset",
            decor_selection["artwork"],
        )
    ]
    for slot_name, path in PLANT_SLOT_PATHS.items():
        variant_authored.append(
            _set_variant_selection(
                stage, path, "plantAsset", decor_selection["plants"][slot_name]
            )
        )
    replacement = decor_selection["lamp_replacement"]
    replacement_mode = str(replacement["mode"])
    replacement_slots = tuple(map(str, replacement["slots"]))
    replacement_lamp = str(replacement["lamp"])
    if replacement_mode not in LAMP_REPLACEMENT_MODES:
        raise ValueError(f"unknown TV-side lamp mode {replacement_mode!r}")
    expected_slots = LAMP_REPLACEMENT_MODES[replacement_mode]
    if replacement_slots != expected_slots:
        raise ValueError(
            f"lamp mode {replacement_mode!r} requires slots "
            f"{expected_slots}, got {replacement_slots}"
        )
    if replacement_lamp not in LAMP_VARIANTS:
        raise ValueError(f"unknown floor lamp variant {replacement_lamp!r}")
    visibility_authored = []
    for slot_name in TV_LAMP_SLOT_PATHS:
        selected = slot_name in replacement_slots
        visibility_authored.append(
            _set_visibility(
                stage,
                TV_PLANT_STAND_SLOT_PATHS[slot_name],
                "invisible" if selected else "inherited",
            )
        )
        variant_authored.append(
            _set_variant_selection(
                stage,
                TV_LAMP_SLOT_PATHS[slot_name],
                "lampAsset",
                replacement_lamp if selected else "none",
            )
        )
    return {
        "schema": "real2sim.livingroom_environment_appearance_applied.v6",
        "style": plan["style"],
        "floor_texture_asset": floor_texture_asset,
        "policy_version": plan.get("policy_version", POLICY_VERSION),
        "authored_attribute_paths": authored,
        "authored_variant_selections": variant_authored,
        "authored_visibility_selections": visibility_authored,
        "decor_selection": decor_selection,
        "authored_attribute_kinds": [
            "shader_input", "render_only_variant_selection"
        ],
        "furniture_transforms_unchanged": True,
        "physics_effect": "none",
    }
