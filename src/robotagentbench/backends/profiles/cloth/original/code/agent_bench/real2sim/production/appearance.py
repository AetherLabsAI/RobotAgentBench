#!/usr/bin/env python3
"""Opt-in coordinated whole-House appearance randomization v3.

One plan now owns the living-room furniture/materials, room lighting, and the
current l2 kitchen material choices.  The application path requires the
mesh311 USD interpreter and refuses to touch a baseline or a working scene
without an open-session sidecar.  It authors shader/light attributes only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TELEOP = REPO / "real2sim" / "teleop"
if str(TELEOP) not in sys.path:
    sys.path.insert(0, str(TELEOP))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import contract as production_contract  # noqa: E402
import livingroom_environment_randomization as environment_randomization  # noqa: E402
import house_scene_authority  # noqa: E402
from git_provenance import source_revision  # noqa: E402


SCHEMA = "real2sim.livingroom_appearance_plan.v3"
ENABLE_ENV = "REAL2SIM_APPEARANCE_PROFILE_V2"
REGISTRY_PATH = HERE / "whole_house_appearance_registry_v1.json"
HARMONY_SCHEMA = "real2sim.whole_house_appearance_harmony.v1"
REQUIRED_RANDOMIZERS = frozenset({
    "kitchen",
    "central_table",
    "livingroom_lamps",
    "livingroom_plants",
})


def load_randomization_registry(
    path: Path | str = REGISTRY_PATH,
) -> dict[str, Any]:
    registry = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        registry.get("schema")
        != "real2sim.whole_house_appearance_registry.v1"
        or registry.get("status") != "current"
        or registry.get("activation_env") != f"{ENABLE_ENV}=1"
        or registry.get("entrypoint") != "real2sim/production/appearance.py"
    ):
        raise ValueError("whole-House appearance registry is not current v1")
    components = registry.get("components")
    if not isinstance(components, Mapping) or set(components) != REQUIRED_RANDOMIZERS:
        raise ValueError(
            "whole-House appearance registry must contain exactly "
            + ", ".join(sorted(REQUIRED_RANDOMIZERS))
        )
    for component_id, component in components.items():
        owner = REPO / str(component.get("owner", ""))
        if not owner.is_file():
            raise FileNotFoundError(
                f"registered randomizer owner is missing for {component_id}: {owner}"
            )
        policy = component.get("policy")
        if policy and not (REPO / str(policy)).is_file():
            raise FileNotFoundError(
                f"registered randomizer policy is missing for {component_id}: {policy}"
            )
    harmony = registry.get("harmony_policy")
    if (
        not isinstance(harmony, Mapping)
        or harmony.get("schema") != HARMONY_SCHEMA
        or harmony.get("status") != "enforced"
        or harmony.get("failure_mode") != "fail_closed_before_scene_authoring"
        or not isinstance(harmony.get("palettes"), Mapping)
        or not harmony["palettes"]
    ):
        raise ValueError("whole-House harmony policy is missing or not enforced")
    required = {
        "environment_style", "lighting_schemes", "central_table_options",
        "stove_rules", "artworks", "lamps", "plants",
    }
    for palette_id, palette in harmony["palettes"].items():
        if not isinstance(palette, Mapping) or set(palette) != required:
            raise ValueError(
                f"harmony palette {palette_id!r} must contain exactly "
                + ", ".join(sorted(required)))
        for key in required - {"environment_style"}:
            values = palette[key]
            minimum = 5 if key == "plants" else 1
            if (
                not isinstance(values, list)
                or len(values) < minimum
                or len(values) != len(set(map(str, values)))
            ):
                raise ValueError(
                    f"harmony palette {palette_id!r} has an invalid {key} pool")
    return registry


def delivery_palettes(
    registry: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Return the four ordered palettes used once per trajectory core."""

    effective = dict(registry or load_randomization_registry())
    harmony = effective["harmony_policy"]
    values = harmony.get("delivery_palette_ids")
    expected_policy = "one_fixed_palette_per_variant_each_core"
    if (
        harmony.get("delivery_variant_palette_policy") != expected_policy
        or not isinstance(values, list)
        or len(values) != 4
        or len(set(map(str, values))) != 4
        or any(value not in harmony["palettes"] for value in values)
    ):
        raise ValueError("four-palette delivery policy is not enforced")
    return tuple(map(str, values))


def _rng(task_id: str, gen_seed: int) -> random.Random:
    return random.Random(f"{SCHEMA}:house-environment:{gen_seed}")


def _sample(bounds: Sequence[float], rng: random.Random) -> float:
    low, high = (float(value) for value in bounds)
    if not (math.isfinite(low) and math.isfinite(high) and 0 < low <= high):
        raise ValueError(f"invalid positive range: {bounds}")
    return rng.uniform(low, high)


def build_plan(
    *,
    task_id: str,
    gen_seed: int,
    style: str | None = None,
    lighting: str | None = None,
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    registry = load_randomization_registry()
    effective = dict(contract or production_contract.load_contract())
    production_contract.validate_identity(
        str(task_id), "" if str(task_id).startswith("family7:")
        else "00000000-0000-4000-8000-000000000000", int(gen_seed)
    )
    appearance = effective["appearance"]
    styles = appearance["coordinated_styles"]
    schemes = appearance["lighting_schemes"]
    harmony = registry["harmony_policy"]
    palettes = harmony["palettes"]
    if set(palettes) != set(styles):
        raise ValueError(
            "harmony palettes must cover every coordinated furniture style")
    import kitchen_material_randomization

    registered_table_options = {
        str(option["id"])
        for option in kitchen_material_randomization.load_table_registry()["options"]
    }
    registered_stove_rules = {
        str(rule["id"])
        for rule in kitchen_material_randomization.load_stove_registry()["rules"]
    }
    for palette_id, registered_palette in palettes.items():
        expected_environment = (
            environment_randomization.FURNITURE_STYLE_TO_ENVIRONMENT.get(
                str(palette_id)))
        if registered_palette["environment_style"] != expected_environment:
            raise ValueError(
                f"harmony palette {palette_id!r} has the wrong environment style")
        domains = {
            "lighting_schemes": set(schemes),
            "central_table_options": registered_table_options,
            "stove_rules": registered_stove_rules,
            "artworks": set(environment_randomization.ARTWORK_VARIANTS),
            "lamps": set(environment_randomization.LAMP_VARIANTS),
            "plants": set(environment_randomization.PLANT_VARIANTS),
        }
        for component, registered_values in domains.items():
            unknown = (
                set(map(str, registered_palette[component]))
                - registered_values
            )
            if unknown:
                raise ValueError(
                    f"harmony palette {palette_id!r} has unknown {component}: "
                    + repr(sorted(unknown)))
    rng = _rng(str(task_id), int(gen_seed))
    style_name = style or sorted(styles)[rng.randrange(len(styles))]
    if style_name not in styles:
        raise ValueError(f"unknown coordinated style {style_name!r}")
    palette = palettes[style_name]
    allowed_lighting = tuple(map(str, palette["lighting_schemes"]))
    lighting_name = lighting or allowed_lighting[rng.randrange(len(allowed_lighting))]
    if lighting_name not in schemes:
        raise ValueError(f"unknown lighting scheme {lighting_name!r}")
    if lighting_name not in allowed_lighting:
        raise ValueError(
            f"lighting scheme {lighting_name!r} is incompatible with "
            f"harmony palette {style_name!r}")

    legacy_policy = json.loads(
        (REPO / appearance["legacy_policy"]).read_text(encoding="utf-8")
    )
    tint_paths = {
        entry["id"]: entry["prim_path"]
        for entry in legacy_policy["tintable_prims"]
    }
    materials = []
    for furniture_id, look_name in sorted(styles[style_name].items()):
        if furniture_id not in tint_paths:
            raise ValueError(
                f"appearance style references non-tintable {furniture_id!r}"
            )
        look = appearance["material_looks"][look_name]
        materials.append({
            "furniture_id": furniture_id,
            "prim_path": tint_paths[furniture_id],
            "look": look_name,
            "rgb_scale": [float(value) for value in look["rgb_scale"]],
            "roughness": look["roughness"],
            "metallic": look["metallic"],
        })

    scheme = schemes[lighting_name]
    lights = []
    for group_name, group in sorted(legacy_policy["lights"]["groups"].items()):
        scale_key = f"{group_name}_scale"
        temperature_key = f"{group_name}_temperature_k"
        if scale_key not in scheme or temperature_key not in scheme:
            raise ValueError(
                f"lighting scheme {lighting_name} lacks {group_name} controls"
            )
        group_scale = _sample(scheme[scale_key], rng)
        temperature = _sample(scheme[temperature_key], rng)
        jitter = group["per_light_jitter_range"]
        for path in group["paths"]:
            lights.append({
                "prim_path": path,
                "group": group_name,
                "intensity_scale": round(group_scale * _sample(jitter, rng), 5),
                "color_temperature_k": round(temperature, 1),
            })

    environment = environment_randomization.build_plan(
        task_id=str(task_id),
        gen_seed=int(gen_seed),
        furniture_style=style_name,
        decor_constraints={
            "artworks": palette["artworks"],
            "lamps": palette["lamps"],
            "plants": palette["plants"],
        },
    )
    kitchen_materials = kitchen_material_randomization.build_randomization_plan(
        task_id=str(task_id),
        seed_override=int(gen_seed),
        allowed_table_option_ids=palette["central_table_options"],
        allowed_stove_rule_ids=palette["stove_rules"],
    )

    plan = {
        "schema": SCHEMA,
        "contract_source_revision": source_revision((
            production_contract.CONTRACT_PATH,)),
        "task_id": str(task_id),
        "gen_seed": int(gen_seed),
        "style": style_name,
        "lighting_scheme": lighting_name,
        "materials": materials,
        "lights": lights,
        "environment": environment,
        "kitchen_materials": kitchen_materials,
        "harmony": {
            "schema": HARMONY_SCHEMA,
            "policy": HARMONY_SCHEMA,
            "status": "passed",
            "palette_id": style_name,
            "selection_mode": harmony["selection"],
            "failure_mode": harmony["failure_mode"],
            "selected": {
                "furniture_style": style_name,
                "environment_style": environment["style"],
                "lighting_scheme": lighting_name,
                "central_table_option": kitchen_materials[
                    "central_table"]["option_id"],
                "stove_rule": kitchen_materials["stove"]["rule_id"],
                "artwork": environment["decor_selection"]["artwork"],
                "lamp": environment["decor_selection"]
                    ["lamp_replacement"]["lamp"],
                "plants": sorted(
                    environment["decor_selection"]["plants"].values()),
            },
        },
        "randomization_registry": {
            "schema": registry["schema"],
            "source_revision": source_revision((REGISTRY_PATH, Path(__file__))),
            "activation_env": registry["activation_env"],
            "entrypoint": registry["entrypoint"],
            "application_groups": registry["application_groups"],
            "components": registry["components"],
        },
        "excluded_semantic_prims": sorted(
            legacy_policy["excluded_semantic_prims"]),
        "physics_effect": "none",
    }
    validate_harmony(plan, registry=registry)
    return plan


def validate_harmony(
    plan: Mapping[str, Any], *,
    registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed unless every selected whole-House component fits one palette."""
    effective = dict(registry or load_randomization_registry())
    policy = effective["harmony_policy"]
    style = str(plan.get("style", ""))
    palette = policy["palettes"].get(style)
    if not isinstance(palette, Mapping):
        raise ValueError(f"appearance plan has no harmony palette for {style!r}")
    environment = plan.get("environment")
    kitchen = plan.get("kitchen_materials")
    evidence = plan.get("harmony")
    if not all(isinstance(value, Mapping) for value in (
        environment, kitchen, evidence,
    )):
        raise ValueError("appearance plan lacks harmony inputs or evidence")
    decor = environment.get("decor_selection")
    if not isinstance(decor, Mapping):
        raise ValueError("appearance plan lacks coordinated decor selection")
    replacement = decor.get("lamp_replacement")
    plants = decor.get("plants")
    if not isinstance(replacement, Mapping) or not isinstance(plants, Mapping):
        raise ValueError("appearance plan lacks lamp or plant selections")
    selected = {
        "environment_style": environment.get("style"),
        "lighting_scheme": plan.get("lighting_scheme"),
        "central_table_option": kitchen.get("central_table", {}).get("option_id"),
        "stove_rule": kitchen.get("stove", {}).get("rule_id"),
        "artwork": decor.get("artwork"),
        "lamp": replacement.get("lamp"),
    }
    requirements = {
        "environment_style": (selected["environment_style"],
                              [palette["environment_style"]]),
        "lighting_scheme": (selected["lighting_scheme"],
                            palette["lighting_schemes"]),
        "central_table_option": (selected["central_table_option"],
                                 palette["central_table_options"]),
        "stove_rule": (selected["stove_rule"], palette["stove_rules"]),
        "artwork": (selected["artwork"], palette["artworks"]),
        "lamp": (selected["lamp"], palette["lamps"]),
    }
    for component, (value, allowed) in requirements.items():
        if value not in allowed:
            raise ValueError(
                f"harmony gate rejected {component}={value!r} for {style!r}")
    selected_plants = list(map(str, plants.values()))
    if (
        len(selected_plants) != len(environment_randomization.PLANT_SLOT_PATHS)
        or len(selected_plants) != len(set(selected_plants))
        or not set(selected_plants).issubset(set(map(str, palette["plants"])))
    ):
        raise ValueError(f"harmony gate rejected plant selections for {style!r}")
    if (
        evidence.get("schema") != HARMONY_SCHEMA
        or evidence.get("policy") != HARMONY_SCHEMA
        or evidence.get("status") != "passed"
        or evidence.get("palette_id") != style
    ):
        raise ValueError("appearance harmony evidence is missing or drifted")
    return {
        "schema": HARMONY_SCHEMA,
        "status": "passed",
        "palette_id": style,
        "policy": HARMONY_SCHEMA,
    }


def _session_sidecar(scene: Path) -> Path:
    return scene.with_suffix(scene.suffix + ".session.json")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=path.name + ".", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _set_material_response(stage, entry: Mapping[str, Any]) -> list[str]:
    from pxr import Sdf, Usd, UsdShade

    root = stage.GetPrimAtPath(str(entry["prim_path"]))
    if not root or not root.IsValid():
        raise ValueError(f"material root is missing: {entry['prim_path']}")
    authored: list[str] = []
    roughness = entry["roughness"]
    metallic = entry["metallic"]
    if roughness is None and metallic is None:
        return authored
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(prim)
        if shader.GetIdAttr().Get() == "UsdPreviewSurface":
            for name, value in (("roughness", roughness), ("metallic", metallic)):
                if value is None:
                    continue
                attribute = shader.CreateInput(name, Sdf.ValueTypeNames.Float)
                attribute.Set(float(value))
                authored.append(str(attribute.GetAttr().GetPath()))
            continue
        if not prim.GetAttribute("info:mdl:sourceAsset").HasAuthoredValue():
            continue
        candidates = {
            "roughness": ("reflection_roughness_constant", "roughness"),
            "metallic": ("metallic_constant", "metallic"),
        }
        for kind, value in (("roughness", roughness), ("metallic", metallic)):
            if value is None:
                continue
            for input_name in candidates[kind]:
                attribute = shader.GetInput(input_name)
                if attribute and attribute.GetAttr().IsValid():
                    attribute.Set(float(value))
                    authored.append(str(attribute.GetAttr().GetPath()))
                    break
    if (roughness is not None or metallic is not None) and not authored:
        raise ValueError(
            f"named material look found no roughness/metallic shader input "
            f"under {entry['prim_path']}"
        )
    return authored


def apply_plan(scene_path: Path | str, plan: Mapping[str, Any]) -> dict[str, Any]:
    if os.environ.get(ENABLE_ENV) != "1":
        raise RuntimeError(
            f"v2 appearance application is opt-in; set {ENABLE_ENV}=1")
    scene = Path(scene_path).resolve()
    house_scene_authority.require_current_scene(scene, scene.parent)
    sidecar = _session_sidecar(scene)
    if not sidecar.is_file():
        raise RuntimeError(
            f"open a scene session before appearance authoring; missing {sidecar}")
    if plan.get("schema") != SCHEMA:
        raise ValueError("appearance plan has the wrong schema")
    registry = load_randomization_registry()
    validate_harmony(plan, registry=registry)
    planned_registry = plan.get("randomization_registry")
    if not isinstance(planned_registry, Mapping):
        raise ValueError("appearance plan has no randomization registry")
    if set(planned_registry.get("components", {})) != REQUIRED_RANDOMIZERS:
        raise ValueError("appearance plan does not cover all four randomizers")

    existing_path = scene.with_suffix(scene.suffix + ".episode_appearance_v3.json")
    session = json.loads(sidecar.read_text(encoding="utf-8"))
    environment_identity = {"seed": int(plan["gen_seed"]), "palette": plan["style"],
        "lighting_scheme": plan["lighting_scheme"],
        "topology_id": plan.get("topology_visual", {}).get("topology_id")}
    if session.get("shared_environment") is not None:
        if session["shared_environment"] != environment_identity:
            raise ValueError("environment is fixed for this session; open a new session to randomize")
        previous = json.loads(existing_path.read_text(encoding="utf-8"))
        return {**previous, "reused_session_environment": True}

    import episode_appearance_randomization as legacy

    policy = legacy.load_policy()
    tints = [
        {
            "id": entry["furniture_id"],
            "prim_path": entry["prim_path"],
            "rgb_scale": entry["rgb_scale"],
        }
        for entry in plan["materials"]
    ]
    legacy_plan = legacy.build_appearance_plan(
        task_id=str(plan["task_id"]),
        seed_override=int(plan["gen_seed"]),
        policy=policy,
    )
    legacy_plan["lights"] = list(plan["lights"])
    legacy_plan["tints"] = tints
    legacy_plan["kitchen_materials"] = dict(plan["kitchen_materials"])
    legacy_manifest = legacy.apply_plan_to_stage(
        scene_path=scene, plan=legacy_plan, policy=policy)

    from pxr import Usd

    stage = Usd.Stage.Open(str(scene))
    if not stage:
        raise RuntimeError(f"could not reopen appearance scene {scene}")
    response_attributes: list[str] = []
    for entry in plan["materials"]:
        response_attributes.extend(_set_material_response(stage, entry))
    environment_manifest = environment_randomization.apply_plan_to_stage(
        stage, plan["environment"]
    )
    stage.GetRootLayer().Save()
    topology_manifest = None
    topology_plan = plan.get("topology_visual")
    if topology_plan is not None:
        import house_topology_visual

        topology_id = str(topology_plan.get("topology_id", ""))
        topology_catalog_path = topology_plan.get("catalog_path")
        topology_catalog_override = (
            None
            if topology_catalog_path is None
            else house_topology_visual.load_overlay_catalog(
                Path(str(topology_catalog_path)))
        )
        candidate, topology_catalog = house_topology_visual.candidate_for_id(
            topology_id, topology_catalog_override)
        topology_manifest = house_topology_visual.apply_visual_plan(
            scene,
            candidate,
            topology_catalog["interaction_modules"],
            topology_plan,
        )
    import house_environment_appearance
    stage = Usd.Stage.Open(str(scene))
    shared_environment = house_environment_appearance.apply(stage, plan)
    stage.GetRootLayer().Save()
    result = {
        **dict(plan),
        "shared_environment": shared_environment,
        "schema": "real2sim.livingroom_appearance_applied.v3",
        "scene_path": str(scene),
        "session_sidecar": str(sidecar),
        "legacy_authored_attribute_paths": legacy_manifest[
            "authored_attribute_paths"],
        "kitchen_material_application": legacy_manifest[
            "kitchen_material_application"],
        "material_response_attribute_paths": sorted(response_attributes),
        "environment": environment_manifest,
        "topology_visual": topology_manifest,
        "randomization_registry": dict(planned_registry),
        "furniture_transforms_unchanged": True,
        "semantic_prims_untouched": True,
    }
    output = scene.with_suffix(scene.suffix + ".episode_appearance_v3.json")
    _atomic_json(output, result)
    session["shared_environment"] = environment_identity
    _atomic_json(sidecar, session)
    result["sidecar_path"] = str(output)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--gen-seed", required=True, type=int)
    parser.add_argument("--style")
    parser.add_argument("--lighting")
    parser.add_argument(
        "--topology-id",
        help=(
            "optional full-House topology candidate already authored into "
            "the disposable scene; relocates decor and creates room lights"
        ),
    )
    parser.add_argument(
        "--topology-catalog", type=Path,
        help="optional final-delivery topology catalog used only by this canary",
    )
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        plan = build_plan(
            task_id=args.task,
            gen_seed=args.gen_seed,
            style=args.style,
            lighting=args.lighting,
        )
        if args.topology_id:
            import house_topology_visual

            topology_catalog_override = (
                None
                if args.topology_catalog is None
                else house_topology_visual.load_overlay_catalog(
                    args.topology_catalog)
            )
            candidate, topology_catalog = house_topology_visual.candidate_for_id(
                args.topology_id, topology_catalog_override)
            plan["topology_visual"] = house_topology_visual.build_visual_plan(
                candidate,
                topology_catalog["interaction_modules"],
                plan,
            )
            if args.topology_catalog is not None:
                plan["topology_visual"]["catalog_path"] = str(
                    args.topology_catalog.resolve())
        result = plan if args.dry_run else apply_plan(args.scene, plan)
        print(json.dumps({"ok": True, "appearance": result}, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({
            "ok": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
