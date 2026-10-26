#!/usr/bin/env python3
"""Deterministic, appearance-only Kitchen material randomization.

This module owns two independent but jointly seeded choices:

* one of the seven owner-approved central-table material combinations; and
* one of the sixteen legal stove-area combinations defined by four conditional
  countertop rules (two lower-cabinet choices and two upper-cabinet choices
  per countertop).

The planner is pure Python and can be tested without Isaac/USD.  The apply path
is deliberately narrow: it authors materials, material bindings, and the box
UV primvars needed by textured central-table choices.  It snapshots every
furniture transform before authoring and fails if any local-to-world matrix
changes.  Physics, collision, camera, light, geometry, and object poses are
outside this script's scope.

Production application is allowed only on the canonical ``scene_room.usda``
working scene while an open scene-session sidecar owns it.  ``--allow-test-scene``
exists only for disposable validation copies; baseline files are always
rejected.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
TABLE_REGISTRY_PATH = HERE / "central_table_material_random_options.json"
STOVE_REGISTRY_PATH = HERE / "stove_material_random_options.json"
POLICY_VERSION = "kitchen-material-randomization-v1-2026-08-08"
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}$"
)
SAFE_SEED_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
SAFE_PRIM_TOKEN_RE = re.compile(r"[^A-Za-z0-9_]", re.ASCII)


def _read_json(path: Path | str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"registry must be a JSON object: {path}")
    return payload


def load_table_registry(
    path: Path | str = TABLE_REGISTRY_PATH,
) -> dict[str, Any]:
    payload = _read_json(path)
    if (
        payload.get("schema")
        != "real2sim.central_table_material_random_options.v1"
        or payload.get("status") != "approved_options"
        or payload.get("application_status") != "wired_standalone"
    ):
        raise ValueError("central-table material registry is not wired v1")
    _validate_table_registry(payload)
    return payload


def load_stove_registry(
    path: Path | str = STOVE_REGISTRY_PATH,
) -> dict[str, Any]:
    payload = _read_json(path)
    if (
        payload.get("schema")
        != "real2sim.stove_material_random_options.v1"
        or payload.get("status") != "approved_options"
        or payload.get("application_status") != "wired_standalone"
    ):
        raise ValueError("stove material registry is not wired v1")
    _validate_stove_registry(payload)
    return payload


def _validate_material_spec(spec: Mapping[str, Any], label: str) -> None:
    kind = str(spec.get("kind", ""))
    if kind == "existing_material":
        path = str(spec.get("material_path", ""))
        if not path.startswith("/World/"):
            raise ValueError(f"{label}: invalid existing material path")
        return
    if kind == "inline_preview_surface":
        color = spec.get("diffuse_color")
        if not isinstance(color, list) or len(color) != 3:
            raise ValueError(f"{label}: diffuse_color must have three values")
    elif kind == "textured_preview_surface":
        texture = str(spec.get("texture", ""))
        scale = spec.get("rgb_scale")
        if not texture or Path(texture).is_absolute() or ".." in Path(texture).parts:
            raise ValueError(f"{label}: texture must be a safe relative path")
        if not isinstance(scale, list) or len(scale) != 3:
            raise ValueError(f"{label}: rgb_scale must have three values")
    else:
        raise ValueError(f"{label}: unsupported material kind {kind!r}")
    for key in ("roughness", "metallic"):
        value = float(spec.get(key, -1.0))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{label}: {key} is outside [0, 1]")


def _validate_table_registry(payload: Mapping[str, Any]) -> None:
    if payload.get("appearance_only") is not True:
        raise ValueError("central-table registry must be appearance-only")
    if payload.get("geometry_physics_transforms_unchanged") is not True:
        raise ValueError("central-table registry must freeze geometry and physics")
    targets = payload.get("target_prims", {})
    if set(targets) != {"top", "body"}:
        raise ValueError("central-table registry must define top and body targets")
    if len(set(map(str, targets.values()))) != 2:
        raise ValueError("central-table targets must be distinct")
    options = payload.get("options")
    if not isinstance(options, list) or len(options) != 7:
        raise ValueError("central-table registry must contain seven options")
    ids: set[str] = set()
    for option in options:
        option_id = str(option.get("id", ""))
        if not option_id or option_id in ids:
            raise ValueError("central-table option ids must be unique")
        ids.add(option_id)
        _validate_material_spec(option["top"], f"table {option_id} top")
        body = option["body"]
        if body.get("kind") != "same_as_top":
            _validate_material_spec(body, f"table {option_id} body")


def _validate_stove_registry(payload: Mapping[str, Any]) -> None:
    if payload.get("appearance_only") is not True:
        raise ValueError("stove registry must be appearance-only")
    if payload.get("geometry_physics_transforms_unchanged") is not True:
        raise ValueError("stove registry must freeze geometry and physics")
    lower = list(map(str, payload.get("lower_roots", [])))
    upper = list(map(str, payload.get("upper_roots", [])))
    if not lower or not upper or len(lower) != len(set(lower)):
        raise ValueError("stove lower roots are empty or duplicated")
    if len(upper) != len(set(upper)) or set(lower) & set(upper):
        raise ValueError("stove upper roots are duplicated or overlap lower roots")
    library = payload.get("material_library")
    if not isinstance(library, dict) or not library:
        raise ValueError("stove material library is empty")
    for material_id, spec in library.items():
        _validate_material_spec(spec, f"stove material {material_id}")
    rules = payload.get("rules")
    if not isinstance(rules, list) or len(rules) != 4:
        raise ValueError("stove registry must define four countertop rules")
    rule_ids: set[str] = set()
    combinations = 0
    for rule in rules:
        rule_id = str(rule.get("id", ""))
        if not rule_id or rule_id in rule_ids:
            raise ValueError("stove rule ids must be unique")
        rule_ids.add(rule_id)
        lower_choices = list(rule.get("lower_choices", []))
        upper_choices = list(rule.get("upper_choices", []))
        if len(lower_choices) != 2 or len(upper_choices) != 2:
            raise ValueError(f"{rule_id}: expected 2 lower and 2 upper choices")
        referenced = [rule.get("countertop"), *lower_choices, *upper_choices]
        missing = [name for name in referenced if name not in library]
        if missing:
            raise ValueError(f"{rule_id}: unknown materials {missing}")
        combinations += len(lower_choices) * len(upper_choices)
    if combinations != 16:
        raise ValueError("stove registry must expose exactly 16 legal combinations")
    selectors = payload.get("target_selection", {})
    excluded = set(map(str, selectors.get("excluded_materials", [])))
    selected_sources = {
        str(selectors.get("worktop_source_material", "")),
        str(selectors.get("lower_cabinet_source_material", "")),
        str(selectors.get("lower_handle_source_material", "")),
        str(selectors.get("upper_cabinet_source_material", "")),
    }
    if "" in selected_sources or selected_sources & excluded:
        raise ValueError("stove selectors overlap excluded materials")
    excluded_components = list(
        map(str, selectors.get("lower_cabinet_excluded_path_components", []))
    )
    if "Oven" not in excluded_components or any(
        not component or "/" in component for component in excluded_components
    ):
        raise ValueError("stove lower-cabinet selector must exclude the Oven subtree")
    full_recolor_roots = list(
        map(str, selectors.get("upper_full_recolor_roots", []))
    )
    if (
        not full_recolor_roots
        or len(full_recolor_roots) != len(set(full_recolor_roots))
        or not set(full_recolor_roots).issubset(set(upper))
    ):
        raise ValueError(
            "stove full-recolor roots must be unique registered upper roots"
        )
    allowed_full_recolor_materials = set(
        map(
            str,
            selectors.get(
                "upper_full_recolor_allowed_excluded_materials", []
            ),
        )
    )
    if (
        not allowed_full_recolor_materials
        or not allowed_full_recolor_materials.issubset(excluded)
    ):
        raise ValueError(
            "stove full-recolor overrides must be registered exclusions"
        )
    extra_paths = list(
        map(str, selectors.get("upper_cabinet_extra_paths", []))
    )
    if (
        not extra_paths
        or len(extra_paths) != len(set(extra_paths))
        or any(not path.startswith("/World/") for path in extra_paths)
    ):
        raise ValueError("stove upper extra paths are empty, duplicated, or invalid")


def _identity_token(
    *,
    task_id: str,
    episode_uid: str = "",
    seed_override: str | int | None = None,
) -> str:
    if seed_override is not None and str(seed_override).strip():
        token = str(seed_override).strip()
        if not SAFE_SEED_RE.fullmatch(token):
            raise ValueError("randomization seed contains unsafe characters")
        return "benchmark:" + token
    uid = str(episode_uid).strip().lower()
    if uid:
        if not UUID_RE.fullmatch(uid):
            raise ValueError(
                "kitchen material randomization requires an exact UUID"
            )
        return "real:" + uid
    return "manual:" + str(task_id)


def randomization_identity(
    *,
    task_id: str,
    episode_uid: str = "",
    seed_override: str | int | None = None,
) -> dict[str, Any]:
    token = _identity_token(
        task_id=str(task_id),
        episode_uid=episode_uid,
        seed_override=seed_override,
    )
    seed_u64 = random.Random(
        POLICY_VERSION + "\0kitchen-materials\0"
        + ("house-environment" if seed_override is not None else str(task_id)) + "\0" + token
    ).getrandbits(64)
    return {
        "schema": "real2sim.kitchen_material_randomization.identity.v1",
        "policy_version": POLICY_VERSION,
        "task_id": str(task_id),
        "episode_token": token,
        "seed_u64": seed_u64,
    }


def build_randomization_plan(
    *,
    task_id: str,
    episode_uid: str = "",
    seed_override: str | int | None = None,
    table_registry: Mapping[str, Any] | None = None,
    stove_registry: Mapping[str, Any] | None = None,
    allowed_table_option_ids: Sequence[str] | None = None,
    allowed_stove_rule_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Choose one reproducible table option and one legal stove combination."""
    table = deepcopy(dict(table_registry or load_table_registry()))
    stove = deepcopy(dict(stove_registry or load_stove_registry()))
    _validate_table_registry(table)
    _validate_stove_registry(stove)
    identity = randomization_identity(
        task_id=str(task_id),
        episode_uid=episode_uid,
        seed_override=seed_override,
    )
    rng = random.Random(identity["seed_u64"])
    table_ids = (
        set(map(str, allowed_table_option_ids))
        if allowed_table_option_ids is not None else None
    )
    stove_ids = (
        set(map(str, allowed_stove_rule_ids))
        if allowed_stove_rule_ids is not None else None
    )
    table_options = [
        option for option in table["options"]
        if table_ids is None or str(option["id"]) in table_ids
    ]
    stove_rules = [
        rule for rule in stove["rules"]
        if stove_ids is None or str(rule["id"]) in stove_ids
    ]
    known_table_ids = {str(option["id"]) for option in table["options"]}
    known_stove_ids = {str(rule["id"]) for rule in stove["rules"]}
    if table_ids is not None and (
        not table_ids or table_ids - known_table_ids
    ):
        raise ValueError(
            "allowed central-table options are empty or unknown: "
            + repr(sorted(table_ids - known_table_ids)))
    if stove_ids is not None and (
        not stove_ids or stove_ids - known_stove_ids
    ):
        raise ValueError(
            "allowed stove rules are empty or unknown: "
            + repr(sorted(stove_ids - known_stove_ids)))
    table_option = deepcopy(rng.choice(table_options))
    stove_rule = deepcopy(rng.choice(stove_rules))
    lower = str(rng.choice(stove_rule["lower_choices"]))
    upper = str(rng.choice(stove_rule["upper_choices"]))
    return {
        **identity,
        "schema": "real2sim.kitchen_material_randomization.plan.v1",
        "physics_effect": "none",
        "geometry_physics_transforms_unchanged": True,
        "selection_mode": {
            "central_table": f"uniform_{len(table_options)}",
            "stove": (
                f"uniform_{len(stove_rules)}_countertops_then_uniform_2x2"
            ),
        },
        "harmony_constraints": {
            "enforced": bool(
                allowed_table_option_ids is not None
                or allowed_stove_rule_ids is not None
            ),
            "allowed_central_table_option_ids": sorted(
                str(option["id"]) for option in table_options),
            "allowed_stove_rule_ids": sorted(
                str(rule["id"]) for rule in stove_rules),
        },
        "central_table": {
            "option_id": str(table_option["id"]),
            "label": str(table_option["label"]),
        },
        "stove": {
            "rule_id": str(stove_rule["id"]),
            "label": str(stove_rule["label"]),
            "countertop_material_id": str(stove_rule["countertop"]),
            "lower_material_id": lower,
            "lower_handles_material_id": lower,
            "upper_material_id": upper,
            "upper_handles_unchanged": True,
        },
    }


def _matrix_values(matrix: Any) -> tuple[float, ...]:
    return tuple(
        float(matrix[row][column])
        for row in range(4)
        for column in range(4)
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _validate_working_scene(scene: Path, *, allow_test_scene: bool) -> None:
    if "baseline" in str(scene).lower():
        raise ValueError("refusing to modify a baseline scene")
    if not scene.is_file():
        raise FileNotFoundError(f"scene does not exist: {scene}")
    if allow_test_scene:
        return
    if scene.name != "scene_room.usda":
        raise ValueError(
            "production material randomization requires canonical scene_room.usda"
        )
    session_path = scene.with_suffix(scene.suffix + ".session.json")
    if not session_path.is_file():
        raise ValueError("scene_room.usda has no open session sidecar")
    session = _read_json(session_path)
    if (
        session.get("schema") != "real2sim.scene_session.v1"
        or session.get("status") != "open"
        or Path(str(session.get("scene", ""))).resolve() != scene.resolve()
    ):
        raise ValueError("scene_room.usda session sidecar is not a valid open owner")


def _safe_prim_token(value: str) -> str:
    token = SAFE_PRIM_TOKEN_RE.sub("_", value)
    if not token or token[0].isdigit():
        token = "M_" + token
    return token


def _material_from_spec(
    stage: Any,
    spec: Mapping[str, Any],
    *,
    material_id: str,
    scene: Path,
) -> tuple[Any, list[str]]:
    from pxr import Gf, Sdf, UsdShade

    kind = str(spec["kind"])
    if kind == "existing_material":
        path = str(spec["material_path"])
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            raise ValueError(f"existing material is missing: {path}")
        material = UsdShade.Material(prim)
        if not material:
            raise ValueError(f"prim is not a material: {path}")
        return material, []

    base_path = (
        "/World/Looks/KitchenMaterialRandom/" + _safe_prim_token(material_id)
    )
    material = UsdShade.Material.Define(stage, base_path)
    shader = UsdShade.Shader.Define(stage, base_path + "/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    authored: list[str] = [
        str(shader.GetIdAttr().GetPath()),
    ]

    def remember(port: Any) -> Any:
        authored.append(str(port.GetAttr().GetPath()))
        return port

    remember(
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float)
    ).Set(float(spec["roughness"]))
    remember(
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float)
    ).Set(float(spec["metallic"]))

    if kind == "inline_preview_surface":
        color = [float(value) for value in spec["diffuse_color"]]
        remember(
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
        ).Set(Gf.Vec3f(*color))
    elif kind == "textured_preview_surface":
        texture_path = Path(str(spec["texture"]))
        resolved_texture = (scene.parent / texture_path).resolve()
        if not resolved_texture.is_file():
            raise FileNotFoundError(
                f"material texture is missing: {resolved_texture}"
            )
        texture = UsdShade.Shader.Define(stage, base_path + "/Texture")
        texture.CreateIdAttr("UsdUVTexture")
        authored.append(str(texture.GetIdAttr().GetPath()))
        remember(
            texture.CreateInput("file", Sdf.ValueTypeNames.Asset)
        ).Set(Sdf.AssetPath(texture_path.as_posix()))
        remember(
            texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token)
        ).Set("sRGB")
        remember(
            texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token)
        ).Set("repeat")
        remember(
            texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token)
        ).Set("repeat")
        scale = [float(value) for value in spec.get("rgb_scale", [1, 1, 1])]
        remember(
            texture.CreateInput("scale", Sdf.ValueTypeNames.Float4)
        ).Set(Gf.Vec4f(scale[0], scale[1], scale[2], 1.0))
        texture_rgb = remember(
            texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        )
        reader = UsdShade.Shader.Define(stage, base_path + "/PrimvarReader")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        authored.append(str(reader.GetIdAttr().GetPath()))
        remember(
            reader.CreateInput("varname", Sdf.ValueTypeNames.Token)
        ).Set("st")
        reader_result = remember(
            reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        )
        remember(
            texture.CreateInput("st", Sdf.ValueTypeNames.Float2)
        ).ConnectToSource(reader.ConnectableAPI(), "result")
        remember(
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
        ).ConnectToSource(texture.ConnectableAPI(), "rgb")
        del texture_rgb, reader_result
    else:
        raise ValueError(f"unsupported material kind: {kind}")

    surface = remember(material.CreateSurfaceOutput())
    surface.ConnectToSource(shader.ConnectableAPI(), "surface")
    return material, authored


def _bind_material(prim: Any, material: Any) -> str:
    from pxr import UsdShade

    if not prim or not prim.IsValid():
        raise ValueError("cannot bind material to an invalid prim")
    api = UsdShade.MaterialBindingAPI.Apply(prim)
    api.Bind(material)
    relation = prim.GetRelationship("material:binding")
    if not relation:
        raise RuntimeError(f"material binding was not authored on {prim.GetPath()}")
    return str(relation.GetPath())


def _direct_material_path(prim: Any) -> str:
    relation = prim.GetRelationship("material:binding")
    if not relation:
        return ""
    targets = relation.GetTargets()
    return str(targets[0]) if len(targets) == 1 else ""


def _snapshot_protected_material_bindings(
    stage: Any,
    roots: Sequence[str],
    *,
    excluded_materials: set[str],
    excluded_components: Sequence[str],
    allowed_override_roots: Sequence[str] = (),
    allowed_override_materials: set[str] | None = None,
) -> dict[str, str]:
    """Pin appliance/glass/interior bindings that this script must not alter."""
    from pxr import Usd, UsdShade

    snapshot: dict[str, str] = {}
    allowed_materials = allowed_override_materials or set()
    for root_path in roots:
        root = stage.GetPrimAtPath(root_path)
        if not root or not root.IsValid():
            raise ValueError(f"registered furniture root is missing: {root_path}")
        for prim in Usd.PrimRange(root):
            path = str(prim.GetPath())
            components = path.split("/")
            direct = _direct_material_path(prim)
            approved_override = direct in allowed_materials and any(
                path == allowed_root or path.startswith(allowed_root + "/")
                for allowed_root in allowed_override_roots
            )
            if approved_override:
                continue
            protected_subtree = any(
                component.startswith(excluded)
                for component in components
                for excluded in excluded_components
            )
            if not protected_subtree and direct not in excluded_materials:
                continue
            if protected_subtree:
                material, _ = UsdShade.MaterialBindingAPI(
                    prim
                ).ComputeBoundMaterial()
                snapshot[path] = str(material.GetPath()) if material else ""
            else:
                snapshot[path] = direct
    return snapshot


def _assert_protected_material_bindings_unchanged(
    stage: Any, expected: Mapping[str, str]
) -> None:
    from pxr import UsdShade

    for path, material_path in expected.items():
        prim = stage.GetPrimAtPath(path)
        direct = _direct_material_path(prim)
        if direct == material_path:
            continue
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        observed = str(material.GetPath()) if material else ""
        if observed != material_path:
            raise ValueError(
                f"protected material binding changed: {path}: "
                f"{material_path} -> {observed}"
            )


def _prim_range_paths_with_direct_material(
    stage: Any,
    roots: Sequence[str],
    material_path: str,
    *,
    required_component: str = "",
    excluded_components: Sequence[str] = (),
) -> list[str]:
    from pxr import Usd

    selected: set[str] = set()
    for root_path in roots:
        root = stage.GetPrimAtPath(root_path)
        if not root or not root.IsValid():
            raise ValueError(f"registered furniture root is missing: {root_path}")
        for prim in Usd.PrimRange(root):
            path = str(prim.GetPath())
            if required_component and not any(
                component.startswith(required_component)
                for component in path.split("/")
            ):
                continue
            if any(
                component.startswith(excluded)
                for component in path.split("/")
                for excluded in excluded_components
            ):
                continue
            if _direct_material_path(prim) == material_path:
                selected.add(path)
    return sorted(selected)


def _include_direct_geom_subset_paths(
    stage: Any,
    paths: Sequence[str],
    *,
    excluded_materials: set[str] | None = None,
) -> list[str]:
    """Include a selected mesh's own face subsets, but not nested components.

    Some modular cabinet assets remap their door-face materials into local
    Section0/Section1 subsets.  Selecting only the parent door therefore
    leaves the visible faces unchanged.  Restricting expansion to direct
    GeomSubset children fixes those faces without reaching nested knobs,
    appliance glass, shelves, or interiors.
    """
    from pxr import UsdGeom

    excluded = excluded_materials or set()
    expanded = set(map(str, paths))
    for path in tuple(expanded):
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            raise ValueError(f"material target prim is missing: {path}")
        expanded.update(
            str(child.GetPath())
            for child in prim.GetChildren()
            if child.IsA(UsdGeom.Subset)
            and _direct_material_path(child) not in excluded
        )
    return sorted(expanded)


def _include_upper_cabinet_exterior_shell_paths(
    stage: Any,
    roots: Sequence[str],
    paths: Sequence[str],
    *,
    excluded_materials: set[str] | None = None,
) -> list[str]:
    """Include remapped exterior carcass faces without coloring interiors.

    The imported upper-cabinet assets keep the visible carcass/top face on the
    direct ``Body/Section0`` subset and the narrow face-frame strips on direct
    ``Body/Join_*_Body_*`` meshes.  Their bindings point at asset-local
    ``UnrealMaterial`` prims after reference composition, so source-material
    selection cannot discover them.  Select only these two structural forms;
    shelves, ``Section1`` interiors, doors, handles, glass, and dish racks stay
    outside this expansion.
    """
    from pxr import UsdGeom

    excluded = excluded_materials or set()
    expanded = set(map(str, paths))
    for root_path in roots:
        body = stage.GetPrimAtPath(str(root_path) + "/Body")
        if not body or not body.IsValid():
            raise ValueError(
                f"registered upper-cabinet body is missing: {root_path}/Body"
            )
        for child in body.GetChildren():
            name = child.GetName()
            is_exterior_subset = (
                name == "Section0" and child.IsA(UsdGeom.Subset)
            )
            is_face_frame = (
                name.startswith("Join_")
                and "_Body_" in name
                and child.IsA(UsdGeom.Mesh)
            )
            if not (is_exterior_subset or is_face_frame):
                continue
            if _direct_material_path(child) in excluded:
                continue
            expanded.add(str(child.GetPath()))
    return sorted(expanded)


def _include_upper_full_recolor_paths(
    stage: Any,
    roots: Sequence[str],
    paths: Sequence[str],
    *,
    excluded_materials: set[str],
    allowed_excluded_materials: set[str],
) -> tuple[list[str], set[str]]:
    """Make registered open-grid cabinets follow the upper-cabinet color.

    These doorless modules expose their carcass interiors, shelves, and
    asset-local Section2 faces directly to the room.  Every mesh and face
    subset under their Body is therefore appearance-owned by the upper-cabinet
    choice.  Only explicitly registered excluded materials may be overridden;
    discovering glass, metal, or appliance material fails closed.
    """
    from pxr import Usd, UsdGeom

    expanded = set(map(str, paths))
    full_recolor_paths: set[str] = set()
    for root_path in roots:
        body = stage.GetPrimAtPath(str(root_path) + "/Body")
        if not body or not body.IsValid():
            raise ValueError(
                f"registered open-grid upper body is missing: {root_path}/Body"
            )
        for prim in Usd.PrimRange(body):
            if not (prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Subset)):
                continue
            direct = _direct_material_path(prim)
            if direct in excluded_materials and direct not in (
                allowed_excluded_materials
            ):
                raise ValueError(
                    f"open-grid upper contains a protected material: {prim.GetPath()}"
                )
            path = str(prim.GetPath())
            expanded.add(path)
            full_recolor_paths.add(path)
    return sorted(expanded), full_recolor_paths


def _include_registered_upper_extra_paths(
    stage: Any,
    registered_paths: Sequence[str],
    paths: Sequence[str],
    *,
    expected_source_material: str,
) -> list[str]:
    """Include appearance-owned upper fillers outside the upper root list."""
    expanded = set(map(str, paths))
    for path in registered_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            raise ValueError(f"registered upper extra prim is missing: {path}")
        direct = _direct_material_path(prim)
        if direct != expected_source_material:
            raise ValueError(
                f"registered upper extra prim has unexpected source material: "
                f"{path}: {direct}"
            )
        expanded.add(str(path))
    return sorted(expanded)


def _snapshot_transforms(stage: Any, paths: Sequence[str]) -> dict[str, tuple]:
    from pxr import Usd, UsdGeom

    snapshot: dict[str, tuple] = {}
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            raise ValueError(f"protected furniture prim is missing: {path}")
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            raise ValueError(f"protected furniture prim is not xformable: {path}")
        snapshot[path] = _matrix_values(
            xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        )
    return snapshot


def _assert_transforms_unchanged(stage: Any, expected: Mapping[str, tuple]) -> None:
    from pxr import Usd, UsdGeom

    for path, matrix in expected.items():
        observed = _matrix_values(
            UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
        )
        if observed != matrix:
            raise ValueError(f"material randomization moved furniture: {path}")


def _author_cube_uv(prim: Any, mapping: Mapping[str, Any]) -> str:
    from pxr import Gf, Sdf, UsdGeom

    if mapping.get("mode") != "box_facevarying_uv_v1":
        raise ValueError("unsupported central-table UV mapping")
    face_values = [Gf.Vec2f(*map(float, value)) for value in mapping["face_uvs"]]
    repeat = int(mapping["repeat_for_cube_faces"])
    st = UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.faceVarying,
    )
    st.Set(face_values * repeat)
    return str(st.GetAttr().GetPath())


def _table_option(registry: Mapping[str, Any], option_id: str) -> dict[str, Any]:
    for option in registry["options"]:
        if option["id"] == option_id:
            return deepcopy(option)
    raise ValueError(f"unknown central-table option: {option_id}")


def _stove_rule(registry: Mapping[str, Any], rule_id: str) -> dict[str, Any]:
    for rule in registry["rules"]:
        if rule["id"] == rule_id:
            return deepcopy(rule)
    raise ValueError(f"unknown stove rule: {rule_id}")


def _validate_plan_against_registries(
    plan: Mapping[str, Any],
    table: Mapping[str, Any],
    stove: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if plan.get("schema") != "real2sim.kitchen_material_randomization.plan.v1":
        raise ValueError("not a kitchen material randomization v1 plan")
    if plan.get("physics_effect") != "none":
        raise ValueError("kitchen material plan is not appearance-only")
    option = _table_option(table, str(plan["central_table"]["option_id"]))
    rule = _stove_rule(stove, str(plan["stove"]["rule_id"]))
    selection = plan["stove"]
    if str(selection["countertop_material_id"]) != str(rule["countertop"]):
        raise ValueError("stove countertop violates its conditional rule")
    if str(selection["lower_material_id"]) not in rule["lower_choices"]:
        raise ValueError("stove lower cabinet violates its conditional rule")
    if str(selection["upper_material_id"]) not in rule["upper_choices"]:
        raise ValueError("stove upper cabinet violates its conditional rule")
    if selection.get("lower_handles_material_id") != selection.get(
        "lower_material_id"
    ):
        raise ValueError("lower handles must follow the lower cabinet")
    return option, rule


def apply_plan_to_stage(
    *,
    scene_path: Path | str,
    plan: Mapping[str, Any],
    table_registry: Mapping[str, Any] | None = None,
    stove_registry: Mapping[str, Any] | None = None,
    allow_test_scene: bool = False,
) -> dict[str, Any]:
    """Apply only registered material/UV changes to a session working scene."""
    from pxr import Usd, UsdGeom

    scene = Path(scene_path).resolve()
    _validate_working_scene(scene, allow_test_scene=allow_test_scene)
    table = deepcopy(dict(table_registry or load_table_registry()))
    stove = deepcopy(dict(stove_registry or load_stove_registry()))
    _validate_table_registry(table)
    _validate_stove_registry(stove)
    table_option, _ = _validate_plan_against_registries(plan, table, stove)

    stage = Usd.Stage.Open(str(scene))
    if not stage:
        raise ValueError(f"could not open scene: {scene}")
    protected_paths = [
        str(table["target_prims"]["top"]),
        str(table["target_prims"]["body"]),
        *map(str, stove["lower_roots"]),
        *map(str, stove["upper_roots"]),
        *map(str, stove["target_selection"]["upper_cabinet_extra_paths"]),
    ]
    transforms_before = _snapshot_transforms(stage, protected_paths)

    selectors = stove["target_selection"]
    excluded = set(map(str, selectors["excluded_materials"]))
    full_recolor_roots = list(
        map(str, selectors["upper_full_recolor_roots"])
    )
    allowed_full_recolor_materials = set(
        map(
            str,
            selectors["upper_full_recolor_allowed_excluded_materials"],
        )
    )
    protected_bindings_before = _snapshot_protected_material_bindings(
        stage,
        [*stove["lower_roots"], *stove["upper_roots"]],
        excluded_materials=excluded,
        excluded_components=tuple(
            map(str, selectors["lower_cabinet_excluded_path_components"])
        ),
        allowed_override_roots=full_recolor_roots,
        allowed_override_materials=allowed_full_recolor_materials,
    )
    worktop_paths = _prim_range_paths_with_direct_material(
        stage,
        stove["lower_roots"],
        str(selectors["worktop_source_material"]),
        required_component=str(selectors["worktop_path_component"]),
    )
    lower_paths = _prim_range_paths_with_direct_material(
        stage,
        stove["lower_roots"],
        str(selectors["lower_cabinet_source_material"]),
        excluded_components=tuple(
            map(str, selectors["lower_cabinet_excluded_path_components"])
        ),
    )
    lower_paths = _include_direct_geom_subset_paths(
        stage, lower_paths, excluded_materials=excluded
    )
    lower_handle_paths = _prim_range_paths_with_direct_material(
        stage,
        stove["lower_roots"],
        str(selectors["lower_handle_source_material"]),
        required_component=str(selectors["lower_handle_path_component"]),
    )
    lower_handle_paths = _include_direct_geom_subset_paths(
        stage, lower_handle_paths, excluded_materials=excluded
    )
    upper_paths = _prim_range_paths_with_direct_material(
        stage,
        stove["upper_roots"],
        str(selectors["upper_cabinet_source_material"]),
    )
    upper_paths = _include_direct_geom_subset_paths(
        stage, upper_paths, excluded_materials=excluded
    )
    upper_paths = _include_upper_cabinet_exterior_shell_paths(
        stage,
        stove["upper_roots"],
        upper_paths,
        excluded_materials=excluded,
    )
    upper_paths, full_recolor_paths = _include_upper_full_recolor_paths(
        stage,
        full_recolor_roots,
        upper_paths,
        excluded_materials=excluded,
        allowed_excluded_materials=allowed_full_recolor_materials,
    )
    upper_paths = _include_registered_upper_extra_paths(
        stage,
        selectors["upper_cabinet_extra_paths"],
        upper_paths,
        expected_source_material=str(selectors["upper_cabinet_source_material"]),
    )
    target_groups = {
        "stove_worktops": worktop_paths,
        "stove_lower_cabinets": lower_paths,
        "stove_lower_handles": lower_handle_paths,
        "stove_upper_cabinets": upper_paths,
    }
    empty = [name for name, paths in target_groups.items() if not paths]
    if empty:
        raise ValueError(f"no registered target bindings found for: {empty}")
    all_stove_targets = [path for paths in target_groups.values() for path in paths]
    if len(all_stove_targets) != len(set(all_stove_targets)):
        raise ValueError("stove material target groups overlap")
    excluded_violations = []
    for path in all_stove_targets:
        direct = _direct_material_path(stage.GetPrimAtPath(path))
        if direct not in excluded:
            continue
        if path in full_recolor_paths and direct in allowed_full_recolor_materials:
            continue
        excluded_violations.append(path)
    if excluded_violations:
        raise ValueError(
            "stove target selection touched an excluded material: "
            f"{excluded_violations}"
        )

    authored: list[str] = []
    created_material_properties: list[str] = []
    material_cache: dict[str, Any] = {}

    def material_for(material_id: str, spec: Mapping[str, Any]) -> Any:
        if material_id not in material_cache:
            material, properties = _material_from_spec(
                stage,
                spec,
                material_id=material_id,
                scene=scene,
            )
            material_cache[material_id] = material
            created_material_properties.extend(properties)
        return material_cache[material_id]

    top_spec = table_option["top"]
    top_material = material_for(
        "table_" + str(table_option["id"]) + "_top", top_spec
    )
    body_spec = table_option["body"]
    if body_spec["kind"] == "same_as_top":
        body_material = top_material
    else:
        body_material = material_for(
            "table_" + str(table_option["id"]) + "_body", body_spec
        )
    top_prim = stage.GetPrimAtPath(str(table["target_prims"]["top"]))
    body_prim = stage.GetPrimAtPath(str(table["target_prims"]["body"]))
    authored.append(_bind_material(top_prim, top_material))
    authored.append(_bind_material(body_prim, body_material))
    top_needs_uv = top_spec["kind"] == "textured_preview_surface" or str(
        top_spec.get("material_path", "")
    ).endswith("/DarkWorktop")
    if top_needs_uv:
        authored.append(_author_cube_uv(top_prim, table["texture_mapping"]))
    if body_spec["kind"] == "same_as_top" and top_spec["kind"] == (
        "textured_preview_surface"
    ):
        authored.append(_author_cube_uv(body_prim, table["texture_mapping"]))

    library = stove["material_library"]
    stove_plan = plan["stove"]
    countertop_id = str(stove_plan["countertop_material_id"])
    lower_id = str(stove_plan["lower_material_id"])
    upper_id = str(stove_plan["upper_material_id"])
    countertop_material = material_for(
        "stove_" + countertop_id, library[countertop_id]
    )
    lower_material = material_for("stove_" + lower_id, library[lower_id])
    upper_material = material_for("stove_" + upper_id, library[upper_id])
    for path in worktop_paths:
        authored.append(_bind_material(stage.GetPrimAtPath(path), countertop_material))
    for path in [*lower_paths, *lower_handle_paths]:
        authored.append(_bind_material(stage.GetPrimAtPath(path), lower_material))
    for path in upper_paths:
        authored.append(_bind_material(stage.GetPrimAtPath(path), upper_material))

    expected_bindings = {
        **{path: str(countertop_material.GetPath()) for path in worktop_paths},
        **{
            path: str(lower_material.GetPath())
            for path in [*lower_paths, *lower_handle_paths]
        },
        **{path: str(upper_material.GetPath()) for path in upper_paths},
        str(table["target_prims"]["top"]): str(top_material.GetPath()),
        str(table["target_prims"]["body"]): str(body_material.GetPath()),
    }
    failed_bindings = {
        path: _direct_material_path(stage.GetPrimAtPath(path))
        for path, expected in expected_bindings.items()
        if _direct_material_path(stage.GetPrimAtPath(path)) != expected
    }
    if failed_bindings:
        raise ValueError(f"material bindings did not apply: {failed_bindings}")
    _assert_protected_material_bindings_unchanged(
        stage, protected_bindings_before
    )

    all_authored = sorted(set([*authored, *created_material_properties]))
    forbidden_tokens = ("xformop", "physics:", "physx", "collision:")
    forbidden = [
        path for path in all_authored
        if any(token in path.lower() for token in forbidden_tokens)
    ]
    if forbidden:
        raise ValueError(f"appearance script authored forbidden properties: {forbidden}")
    _assert_transforms_unchanged(stage, transforms_before)
    stage.GetRootLayer().Save()

    manifest = {
        **dict(plan),
        "schema": "real2sim.kitchen_material_randomization.applied.v1",
        "scene_path": str(scene),
        "authored_property_paths": all_authored,
        "target_prim_paths": {
            "central_table_top": str(table["target_prims"]["top"]),
            "central_table_body": str(table["target_prims"]["body"]),
            **target_groups,
        },
        "target_counts": {name: len(paths) for name, paths in target_groups.items()},
        "excluded_materials_untouched": sorted(excluded),
        "protected_material_binding_count": len(protected_bindings_before),
        "material_bindings_verified": True,
        "furniture_transforms_unchanged": True,
        "physics_effect": "none",
    }
    sidecar = scene.with_suffix(
        scene.suffix + ".kitchen_material_randomization.json"
    )
    _atomic_json(sidecar, manifest)
    manifest["sidecar_path"] = str(sidecar)
    return manifest


def _bootstrap_usd_runtime() -> Any | None:
    """Start a headless SimulationApp only when the interpreter lacks pxr."""
    try:
        import pxr  # noqa: F401

        return None
    except ModuleNotFoundError:
        try:
            from isaacsim import SimulationApp
        except ModuleNotFoundError:
            from omni.isaac.kit import SimulationApp
        return SimulationApp({"headless": True})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--episode-uid", default="")
    parser.add_argument("--seed", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-test-scene", action="store_true")
    args = parser.parse_args(argv)
    app = None
    try:
        table = load_table_registry()
        stove = load_stove_registry()
        plan = build_randomization_plan(
            task_id=str(args.task_id),
            episode_uid=str(args.episode_uid),
            seed_override=args.seed,
            table_registry=table,
            stove_registry=stove,
        )
        if args.dry_run:
            result = plan
        else:
            app = _bootstrap_usd_runtime()
            result = apply_plan_to_stage(
                scene_path=args.scene,
                plan=plan,
                table_registry=table,
                stove_registry=stove,
                allow_test_scene=args.allow_test_scene,
            )
        print(json.dumps({"ok": True, "randomization": result}, sort_keys=True))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {"ok": False, "reason": f"{type(exc).__name__}: {exc}"},
                sort_keys=True,
            )
        )
        return 2
    finally:
        if app is not None:
            app.close()


if __name__ == "__main__":
    raise SystemExit(main())
