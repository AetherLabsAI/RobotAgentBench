#!/usr/bin/env python3
"""Deterministic per-episode appearance randomization (lights + tints).

Randomizes what nobody reasons over visually: room-shell colours, furniture
material tints, and the ceiling/daylight RectLights.  Appearance-semantic
assets — branded shelf stock, colour-identified trash bins, every task target
prop — are excluded by policy and asserted untouched.

Architecture mirrors ``episode_scene_randomization``: the planner is pure
Python (testable without USD); the application path runs under the mesh311
interpreter, authors ONLY shader inputs and light attributes into the
canonical working scene (which every session restores from the read-only
baseline, so overrides are episode-scoped by construction), verifies that no
protected furniture transform moved, and writes an atomic JSON sidecar.

The policy version, benchmark task id, and exact REAL episode UUID or benchmark
seed select a stream decorrelated from the clutter planner's, so the same
inputs always produce the same look. Git records policy source changes.

Default is OFF: ``make_compare3`` only calls this when
``REAL2SIM_APPEARANCE_RANDOMIZATION=1`` — unset, the pipeline is bit-identical
to history.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

POLICY_PATH = Path(__file__).with_name("episode_appearance_policy.json")
POLICY_VERSION = "episode-appearance-v2-2026-08-08"
ENABLE_ENV = "REAL2SIM_APPEARANCE_RANDOMIZATION"

from machine_paths import MESH311_PYTHON  # noqa: E402

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}$"
)
_KITCHEN_MATERIAL_MODULE = None


def _kitchen_material_module():
    """Load the reviewed kitchen owner without assuming repo/runtime layout."""
    global _KITCHEN_MATERIAL_MODULE
    if _KITCHEN_MATERIAL_MODULE is not None:
        return _KITCHEN_MATERIAL_MODULE
    here = Path(__file__).resolve()
    candidates = (
        here.with_name("kitchen_material_randomization.py"),
        here.parents[1] / "production" / "kitchen_material_randomization.py",
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            "kitchen_material_randomization.py is not deployed with appearance randomization"
        )
    name = "_episode_kitchen_material_randomization"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load kitchen material owner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _KITCHEN_MATERIAL_MODULE = module
    return module


def load_policy(path: Path | str = POLICY_PATH) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        payload.get("schema") != "real2sim.episode_appearance_policy.v1"
        or payload.get("status") != "current"
    ):
        raise ValueError("episode appearance policy is not current v1")
    return payload


def _seed_from_explicit_identity(*parts: str) -> int:
    return random.Random("\0".join(parts)).getrandbits(64)


def randomization_enabled(environ: Mapping | None = None) -> bool:
    """Appearance randomization is opt-in per process, identity by default."""
    return (environ if environ is not None else os.environ).get(
        ENABLE_ENV) == "1"


def _identity_token(
    *,
    task_id: str,
    episode_uid: str = "",
    seed_override: str | int | None = None,
) -> str:
    if seed_override is not None and str(seed_override).strip():
        token = str(seed_override).strip()
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", token):
            raise ValueError(
                "SCENE_RANDOMIZATION_SEED contains unsafe characters")
        return "benchmark:" + token
    uid = str(episode_uid).strip().lower()
    if uid:
        if not UUID_RE.fullmatch(uid):
            raise ValueError(
                "episode appearance randomization requires an exact UUID")
        return "real:" + uid
    return "manual:" + str(task_id)


def appearance_identity(
    task_id: str,
    episode_uid: str = "",
    seed_override: str | int | None = None,
) -> dict:
    token = _identity_token(
        task_id=str(task_id), episode_uid=episode_uid,
        seed_override=seed_override)
    seed = _seed_from_explicit_identity(
        POLICY_VERSION, "appearance", str(task_id), token)
    return {
        "schema": "real2sim.episode_appearance.identity.v1",
        "policy_version": POLICY_VERSION,
        "task_id": str(task_id),
        "episode_token": token,
        "seed_u64": seed,
    }


def _check_range(bounds: Sequence[float], label: str) -> tuple[float, float]:
    low, high = float(bounds[0]), float(bounds[1])
    if not (math.isfinite(low) and math.isfinite(high) and 0.0 < low <= high):
        raise ValueError(f"invalid range for {label}: {bounds}")
    return low, high


def build_appearance_plan(
    *,
    task_id: str,
    episode_uid: str = "",
    seed_override: str | int | None = None,
    policy: Mapping | None = None,
) -> dict:
    """Sample one deterministic look: light scales/temperatures + RGB tints."""
    effective = dict(policy or load_policy())
    identity = appearance_identity(
        str(task_id), episode_uid, seed_override)
    rng = random.Random(identity["seed_u64"])

    lights = []
    for group_name in sorted(effective["lights"]["groups"]):
        group = effective["lights"]["groups"][group_name]
        scale_low, scale_high = _check_range(
            group["intensity_scale_range"], f"{group_name} intensity")
        temp_low, temp_high = _check_range(
            group["color_temperature_range_k"], f"{group_name} temperature")
        jitter_low, jitter_high = _check_range(
            group["per_light_jitter_range"], f"{group_name} jitter")
        group_scale = rng.uniform(scale_low, scale_high)
        group_temperature = rng.uniform(temp_low, temp_high)
        for path in group["paths"]:
            lights.append({
                "prim_path": str(path),
                "group": group_name,
                "intensity_scale": round(
                    group_scale * rng.uniform(jitter_low, jitter_high), 5),
                "color_temperature_k": round(group_temperature, 1),
            })

    tints = []
    for entry in effective["tintable_prims"]:
        low, high = _check_range(
            entry["rgb_scale_range"], str(entry["id"]))
        tints.append({
            "id": str(entry["id"]),
            "prim_path": str(entry["prim_path"]),
            "rgb_scale": [round(rng.uniform(low, high), 5)
                          for _ in range(3)],
        })

    kitchen_materials = _kitchen_material_module().build_randomization_plan(
        task_id=str(task_id),
        episode_uid=episode_uid,
        seed_override=seed_override,
    )
    return {
        **identity,
        "schema": "real2sim.episode_appearance.plan.v2",
        "lights": lights,
        "tints": tints,
        "kitchen_materials": kitchen_materials,
        "physics_effect": "none",
        "excluded_semantic_prims": sorted(
            effective["excluded_semantic_prims"]),
    }


# --------------------------------------------------------------------------
# USD application path (mesh311 interpreter only below this line)
# --------------------------------------------------------------------------

def _matrix_values(matrix) -> tuple[float, ...]:
    return tuple(
        float(matrix[row][column])
        for row in range(4)
        for column in range(4)
    )


def _under_any(path: str, roots: Sequence[str]) -> bool:
    return any(path == root or path.startswith(root + "/") for root in roots)


def _tint_prim_materials(stage, root_path: str, rgb: Sequence[float],
                         excluded: Sequence[str]) -> list[str]:
    """Multiply every diffuse contribution below ``root_path`` by ``rgb``.

    Textured materials: author ``inputs:scale`` on each UsdUVTexture that
    feeds a UsdPreviewSurface ``diffuseColor``.  Flat materials: multiply the
    authored ``diffuseColor`` value.  Returns the authored attribute paths.
    """
    from pxr import Gf, Usd, UsdShade

    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        raise ValueError(f"tintable prim is missing: {root_path}")
    authored = []
    for prim in Usd.PrimRange(root):
        if _under_any(str(prim.GetPath()), excluded):
            continue
        if not prim.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(prim)
        shader_id = shader.GetIdAttr().Get()
        if shader_id != "UsdPreviewSurface":
            # MDL materials (OmniPBR family) tint via diffuse_tint /
            # diffuse_color_constant instead of a diffuseColor input.
            if prim.GetAttribute("info:mdl:sourceAsset").HasAuthoredValue():
                for name in ("diffuse_tint", "diffuse_color_constant"):
                    mdl_input = shader.GetInput(name)
                    base = mdl_input.Get() if mdl_input else None
                    if base is None:
                        continue
                    mdl_input.Set(Gf.Vec3f(
                        float(base[0]) * float(rgb[0]),
                        float(base[1]) * float(rgb[1]),
                        float(base[2]) * float(rgb[2]),
                    ))
                    authored.append(str(mdl_input.GetAttr().GetPath()))
                    break
            continue
        diffuse = shader.GetInput("diffuseColor")
        if diffuse is None:
            continue
        if diffuse.HasConnectedSource():
            source, _, _ = diffuse.GetConnectedSource()
            texture = UsdShade.Shader(source.GetPrim())
            if texture.GetIdAttr().Get() != "UsdUVTexture":
                continue
            scale_input = texture.GetInput("scale")
            base = scale_input.Get() if scale_input else None
            if base is None:
                base = Gf.Vec4f(1.0, 1.0, 1.0, 1.0)
            value = Gf.Vec4f(
                float(base[0]) * float(rgb[0]),
                float(base[1]) * float(rgb[1]),
                float(base[2]) * float(rgb[2]),
                float(base[3]),
            )
            created = texture.CreateInput(
                "scale", _sdf_float4_type())
            created.Set(value)
            authored.append(str(created.GetAttr().GetPath()))
        else:
            base = diffuse.Get()
            if base is None:
                continue
            value = Gf.Vec3f(
                min(1.0, float(base[0]) * float(rgb[0])),
                min(1.0, float(base[1]) * float(rgb[1])),
                min(1.0, float(base[2]) * float(rgb[2])),
            )
            diffuse.Set(value)
            authored.append(str(diffuse.GetAttr().GetPath()))
    if not authored:
        raise ValueError(
            f"tintable prim has no tintable material: {root_path}")
    return authored


def _sdf_float4_type():
    from pxr import Sdf
    return Sdf.ValueTypeNames.Float4


def _apply_appearance_to_open_stage(stage, plan: Mapping, effective: Mapping):
    """Author only global lights/tints to an open, unsaved stage."""
    from pxr import Usd, UsdGeom, UsdLux

    furniture_before = {}
    for path in effective["protected_furniture_prims"]:
        prim = stage.GetPrimAtPath(str(path))
        if not prim or not prim.IsValid():
            raise ValueError(f"protected furniture prim is missing: {path}")
        furniture_before[str(path)] = _matrix_values(
            UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()))

    excluded = [str(p) for p in effective["excluded_semantic_prims"]]
    authored = []
    for entry in plan["lights"]:
        prim = stage.GetPrimAtPath(str(entry["prim_path"]))
        if not prim or not prim.IsValid():
            raise ValueError(f"light prim is missing: {entry['prim_path']}")
        light = UsdLux.RectLight(prim)
        if not light:
            raise ValueError(f"prim is not a RectLight: {entry['prim_path']}")
        base = light.GetIntensityAttr().Get()
        if base is None or not math.isfinite(float(base)):
            raise ValueError(
                f"light has no finite intensity: {entry['prim_path']}")
        light.GetIntensityAttr().Set(
            float(base) * float(entry["intensity_scale"]))
        light.GetEnableColorTemperatureAttr().Set(True)
        light.GetColorTemperatureAttr().Set(
            float(entry["color_temperature_k"]))
        authored.extend([
            str(light.GetIntensityAttr().GetPath()),
            str(light.GetColorTemperatureAttr().GetPath()),
        ])

    for tint in plan["tints"]:
        if _under_any(str(tint["prim_path"]), excluded):
            raise ValueError(
                f"policy error: tintable prim is excluded: {tint['prim_path']}")
        authored.extend(_tint_prim_materials(
            stage, str(tint["prim_path"]), tint["rgb_scale"], excluded))

    for attr_path in authored:
        if _under_any(attr_path.split(".")[0], excluded):
            raise ValueError(
                f"appearance randomization touched an excluded prim: {attr_path}")

    for path, expected in furniture_before.items():
        prim = stage.GetPrimAtPath(path)
        observed = _matrix_values(
            UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()))
        if observed != expected:
            raise ValueError(
                f"appearance randomization moved protected furniture {path}")
    return sorted(set(authored))


def apply_plan_to_stage(
    *,
    scene_path: Path | str,
    plan: Mapping,
    policy: Mapping | None = None,
    allow_test_scene: bool = False,
) -> dict:
    """Atomically compose global appearance and kitchen material choices."""
    from pxr import Usd

    effective = policy or load_policy()
    if plan.get("schema") != "real2sim.episode_appearance.plan.v2":
        raise ValueError("not an overall appearance v2 plan")
    if plan.get("physics_effect") != "none":
        raise ValueError("appearance plan must not affect physics")
    scene = Path(scene_path).resolve()
    scene_mode = scene.stat().st_mode & 0o777
    kitchen = _kitchen_material_module()
    kitchen._validate_working_scene(
        scene, allow_test_scene=allow_test_scene)
    stage = Usd.Stage.Open(str(scene))
    if not stage:
        raise ValueError(f"could not open scene: {scene}")
    authored = _apply_appearance_to_open_stage(stage, plan, effective)

    temporary_scene = None
    temporary_kitchen_sidecar = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=scene.stem + ".overall-appearance.",
            suffix=scene.suffix,
            dir=scene.parent,
        )
        os.close(descriptor)
        temporary_scene = Path(temporary_name)
        if not stage.GetRootLayer().Export(str(temporary_scene)):
            raise RuntimeError("failed to export temporary appearance stage")
        del stage
        kitchen_applied = kitchen.apply_plan_to_stage(
            scene_path=temporary_scene,
            plan=plan["kitchen_materials"],
            allow_test_scene=True,
        )
        temporary_kitchen_sidecar = Path(kitchen_applied["sidecar_path"])
        os.replace(temporary_scene, scene)
        os.chmod(scene, scene_mode)
        temporary_scene = None

        canonical_kitchen = dict(kitchen_applied)
        canonical_kitchen["scene_path"] = str(scene)
        canonical_kitchen.pop("sidecar_path", None)
        kitchen_sidecar = scene.with_suffix(
            scene.suffix + ".kitchen_material_randomization.json")
        _atomic_json(kitchen_sidecar, canonical_kitchen)
        canonical_kitchen["sidecar_path"] = str(kitchen_sidecar)

        manifest = {
            **dict(plan),
            "schema": "real2sim.episode_appearance.applied.v2",
            "scene_path": str(scene),
            "authored_attribute_paths": authored,
            "kitchen_material_application": {
                key: canonical_kitchen.get(key)
                for key in (
                    "central_table", "stove", "target_counts",
                    "target_prim_paths", "material_bindings_verified",
                    "furniture_transforms_unchanged", "physics_effect",
                    "policy_version", "sidecar_path",
                )
            },
            "furniture_transforms_unchanged": True,
            "physics_effect": "none",
            "atomic_composite_application": True,
        }
        sidecar = scene.with_suffix(scene.suffix + ".episode_appearance.json")
        _atomic_json(sidecar, manifest)
        manifest["sidecar_path"] = str(sidecar)
        return manifest
    finally:
        if temporary_scene is not None:
            temporary_scene.unlink(missing_ok=True)
        if temporary_kitchen_sidecar is not None:
            temporary_kitchen_sidecar.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=path.name + ".", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_appearance_subprocess(
    *,
    scene_path: Path | str,
    task_id: str,
    episode_uid: str = "",
    seed_override: str | int | None = None,
    timeout: int = 180,
) -> dict:
    """Apply with the USD-capable interpreter and return its JSON result."""
    command = [
        str(MESH311_PYTHON),
        str(Path(__file__).resolve()),
        "--scene", str(scene_path),
        "--task-id", str(task_id),
    ]
    if episode_uid:
        command.extend(["--episode-uid", str(episode_uid)])
    if seed_override is not None and str(seed_override).strip():
        command.extend(["--seed", str(seed_override)])
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=timeout,
        check=False)
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "episode appearance randomizer returned invalid JSON: "
            + (result.stdout + result.stderr)[-800:]) from exc
    if result.returncode != 0 or payload.get("ok") is not True:
        raise RuntimeError(
            "episode appearance randomization failed: "
            + str(payload.get("reason", payload)))
    return payload["appearance"]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--episode-uid", default="")
    parser.add_argument("--seed", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-test-scene", action="store_true")
    args = parser.parse_args(argv)
    try:
        policy = load_policy()
        plan = build_appearance_plan(
            task_id=str(args.task_id),
            episode_uid=str(args.episode_uid),
            seed_override=args.seed,
            policy=policy,
        )
        if args.dry_run:
            result = plan
        else:
            result = apply_plan_to_stage(
                scene_path=args.scene,
                plan=plan,
                policy=policy,
                allow_test_scene=args.allow_test_scene,
            )
        print(json.dumps({"ok": True, "appearance": result}, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({
            "ok": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
