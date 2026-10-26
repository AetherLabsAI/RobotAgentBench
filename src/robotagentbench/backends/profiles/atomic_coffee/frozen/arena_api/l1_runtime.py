"""Standalone scene-only wiring for exact L1 appearance planners.

The vendored functions are unchanged. Only filesystem ownership and interpreter
discovery differ from L1's shared working-scene service. No runtime teleports.
"""
import ast
from functools import lru_cache
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

from . import l1_scene
from .bimanual import CONTROL_DT

ROOT = l1_scene.ROOT
VENDOR = l1_scene.VENDOR
PRODUCTION = VENDOR / "real2sim/production"
TELEOP = VENDOR / "real2sim/teleop"
TEXTURES = ROOT / "resources/l1_scene/appearance_assets"
# Upstream calls this field task_id but appearance only uses it as RNG salt.
# A constant scene namespace avoids importing any benchmark task or station.
SCENE_SEED_NAMESPACE = "arena_scene"
PHYSICS_DT = 1 / 60


def is_scene_output_directory(directory):
    """Accept disposable runs or the approved shelf episode output tree only.

    Resolve both the candidate and repository before checking containment so a
    symlink or '..' cannot escape the allowed output trees.
    """
    root = ROOT.resolve()
    directory = Path(directory).resolve()
    return any(directory.is_relative_to(parent) for parent in (
        root / 'runs', root / 'evaluation/tasks/shelf_pick/artifacts/episodes'))


def require_owned_scene(scene, directory):
    scene = Path(scene).resolve(strict=True)
    if (scene.parent != Path(directory).resolve() or scene.name != "scene_room.usda"
            or not is_scene_output_directory(scene.parent) or "baseline" in str(scene).lower()):
        raise ValueError("L1 appearance requires a disposable scene_room.usda in an approved output tree")
    owner = json.loads(scene.with_suffix(scene.suffix + ".session.json").read_text())
    if (owner.get("schema") != "real2sim.scene_session.v1" or owner.get("status") != "open"
            or Path(owner.get("scene", "")).resolve() != scene
            or owner.get("owner") != "arena.l1_runtime.v1"):
        raise ValueError("Scene is not owned by this Arena episode")
    return scene


def _load_with_path_adapters(name, file, skipped_imports, injected):
    """Replace machine/service imports only; retain original function ASTs."""
    module = ModuleType(name)
    module.__file__ = str(file)
    module.__dict__.update(injected)
    tree = ast.parse(file.read_text(), filename=str(file))
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.Import) and any(a.name in skipped_imports for a in node.names)
        or isinstance(node, ast.ImportFrom) and node.module in skipped_imports)]
    sys.modules[name] = module
    exec(compile(tree, str(file), "exec"), module.__dict__)
    return module


@lru_cache(maxsize=1)
def modules():
    # Used in the dedicated worker or CLI, not a model-supplied import path.
    for path in (TELEOP, PRODUCTION):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    legacy = _load_with_path_adapters("episode_appearance_randomization",
        TELEOP / "episode_appearance_randomization.py", {"machine_paths"},
        {"MESH311_PYTHON": Path(sys.executable)})
    appearance = _load_with_path_adapters("appearance", PRODUCTION / "appearance.py",
        {"house_scene_authority"},
        {"house_scene_authority": SimpleNamespace(require_current_scene=require_owned_scene)})
    return SimpleNamespace(appearance=appearance, legacy=legacy)


def appearance_plan(seed):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Invalid episode seed")
    # Original planner reserves values below 1e6. Preserve and log this mapping.
    gen_seed = 1_000_000 + seed
    contract = json.loads((PRODUCTION / "livingroom_data_production_v1.json").read_text())
    return modules().appearance.build_plan(task_id=SCENE_SEED_NAMESPACE, gen_seed=gen_seed,
                                           contract=contract)


def apply_appearance(scene, plan):
    scene = require_owned_scene(scene, Path(scene).parent)
    # Original shader paths stay exact and relative; dependencies are all in Git.
    for source in TEXTURES.rglob("*.png"):
        destination = scene.parent / source.relative_to(TEXTURES)
        if destination.exists() or destination.is_symlink():
            raise ValueError("Fresh appearance texture destination required: " + str(destination))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(os.path.relpath(source, destination.parent))
    previous = os.environ.get("REAL2SIM_APPEARANCE_PROFILE_V2")
    os.environ["REAL2SIM_APPEARANCE_PROFILE_V2"] = "1"
    try:
        return modules().appearance.apply_plan(scene, plan)
    finally:
        if previous is None:
            os.environ.pop("REAL2SIM_APPEARANCE_PROFILE_V2", None)
        else:
            os.environ["REAL2SIM_APPEARANCE_PROFILE_V2"] = previous


def restore_kitchen_bindings(stage):
    """Exact L1 baseline render bindings; no shader or physics retuning."""
    data = json.loads((ROOT / "resources/l1_scene/kitchen_baseline_bindings.json").read_text())
    changed = []
    for path, relationships in data["bindings"].items():
        prim = stage.GetPrimAtPath(path)
        if not prim:
            raise ValueError("Missing original L1 kitchen prim: " + path)
        for name, spec in relationships.items():
            if name == "material:binding:physics" or not name.startswith("material:binding"):
                raise ValueError("Unexpected non-render binding in source snapshot")
            if any(not stage.GetPrimAtPath(p) for p in spec["targets"]):
                raise ValueError("Missing original L1 material target: " + path)
            rel = prim.CreateRelationship(name)
            rel.SetTargets(spec["targets"])
            if spec["bindMaterialAs"] is not None:
                rel.SetMetadata("bindMaterialAs", spec["bindMaterialAs"])
            changed.append(str(rel.GetPath()))
    return dict(source_commit=data["source_commit"], restored_relationships=changed,
                physics_bindings_changed=False)


def open_session(scene):
    scene = Path(scene).resolve()
    sidecar = scene.with_suffix(scene.suffix + ".session.json")
    with sidecar.open("x") as stream:
        json.dump(dict(schema="real2sim.scene_session.v1", status="open", scene=str(scene),
                       owner="arena.l1_runtime.v1"), stream, indent=2)


def layout_ordinal(seed):
    from vendor.l1_scene.real2sim.production import house_layout
    # Original permutation implementation, supplied the explicit API seed.
    return house_layout.task_layout_permutation(SCENE_SEED_NAMESPACE, seed=seed)[0]
