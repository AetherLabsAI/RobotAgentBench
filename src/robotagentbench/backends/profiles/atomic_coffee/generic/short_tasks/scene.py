"""Local scene planning and scene-owner hooks; no deployment or robot actions.

The friend CPU USD builder supplies a prepared manifest/stage. This module
resolves its recipe, checks source integrity and installs task-owned mechanisms
before physics is attached. Planar feasibility is never native acceptance.
"""
from contextlib import contextmanager
from copy import deepcopy
from importlib import import_module
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import shutil
import uuid

from .common.contracts import TaskError
from . import registry

ROOT = Path(__file__).resolve().parents[1]
SCENE_ROOT = ROOT / "openend/tasks/scene_requirements"


def long_task_cache_root(output):
    """Select only this approved Long Task package's cache; no request paths."""
    owner = ROOT / 'short_tasks/agent_probe_v1/long_task_eval_setup_v1'
    output = Path(output).absolute()
    if not output.is_relative_to(owner):
        return None
    if output == owner or owner.resolve() != owner or output.resolve() != output:
        raise ValueError('Long Task output must be canonical and below its owned root')
    cache = owner / 'generated_rooms'
    if cache.resolve() != cache:
        raise ValueError('Long Task cache must not use symbolic links')
    return cache


def composition_cache_root(cache_root=None):
    """Keep the old default; only the explicit task-owned cache is additional."""
    from short_tasks.tasks.robot_pose_alignment_v1.temporary_outputs import generated
    if cache_root is None:
        return generated(SCENE_ROOT / 'generated_rooms')
    path = Path(cache_root).absolute()
    expected = ROOT / 'short_tasks/agent_probe_v1/long_task_eval_setup_v1/generated_rooms'
    if path != expected or path.resolve() != expected:
        raise ValueError('Only the approved Long Task generation cache is allowed')
    return path


def verify_scene_sources():
    record = json.loads((Path(__file__).parent / "scene_sources.json").read_text())
    verify_manifest(dict(source_hashes={str(ROOT / p): h for p, h in record["sources"].items()}))
    return dict(verified=True, file_count=len(record["sources"]))


@contextmanager
def _scene_imports():
    previous = list(sys.path)
    sys.path.insert(0, str(SCENE_ROOT))
    try:
        yield
    finally:
        sys.path[:] = previous


def _candidate(recipe, resolved, layout_id, rerolls):
    data = deepcopy(resolved)
    layouts = json.loads((SCENE_ROOT / "placement_sources/random-layouts-v1.json").read_text())["layouts"]
    data["layout"] = next(v for v in layouts if v["id"] == layout_id)
    data["request"]["layout_id"] = layout_id
    with _scene_imports():
        data["requirements"] = import_module("planner_v2").compile_request(data["request"])
    data["layout_selection"].update(selected=layout_id, rerolls=rerolls)
    with tempfile.TemporaryDirectory(prefix="short-task-plan-") as directory:
        path = Path(directory) / "candidate.json"
        path.write_text(json.dumps(data))
        try:
            result = subprocess.run(["node", str(SCENE_ROOT / "room_plan.cjs"), str(path)],
                capture_output=True, text=True, timeout=30, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise TaskError("Local scene planner failed: " + str(exc), kind="load", reason="scene_planner_unavailable") from exc
    if result.returncode not in (0, 2):
        raise TaskError("Local scene planner failed: " + result.stderr[:800], kind="load", reason="scene_planner_failed")
    try:
        candidate = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise TaskError("Scene planner returned malformed output", kind="load") from exc
    return candidate


def plan_scene(request, *, candidate_builder=None):
    """Deterministic, bounded layout selection, with no USD/GPU dependency."""
    plan = registry.resolve(request)
    if not plan["available"]:
        return dict(schema="openend.short_task.scene_plan.v1", task_plan=plan,
                    scene_status="unavailable", scene_generated=False, native_physics_validated=False)
    if registry.is_gpu_cloth_preview(plan):
        return dict(schema="openend.short_task.scene_plan.v1", task_plan=plan,
                    scene_status="scene_owner_required", scene_generated=False,
                    native_physics_validated=False, actual_layout_id=None,
                    public_task_count=1, native_load_available=True,
                    composition_owner="gpu_cloth_scene_owner.v1",
                    message="Bind an explicitly prepared GPU cloth scene; the CPU room planner cannot create deformable tasks.")
    verify_scene_sources()
    recipe = plan["scene_request"]
    try:
        with _scene_imports():
            resolved = import_module("room_request").resolve(recipe)
    except (ValueError, KeyError) as exc:
        raise TaskError(str(exc), kind="load", reason="scene_recipe_rejected") from exc
    attempts = []
    build = candidate_builder or _candidate
    for index, layout_id in enumerate(resolved["layout_candidates"]):
        candidate = build(recipe, resolved, layout_id, index)
        status = candidate["plan"]["status"]
        attempts.append(dict(layout_id=layout_id, status=status))
        if status == "planar_candidate":
            if plan["task_name"] == "stock_wooden_tray":
                from .tasks.stock_wooden_tray.generation import POLICY
                # Sole default for newly prepared ordinary shelf episodes.
                candidate["task_scene_policy"] = dict(POLICY)
            return dict(schema="openend.short_task.scene_plan.v1", task_plan=plan,
                scene_status="planar_candidate", actual_layout_id=layout_id,
                attempts=attempts, recipe=candidate, scene_generated=False,
                native_physics_validated=False, public_task_count=1,
                composition_owner="friend_scene_requirements",
                native_load_available=plan.get("scene_runtime_available", True))
    raise TaskError("No feasible layout within the seeded, non-repeating candidate list", kind="layout", reason="layout_candidates_exhausted")


def verify_manifest(manifest):
    """No source refresh or fallback asset substitution at episode admission."""
    hashes = manifest.get("source_hashes", {})
    if not hashes:
        raise TaskError("Scene owner must supply source hashes", kind="binding", reason="unverified_scene_sources")
    for path, expected in hashes.items():
        source = Path(path)
        if not source.is_file():
            raise TaskError("Scene source missing: " + path, kind="load", reason="scene_asset_missing")
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise TaskError("Scene source changed: " + path, kind="binding", reason="scene_source_changed")


def prepare_runtime_scene(plan, stage, manifest):
    """Caller must invoke before PhysX attachment, using a fresh scene layer."""
    verify_manifest(manifest)
    result = deepcopy(manifest)
    hook = plan.get("scene_preparation")
    if hook:
        name, function = hook.rsplit(".", 1)
        if not name.startswith("short_tasks.tasks."):
            raise TaskError("Scene hook is outside Short Tasks", kind="binding")
        for artifact in plan.get("required_artifacts", []):
            path = artifact.get("path")
            if path and not Path(path).is_file():
                raise TaskError("Required task asset missing: " + path, kind="load", reason="task_scene_asset_missing")
        from pxr import Usd
        layer = stage.GetSessionLayer()
        before = layer.ExportToString()
        try:
            with Usd.EditContext(stage, layer):
                result = getattr(import_module(name), function)(plan, stage, result)
        except FileNotFoundError as exc:
            layer.ImportFromString(before)
            raise TaskError(str(exc), kind="load", reason="task_scene_asset_missing") from exc
        except Exception:
            layer.ImportFromString(before)
            raise
    return result


class LocalFriendProvider:
    """Run the existing CPU USD composer in a fresh local episode directory.

    The upstream composer requires its generated_rooms parent. Keep that local
    build cache, then copy its output into the caller's fresh output directory.
    This never calls generate_room.py, SSH, native simulation or deployment.
    """
    def __init__(self, *, python_executable=None, cache_root=None):
        self.python = python_executable or sys.executable
        self.cache_root = cache_root

    def compose(self, recipe, output):
        if self.cache_root is not None and Path(output).absolute().resolve() != Path(output).absolute():
            raise ValueError('Long Task output must not use symbolic links')
        output = Path(output).resolve()
        if output.exists():
            raise TaskError("Composition output must be a fresh directory", kind="input")
        verify_scene_sources()
        layout = recipe["layout"]
        registry_root = ROOT / "layout_task_activation_v1/task_scene_registration_v1"
        dependencies = {layout["scene"]: layout["scene_sha256"]}
        dependencies.update(json.loads((SCENE_ROOT / "station_geometry_v2.json").read_text())["source_hashes"])
        for relative in ("room_demo_v6/room.usda", "room_demo_v6/composition.json", "room_preview_plan_seed42.json"):
            source = registry_root / relative
            if not source.is_file():
                raise TaskError("Original local composition source missing: " + str(source), kind="load", reason="scene_asset_missing")
        verify_manifest(dict(source_hashes=dependencies))
        generated = composition_cache_root(self.cache_root)
        if self.cache_root is not None and long_task_cache_root(output) != generated:
            raise ValueError('Long Task cache requires output in the same owned package')
        generated.mkdir(exist_ok=True)
        cache = generated / ("short_task_" + uuid.uuid4().hex)
        cache.mkdir(exist_ok=False)
        source = cache / "placement.json"
        source.write_text(json.dumps(recipe, ensure_ascii=False, indent=2))
        try:
            process = subprocess.run([self.python, str(SCENE_ROOT / "build_selected_room.py"), str(source)],
                cwd=str(SCENE_ROOT), capture_output=True, text=True, check=False, timeout=180)
        except (OSError, subprocess.TimeoutExpired) as exc:
            (cache / "composition_failure.txt").write_text(str(exc))
            raise TaskError("Local CPU USD composition failed: " + str(exc), kind="load", reason="scene_composition_failed") from exc
        (cache / "composition_log.txt").write_text(process.stdout + process.stderr)
        if process.returncode:
            raise TaskError("Local CPU USD composer rejected the scene; evidence: " + str(cache / "composition_log.txt"),
                            kind="load", reason="scene_composition_failed")
        manifest_path = cache / "composition.json"
        if not manifest_path.is_file():
            raise TaskError("CPU composer produced no real scene manifest", kind="load")
        shutil.copytree(cache, output)
        manifest = json.loads((output / "composition.json").read_text())
        manifest["scene"] = str(output / "room.usda")
        manifest["short_task_composition"] = dict(owner="friend_cpu_usd_composer",
            cache=str(cache), initialization_owner="unified_scene", simulation_run=False)
        (output / "composition.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        return manifest


def compose_with_provider(request, provider, output):
    """Explicit local builder boundary; it must return an actual scene manifest.

    provider.compose receives the existing friend's resolved scene recipe.
    It owns asset placement/initialization; the scorer never authors them.
    Remote generate_room.py is deliberately not used by this entry.
    """
    result = plan_scene(request)
    if result["scene_status"] == "unavailable":
        return result
    preview = registry.is_gpu_cloth_preview(result["task_plan"])
    if preview and provider is None:
        raise TaskError("GPU cloth needs an explicit GPU scene owner; the default CPU room composer is incompatible",
                        kind="load", reason="gpu_cloth_scene_owner_required")
    recipe = result["task_plan"] if preview else result["recipe"]
    manifest = (provider or LocalFriendProvider()).compose(deepcopy(recipe), Path(output))
    verify_manifest(manifest)
    if manifest.get("scene_generated") is not True or (not preview and manifest.get("layout_id") != result["actual_layout_id"]):
        raise TaskError("Scene provider did not compose the selected real layout", kind="binding")
    registry.bind(result["task_plan"], manifest)
    return manifest
