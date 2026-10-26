"""Guarded install of the five explicitly authorized shared wiring files.

Existing records and bytes are retained under this task. No asset/history file,
default physics setting, unrelated task implementation or live job is touched.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROPOSED = HERE/"integration_v1/proposed"
SHARED = (
    "short_tasks/registry.py", "short_tasks/agent_probe_v1/prepare_task.py",
    "openend/tasks/scene_requirements/room_request.py",
    "openend/tasks/scene_requirements/planner_v2.py",
    "openend/tasks/scene_requirements/build_selected_room.py",
)


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    from short_tasks import registry
    if HERE != ROOT/"short_tasks/tasks/microwave_heat_take" or HERE.is_symlink():
        raise ValueError("Wrong deployment root")
    receipt = HERE/"integration_v1/installed.json"
    if receipt.exists():
        raise ValueError("Existing installation receipt must not be overwritten")
    expected = json.loads((HERE/"integration_v1/shared_before.json").read_text())
    changes, before = {}, {}
    for rel in SHARED:
        target, candidate = ROOT/rel, PROPOSED/rel
        if target.is_symlink() or digest(target) != expected[rel]:
            raise ValueError("Concurrent or unreviewed source change: "+rel)
        text = candidate.read_text(); ast.parse(text, filename=rel)
        changes[rel] = text.encode()
        before[rel] = digest(target)
    # Only current registered source records, never historical copies.
    records = [ROOT/"short_tasks/scene_sources.json"]
    skipped_legacy = []
    for name in registry.TASK_NAMES:
        try:
            records.append(registry.task_directory(name)/"component_sources.json")
        except ModuleNotFoundError as exc:
            # Some pre-existing optional families are absent from this full
            # source installation. Do not install them or mask their status.
            skipped_legacy.append(dict(task=name, reason=str(exc)))
    record_updates = {}
    after = {rel: hashlib.sha256(content).hexdigest() for rel,content in changes.items()}
    for p in dict.fromkeys(records):
        if not p.exists():
            continue
        if p.is_symlink() or not p.resolve().is_relative_to(ROOT):
            raise ValueError("Invalid current source record")
        data = json.loads(p.read_text()); modified = False
        for field in ("sources", "files_sha256", "source_files"):
            mapping = data.get(field)
            if not isinstance(mapping, dict):
                continue
            for rel in SHARED:
                if rel in mapping:
                    if mapping[rel] != before[rel]:
                        raise ValueError("Existing source record is already inconsistent: "+str(p))
                    mapping[rel] = after[rel]; modified = True
        if modified:
            record_updates[str(p.relative_to(ROOT))] = (json.dumps(data, indent=2, ensure_ascii=False)+"\n").encode()
    changes.update(record_updates)
    before.update({rel:digest(ROOT/rel) for rel in record_updates})
    backup = HERE/"integration_v1/before"
    backup.mkdir(parents=True, exist_ok=False)
    staging = HERE/"integration_v1/staged"
    staging.mkdir()
    for rel, content in changes.items():
        p, dst = ROOT/rel, backup/rel
        dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dst)
        candidate = staging/rel; candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_bytes(content); candidate.chmod(stat.S_IMODE(p.stat().st_mode))
    # Last compare immediately before replacing exact allowlisted files.
    if any(digest(ROOT/rel) != value for rel,value in before.items()):
        raise ValueError("Concurrent source edit during preflight")
    committed = []
    try:
        for rel in changes:
            os.replace(staging/rel, ROOT/rel); committed.append(rel)
    except BaseException:
        # Only our own just-installed bytes may be restored. Never clobber a
        # concurrent editor while handling an installation error.
        for rel in reversed(committed):
            if digest(ROOT/rel) == hashlib.sha256(changes[rel]).hexdigest():
                shutil.copy2(backup/rel, ROOT/rel)
        raise
    pins = json.loads((HERE/"source_manifest.json").read_text())["files"]
    pins.update(after)
    for p in HERE.glob("*.py"):
        if not p.name.startswith("test_"):
            pins[str(p.relative_to(ROOT))] = digest(p)
    for rel in ("short_tasks/scene.py", "short_tasks/runtime.py",
                "openend/tasks/meal_tray/microwave_initialization.py",
                "openend/tasks/meal_tray/rotary_component/geometry.py",
                "evaluation/tasks/microwave_take/isaac_process.py",
                "short_tasks/tasks/interaction_progress_v1/native.py",
                "short_tasks/tasks/interaction_progress_v1/geometry.py",
                "short_tasks/tasks/interaction_progress_v1/contact.py"):
        pins[rel] = digest(ROOT/rel)
    (HERE/"component_sources.json").write_text(json.dumps(dict(
        schema="openend.short_task.component_sources.v1", sources=pins), indent=2)+"\n")
    value = dict(status="installed_pending_validation", shared_python=list(SHARED),
        current_source_records_updated=list(record_updates), source_before=before,
        source_after={rel:digest(ROOT/rel) for rel in changes},
        task_component_sources=len(pins), authorized_scope="standalone microwave Short Task wiring",
        assets_changed=False, physics_control_camera_timeout_defaults_changed=False)
    value["preexisting_missing_optional_families"] = skipped_legacy
    receipt.write_text(json.dumps(value, indent=2)+"\n")
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
