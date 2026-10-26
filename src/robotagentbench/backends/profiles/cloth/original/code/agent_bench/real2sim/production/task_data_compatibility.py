#!/usr/bin/env python3
"""Separate task-semantic success from current-codebase compatibility.

Task success answers whether the recorded action completed the Sheet task.
Compatibility answers whether that action/state pair can be published as data
from the current House runtime.  Old or unknown physics provenance never turns
an otherwise successful action into a semantic failure: it requests a
same-action replay on the current baseline instead.
"""

from __future__ import annotations

import argparse
import functools
import json
import re
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from .benchmark_eval_registry import CATALOG
    from .git_provenance import SCHEMA as GIT_SOURCE_SCHEMA
    from .git_provenance import source_revision
    from .task_success import evaluate_task_success, success_contract
except ImportError:  # Direct script execution.
    from benchmark_eval_registry import CATALOG
    from git_provenance import SCHEMA as GIT_SOURCE_SCHEMA
    from git_provenance import source_revision
    from task_success import evaluate_task_success, success_contract


REPO = Path(__file__).resolve().parents[2]
SCHEMA = "real2sim.task_data_compatibility.v1"
AUDIT_SCHEMA = "real2sim.task_data_compatibility_audit.v1"
DEFAULT_ENVIRONMENT = "canonical_house_default"
SHARED_RUNTIME_PATHS = (
    "arena_scene/v1_2/demo_sim/scene_room_baseline.usda",
    "real2sim/production/benchmark_eval_registry.py",
    "real2sim/production/house_task_environment.py",
    "real2sim/production/task_success.py",
    "real2sim/teleop/robot_positions.py",
)
DISPOSITIONS = (
    "current",
    "same_action_replay_required",
    "evidence_upgrade_and_replay_required",
    "owner_checker_and_replay_required",
    "invalid_action_or_identity",
    "reject_semantic_failure",
    "no_data",
)
TASK_KEY = re.compile(r"^benchmark_(\d+)_(\d+)$")
REMOTE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
REMOTE_PATH = re.compile(r"^/[A-Za-z0-9_./-]+$")

REMOTE_INVENTORY_PROGRAM = r'''\
import collections
import json
import re
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
source_checkout = Path(sys.argv[2]) if len(sys.argv) > 2 else None
task_pattern = re.compile(r"benchmark_(\d+)_(\d+)$")

def load(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}

rows = []
unknown = []
for task_directory in sorted(root.glob("benchmark_*")):
    if not task_directory.is_dir():
        continue
    match = task_pattern.fullmatch(task_directory.name)
    if not match:
        unknown.append(task_directory.name)
        continue
    task_id = f"{int(match.group(1))}.{int(match.group(2))}"
    counters = collections.Counter()
    validation = collections.Counter()
    schemas = collections.Counter()
    direct_claims = collections.Counter()
    canonical_episodes = [
        path for path in (task_directory / "trajectories").glob("episode_*")
        if path.is_dir()
    ]
    direct_episodes = [
        path for path in task_directory.glob("episode_*")
        if path.is_dir()
    ]
    episodes = sorted([*canonical_episodes, *direct_episodes])
    for episode in episodes:
        for name in (
            "action.npz", "full_state.npz", "pick_contact_trace.npz",
            "place_metrics.npz", "task_metrics.npz", "metrics.npz",
            "task_metrics.json", "metrics.json", "episode.json",
            "result.json", "validation.json", "score.json",
        ):
            counters[name] += int((episode / name).is_file())
        episode_json = load(episode / "episode.json")
        result_json = load(episode / "result.json")
        validation_json = load(episode / "validation.json")
        score_json = load(episode / "score.json")
        identity_values = [episode_json, result_json]
        config = result_json.get("config")
        if isinstance(config, dict):
            identity_values.append(config)
            request = config.get("task_owned_request")
            if isinstance(request, dict):
                identity_values.append(request)
        for value in identity_values:
            claim = value.get("benchmark_subtask")
            key = value.get("task_key")
            if claim:
                direct_claims[str(claim)] += 1
            elif key:
                direct_claims[str(key)] += 1
        validation[str(validation_json.get("status", "missing"))] += 1
        for value in (episode_json, result_json, validation_json, score_json):
            if value.get("schema"):
                schemas[str(value["schema"])] += 1
        counters["old_score_pass"] += int(
            score_json.get("ok") is True
            and score_json.get("failed_gates") in (None, [])
        )
    rows.append({
        "benchmark_subtask": task_id,
        "directory": task_directory.name,
        "episode_count": len(episodes),
        "episode_layout_counts": {
            "canonical_trajectories": len(canonical_episodes),
            "direct_legacy": len(direct_episodes),
        },
        "artifact_counts": dict(counters),
        "direct_claim_counts": dict(direct_claims),
        "validation_status_counts": dict(validation),
        "schema_counts": dict(schemas),
    })

source = {}
if source_checkout is not None and (source_checkout / ".git").exists():
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source_checkout,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=source_checkout, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    source = {
        "checkout": str(source_checkout),
        "git_commit": head.stdout.strip(),
        "worktree_dirty": bool(status.stdout.strip()),
        "changed_path_count": len(status.stdout.splitlines()),
    }

print(json.dumps({
    "root": str(root),
    "source_checkout": source,
    "unknown_task_directories": unknown,
    "tasks": rows,
}, sort_keys=True))
'''


def _task_sort_key(value: str) -> tuple[int, int]:
    return tuple(map(int, str(value).split(".")))  # type: ignore[return-value]


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _normalize_claim(value: Any) -> str | None:
    text = str(value or "").strip()
    match = TASK_KEY.fullmatch(text)
    if match:
        return f"{int(match.group(1))}.{int(match.group(2))}"
    match = re.fullmatch(r"(\d+)\.(\d+)", text)
    if match:
        return f"{int(match.group(1))}.{int(match.group(2))}"
    return None


def _direct_task_claims(directory: Path) -> set[str]:
    claims: set[str] = set()
    for name in ("episode.json", "result.json"):
        value = _json(directory / name)
        sources: list[Mapping[str, Any]] = [value]
        config = value.get("config")
        if isinstance(config, Mapping):
            sources.append(config)
            request = config.get("task_owned_request")
            if isinstance(request, Mapping):
                sources.append(request)
        for source in sources:
            for field in (
                "benchmark_subtask", "benchmark_task_id",
                "task_key", "benchmark_task_key",
            ):
                claim = _normalize_claim(source.get(field))
                if claim is not None:
                    claims.add(claim)
    return claims


def _directory_class(directory: Path) -> str:
    text = directory.as_posix()
    if re.search(
        r"/task_owned_50x4/benchmark_\d+_\d+/"
        r"trajectories/episode_[^/]+$",
        text,
    ):
        return "canonical_task_owned"
    if "/quarantine/" in text:
        return "quarantine"
    if "/_superseded" in text or "superseded" in text:
        return "superseded"
    if "staging" in text:
        return "staging"
    if "canary" in text:
        return "canary"
    if "/work/" in text or "/output/" in text:
        return "work_or_output"
    return "historical_or_experiment"


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


@functools.lru_cache(maxsize=None)
def _expected_source_paths(task_id: str) -> tuple[str, ...]:
    paths = set(SHARED_RUNTIME_PATHS)
    checker = success_contract(task_id).get("checker")
    if checker:
        path = str(checker).partition(":")[0]
        if path:
            paths.add(path)
    return tuple(sorted(paths))


def current_compatibility_source_revision(task_id: str) -> dict[str, Any]:
    """Record the Git-owned inputs needed to call one episode current."""
    if str(task_id) not in CATALOG:
        raise KeyError(f"unknown benchmark task {task_id}")
    return source_revision(_expected_source_paths(str(task_id)), repo=REPO)


@functools.lru_cache(maxsize=None)
def _current_source_state(task_id: str) -> dict[str, Any]:
    paths = _expected_source_paths(task_id)
    missing = [path for path in paths if not (REPO / path).is_file()]
    untracked = []
    for path in paths:
        if _git("ls-files", "--error-unmatch", "--", path).returncode:
            untracked.append(path)
    status = _git(
        "status", "--porcelain=v1", "--untracked-files=all", "--", *paths
    )
    dirty = sorted({
        line[3:]
        for line in status.stdout.splitlines()
        if len(line) > 3
    })
    commit_result = _git("rev-parse", "HEAD")
    released = not missing and not untracked and not dirty
    return {
        "status": "released" if released else "unreleased_worktree",
        "git_commit": commit_result.stdout.strip(),
        "expected_paths": list(paths),
        "missing_paths": missing,
        "untracked_paths": sorted(untracked),
        "dirty_paths": dirty,
    }


def _walk_sources(value: Any, location: str = ""):
    if isinstance(value, Mapping):
        if value.get("schema") == GIT_SOURCE_SCHEMA:
            yield location or "/", dict(value)
        for key, child in value.items():
            yield from _walk_sources(child, location + "/" + str(key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_sources(child, location + f"/{index}")


@functools.lru_cache(maxsize=4096)
def _recorded_source_current_failure(
    commit: str, paths: tuple[str, ...]
) -> str | None:
    if not commit or not paths:
        return "invalid_recorded_source"
    if _git("cat-file", "-e", f"{commit}^{{commit}}").returncode:
        return "recorded_commit_unavailable"
    result = _git("diff", "--quiet", commit, "HEAD", "--", *paths)
    if result.returncode == 1:
        return "recorded_source_changed"
    if result.returncode:
        return "recorded_source_comparison_failed"
    return None


def _source_compatibility(
    task_id: str, metadata: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    expected = set(_expected_source_paths(task_id))
    records = []
    covered: set[str] = set()
    failures = []
    for filename, value in metadata.items():
        for location, source in _walk_sources(value):
            paths_value = source.get("paths")
            paths = tuple(
                sorted(map(str, paths_value))
                if isinstance(paths_value, list) else ()
            )
            commit = str(source.get("git_commit", ""))
            covered.update(paths)
            failure = _recorded_source_current_failure(commit, paths)
            if source.get("worktree_dirty") is True:
                failure = failure or "recorded_source_was_dirty"
            if failure:
                failures.append(f"{filename}{location}: {failure}")
            records.append({
                "file": filename,
                "location": location,
                "git_commit": commit,
                "paths": list(paths),
                "failure": failure,
            })
    missing_expected = sorted(expected - covered)
    if not records:
        status = "missing"
    elif failures:
        status = "changed_or_unusable"
    elif missing_expected:
        status = "incomplete"
    else:
        status = "current"
    return {
        "status": status,
        "expected_paths": sorted(expected),
        "missing_expected_paths": missing_expected,
        "records": records,
        "reasons": failures,
    }


def _environment_compatibility(
    metadata: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    scopes = sorted({
        str(value["environment_scope"])
        for value in metadata.values()
        if value.get("environment_scope") not in (None, "")
    })
    if scopes == [DEFAULT_ENVIRONMENT]:
        status = "declared_current_default"
    elif scopes:
        status = "declared_nondefault_or_conflicting"
    else:
        status = "not_recorded"
    return {"status": status, "declared_scopes": scopes}


def evaluate_episode_compatibility(
    benchmark_subtask: str,
    episode_directory: Path | str,
) -> dict[str, Any]:
    """Evaluate semantic success and current-runtime reuse separately."""
    task_id = str(benchmark_subtask)
    directory = Path(episode_directory).resolve()
    if task_id not in CATALOG:
        raise KeyError(f"unknown benchmark task {task_id}")
    if not directory.is_dir():
        raise ValueError(f"episode directory is missing: {directory}")
    metadata = {
        name: _json(directory / name)
        for name in (
            "episode.json", "result.json", "validation.json",
            "action_derivation.json", "score.json",
        )
    }
    semantic = evaluate_task_success(task_id, directory)
    source = _source_compatibility(task_id, metadata)
    current_source = _current_source_state(task_id)
    environment = _environment_compatibility(metadata)
    contract = success_contract(task_id)
    components = semantic.get("components", {})
    identity_ok = (
        components.get("task_identity", {}).get("status") == "passed"
    )
    action_ok = (
        components.get("action_state_integrity", {}).get("status") == "passed"
    )
    action_reusable = bool(identity_ok and action_ok)

    if semantic.get("status") == "failed":
        disposition = "reject_semantic_failure"
    elif not action_reusable:
        disposition = "invalid_action_or_identity"
    elif contract.get("status") != "registered":
        disposition = "owner_checker_and_replay_required"
    elif semantic.get("status") != "passed":
        disposition = "evidence_upgrade_and_replay_required"
    elif (
        source["status"] == "current"
        and current_source["status"] == "released"
        and environment["status"] == "declared_current_default"
    ):
        disposition = "current"
    else:
        disposition = "same_action_replay_required"

    return {
        "schema": SCHEMA,
        "benchmark_subtask": task_id,
        "owner": CATALOG[task_id]["task"].get("owner", ""),
        "episode_directory": str(directory),
        "task_success": semantic,
        "source_compatibility": source,
        "current_source_release": current_source,
        "environment_compatibility": environment,
        "checker_registration_status": contract.get("status"),
        "action_reusable_for_same_action_replay": action_reusable,
        "disposition": disposition,
    }


def _task_disposition(
    episode_dispositions: Counter[str], *, checker_status: str
) -> str:
    if not episode_dispositions:
        return "no_data"
    for value in (
        "reject_semantic_failure",
        "invalid_action_or_identity",
        "owner_checker_and_replay_required",
        "evidence_upgrade_and_replay_required",
        "same_action_replay_required",
    ):
        if episode_dispositions.get(value):
            return value
    if checker_status != "registered":
        return "owner_checker_and_replay_required"
    return "current"


def _task_episode_directories(task_directory: Path) -> list[Path]:
    """Find canonical cores and older direct replay-artifact cores."""
    return sorted([
        *(
            path
            for path in (task_directory / "trajectories").glob("episode_*")
            if path.is_dir()
        ),
        *(
            path
            for path in task_directory.glob("episode_*")
            if path.is_dir()
        ),
    ])


def audit_root(root: Path | str) -> dict[str, Any]:
    """Audit one canonical task-owned root and always report all 103 tasks."""
    directory = Path(root).resolve()
    task_rows = []
    episode_rows = []
    disposition_counts: Counter[str] = Counter()
    success_counts: Counter[str] = Counter()
    for task_id in sorted(CATALOG, key=_task_sort_key):
        contract = success_contract(task_id)
        task_directory = directory / (
            "benchmark_" + task_id.replace(".", "_")
        )
        episodes = _task_episode_directories(task_directory)
        episode_dispositions: Counter[str] = Counter()
        task_success_counts: Counter[str] = Counter()
        source_counts: Counter[str] = Counter()
        environment_counts: Counter[str] = Counter()
        reusable_actions = 0
        for episode in episodes:
            value = evaluate_episode_compatibility(task_id, episode)
            disposition = str(value["disposition"])
            success_status = str(value["task_success"]["status"])
            episode_dispositions[disposition] += 1
            disposition_counts[disposition] += 1
            task_success_counts[success_status] += 1
            success_counts[success_status] += 1
            source_counts[str(value["source_compatibility"]["status"])] += 1
            environment_counts[
                str(value["environment_compatibility"]["status"])
            ] += 1
            reusable_actions += int(
                value["action_reusable_for_same_action_replay"]
            )
            episode_rows.append({
                "benchmark_subtask": task_id,
                "episode": episode.name,
                "episode_layout": (
                    "canonical_trajectories"
                    if episode.parent.name == "trajectories"
                    else "direct_legacy"
                ),
                "task_success_status": success_status,
                "disposition": disposition,
                "action_reusable_for_same_action_replay": value[
                    "action_reusable_for_same_action_replay"
                ],
                "source_compatibility_status": value[
                    "source_compatibility"
                ]["status"],
                "environment_compatibility_status": value[
                    "environment_compatibility"
                ]["status"],
                "reasons": value["task_success"].get("reasons", []),
            })
        task_disposition = _task_disposition(
            episode_dispositions,
            checker_status=str(contract.get("status")),
        )
        task_rows.append({
            "benchmark_subtask": task_id,
            "owner": CATALOG[task_id]["task"].get("owner", ""),
            "checker_registration_status": contract.get("status"),
            "checker": contract.get("checker"),
            "episode_count": len(episodes),
            "task_success_counts": {
                status: task_success_counts.get(status, 0)
                for status in ("passed", "failed", "unverified")
            },
            "compatibility_disposition": task_disposition,
            "episode_disposition_counts": dict(episode_dispositions),
            "source_compatibility_counts": dict(source_counts),
            "environment_compatibility_counts": dict(environment_counts),
            "same_action_replayable_episode_count": reusable_actions,
            "owner_requirement": contract.get("owner_requirement"),
        })
    return {
        "schema": AUDIT_SCHEMA,
        "root": str(directory),
        "catalog_task_count": len(task_rows),
        "episode_count": len(episode_rows),
        "tasks_with_data": sum(row["episode_count"] > 0 for row in task_rows),
        "task_success_counts": {
            status: success_counts.get(status, 0)
            for status in ("passed", "failed", "unverified")
        },
        "episode_disposition_counts": {
            status: disposition_counts.get(status, 0)
            for status in DISPOSITIONS if status != "no_data"
        },
        "task_disposition_counts": dict(Counter(
            row["compatibility_disposition"] for row in task_rows
        )),
        "tasks": task_rows,
        "episodes": episode_rows,
    }


def audit_workspace(roots: Sequence[Path | str]) -> dict[str, Any]:
    """Audit every directly task-bound action file below local data roots.

    Directory names and old runtime labels are intentionally not used as task
    identity.  Unbound files remain visible in aggregate but cannot be scored
    until an owner supplies a direct Sheet task claim.
    """
    resolved_roots = [Path(root).resolve() for root in roots]
    action_paths = sorted({
        path.resolve()
        for root in resolved_roots
        for path in root.rglob("action.npz")
        if path.is_file()
    })
    task_counters: dict[str, dict[str, Counter[str] | int]] = {
        task_id: {
            "candidate_count": 0,
            "action_state_pair_count": 0,
            "task_success": Counter(),
            "disposition": Counter(),
            "directory_class": Counter(),
            "reason": Counter(),
        }
        for task_id in CATALOG
    }
    unbound = 0
    conflicting_claims = 0
    old_claims: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    scored = 0
    for action_path in action_paths:
        directory = action_path.parent
        directory_class = _directory_class(directory)
        class_counts[directory_class] += 1
        claims = _direct_task_claims(directory)
        if not claims:
            unbound += 1
            continue
        current_claims = claims.intersection(CATALOG)
        old_claims.update(claims - set(CATALOG))
        if len(current_claims) != 1 or len(claims) != 1:
            conflicting_claims += 1
            continue
        task_id = next(iter(current_claims))
        value = evaluate_episode_compatibility(task_id, directory)
        row = task_counters[task_id]
        row["candidate_count"] = int(row["candidate_count"]) + 1
        paired = (directory / "full_state.npz").is_file()
        row["action_state_pair_count"] = (
            int(row["action_state_pair_count"]) + int(paired)
        )
        success_status = str(value["task_success"]["status"])
        disposition = str(value["disposition"])
        task_success_counter = row["task_success"]
        disposition_counter = row["disposition"]
        directory_counter = row["directory_class"]
        reason_counter = row["reason"]
        assert isinstance(task_success_counter, Counter)
        assert isinstance(disposition_counter, Counter)
        assert isinstance(directory_counter, Counter)
        assert isinstance(reason_counter, Counter)
        task_success_counter[success_status] += 1
        disposition_counter[disposition] += 1
        directory_counter[directory_class] += 1
        reason_counter.update(map(
            str, value["task_success"].get("reasons", ())
        ))
        scored += 1

    task_rows = []
    for task_id in sorted(CATALOG, key=_task_sort_key):
        values = task_counters[task_id]
        success_counter = values["task_success"]
        disposition_counter = values["disposition"]
        directory_counter = values["directory_class"]
        reason_counter = values["reason"]
        assert isinstance(success_counter, Counter)
        assert isinstance(disposition_counter, Counter)
        assert isinstance(directory_counter, Counter)
        assert isinstance(reason_counter, Counter)
        task_rows.append({
            "benchmark_subtask": task_id,
            "owner": CATALOG[task_id]["task"].get("owner", ""),
            "checker_registration_status": success_contract(task_id).get(
                "status"
            ),
            "candidate_count": values["candidate_count"],
            "action_state_pair_count": values["action_state_pair_count"],
            "task_success_counts": {
                status: success_counter.get(status, 0)
                for status in ("passed", "failed", "unverified")
            },
            "disposition_counts": dict(disposition_counter),
            "directory_class_counts": dict(directory_counter),
            "reason_counts": dict(reason_counter.most_common()),
        })
    return {
        "schema": "real2sim.workspace_task_data_compatibility_audit.v1",
        "roots": [str(root) for root in resolved_roots],
        "catalog_task_count": len(task_rows),
        "action_file_count": len(action_paths),
        "direct_current_task_candidate_count": scored,
        "tasks_with_direct_current_candidates": sum(
            int(row["candidate_count"]) > 0 for row in task_rows
        ),
        "unbound_action_file_count": unbound,
        "conflicting_or_mixed_claim_count": conflicting_claims,
        "old_or_noncurrent_claim_counts": dict(old_claims),
        "directory_class_counts": dict(class_counts),
        "task_success_counts": dict(sum(
            (
                Counter(row["task_success_counts"])
                for row in task_rows
            ),
            Counter(),
        )),
        "disposition_counts": dict(sum(
            (
                Counter(row["disposition_counts"])
                for row in task_rows
            ),
            Counter(),
        )),
        "tasks": task_rows,
    }


def audit_remote_task_owned_root(
    host: str,
    root: str,
    *,
    source_checkout: str | None = None,
) -> dict[str, Any]:
    """Read one owner/server root without modifying its worktree or data."""
    if not REMOTE_NAME.fullmatch(str(host)):
        raise ValueError(f"invalid remote host name: {host!r}")
    for value in (root, source_checkout):
        if value is not None and not REMOTE_PATH.fullmatch(str(value)):
            raise ValueError(f"invalid absolute remote path: {value!r}")
    command = [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
        str(host), "python3", "-", str(root),
    ]
    if source_checkout is not None:
        command.append(str(source_checkout))
    completed = subprocess.run(
        command,
        input=REMOTE_INVENTORY_PROGRAM,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )
    if completed.returncode:
        raise RuntimeError(
            f"remote inventory failed on {host}: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    try:
        observed = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("remote inventory returned invalid JSON") from exc
    observed_by_task = {
        str(row["benchmark_subtask"]): row
        for row in observed.get("tasks", ())
        if isinstance(row, Mapping) and row.get("benchmark_subtask")
    }
    task_rows = []
    for task_id in sorted(CATALOG, key=_task_sort_key):
        row = dict(observed_by_task.get(task_id, {}))
        contract = success_contract(task_id)
        artifact_counts = row.get("artifact_counts", {})
        episode_count = int(row.get("episode_count", 0))
        metrics_count = sum(
            int(artifact_counts.get(name, 0))
            for name in (
                "task_metrics.npz", "metrics.npz",
                "task_metrics.json", "metrics.json",
            )
        )
        if not episode_count:
            disposition = "no_data"
        elif contract.get("status") != "registered":
            disposition = "owner_checker_and_replay_required"
        elif contract.get("adapter") == "metrics" and not metrics_count:
            disposition = "evidence_upgrade_and_replay_required"
        else:
            disposition = "same_action_replay_and_current_check_required"
        task_rows.append({
            "benchmark_subtask": task_id,
            "owner": CATALOG[task_id]["task"].get("owner", ""),
            "checker_registration_status": contract.get("status"),
            "episode_count": episode_count,
            "artifact_counts": artifact_counts,
            "direct_claim_counts": row.get("direct_claim_counts", {}),
            "episode_layout_counts": row.get(
                "episode_layout_counts", {}
            ),
            "old_validation_status_counts": row.get(
                "validation_status_counts", {}
            ),
            "old_score_pass_count": int(
                artifact_counts.get("old_score_pass", 0)
            ),
            "compatibility_disposition": disposition,
        })
    return {
        "schema": "real2sim.remote_task_data_inventory.v1",
        "host": str(host),
        "root": str(root),
        "read_only": True,
        "source_checkout": observed.get("source_checkout", {}),
        "catalog_task_count": len(task_rows),
        "tasks_with_data": sum(row["episode_count"] > 0 for row in task_rows),
        "episode_count": sum(row["episode_count"] for row in task_rows),
        "unknown_task_directories": observed.get(
            "unknown_task_directories", []
        ),
        "task_disposition_counts": dict(Counter(
            row["compatibility_disposition"] for row in task_rows
        )),
        "tasks": task_rows,
    }


def _emit(value: Mapping[str, Any], output: Path | None) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check")
    check.add_argument("task")
    check.add_argument("episode", type=Path)
    check.add_argument("--output", type=Path)
    audit = subparsers.add_parser("audit")
    audit.add_argument("root", type=Path)
    audit.add_argument("--output", type=Path)
    workspace = subparsers.add_parser("audit-workspace")
    workspace.add_argument("roots", nargs="+", type=Path)
    workspace.add_argument("--output", type=Path)
    remote = subparsers.add_parser("audit-remote")
    remote.add_argument("host")
    remote.add_argument("root")
    remote.add_argument("--source-checkout")
    remote.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "check":
        value = evaluate_episode_compatibility(args.task, args.episode)
    elif args.command == "audit":
        value = audit_root(args.root)
    elif args.command == "audit-workspace":
        value = audit_workspace(args.roots)
    else:
        value = audit_remote_task_owned_root(
            args.host, args.root, source_checkout=args.source_checkout
        )
    _emit(value, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
