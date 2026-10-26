"""Git-backed provenance for repository-owned production inputs.

Repository source is versioned by Git.  Runtime evidence records the commit,
the repository-relative paths that participated, and whether those paths had
uncommitted changes.  It deliberately does not create a second, per-file
checksum system beside Git.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
SCHEMA = "agent_bench.git_source.v1"


def _run_git(*args: str, repo: Path = REPO) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    # Porcelain status uses leading spaces as data (for example `` M path``).
    # Remove only record terminators so callers can parse Git output exactly.
    return completed.stdout.rstrip("\r\n")


def repo_relative(path: Path | str, *, repo: Path = REPO) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo / candidate
    try:
        return candidate.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"path is outside the Git repository: {path}") from exc


def source_revision(
    paths: Iterable[Path | str], *, repo: Path = REPO,
) -> dict[str, Any]:
    """Describe tracked source using one Git revision plus path names.

    Dirty paths are reported, not rejected.  This keeps provenance honest
    during development without turning version metadata into a runtime lock.
    """
    relative = sorted({repo_relative(path, repo=repo) for path in paths})
    if not relative:
        raise ValueError("source revision requires at least one path")
    missing = [name for name in relative if not (repo / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "source revision paths are missing: " + ", ".join(missing))
    for name in relative:
        _run_git("ls-files", "--error-unmatch", "--", name, repo=repo)
    preferred_remote = "agent-bench" if _run_git(
        "remote", repo=repo
    ).splitlines().count("agent-bench") else "origin"
    repository = _run_git(
        "remote", "get-url", preferred_remote, repo=repo)
    status = _run_git(
        "status", "--porcelain=v1", "--untracked-files=all", "--",
        *relative, repo=repo,
    )
    dirty_paths = sorted({
        line[3:] for line in status.splitlines() if len(line) > 3
    })
    return {
        "schema": SCHEMA,
        "repository": repository,
        "git_commit": _run_git("rev-parse", "HEAD", repo=repo),
        "paths": relative,
        "worktree_dirty": bool(dirty_paths),
        "dirty_paths": dirty_paths,
    }


def source_revision_failure(
    value: Any, *, expected_paths: Iterable[Path | str], repo: Path = REPO,
) -> str | None:
    """Validate the shape of recorded Git provenance without pinning HEAD."""
    if not isinstance(value, Mapping) or value.get("schema") != SCHEMA:
        return "missing_or_invalid_git_source_revision"
    expected = sorted({repo_relative(path, repo=repo) for path in expected_paths})
    if value.get("paths") != expected:
        return "git_source_paths_mismatch"
    if not isinstance(value.get("repository"), str) or not value["repository"]:
        return "git_source_repository_missing"
    if not isinstance(value.get("git_commit"), str) or not value["git_commit"]:
        return "git_source_commit_missing"
    if not isinstance(value.get("worktree_dirty"), bool):
        return "git_source_worktree_state_missing"
    dirty = value.get("dirty_paths")
    if not isinstance(dirty, list) or any(not isinstance(path, str) for path in dirty):
        return "git_source_dirty_paths_invalid"
    if bool(dirty) != value["worktree_dirty"]:
        return "git_source_worktree_state_inconsistent"
    return None


def source_revision_current_failure(
    value: Any,
    *,
    expected_paths: Iterable[Path | str] | None = None,
    repo: Path = REPO,
) -> str | None:
    """Return why recorded source no longer matches the current Git tree.

    This comparison stays inside Git: the recorded commit is compared with
    ``HEAD`` only for the participating paths.  An unrelated documentation or
    GUI commit therefore does not invalidate an accepted trajectory/render,
    while a relevant source change does.  Dirty participating paths cannot be
    called current because they are not represented by either Git revision.
    """

    if expected_paths is None:
        if not isinstance(value, Mapping):
            return "missing_or_invalid_git_source_revision"
        recorded_paths = value.get("paths")
        if not isinstance(recorded_paths, list):
            return "git_source_paths_mismatch"
        expected_paths = recorded_paths
    expected = tuple(expected_paths)
    failure = source_revision_failure(
        value, expected_paths=expected, repo=repo)
    if failure is not None:
        return failure
    if value["worktree_dirty"]:
        return "recorded_git_source_was_dirty"
    try:
        current = source_revision(expected, repo=repo)
    except (FileNotFoundError, RuntimeError, ValueError):
        return "current_git_source_unavailable"
    if current["worktree_dirty"]:
        return "current_git_source_is_dirty"

    commit = str(value["git_commit"])
    try:
        _run_git("cat-file", "-e", f"{commit}^{{commit}}", repo=repo)
    except RuntimeError:
        return "recorded_git_commit_unavailable"
    completed = subprocess.run(
        ["git", "diff", "--quiet", commit, "HEAD", "--", *current["paths"]],
        cwd=repo,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode == 1:
        return "git_source_changed_since_capture"
    if completed.returncode != 0:
        return "git_source_comparison_failed"
    return None
