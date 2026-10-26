"""Subprocess runner that cannot leave a timed-out managed process tree alive."""
from __future__ import annotations

import os
import signal
import subprocess

from managed_place_lock import lock_pass_fds


TERMINATE_GRACE_S = 10


def _text(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _merge(partial, final):
    partial = _text(partial)
    final = _text(final)
    if not partial:
        return final
    if not final or final.startswith(partial):
        return final or partial
    return partial + final


def _descendant_pids(root_pid):
    """Snapshot Linux descendants without adding a runtime dependency."""
    parent_by_pid = {}
    try:
        entries = os.scandir("/proc")
    except OSError:
        return []
    with entries:
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                with open(
                        f"/proc/{entry.name}/stat", encoding="ascii") as stream:
                    raw = stream.read()
                # comm is parenthesized and may contain spaces or ')'.
                fields = raw[raw.rfind(")") + 2:].split()
                parent_by_pid[int(entry.name)] = int(fields[1])
            except (IndexError, OSError, ValueError):
                continue
    descendants = []
    frontier = [int(root_pid)]
    seen = set(frontier)
    while frontier:
        parents = set(frontier)
        frontier = []
        for pid, parent in parent_by_pid.items():
            if parent in parents and pid not in seen:
                seen.add(pid)
                descendants.append(pid)
                frontier.append(pid)
    return descendants


def _signal_pids(pids, sig):
    for pid in pids:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def _stop_process_tree(process, stdout, stderr, isolated_group):
    known_descendants = _descendant_pids(process.pid)
    if isolated_group:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    else:
        # A lock-inheriting child deliberately stays in its managed parent's
        # process group, so an outer overnight killpg cannot be escaped by
        # nested init/docker sessions.  Kill only this child's snapshotted tree
        # for its own timeout; never signal the parent's whole group.
        _signal_pids([process.pid, *known_descendants], signal.SIGTERM)
    try:
        final_stdout, final_stderr = process.communicate(
            timeout=TERMINATE_GRACE_S)
    except subprocess.TimeoutExpired as exc:
        stdout = _merge(stdout, exc.output)
        stderr = _merge(stderr, exc.stderr)
        if isolated_group:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            current = _descendant_pids(process.pid)
            pids = list(dict.fromkeys(
                [process.pid, *known_descendants, *current]))
            _signal_pids(pids, signal.SIGKILL)
        final_stdout, final_stderr = process.communicate()
    return _merge(stdout, final_stdout), _merge(stderr, final_stderr)


def run_argv(command, timeout=None, *, cwd=None, env=None):
    """Run argv in a new session, preserving subprocess.run timeout semantics."""
    pass_fds = lock_pass_fds(env)
    isolated_group = not pass_fds
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        # Managed descendants stay in the outer runner's group.  The inherited
        # OFD lock prevents concurrency if a parent alone is SIGKILLed, while an
        # outer group timeout now reaches every nested host subprocess.
        start_new_session=isolated_group,
        pass_fds=pass_fds,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        stdout, stderr = _stop_process_tree(
            process, exc.output, exc.stderr, isolated_group)
        raise subprocess.TimeoutExpired(
            command, timeout, output=stdout, stderr=stderr) from None
    return subprocess.CompletedProcess(
        command, process.returncode, stdout, stderr)
