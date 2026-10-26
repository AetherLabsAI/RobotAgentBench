"""Bounded shutdown of an owned worker (including shell-launcher descendants)."""
import os
import signal
import subprocess
from time import monotonic, sleep

# Use the existing task-worker grace for all Isaac workers, not a shorter
# execution-only path. These are shutdown waits, never episode time limits.
GRACE_SECONDS = 60
TERMINATE_SECONDS = 5
KILL_SECONDS = 5


def shutdown_worker(process, *, process_group=None, grace_seconds=GRACE_SECONDS,
                    terminate_seconds=TERMINATE_SECONDS, kill_seconds=KILL_SECONDS):
    if process_group is not None:
        if (os.name != "posix" or type(process_group) is not int or process_group <= 1
                or process_group != process.pid or process_group == os.getpgrp()):
            raise ValueError("Refusing to signal an unowned worker process group")
    started = monotonic()
    report = dict(pid=process.pid, process_group=process_group,
                  grace_seconds=grace_seconds, forced=False,
                  terminate_sent=False, kill_sent=False, returncode=None,
                  reaped=False, completed=False)

    def group_alive():
        if process_group is None:
            return False
        try:
            os.killpg(process_group, 0)
            return True
        except ProcessLookupError:
            return False

    def signal_worker(sig):
        try:
            if process_group is not None:
                os.killpg(process_group, sig)
            elif sig == signal.SIGTERM:
                process.terminate()
            else:
                process.kill()
            return True
        except ProcessLookupError:
            return False

    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        report["forced"] = True
        report["reason"] = "grace_expired"
    else:
        if group_alive():
            report["forced"] = True
            report["reason"] = "launcher_exited_with_descendants"

    if report["forced"]:
        deadline = monotonic() + terminate_seconds
        report["terminate_sent"] = signal_worker(signal.SIGTERM)
        try:
            process.wait(timeout=terminate_seconds)
        except subprocess.TimeoutExpired:
            pass
        # A shell launcher can exit before its worker. Keep the same TERM
        # grace for those descendants, then clean up the owned group as well.
        while group_alive() and monotonic() < deadline:
            sleep(min(0.05, max(0.0, deadline - monotonic())))
        if group_alive() or process.returncode is None:
            report["kill_sent"] = signal_worker(signal.SIGKILL)
        try:
            process.wait(timeout=kill_seconds)
        except subprocess.TimeoutExpired:
            report["reason"] = "worker_not_reaped_after_kill"

    report.update(returncode=process.returncode, reaped=process.returncode is not None,
                  completed=process.returncode == 0 and not report["forced"],
                  elapsed_seconds=monotonic() - started)
    if process.returncode not in (None, 0) and not report.get("reason"):
        report["reason"] = "nonzero_exit"
    return report
