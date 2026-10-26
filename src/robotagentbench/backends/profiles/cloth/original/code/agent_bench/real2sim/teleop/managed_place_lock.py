"""Fail-closed ownership handoff for the shared managed-place runtime lock.

Top-level runners own the ``flock``.  Their child ``make_compare3`` /
``init_task`` / ``init_place`` processes cannot reacquire that lock, so they
inherit a random token written while the parent holds it.  Merely setting the
old boolean environment flag is intentionally insufficient.

Both lock paths are scoped to the resolved scene root (``REAL2SIM_DEMO_ROOT``),
because the state they guard — ``scene_room.usda``, ``playback_traj.npz``, the
``cam_preview`` render intermediates — all lives under that root.  Two lanes on
one host with different scene roots touch disjoint state and therefore get
disjoint locks; two runners on the SAME root still serialize exactly as before.
(The pre-2026-08-05 path was the host-global
``/tmp/real2sim_place_overnight.lock``, which starved co-tenant lanes on l1
even though they shared nothing but the machine.)
"""
from __future__ import annotations

import fcntl
import os
import re
import secrets
import stat

from machine_paths import DEMO_ROOT


def _scene_scope() -> str:
    """Return a readable filesystem-safe label for the resolved scene root."""
    root = os.path.realpath(str(DEMO_ROOT))
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", root).strip("_")


PLACE_LOCK = f"/tmp/real2sim_place_{_scene_scope()}.lock"
# The scene-transaction lock the pick/throw batch runners and make_compare3
# hold while mutating/reading the working scene.  Same scope, same reasoning.
SCENE_TRANSACTION_LOCK = f"/tmp/real2sim_scene_transaction_{_scene_scope()}.lock"
LOCK_TOKEN_ENV = "REAL2SIM_PLACE_LOCK_TOKEN"
LOCK_FD_ENV = "REAL2SIM_PLACE_LOCK_FD"
_TOKEN_PATTERN = re.compile(r"[0-9a-f]{64}")
_FD_PATTERN = re.compile(r"(?:0|[1-9][0-9]{0,9})")


def _read_token(stream) -> str:
    stream.seek(0)
    return stream.read().strip()


def _environment_fd(environment) -> int | None:
    raw = environment.get(LOCK_FD_ENV, "")
    if not isinstance(raw, str) or _FD_PATTERN.fullmatch(raw) is None:
        return None
    return int(raw)


def inherited_lock_stream(path=PLACE_LOCK, environment=None):
    """Duplicate and verify an inherited open-file-description lock.

    The duplicate is important: unlike a liveness-only token check, it keeps
    the kernel ``flock`` held if the parent runner is SIGKILLed while this child
    is still rendering.  ``flock`` on a duplicate of the owning open file
    description succeeds; a spoofed, separately opened descriptor is rejected
    while another owner holds the path, or safely acquires the lock itself when
    the path is currently unlocked.
    """
    environment = os.environ if environment is None else environment
    token = environment.get(LOCK_TOKEN_ENV, "")
    descriptor = _environment_fd(environment)
    if (not isinstance(token, str)
            or _TOKEN_PATTERN.fullmatch(token) is None
            or descriptor is None):
        return None
    duplicate = None
    try:
        duplicate = os.dup(descriptor)
        os.set_inheritable(duplicate, True)
        descriptor_stat = os.fstat(duplicate)
        path_stat = os.stat(path)
        if (not stat.S_ISREG(descriptor_stat.st_mode)
                or descriptor_stat.st_dev != path_stat.st_dev
                or descriptor_stat.st_ino != path_stat.st_ino):
            raise ValueError("inherited descriptor does not identify lock path")
        raw_token = os.pread(duplicate, 4096, 0).decode("ascii").strip()
        if raw_token != token:
            raise ValueError("inherited lock token mismatch")
        # Same-OFD inherited descriptors succeed here.  A distinct descriptor
        # cannot borrow somebody else's readable token because its flock fails.
        fcntl.flock(duplicate, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return os.fdopen(duplicate, "a+", encoding="ascii")
    except (BlockingIOError, OSError, UnicodeError, ValueError):
        if duplicate is not None:
            os.close(duplicate)
        return None


def inherited_lock_is_valid(path=PLACE_LOCK, environment=None) -> bool:
    """Return true only for a token-matched descriptor that can retain the lock."""
    environment = os.environ if environment is None else environment
    stream = inherited_lock_stream(path, environment)
    if stream is None:
        return False
    try:
        return True
    finally:
        stream.close()


def acquire_lock(path=PLACE_LOCK, environment=None):
    """Acquire the lock, publish a fresh child token, and return its open stream."""
    environment = os.environ if environment is None else environment
    # ``a+`` is important: opening with ``w`` would truncate the current
    # owner's handoff token before a competing flock attempt fails.
    stream = open(path, "a+", encoding="ascii")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except Exception:
        stream.close()
        raise
    token = secrets.token_hex(32)
    stream.seek(0)
    stream.truncate()
    stream.write(token + "\n")
    stream.flush()
    os.fsync(stream.fileno())
    os.set_inheritable(stream.fileno(), True)
    environment[LOCK_TOKEN_ENV] = token
    environment[LOCK_FD_ENV] = str(stream.fileno())
    # Do not let a caller-supplied legacy boolean masquerade as ownership.
    environment.pop("REAL2SIM_PLACE_LOCK_HELD", None)
    return stream


def lock_pass_fds(environment=None) -> tuple[int, ...]:
    """Return the managed lock fd for ``Popen(pass_fds=...)`` when it is open."""
    environment = os.environ if environment is None else environment
    descriptor = _environment_fd(environment)
    if descriptor is None:
        return ()
    try:
        os.fstat(descriptor)
    except OSError:
        return ()
    return (descriptor,)


def acquire_or_inherit_lock(path=PLACE_LOCK, environment=None):
    """Retain an inherited OFD lock, otherwise acquire the lock directly."""
    environment = os.environ if environment is None else environment
    stream = inherited_lock_stream(path, environment)
    if stream is not None:
        environment.pop("REAL2SIM_PLACE_LOCK_HELD", None)
        environment[LOCK_FD_ENV] = str(stream.fileno())
        os.set_inheritable(stream.fileno(), True)
        return stream
    environment.pop(LOCK_TOKEN_ENV, None)
    environment.pop(LOCK_FD_ENV, None)
    return acquire_lock(path, environment)
