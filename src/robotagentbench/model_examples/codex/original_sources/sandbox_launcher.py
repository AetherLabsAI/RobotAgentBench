"""Trusted HOST launcher. Never expose this host process as an agent shell.

Only agent_workspace is mounted. No caller-selected mount/image/network flags.
This is code execution isolation, NOT the observation/action broker.
"""
import argparse
import json
import os
from pathlib import Path
import re
import selectors
import stat
import subprocess
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parent
TASK_ROOT = ROOT.parent
IMAGE = 'sha256:4ccf7e950d48f9b5477292bbc123b676fdbad039c6d51573d969008d8ffd5dc6'
LABEL = 'arena.gpt-policy-sandbox'
OUTPUT_LIMIT = 1024 * 1024


def validate_workspace(workspace):
    WORKSPACE = Path(workspace)
    from .batch_storage import permits
    if not (WORKSPACE.resolve().is_relative_to(TASK_ROOT.resolve()) or permits(WORKSPACE)):
        raise ValueError('Task-owned sandbox workspace required')
    if not WORKSPACE.is_dir() or WORKSPACE.is_symlink() or WORKSPACE.resolve() != WORKSPACE:
        raise ValueError('Expected the real project/agent_workspace directory')
    if os.getuid() == 0 or WORKSPACE.stat().st_uid != os.getuid():
        raise ValueError('Use the non-root workspace owner')
    # A bind mount cannot distinguish an externally imported hardlink. Do not
    # seed this folder with linked credentials, special files or host sockets.
    for parent, dirs, files in os.walk(WORKSPACE, followlinks=False):
        for name in dirs + files:
            path = Path(parent) / name
            mode = path.lstat()
            if stat.S_ISLNK(mode.st_mode):
                continue  # Resolves within the container namespace, not the host.
            if stat.S_ISREG(mode.st_mode) and mode.st_nlink == 1:
                continue
            if stat.S_ISDIR(mode.st_mode):
                continue
            raise ValueError('Workspace contains a special file or hardlinked file')


def docker(*args, **kwargs):
    # Never forward caller environment (keys/proxy settings) into the container.
    return subprocess.run(['docker', *args], capture_output=True, text=True,
                          timeout=30, check=True, **kwargs).stdout


def create_arguments(name, command, workspace):
    WORKSPACE = Path(workspace)
    validate_workspace(WORKSPACE)
    if not command or not all(isinstance(v, str) and '\0' not in v for v in command):
        raise ValueError('Expected a command argument list')
    return ['create', '--name', name, '--label', LABEL + '=v1',
        '--runtime', 'runc', '--pull', 'never', '--network', 'none', '--ipc', 'none',
        '--cgroupns', 'private', '--read-only', '--cap-drop', 'ALL',
        '--security-opt', 'no-new-privileges=true', '--user', f'{os.getuid()}:{os.getgid()}',
        '--pids-limit', '64', '--memory', '512m', '--memory-swap', '512m', '--cpus', '1',
        '--ulimit', 'nofile=256:256', '--ulimit', 'fsize=67108864:67108864',
        '--ulimit', 'core=0:0', '--log-driver', 'none', '--hostname', 'policy-sandbox',
        '--tmpfs', '/tmp:rw,noexec,nosuid,nodev,size=32m,mode=1777',
        '--mount', f'type=bind,source={WORKSPACE},target=/workspace,bind-recursive=disabled',
        '--workdir', '/workspace', '--entrypoint', '/usr/bin/env', IMAGE,
        '-i', 'PATH=/usr/local/bin:/usr/bin:/bin', 'HOME=/workspace', 'TMPDIR=/tmp',
        'LANG=C.UTF-8', 'PYTHONDONTWRITEBYTECODE=1',
        'OPENBLAS_NUM_THREADS=1', 'OMP_NUM_THREADS=1', *command]
