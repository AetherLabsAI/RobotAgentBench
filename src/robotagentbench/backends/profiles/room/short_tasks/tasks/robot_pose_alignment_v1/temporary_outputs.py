"""Opt-in output relocation only; no scene, score or simulation settings."""
import json
import os
from pathlib import Path

EXPECTED = Path('/dev/shm/short_task_heads_fixed_v3_20261013')


def root():
    value = os.environ.get('ARENA_INITIAL_HEAD_OUTPUT_ROOT')
    if not value:
        return None
    path = Path(value)
    if path != EXPECTED or path.is_symlink() or path.resolve() != EXPECTED:
        raise ValueError('Only the explicitly approved head-gallery output root is allowed')
    marker = json.loads((path/'owner.json').read_text())
    if marker != {'purpose':'short_task_fixed_v3_initial_heads', 'gpus':[2,4]}:
        raise ValueError('Wrong temporary output owner')
    return path


def permits(path):
    parent = root()
    return parent is not None and Path(path).resolve().is_relative_to(parent) and Path(path).resolve() != parent


def generated(default):
    parent = root()
    return parent/'generated_rooms' if parent else default


def cutlery(default, case):
    parent = root()
    return parent/'cutlery_layers'/case/'random.usda' if parent else default
