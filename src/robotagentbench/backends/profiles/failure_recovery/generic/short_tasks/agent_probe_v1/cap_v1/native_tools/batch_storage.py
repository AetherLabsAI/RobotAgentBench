"""Explicit trusted policy workspaces; no model-selected storage root."""
from pathlib import Path
import re

ROOT = Path('/mnt/agent-bench-data/lingjun/astra_all14_20261015')
LONG_ROLLOUT_ROOT = (Path(__file__).resolve().parents[2] /
    'long_task_eval_setup_v1/evening_progress/trajectory_followup/native_validation')


def permits(path):
    raw = Path(path)
    target = raw.resolve()
    if ROOT.is_dir() and ROOT.resolve() == ROOT and target != ROOT and target.is_relative_to(ROOT):
        return True
    # This task's existing non-root sandbox may mount only a single fresh
    # policy_workspace directly inside one admitted run directory.
    return (raw == target and LONG_ROLLOUT_ROOT.is_dir()
        and LONG_ROLLOUT_ROOT.resolve() == LONG_ROLLOUT_ROOT
        and target.parent.parent == LONG_ROLLOUT_ROOT
        and target.name == 'policy_workspace'
        and re.fullmatch(r'[a-zA-Z0-9_-]{8,100}', target.parent.name) is not None)
