"""Trusted, task-local initial checkpoints for the requested 20-case API audit.

Only catalogued IDs are accepted. HTTP cannot provide a path or module.
This selects initial state; it never changes physical parameters or scoring.
"""
import json
import hashlib
from pathlib import Path

TASK = Path(__file__).resolve().parent
CATALOG = TASK / 'references/api20_20261002/catalog.json'
DEFAULT = TASK / 'references/l1_beef_camv1_001569_20260913'

def checkpoint_for(case_id=None):
    if case_id is None:
        return DEFAULT
    if not isinstance(case_id, str) or len(case_id) != 14 or not case_id.startswith('episode_') or not case_id[8:].isdigit():
        raise ValueError('Invalid microwave replay case ID')
    catalog = json.loads(CATALOG.read_text())
    if catalog.get('schema') != 'microwave_put.api20_cases.v1' or case_id not in catalog['cases']:
        raise ValueError('Uncatalogued microwave replay case')
    raw = CATALOG.parent / case_id
    path = raw.resolve(strict=True)
    if path.parent != CATALOG.parent.resolve() or raw.is_symlink() or path.name != case_id:
        raise ValueError('Replay checkpoint escaped the task catalog')
    if hashlib.sha256((path/'source_manifest.json').read_bytes()).hexdigest() != catalog['cases'][case_id]['checkpoint_manifest_sha256']:
        raise ValueError('Replay checkpoint manifest changed after selection')
    return path
