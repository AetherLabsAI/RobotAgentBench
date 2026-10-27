"""Exact static inputs omitted by the original Python-only cloth export.

The original entry resolves its station through the Sheet-backed registry.
Keep both original JSON documents intact; do not duplicate the station values
or replace the registry. The archival acquisition manifest remains pinned and
unchanged beside the derived, additive runtime manifest.
"""
from copy import deepcopy
import hashlib
from pathlib import Path


PREFIX = 'code/agent_bench/real2sim/production/'
FILES = {
    PREFIX + 'kitchen_task_components_v1.json': {
        'sha256': '50bb09dbda5fed581d2a8ee96d71112e435b4d8c47abc354530bd7e78a4abcae',
        'bytes': 8408,
    },
    PREFIX + 'google_sheet_tasks_current.json': {
        'sha256': 'f645c7a0376c4104d9a26d528a81356e45da28c8fa2b1c9ce425d40ada0e4b82',
        'bytes': 48052,
    },
}
PROVENANCE = {
    'repository': 'https://github.com/Wr3ck1Am/agent_bench',
    'retrieved_checkout_commit': '1b19fb0e20f5b32dd0b4b88ca5e6c8ef767d1910',
    'last_data_commit': 'b24fe21f48a77820a24c9c74291d211be6cdbc39',
    'tracked_files_clean': True,
    'git_blobs': {
        'real2sim/production/kitchen_task_components_v1.json': '3665fcea19fa0f297548c88222049a20354fcb5f',
        'real2sim/production/google_sheet_tasks_current.json': '55c08f6f32f60c889d3ec7c7826643b59558f64f',
    },
    'unchanged_reader_sha256': '48b87b9de95e48cfd2fe1d860c066effc8adb9d1b90f3e2dcbe039188684b9fa',
    'original_source_edited': False,
    'physics_scoring_modified': False,
}


def require_inputs(original_root, pins, manifest):
    """Require the added exact inputs in both the owner and source manifests."""
    for relative, expected in FILES.items():
        path = Path(original_root) / relative
        if pins.get(str(path)) != expected['sha256']:
            raise ValueError('Original cloth static input is not pinned: ' + relative)
        if manifest.get(relative) != expected:
            raise ValueError('Original cloth static input differs from source manifest: ' + relative)


def extend_manifest(document, original_root, pins, *, upstream_sha256):
    """Add only reviewed source data, after checking its installed bytes."""
    if not isinstance(document.get('files'), dict):
        raise ValueError('Original cloth acquisition manifest required')
    derived = deepcopy(document)
    if 'static_dependency_delivery' in derived:
        raise ValueError('Use the original cloth acquisition manifest, not a derived copy')
    for relative, expected in FILES.items():
        path = Path(original_root) / relative
        if path.resolve() != path or not path.is_file():
            raise ValueError('Original cloth static input must be canonical: ' + relative)
        if (path.stat().st_size != expected['bytes']
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected['sha256']
                or pins.get(str(path)) != expected['sha256']):
            raise ValueError('Original cloth static input changed: ' + relative)
        old = derived['files'].get(relative)
        if old is not None and old != expected:
            raise ValueError('Conflicting original cloth source manifest entry: ' + relative)
        derived['files'][relative] = dict(expected)
    derived['static_dependency_delivery'] = dict(
        schema='robotagentbench.cloth_static_dependencies.v1',
        upstream_manifest_sha256=upstream_sha256, provenance=deepcopy(PROVENANCE),
        added_files=list(FILES))
    require_inputs(original_root, pins, derived['files'])
    return derived
