"""Preserve the original trash selector's asset identities after relocation.

The original adapter compares resolved path names, not only asset contents.
Its frozen catalogue keeps container-only logical paths. Scene restoration may
use installed Runtime paths, but the generated candidate metadata must retain
the catalogue identity. No original catalogue, adapter or asset is modified.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

_CATALOG = 'short_task_ContextReasoning/asset_pools/trash_selection_v1/catalog.json'
_CATALOG_SHA = '76a4d2314357c39774d5fd1a0864961cf323c5f1a9e1e60763889754cb78d7f3'
_LOGICAL_ASSETS = Path('/home/lingjun/robot_agent_arena/api-protocol/all_assets')


def _verify(path, expected, *, canonical=True):
    path = Path(path)
    if (not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected)
            or not path.is_file() or (canonical and path.resolve(strict=True) != path)):
        raise ValueError('Exact installed trash asset identity required: ' + str(path))
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != expected:
        raise ValueError('Original trash asset bytes differ: ' + str(path))


def restore_trash_entries(manifest, *, source, runtime, metadata):
    """Return a copied manifest and exact alias evidence, without source writes."""
    source, runtime = Path(source), Path(runtime)
    if (source.resolve(strict=True) != source or runtime.resolve(strict=True) != runtime
            or not source.is_relative_to(runtime) or metadata.get('profile') != 'room'):
        raise ValueError('Canonical installed room Runtime required')
    if metadata.get('source_files', {}).get(_CATALOG) != _CATALOG_SHA:
        raise ValueError('Original trash catalogue pin differs')
    _verify(source / _CATALOG, _CATALOG_SHA)
    catalog = json.loads((source / _CATALOG).read_text())
    if catalog.get('schema') != 'context_reasoning.trash_assets.v1':
        raise ValueError('Original trash catalogue schema required')
    approved = {}
    for asset in catalog['assets']:
        identifier = asset['all_assets_id']
        if (not isinstance(identifier, str) or not re.fullmatch('context_trash_[a-z0-9_]+', identifier)
                or identifier in approved):
            raise ValueError('Unique original trash asset identifier required')
        relative = 'all_assets/' + identifier + '/asset.usda'
        logical = _LOGICAL_ASSETS / identifier / 'asset.usda'
        if asset['entry'] != str(logical):
            raise ValueError('Original trash catalogue logical entry differs')
        pin = asset['sha256']
        library = 'libraries/current_assets/' + relative
        if (metadata.get('source_files', {}).get(relative) != pin
                or metadata.get('library_files', {}).get(library) != pin):
            raise ValueError('Trash source/library metadata differs from original catalogue')
        _verify(source / relative, pin)
        _verify(runtime / library, pin)
        # This is the original read-only container alias, not a host path to
        # create. Its bytes must independently match both installed bindings.
        _verify(logical, pin, canonical=False)
        approved[identifier] = (source / relative, logical, pin)
    result = deepcopy(manifest)
    rows = [row for row in result['objects'] if row.get('task_id') == 'select_trash']
    if len(rows) not in (4, 5) or len({row['instance_id'] for row in rows}) != len(rows):
        raise ValueError('Original four/five distinct trash candidates required')
    evidence = []
    for row in rows:
        if row['id'] not in approved:
            raise ValueError('Candidate is absent from the original trash catalogue')
        installed, logical, pin = approved[row['id']]
        entry = Path(row['entry'])
        if entry != installed and entry != logical:
            raise ValueError('Candidate entry is not its exact installed or original identity')
        _verify(entry, pin, canonical=entry == installed)
        row['entry'] = str(logical)
        evidence.append(dict(instance_id=row['instance_id'], asset=row['id'],
                             installed_entry=str(installed), original_entry=str(logical), sha256=pin))
    return result, dict(catalog_sha256=_CATALOG_SHA, entries=evidence,
                        all_catalog_aliases_verified=len(approved),
                        source_assets_modified=False, physics_scoring_modified=False)
