"""Called by the unified scene producer, not by scoring to guess object purpose."""
from copy import deepcopy
from .registration import CATEGORIES, ASSETS, inventory, prim_path

DEFAULT_EMPTY = {'paper_ball': 'paper', 'empty_can': 'can', 'paper_cup': 'cup'}
# Source project's approved semantic mapping, not runtime color recognition.
SOURCE_BIN_ROLES = (
    ('ordinary_trash_bin', '/World/Assets/TrashBinRed'),
    ('recycling_trash_bin', '/World/Assets/TrashBinTeal'),
)


def mark_object(obj, *, empty_packaging=False):
    """Explicit creation context distinguishes shelf product from empty template."""
    result = deepcopy(obj)
    if result['id'] in ASSETS:
        category = DEFAULT_EMPTY.get(result['id'])
        if empty_packaging:
            if result['id'] != 'zbar_t':
                raise ValueError('Empty packaging context requires packaging asset')
            category = 'snack'
        existing = result.get('trash_category', category)
        if existing != category:
            raise ValueError('Conflicting scene-owned garbage semantics')
        result['trash_category'] = category
    return result


def bin_roles(stage):
    """Bind existing authored semantic bin roles; verify existence, never create."""
    result = []
    for role, path in SOURCE_BIN_ROLES:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsActive():
            raise ValueError('Missing approved scene bin: ' + role)
        result.append(dict(role=role, path=path))
    return result


def annotate_manifest(manifest, *, empty_packaging_paths, destinations):
    """Metadata-only copy with explicit producer-supplied instance provenance."""
    result = deepcopy(manifest)
    paths = list(empty_packaging_paths)
    if len(paths) != len(set(paths)):
        raise ValueError('Repeated packaging instance declaration')
    expected = {prim_path(p) for p in paths}
    actual = {o['path'] for o in result['objects'] if o['id'] == 'zbar_t'}
    if not expected <= actual:
        raise ValueError('Packaging declaration references missing/wrong asset')
    result['objects'] = [mark_object(o, empty_packaging=o['path'] in expected) for o in result['objects']]
    prior = result.get('task_destinations', [])
    other = [r for r in prior if r['role'] not in {x[1] for x in CATEGORIES.values()}]
    garbage = [r for r in prior if r['role'] in {x[1] for x in CATEGORIES.values()}]
    if garbage and garbage != destinations:
        raise ValueError('Conflicting existing bin role mapping')
    result['task_destinations'] = other + deepcopy(destinations)
    result['trash_semantics_version'] = 1
    inventory(result)
    return result
