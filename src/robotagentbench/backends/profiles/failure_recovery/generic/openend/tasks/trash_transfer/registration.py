"""Pure preflight. Scene semantics, never location/name heuristics, select trash.

No USD edits, resets, scoring events or global registry mutation occur here.
"""
from copy import deepcopy
from hashlib import sha256
import re

TYPE = 'trash_transfer'
CATEGORIES = {
    'paper': ('paper_ball', 'ordinary_trash_bin', '18-1'),
    'snack': ('zbar_t', 'ordinary_trash_bin', '18-2'),
    'can': ('empty_can', 'recycling_trash_bin', '18-3'),
    'cup': ('paper_cup', 'ordinary_trash_bin', '18-4'),
}
ASSETS = frozenset(row[0] for row in CATEGORIES.values())


def prim_path(value):
    if not isinstance(value, str) or not re.fullmatch(r'/World(?:/[A-Za-z_][A-Za-z0-9_]*)+', value):
        raise ValueError('Canonical concrete scene prim path required')
    return value


def validate_parameters(parameters):
    if not isinstance(parameters, dict):
        raise ValueError('Explicit garbage parameters required')
    if set(parameters) == {'auto_bind'} and parameters['auto_bind'] is True:
        return
    if set(parameters) != {'target_path', 'category', 'destination_path'}:
        raise ValueError('Use auto_bind=true or an exact garbage binding')
    prim_path(parameters['target_path']); prim_path(parameters['destination_path'])
    if parameters['category'] not in CATEGORIES:
        raise ValueError('Unknown garbage category')
    if parameters['target_path'] == parameters['destination_path']:
        raise ValueError('Garbage and bin must be distinct')


def inventory(manifest):
    if manifest.get('trash_semantics_version') != 1:
        raise ValueError('Unified scene must emit explicit trash semantics v1')
    objects = manifest.get('objects')
    if not isinstance(objects, list):
        raise ValueError('Scene instance inventory required')
    seen = set(); trash = []
    for obj in objects:
        path = prim_path(obj['path'])
        if path in seen:
            raise ValueError('Duplicate scene instance path: ' + path)
        seen.add(path)
        if obj.get('id') in ASSETS and 'trash_category' not in obj:
            raise ValueError('Missing per-instance trash classification: ' + path)
        category = obj.get('trash_category')
        if category is None:
            continue
        if category not in CATEGORIES or obj.get('id') != CATEGORIES[category][0]:
            raise ValueError('Trash category and asset identity disagree: ' + path)
        trash.append(obj)
    destinations = manifest.get('task_destinations', [])
    if not isinstance(destinations, list):
        raise ValueError('Explicit destination role declarations required')
    bins = {}; used_paths = set()
    for row in destinations:
        role = row.get('role')
        if role not in {v[1] for v in CATEGORIES.values()}:
            continue
        path = prim_path(row['path'])
        if role in bins or path in used_paths:
            raise ValueError('Ambiguous garbage bin role/instance')
        bins[role] = path; used_paths.add(path)
    for obj in trash:
        role = CATEGORIES[obj['trash_category']][1]
        if role not in bins:
            raise ValueError('Missing garbage bin role: ' + role)
        if bins[role] == obj['path']:
            raise ValueError('Garbage is its own destination')
    return sorted(trash, key=lambda o: o['path']), bins


def resolve_object(parameters, manifest):
    validate_parameters(parameters)
    if 'auto_bind' in parameters:
        raise ValueError('Expand auto registration before binding native readers')
    objects, bins = inventory(manifest)
    matches = [o for o in objects if o['path'] == parameters['target_path']]
    if len(matches) != 1:
        raise ValueError('Exact marked garbage instance required')
    obj = matches[0]; category = obj['trash_category']
    if parameters['category'] != category or parameters['destination_path'] != bins[CATEGORIES[category][1]]:
        raise ValueError('Requested garbage category/bin differs from scene semantics')
    return obj


def expand_request(request, manifest):
    """Atomic deterministic expansion; caller owns lifecycle and score instances."""
    result = deepcopy(request)
    tasks = result.get('tasks')
    if not isinstance(tasks, list):
        raise ValueError('Explicit task list required')
    expanded = []; auto_count = 0
    for task in tasks:
        if task.get('type') != TYPE:
            expanded.append(task); continue
        validate_parameters(task.get('parameters'))
        if 'auto_bind' not in task['parameters']:
            resolve_object(task['parameters'], manifest)
            expanded.append(task); continue
        auto_count += 1
        if auto_count > 1:
            raise ValueError('Only one automatic garbage declaration per episode')
        objects, bins = inventory(manifest)
        if not objects:
            raise ValueError('Garbage requested but no marked garbage exists')
        prefix = task.get('instance_id')
        if not isinstance(prefix, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', prefix):
            raise ValueError('Safe garbage group ID required')
        for obj in objects:
            category = obj['trash_category']
            # Identity does not depend on array order, pose or current support.
            suffix = sha256(obj['path'].encode()).hexdigest()[:20]
            expanded.append(dict(instance_id=f'{prefix}_{category}_{suffix}', type=TYPE,
                parameters=dict(target_path=obj['path'], category=category,
                                destination_path=bins[CATEGORIES[category][1]])))
    ids = set(); claimed = set()
    for task in expanded:
        instance = task['instance_id']
        if instance in ids:
            raise ValueError('Duplicate task instance ID')
        ids.add(instance)
        target = task['parameters'].get('target_path')
        if target is not None:
            if target in claimed:
                raise ValueError('Duplicate target, including legacy cup registration: ' + target)
            claimed.add(target)
    result['tasks'] = expanded
    return result
