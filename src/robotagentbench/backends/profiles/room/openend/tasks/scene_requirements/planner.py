"""Task-driven inventory proposal, NOT a scene builder or runtime request.

Pure Python. No USD, scoring, physics, spawning, or source-location sampling.
The unified scene adapter must place and validate every proposed instance later.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import re


PORTABLE = {
    'wallet_transfer': ('wallet', 'fixed_cpu_bag'),
    'glasses_transfer': ('glasses_case', 'fixed_cpu_bag'),
    'remote_transfer': ('remote_tv', 'tv_cabinet'),
    'powerbank_transfer': ('power_bank', 'fixed_cpu_bag'),
    'water_bottle_transfer': ('water_bottle', 'fixed_cpu_bag'),
    'umbrella_transfer': ('umbrella', 'fixed_cpu_bag'),
    'book_transfer': ('book_red', 'small_table'),
    'cap_transfer': ('baseball_cap', 'coat_rack_hooks'),
    'notebook_transfer': ('notebook_green', 'coffee_table_drawer_then_close'),
}
SINGLETONS = {asset for asset, _ in PORTABLE.values()} - {'book_red'}
INFRASTRUCTURE = ('fixed_cpu_bag', 'ordinary_trash_bin', 'recycling_trash_bin', 'storage_basket')
TRASH = {
    'paper': ('paper_ball', 'ordinary_trash_bin'),
    'snack': ('zbar_t', 'ordinary_trash_bin'),
    'can': ('empty_can', 'recycling_trash_bin'),
    'cup': ('paper_cup', 'ordinary_trash_bin'),
}
CLOTH_POOL = tuple('cloth_clear_' + name for name in (
    'citrus_can', 'berry_can', 'water_bottle', 'juice_bottle', 'small_bottle',
    'bear', 'rabbit', 'penguin', 'cat', 'car', 'train', 'block', 'house',
    'robot', 'duck', 'glasses_case', 'pouch', 'tape_measure', 'eraser', 'memo_pad'))
KNOWN_ASSETS = SINGLETONS | {'book_red'} | set(CLOTH_POOL) | {x[0] for x in TRASH.values()}
LOCATIONS = {'sofa', 'small_table', 'coffee_table', 'coffee_table.objects',
             'coffee_table.cloth', 'coffee_table.wooden_tray', 'coffee_table.drawer', 'snack_shelf'}
TRASH_ASSETS = {x[0] for x in TRASH.values()}


class RequirementError(ValueError):
    pass


def fields(value, allowed, required=()):
    if not isinstance(value, dict) or set(value) - set(allowed) or set(required) - set(value):
        raise RequirementError('Unknown or missing fields: ' + repr(value))


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', value):
        raise RequirementError('Safe explicit instance ID required')
    return value


def count(value):
    if type(value) is not int or value < 1:
        raise RequirementError('Count must be a positive integer')
    return value


def location(value):
    if value is not None and value not in LOCATIONS:
        raise RequirementError('Unknown source region: ' + repr(value))
    return value


def overlap(a, b):
    """None is scene-selected/unknown, never proof of visible separation."""
    return a is None or b is None or a == b or a.startswith(b + '.') or b.startswith(a + '.')


def rng_for(seed, task_id):
    # Independent streams: adding unrelated tasks must not reshuffle this task.
    digest = hashlib.sha256(f'{seed}:{task_id}'.encode()).digest()
    return random.Random(int.from_bytes(digest, 'big'))


def selections(task, rng):
    kind = task['type']
    pool = sorted(TRASH) if kind == 'trash_transfer' else sorted(CLOTH_POOL)
    if 'items' in task:
        if 'count' in task or 'categories' in task:
            raise RequirementError('Exact items cannot be mixed with random count/categories')
        if not isinstance(task['items'], list) or not task['items']:
            raise RequirementError('Nonempty exact item list required')
        result = []
        for row in task['items']:
            fields(row, ('kind', 'count'), ('kind',))
            exact_pool = pool if kind == 'trash_transfer' else [*pool, 'book_red']
            if row['kind'] not in exact_pool:
                raise RequirementError('Unknown requested item: ' + repr(row['kind']))
            result.extend([row['kind']] * count(row.get('count', 1)))
        if kind == 'cloth_clear' and len(result) > 4:
            raise RequirementError('Cloth-clear supports at most four explicit targets')
        return result
    categories = task.get('categories', [])
    if (not isinstance(categories, list) or any(not isinstance(x, str) for x in categories)
            or len(set(categories)) != len(categories) or any(x not in pool for x in categories)):
        raise RequirementError('Distinct known categories required')
    if kind != 'trash_transfer' and 'categories' in task:
        raise RequirementError('Cloth uses exact items, not garbage categories')
    n = count(task['count']) if 'count' in task else rng.randint(max(2, len(categories)), 4)
    if len(categories) > n:
        raise RequirementError('Requested count cannot cover all required categories')
    if kind == 'trash_transfer':
        # A category restriction means ONLY those categories; each appears >= once.
        result = categories[:] + rng.choices(categories or pool, k=n-len(categories))
        rng.shuffle(result)
        return result
    if n > len(pool):
        raise RequirementError('Random cloth selection exceeds distinct library pool; use exact items')
    return rng.sample(pool, n)


def compile_plan(request):
    fields(request, ('schema', 'seed', 'tasks', 'extras'), ('schema', 'seed', 'tasks'))
    if request['schema'] != 'openend.scene_requirements.request.v1':
        raise RequirementError('Unsupported proposal schema; not a runtime config')
    if type(request['seed']) is not int or not 0 <= request['seed'] < 2**32:
        raise RequirementError('Unsigned 32-bit seed required')
    if not isinstance(request['tasks'], list) or not isinstance(request.get('extras', []), list):
        raise RequirementError('Tasks/extras must be lists')
    objects, tasks, ids, claimed, destinations = [], [], set(), {}, {}

    def reserve_id(value):
        identifier(value)
        if value in ids:
            raise RequirementError('Duplicate task/extra ID: ' + value)
        ids.add(value)

    def need(role, owner):
        destinations.setdefault(role, []).append(owner)

    def add(asset, owner, index, source, category=None, role='target'):
        if asset in SINGLETONS:
            if asset in claimed:
                raise RequirementError(f'Unique object {asset} requested twice: {claimed[asset]} / {owner}')
            claimed[asset] = owner
        ident = f'{owner}--{index:03d}'
        objects.append(dict(instance_id=ident, asset_id=asset, role=role, source=source,
                            source_owner='unified_scene', required_by=[owner] if role == 'target' else [],
                            trash_category=category, pose=None, prim_path=None))
        return ident

    for task in request['tasks']:
        fields(task, ('id', 'type', 'source', 'count', 'items', 'categories', 'destination'), ('id', 'type'))
        reserve_id(task['id']); typ = task['type']; source = location(task.get('source'))
        if 'destination' in task and typ not in ('glasses_transfer','wallet_transfer'):
            raise RequirementError('Destination selection is only for glasses/wallet')
        if typ not in PORTABLE and typ not in ('trash_transfer', 'cloth_clear'):
            raise RequirementError(f'No inventory adapter for {typ!r}; never silently omit a task')
        record = dict(id=task['id'], type=typ, source=source, target_ids=[], destination_roles=[])
        if typ in PORTABLE:
            if typ == 'umbrella_transfer' and source not in (None, 'sofa'):
                raise RequirementError('Umbrella initialization is supported on the sofa only')
            if 'items' in task or 'categories' in task:
                raise RequirementError('Portable tasks use count/source only')
            asset, destination = PORTABLE[typ]
            if typ in ('glasses_transfer','wallet_transfer'):
                destination=task.get('destination','fixed_cpu_bag')
                if destination not in ('fixed_cpu_bag','storage_basket'):raise RequirementError('Unknown portable destination')
                record['parameters']={'destination':destination}
            if asset == 'book_red' and 'count' not in task:
                raise RequirementError('Book count policy not migrated: specify count, no implicit one-book cap')
            n = count(task.get('count', 1))
            if asset in SINGLETONS and n != 1:
                raise RequirementError('Room-wide unique object: ' + asset)
            for i in range(n):
                record['target_ids'].append(add(asset, task['id'], i, source))
            need(destination, task['id']); record['destination_roles'] = [destination]
        else:
            if typ == 'cloth_clear':
                if source not in (None, 'coffee_table.cloth'):
                    raise RequirementError('Cloth-clear must start on the cloth')
                source = record['source'] = 'coffee_table.cloth'
                need('tablecloth_dirty', task['id'])
                need('same_coffee_table_free_space_outside_cloth_and_wooden_tray', task['id'])
            for i, item in enumerate(selections(task, rng_for(request['seed'], task['id']))):
                if typ == 'trash_transfer':
                    asset, destination = TRASH[item]
                    category = item
                else:
                    asset, destination, category = item, 'same_coffee_table_free_space_outside_cloth_and_wooden_tray', None
                record['target_ids'].append(add(asset, task['id'], i, source, category))
                need(destination, task['id'])
                if destination not in record['destination_roles']:
                    record['destination_roles'].append(destination)
            # This describes target membership, not a new runtime success evaluator.
            record['completion_membership'] = 'all_target_ids_not_historical_score_sum'
        record['target_count'] = len(record['target_ids'])
        tasks.append(record)

    for extra in request.get('extras', []):
        fields(extra, ('id', 'asset', 'source', 'count'), ('id', 'asset'))
        reserve_id(extra['id']); asset = extra['asset']; source = location(extra.get('source'))
        if asset not in KNOWN_ASSETS:
            raise RequirementError('No approved extra asset mapping: ' + repr(asset))
        for i in range(count(extra.get('count', 1))):
            add(asset, extra['id'], i, source, role='extra')

    unresolved = validate_semantics(objects, tasks, defer_unknown=True)
    return dict(schema='openend.scene_requirements.plan.v1', seed=request['seed'],
                status='inventory_proposal_only', executable=False,
                infrastructure=[dict(role=r, action='retain_and_bind_existing',
                                     required_by=sorted(set(destinations.get(r, [])))) for r in INFRASTRUCTURE],
                scene_dependencies=[dict(role=r, required_by=sorted(set(v))) for r, v in sorted(destinations.items())
                                    if r not in INFRASTRUCTURE],
                objects=objects, tasks=tasks, unresolved_semantic_pairs=unresolved,
                checks=dict(inventory_structure=True, semantic_scope_resolved=not unresolved,
                            geometry=False, initial_physics=False, runtime_binding=False),
                pending=['unified_scene_placement_and_existing_object_reconciliation',
                         'source_eligibility_and_destination_capacity',
                         'post_placement_semantic_and_visual_disambiguation',
                         'native_initial_acceptance_and_exact_runtime_binding'])


def validate_semantics(objects, tasks, defer_unknown=False):
    """Conservative conflict detection. Metadata never certifies image visibility."""
    unresolved = []
    for task in tasks:
        targets = set(task['target_ids'])
        target_assets = {o['asset_id'] for o in objects if o['instance_id'] in targets}
        for obj in objects:
            if obj['instance_id'] in targets or not overlap(task['source'], obj['source']):
                continue
            # Broad cleanup scope cannot exclude similar objects through a hidden role.
            conflict = (task['type'] == 'cloth_clear' or
                        task['type'] == 'trash_transfer' and obj['asset_id'] in TRASH_ASSETS or
                        task['type'] in PORTABLE and obj['asset_id'] in target_assets)
            if conflict:
                if defer_unknown and (task['source'] is None or obj['source'] is None):
                    unresolved.append([task['id'], obj['instance_id']])
                    continue
                raise RequirementError(f'Ambiguous target scope: {task["id"]} also describes {obj["instance_id"]}; '
                                       'specify disjoint visible sources or revise the explicit target list')
    return unresolved


def verify_placed_inventory(plan, rows, infrastructure_roles):
    """Read-only semantic gate for a FUTURE scene adapter. No physics claims.

    rows must enumerate ALL movable objects, including extras already in a base.
    Furniture and infrastructure are supplied separately by the scene adapter.
    """
    expected = {o['instance_id']: o for o in plan['objects']}
    actual = {}; paths = set()
    for row in rows:
        fields(row, ('instance_id', 'asset_id', 'source', 'prim_path'),
               ('instance_id', 'asset_id', 'source', 'prim_path'))
        key = row['instance_id']; path = row['prim_path']
        if key in actual or key not in expected:
            raise RequirementError('Duplicate or unexpected movable object: ' + str(key))
        if not isinstance(path, str) or not re.fullmatch(r'/World(?:/[A-Za-z_][A-Za-z0-9_]*)+', path) or path in paths:
            raise RequirementError('Unique canonical scene path required')
        location(row['source'])
        if row['source'] is None:
            raise RequirementError('Actual source must be resolved')
        want = expected[key]
        if row['asset_id'] != want['asset_id']:
            raise RequirementError('Wrong asset for instance: ' + key)
        if want['source'] is not None and not (row['source'] == want['source'] or row['source'].startswith(want['source'] + '.')):
            raise RequirementError('Explicit source not fulfilled: ' + key)
        actual[key] = {**want, **row}; paths.add(path)
    if set(expected) != set(actual):
        raise RequirementError('Missing required instances: ' + repr(sorted(set(expected)-set(actual))))
    roles = set(infrastructure_roles)
    required = set(INFRASTRUCTURE) | {r['role'] for r in plan['scene_dependencies']}
    if not required <= roles:
        raise RequirementError('Missing infrastructure/dependency roles: ' + repr(sorted(required-roles)))
    # Narrow scene-random tasks to actual visible location only when all their
    # targets share one source. Otherwise retain broad scope conservatively.
    tasks = deepcopy(plan['tasks'])
    for task in tasks:
        sources = {actual[i]['source'] for i in task['target_ids']}
        if task['source'] is None and len(sources) == 1:
            task['source'] = next(iter(sources))
    validate_semantics(list(actual.values()), tasks)
    return dict(semantic_binding_checked=True, geometry_verified=False, physics_verified=False,
                runtime_request_emitted=False, instance_paths={k: v['prim_path'] for k, v in actual.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('request', type=Path)
    args = ap.parse_args()
    try:
        plan = compile_plan(json.loads(args.request.read_text()))
    except (RequirementError, json.JSONDecodeError) as exc:
        ap.exit(2, str(exc) + '\n')
    print(json.dumps(plan, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
