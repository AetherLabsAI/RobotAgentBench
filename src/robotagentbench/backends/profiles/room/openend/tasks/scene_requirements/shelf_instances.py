"""Independent shelf task identities; no group scorer or physics overrides."""
from dataclasses import replace
from evaluation.tasks.shelf_pick.openended_composite_v1.catalog import normalize_parameters

KIND = 'snack_drink_to_tray'


def validate_instances(tasks):
    rows = [t for t in tasks if t['type'] == KIND]
    if len(rows) > 4:
        raise ValueError('At most four shelf instances')
    for task in rows:
        normalize_parameters(task['parameters'])
    from openend.tasks.meal_tray.product_claims import validate_groups
    validate_groups(tasks)
    if len({t['instance_id'] for t in rows}) != len(rows):
        raise ValueError('Duplicate shelf task instance')
    return rows


def peer_bodies(manifest, instance_id, product, body_by_wrapper):
    """Only exact targets of actually registered sibling tasks are peers.

    Static composition preflight has no registration yet and needs no peers.
    Background stock, packaging and unregistered declarations stay non-targets.
    """
    declared=manifest.get('registration', {}).get('request', {}).get('tasks', [])
    tasks = validate_instances(declared)+[t for t in declared if t['type']=='drink_to_meal_tray']
    if not tasks:
        return frozenset()
    own = [t for t in tasks if t['instance_id'] == instance_id]
    if len(own) != 1 or own[0]['parameters']['product'] != product:
        raise ValueError('Shelf registration does not match native instance')
    bodies = {}
    for task in tasks:
        matches = [o for o in manifest['objects']
                   if o.get('task_role') == 'requested_full_product'
                   and o.get('task_id') == task['instance_id']
                   and o['id'] == task['parameters']['product']]
        if len(matches) != 1 or matches[0]['path'] not in body_by_wrapper:
            raise ValueError('Missing or ambiguous registered shelf body')
        bodies[task['instance_id']] = body_by_wrapper[matches[0]['path']]
    if len(set(bodies.values())) != len(bodies):
        raise ValueError('Two shelf tasks claim one body')
    return frozenset(body for key, body in bodies.items() if key != instance_id)


def scoped_pick(pick, peers):
    """Keep raw facts intact; only classify registered peers as legitimate tasks.

    Never award own grasp from a peer grasp or suppress neighbor collisions.
    """
    return replace(pick, wrong_drinks_grasped=pick.wrong_drinks_grasped - peers)
