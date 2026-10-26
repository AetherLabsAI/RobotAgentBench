"""Exact physical goals from registered objects; never count task stages twice."""
ALLOWED = frozenset(('wallet_transfer','glasses_transfer','powerbank_transfer',
    'water_bottle_transfer','remote_transfer','book_transfer','cap_transfer',
    'notebook_transfer','trash_transfer','snack_drink_to_tray',
    'fridge_to_microwave','microwave_box_to_meal_tray','cutlery_to_meal_tray',
    'meal_tray_transfer'))


def within(path, root):
    return path == root or path.startswith(root + '/')


def catalogue(request, manifest, registration):
    resolved=registration['resolved_objects']
    if registration.get('pending') or not registration.get('native_ready') or len(resolved)<10:
        raise ValueError('At least ten genuinely bound registered task instances required')
    # Expanded garbage members are the original registration's real instances.
    # No synthetic tasks or scoring stages are added here.
    if set(t['type'] for t in request['tasks'])-ALLOWED:
        raise ValueError('Unreviewed task-object binding')
    rows={r['path']:r for r in manifest['objects']}
    owners={}
    for key,path in resolved.items():
        if path not in rows:
            raise ValueError('A target must be an actual manifest object, not a workspace')
        owners.setdefault(path,[]).append(key)
    targets={}
    aliases={}
    for path,keys in sorted(owners.items()):
        keys=sorted(keys)
        # Original registration permits the explicit fridge/microwave food
        # handoff; both stages refer to the same physical box and earn ONE hit.
        if len(keys)>1 and set(keys)!={'fridge','warm_meal'}:
            raise ValueError('Unexpected shared physical object')
        canonical=keys[0]
        targets[canonical]=(path,)
        aliases.update({key:canonical for key in keys})
    if len(targets)<10:
        raise ValueError('At least ten distinct physical target objects required')
    paths=[p[0] for p in targets.values()]
    if any(within(a,b) or within(b,a) for i,a in enumerate(paths) for b in paths[i+1:]):
        raise ValueError('Overlapping physical goal roots')
    return dict(targets=targets,task_to_object=aliases,registered_instances=len(resolved),
                unique_target_count=len(targets),counting='unique_real_physical_objects',
                destinations_added_as_goals=False,score_stages_added_as_goals=False)
