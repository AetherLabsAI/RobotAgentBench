"""One scene-bound group, not one task per clutter prop. No native claims."""
from .inventory import bindings


def declaration(request):
    found = [t for t in request.get('tasks', []) if t.get('type') == 'cloth_clear']
    if len(found) > 1:
        raise ValueError('Only one cloth-clear group per room')
    if not found:
        return None
    task = found[0]
    if set(task) != {'instance_id', 'type', 'parameters'} or task['parameters'] != {}:
        raise ValueError('Cloth-clear binds scene inventory; no task parameter overrides')
    name = task['instance_id']
    if not isinstance(name, str) or not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
        raise ValueError('Safe group instance ID required')
    return task


def resolve(task, manifest):
    declaration({'tasks': [task]})
    group = manifest.get('cloth_clear', {})
    pool = group.get('asset_pool_ids', [])
    clutter = [p for p in pool if isinstance(p, str) and p.startswith('cloth_clear_')]
    if (len(clutter) != 20 or len(set(pool)) != len(pool)
            or set(pool) - set(clutter) not in (set(), {'book_red'})):
        raise ValueError('Complete twenty-asset inventory with optional approved book_red required')
    targets = bindings('scene_registration', manifest, pool)
    by_path = {r['path']: r for r in manifest['objects']}
    cloth = by_path.get(group.get('cloth_path'))
    if cloth is None or cloth['id'] != 'tablecloth_dirty':
        raise ValueError('Manifest cloth role must bind the actual cloth')
    for b in targets:
        row = by_path[b.target_path]
        if row.get('task_role') != 'cloth_clear_target' or row.get('rigid_body_path') != b.target_path + '/Asset':
            raise ValueError('Explicit target role and dynamic body path required')
        if row.get('trash_category'):
            raise ValueError('Cloth-clear target cannot also be registered garbage')
    if group.get('table_path') != '/World/Assets/CoffeeTable':
        raise ValueError('Approved same-table destination required')
    excluded = group.get('exclusion_paths')
    if not isinstance(excluded, list) or len(excluded) != 1 or by_path.get(excluded[0], {}).get('id') != 'japanese_tray':
        raise ValueError('Explicit fixed wooden tray exclusion required')
    paths = [b.target_path for b in targets]
    return dict(instance_id=task['instance_id'], type='cloth_clear', target_paths=paths,
                cloth_path=group['cloth_path'], table_path=group['table_path'], exclusion_paths=excluded,
                target_count=len(paths), maximum_score=3 * len(paths), stage_points=[1, 1, 1],
                registration_status='scene_bound_native_reader_implemented_validation_pending', native_ready=True,
                native_full_trajectory_verified=False,
                success_policy='all_targets_currently_outside_cloth_released_table_supported_stable_1s',
                workspace_id='table_cloth', timeout_owner='room', initialization_owner='unified_scene')


def resolve_request(request, manifest):
    task = declaration(request)
    if task is None:
        return []
    record = resolve(task, manifest)
    targets = set(record['target_paths'])
    for other in request['tasks']:
        if other is not task and other.get('parameters', {}).get('target_path') in targets:
            raise ValueError('Target already claimed by cloth-clear group')
    return [record]


class ClothClearRegistration:
    """Registration descriptor only. Must not masquerade as a physical reader."""
    asset = 'tablecloth_dirty'
    physics_hz = 60
    selection_mode = 'contact'

    @staticmethod
    def validate_parameters(parameters):
        if parameters != {}:
            raise ValueError('Cloth-clear has no parameter overrides')

    @staticmethod
    def resolve_object(parameters, manifest):
        ClothClearRegistration.validate_parameters(parameters)
        record = resolve(dict(instance_id='cloth_clear', type='cloth_clear', parameters={}), manifest)
        return next(r for r in manifest['objects'] if r['path'] == record['cloth_path'])

    def __init__(self, *args, **kwargs):
        raise RuntimeError('cloth_clear scene registration is connected; native multi-object evidence reader '
                           'and aggregate completion are not connected. No physical score can be emitted.')
