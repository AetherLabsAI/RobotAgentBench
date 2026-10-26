"""Read-only scene dependency planner. Does not create USD or start a simulator."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build(request, catalog=None):
    catalog = catalog or json.loads((HERE / 'registry.json').read_text())
    allowed = {'layout_id', 'seed', 'tasks', 'initial_task_by_station'}
    if not isinstance(request, dict) or set(request) - allowed:
        raise ValueError('Unknown scene request fields')
    if type(request.get('layout_id')) is not int or request['layout_id'] != catalog['base_scene']['layout_id']:
        raise ValueError('Only the inspected layout 1 is currently supported; no automatic fallback')
    if type(request.get('seed')) is not int or request['seed'] < 0:
        raise ValueError('An explicit non-negative integer seed is required')
    selections = request.get('tasks')
    if not isinstance(selections, list) or not selections:
        raise ValueError('Select at least one concrete task')
    registered = {t['key']: t for t in catalog['tasks']}
    selected = {}; assets = {}; stations = {}; payloads = {}; blockers = []
    for item in selections:
        if not isinstance(item, dict) or set(item) - {'key', 'payload_variant'}:
            raise ValueError('Each task needs a key and, where required, an explicit payload_variant')
        key = item.get('key')
        if not isinstance(key, str) or key not in registered:
            raise ValueError('Unknown task: ' + str(key))
        if key in selected:
            raise ValueError('Duplicate task: ' + key)
        task = registered[key]; selected[key] = task
        dependencies = list(task['assets'])
        if 'payload_options' in task:
            payload = item.get('payload_variant')
            if payload not in task['payload_options']:
                raise ValueError('Explicit supported payload required: ' + key)
            dependencies.append(payload); payloads[key] = payload
        elif 'payload_variant' in item:
            raise ValueError('Payload is not a parameter of this task')
        stations.setdefault(task['station'], []).append(key)
        for asset in dependencies:
            assets.setdefault(asset, {**catalog['assets'][asset], 'required_by': []})['required_by'].append(key)
    initial = request.get('initial_task_by_station', {})
    if not isinstance(initial, dict) or set(initial) - set(stations):
        raise ValueError('Initial task selectors must name selected stations only')
    variants = []
    for station, keys in stations.items():
        chosen = initial.get(station)
        if chosen is not None and chosen not in keys:
            raise ValueError('Initial task must belong to that station and enabled task set')
        if chosen is None and len(keys) == 1:
            chosen = keys[0]
        if chosen is None:
            blockers.append({'code': 'initial_task_required', 'station': station,
                             'detail': 'Task asset union is allowed; incompatible initial states require an explicit scenario selection or approved sequence.'})
        variants.append({'station': station, 'enabled_tasks': keys, 'initial_task': chosen,
                         'initial_state': selected[chosen]['initial_state'] if chosen else None,
                         'payload_variant': payloads.get(chosen)})
        if chosen == 'microwave.put_box':
            blockers.append({'code': 'held_payload_prerequisite', 'station': station,
                             'detail': 'Existing put task starts with the box held in the right hand; navigation-to-grasp precondition is not connected.'})
    blockers.extend([
        {'code': 'station_placement_pending', 'detail': 'Support-local anchors, placement constraints and task-local transforms need validation on this base layout.'},
        {'code': 'asset_contract_pending', 'detail': 'Resolve complete asset dependencies and check existing device compatibility; metadata is not USD acceptance.'},
        {'code': 'runtime_adapter_pending', 'detail': 'Existing reset/evaluator contracts are not connected to this multi-task Action22 environment.'},
        {'code': 'physics_validation_pending', 'detail': 'No generated scene, static composition or native settle test has been run.'}
    ])
    seed = request['seed']
    return {'schema': 'arena.openended.scene_plan.v1', 'status': 'blocked_before_scene_generation',
            'runtime_ready': False, 'usd_written': False, 'random_poses_sampled': False,
            'request': request, 'base_scene': catalog['base_scene'], 'assets': assets,
            'control_contract': dict(catalog['control_contract']),
            'asset_union_semantics': 'Resources to make available, not objects to activate simultaneously. Each station initial state controls instantiated contents and device state.',
            'task_selection_semantics': 'Enabled task pool only; neither uniform sampling nor a sequential workflow has been approved or executed.',
            'station_scenarios': variants, 'blockers': blockers,
            'seed_trace': {'requested': seed, 'derivation_only_not_applied': {
                k: int.from_bytes(hashlib.sha256(f'{seed}:{k}'.encode()).digest()[:4], 'big') for k in sorted(selected)}},
            'randomization': {'policy': 'reuse_verified_task_samplers_after_coordinate_binding',
                              'sources': {k: t['sampler_source'] for k, t in selected.items()},
                              'new_ranges': None, 'resampling': False},
            'activation': 'bind generated real object instances to existing robot sector; do not reset scene on entry',
            'physics_policy': 'preserve approved parameters; mixed-rate scheduling unresolved'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request', type=Path)
    args = parser.parse_args()
    print(json.dumps(build(json.loads(args.request.read_text())), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
