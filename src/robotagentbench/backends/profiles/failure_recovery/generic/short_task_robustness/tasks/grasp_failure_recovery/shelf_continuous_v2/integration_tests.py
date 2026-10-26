"""CPU unified-entry regression and strict identity/launch tests."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from short_tasks import registry
from short_tasks.agent_probe_v1.probe_random_layout import request_for as ordinary_request
from ..common import HERE, write_json
from . import public_entry as entry


def run(output, prepared=None):
    rows = {}
    for name in registry.TASK_NAMES:
        if name.startswith('failure_recovery_'):
            continue
        try:
            rows[name] = dict(plan=registry.resolve(ordinary_request(name)))
        except Exception as exc:
            rows[name] = dict(error=type(exc).__name__+': '+str(exc))
    adapter = registry.failure_recovery_adapter()
    coffee = registry.resolve(adapter.request_for('failure_recovery_coffee_bag'))
    result = dict(ordinary_plans=rows, coffee_plan=coffee)
    if adapter.request_for(entry.NAME)['rule_version'] == entry.VERSION:
        apis = set()
        for seed in range(1, 101):
            req = adapter.request_for(entry.NAME)
            req['seed'] = seed
            plan = registry.resolve(req)
            assert plan['initialization_policy'] == entry.POLICY
            assert plan['score_policy'] == dict(maximum=4, inherited_original=True)
            assert plan['control_hz'] == 20 and plan['no_progress_timeout_s'] is None
            assert plan['physics_policy'] == 'original_atomic_cpu_fixed_240hz'
            apis.add(plan['source_api'])
        assert apis == set(entry.APIS)
        result['random_product_resolve_100_requests'] = True
        req['rule_version'] = adapter.RULE_VERSION
        try:
            registry.resolve(req)
        except Exception:
            result['old_shelf_rule_rejected'] = True
        else:
            raise AssertionError('Old shelf launch cannot silently select a new initial')
        result['pins'] = registry.verify_sources(plan)
        if prepared:
            launch = json.loads(Path(prepared).read_text())
            req, manifest, plan = entry.admit_launch(launch, HERE/'shelf_continuous_v2/public_switch_v2/unused_native_output')
            initial = json.loads(Path(manifest['initial']).read_text())
            assert plan['source_api'] == initial['source_api']
            assert initial['online_ik_calls'] == 0 and initial['runtime_training_dataset_required'] is False
            assert initial['target_xy_continuously_sampled'] and not initial['whole_scene_frozen']
            for key in ('source_api', 'seed', 'initialization_policy'):
                changed = deepcopy(manifest)
                changed[key] = 'tampered'
                try:
                    registry.bind(plan, changed)
                except Exception:
                    pass
                else:
                    raise AssertionError('Tampered manifest admitted: '+key)
            result['prepared_input_checks'] = dict(passed=True, source_api=initial['source_api'],
                instruction=plan['instruction'], profile_id=initial['frozen_profile_id'],
                target=initial['target'], maximum=plan['score_policy']['maximum'])
    else:
        result['old_default_baseline'] = True
    write_json(Path(output), result)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--prepared')
    a = p.parse_args()
    run(a.output, a.prepared)
