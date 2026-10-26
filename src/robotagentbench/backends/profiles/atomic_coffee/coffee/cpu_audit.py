"""Fresh task-local CPU test/composition evidence; no Isaac app/GPU startup."""
import hashlib
import json
from pathlib import Path
import sys
import unittest
import numpy as np
from .build_initial import HERE, build


def audit(output):
    output = Path(output).resolve()
    if output == HERE or not output.is_relative_to(HERE) or output.exists():
        raise ValueError('Fresh audit directory inside this task required')
    output.mkdir(parents=True)
    modules = ['test_rules', 'test_usd', 'test_reader']
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(
        'short_task_robustness.tasks.coffee_machine_push_interference.' + module) for module in modules)
    with (output / 'tests.log').open('x') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError('CPU tests failed: see task-owned tests.log')
    scenes = []
    for seed in (0, 1, 2, 3):
        value = build(output / f'initial_seed{seed}', seed=seed, role='random')
        for row in value['push_routes']['routes']:
            fit = row['palm_fit']
            rotation = np.asarray(fit['orientation'])
            if (not fit['fingers_horizontal'] or abs(float(rotation[2, 1])) > 1e-12
                    or fit['contact_region'] != 'lower_main_case_side_wall'
                    or fit['hopper_contact_allowed']):
                raise RuntimeError('Horizontal lower-case palm contract not met')
        scenes.append(dict(seed=seed, scene=value['scene'],
                           changed_properties=value['approved_dynamic_body']['changed_properties'],
                           selected_side=value['push_routes']['selected_side'],
                           selected_route_static_admitted=value['push_routes']['selected_route_static_admitted'],
                           planned_distance_m=value['push_routes']['planned_distance_m'],
                           source_hashes_verified=True))
    report = dict(tests_run=result.testsRun, errors=len(result.errors), failures=len(result.failures),
                  skipped=len(result.skipped), synthetic_reader_tests=True, cpu_usd_only=True,
                  scenes=scenes, source_files_unchanged=True,
                  physical_settle_verified=False, real_palm_push_verified=False,
                  registered=False, scored_task_verified=False, model_verified=False,
                  task_files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in HERE.iterdir() if p.is_file() and p.suffix in ('.py', '.json', '.md')})
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    audit(sys.argv[1])
