"""Store actual CPU test results in this task directory; no native run."""
import io
import json
import unittest
from pathlib import Path

MODULES = (
    'openend.tasks.scene_requirements.test_shelf_retirement',
    'openend.tasks.scene_requirements.test_shelf_instances',
    'openend.tasks.scene_requirements.test_shelf_runtime',
    'openend.tasks.scene_requirements.test_shelf_legacy_contract',
    'openend.tasks.scene_requirements.test_shelf_request',
    'openend.tests.test_dispatch',
    'openend.tests.test_unified_policy',
    'openend.tasks.remote_transfer.test_room',
)

if __name__ == '__main__':
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(m) for m in MODULES)
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=1).run(suite)
    report = dict(tests=result.testsRun, passed=result.wasSuccessful(),
                  failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
                  scope='synthetic_room_routing_and_score_contracts_not_native_manipulation',
                  modules=MODULES, log=log.getvalue())
    path=Path(__file__).with_name('shelf_retirement_verification_20261006.json')
    with path.open('x') as f: json.dump(report,f,indent=2)
    print(json.dumps(report))
    raise SystemExit(0 if result.wasSuccessful() else 1)
