"""Run the deployed rule suites; record real results, not physics evidence."""
import io,json,sys,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'api-protocol'),str(ROOT/'layout_task_activation_v1/task_scene_registration_v1')]
loader=unittest.TestLoader()
suite=loader.discover(str(ROOT/'openend/tests'),top_level_dir=str(ROOT))
for name in ('openend.tasks.glasses_transfer.test_composite','openend.tasks.wallet_transfer.test_composite',
             'layout_task_activation_v1.task_scene_registration_v1.fridge_to_microwave_v1.test_composite',
             'openend.tasks.scene_requirements.test_runtime_wiring',
             'openend.tasks.scene_requirements.test_shelf_runtime',
             'openend.tasks.scene_requirements.test_shelf_legacy_contract',
             'openend.tasks.glasses_transfer.test_source_agnostic','openend.tasks.wallet_transfer.test_source_agnostic',
             'openend.tasks.all_objects_integration.test_workspace_physics'):
    suite.addTests(loader.loadTestsFromName(name))
stream=io.StringIO();r=unittest.TextTestRunner(stream=stream,verbosity=1).run(suite)
result=dict(tests_run=r.testsRun,success=r.wasSuccessful(),failures=len(r.failures),errors=len(r.errors),
            skipped=len(r.skipped),log=stream.getvalue(),synthetic_rule_tests_only=True,native_physics_validated=False)
out=HERE/'shelf_wiring_regression_20261005.json';out.write_text(json.dumps(result,indent=2))
print(json.dumps(result));raise SystemExit(0 if r.wasSuccessful() else 1)
