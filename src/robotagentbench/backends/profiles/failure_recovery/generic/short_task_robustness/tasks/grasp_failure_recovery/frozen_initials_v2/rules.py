"""Save an actual no-GPU unit-test receipt beneath this task."""
import io
import json
import sys
import unittest
from ..common import HERE,owned,write_json,digest


def main():
    out=owned(HERE/'frozen_initials_v2/verification/rules_v2',fresh=True);out.mkdir(parents=True)
    names=['frozen_initials_v2.test_bank','frozen_initials_v1.test_bank',
        'expansion_v2.test_catalog','expansion_v2.test_ik_branches','expansion_v2.test_operating_station',
        'expansion_v2.test_empty_grasp','expansion_v2.test_arm_first_cup','expansion_v2.test_geometric_direction']
    stream=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromNames(['grasp_failure_recovery.'+n for n in names])
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (out/'unittest.log').write_text(stream.getvalue())
    value=dict(tests_run=result.testsRun,successful=result.wasSuccessful(),errors=len(result.errors),failures=len(result.failures),
        modules=names,python=sys.executable,bank_sha256=digest(HERE/'frozen_initials_v2/compile/cup_training_adapt_v5/bank_v2.json'),
        gpu_used=False,model_called=False)
    write_json(out/'result.json',value);print(json.dumps(value));raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__=='__main__':main()
