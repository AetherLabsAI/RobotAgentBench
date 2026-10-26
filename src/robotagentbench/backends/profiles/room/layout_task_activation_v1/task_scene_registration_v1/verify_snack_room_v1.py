"""Persist exact tests and implementation hashes, not physical success claims."""
import unittest,io,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
modules=['test_snack_room_v1','test_multi_task_dispatch_v1','test_selected_scores_v1',
 'evaluation.tasks.shelf_pick.openended_composite_v1.test_composite']
if __name__=='__main__':
    from evaluation.tasks.shelf_pick.openended_composite_v1 import composite,room_port
    files=[ROOT/'multi_task_room_v2.py',ROOT/'multi_task_episode_snack_v1.json',
           Path(composite.__file__),Path(room_port.__file__)]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    log=io.StringIO()
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(modules))
    report=dict(passed=result.wasSuccessful(),tests_run=result.testsRun,failures=len(result.failures),
                errors=len(result.errors),evidence='synthetic_rules_and_contracts_not_physics',
                source_sha256=hashes,output=log.getvalue())
    (ROOT/'snack_room_rules_v1.json').write_text(json.dumps(report,indent=2)+'\n')
    print(log.getvalue())
    raise SystemExit(0 if result.wasSuccessful() else 1)
