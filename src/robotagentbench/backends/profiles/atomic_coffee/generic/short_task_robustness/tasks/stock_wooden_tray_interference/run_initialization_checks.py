"""Task-owned CPU/USD runner. Uses installed Isaac USD, never SimulationApp."""
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(HERE)]


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(HERE) or output.exists():
        raise ValueError('Fresh task-owned result path required')
    modules = ['short_tasks.tasks.stock_wooden_tray_interference.tests.'+name for name in
        ('test_proximity','test_release_collision','test_frozen_native_route','test_source_range','test_generation','test_initialization','test_prepare_scene','test_reservation','test_random_relocation',
         'test_pick_only','test_integration')]
    modules.append('short_tasks.tasks.stock_wooden_tray.test_adapter')
    suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
    output.mkdir()
    with (output/'tests.txt').open('w') as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    receipt = dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        skips=len(result.skipped),success=result.wasSuccessful(),physical_validation=False,
        gpu_started=False)
    (output/'result.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
    if not result.wasSuccessful():
        print((output/'tests.txt').read_text())
        raise SystemExit(1)


if __name__=='__main__': main()
