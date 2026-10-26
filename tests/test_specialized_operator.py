"""Installed specialized operator/CLI seam; no native or asset acceptance."""
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from robotagentbench import cli
from robotagentbench.simulation.specialized import config, operator


class SpecializedOperator(unittest.TestCase):
    def test_packaged_identity_data_and_fixed_routes(self):
        self.assertEqual(config.HERE.parents[2], Path(cli.__file__).resolve().parent.parent)
        profiles = config.identities()
        self.assertEqual(len(profiles['atomic_coffee']['tasks']), 4)
        self.assertEqual(len(profiles['failure_recovery']['tasks']), 5)
        self.assertEqual(len(profiles['atomic_coffee']['required_startup_resources']), 14)
        self.assertEqual(profiles['failure_recovery']['required_startup_resources'], {})
        self.assertFalse(any(n.split('.')[0] in {'isaacsim', 'omni', 'carb', 'pxr'} for n in sys.modules))

    def test_cli_forwards_only_explicit_operator_modes(self):
        cases = [[], ['--verify'], ['--prepare', '/inputs/fresh.json'],
                 ['--execute', '--launch', '/inputs/launch.json', '--output', '/outputs/fresh',
                  '--dispatch', '/inputs/dispatch.json']]
        prefix = ['/config/operator.json', '--family', 'failure_recovery',
                  '--task', 'failure_recovery_table_waste_empty_grasp']
        for arguments in cases:
            with self.subTest(arguments=arguments), patch.object(operator, 'main', return_value=0) as run:
                self.assertEqual(cli.main(['specialized', *prefix, *arguments]), 0)
                run.assert_called_once_with([*prefix, *arguments])

    def test_default_never_executes_or_starts_sdk(self):
        with patch.object(operator.Path, 'resolve', return_value=Path('/config/operator.json')), patch.object(operator.config, 'read_operator', return_value={}), patch.object(
                operator, 'describe', return_value={'native_started': False}), patch(
                'os.execve', side_effect=AssertionError('No exec in default mode')), patch(
                'subprocess.run', side_effect=AssertionError('No subprocess in default mode')), redirect_stdout(StringIO()):
            self.assertEqual(cli.main(['specialized', '/config/operator.json', '--family',
                'atomic_coffee', '--task', 'coffee_machine_push_interference']), 0)

    def test_configuration_errors_do_not_expose_paths(self):
        with patch.object(operator.Path, 'resolve', return_value=Path('/config/operator.json')), patch.object(operator.config, 'read_operator', side_effect=KeyError('/private/identity')), redirect_stderr(StringIO()) as error:
            self.assertEqual(cli.main(['specialized', '/config/operator.json', '--family',
                'failure_recovery', '--task', 'failure_recovery_sofa']), 2)
        self.assertNotIn('/private/identity', error.getvalue())


if __name__ == '__main__':
    unittest.main()
