"""New installed-package/CLI seam only; no simulator or asset acceptance."""
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from robotagentbench import cli
from robotagentbench.simulation.cloth import config, launcher


class ClothOperator(unittest.TestCase):
    def test_installed_package_and_fixed_children(self):
        self.assertEqual(config.HERE.parents[2], Path(cli.__file__).resolve().parent.parent)
        self.assertEqual(config.selection('spread_tablecloth')['entry'], 'native_measured_mirrored_lay_v50.py')
        self.assertEqual(config.selection('collect_tablecloth')['entry'], 'native_removal_continuous_transport_v54.py')
        self.assertEqual(config.selection('long_task_18')['episode'], 'remove_v179')
        for name in ('supervisor', 'physical'):
            self.assertIn('robotagentbench.simulation.cloth.' + name, config.module_bootstrap(name))
        with self.assertRaises(ValueError):
            config.module_bootstrap('agent_python')
        self.assertFalse(any(n.split('.')[0] in {'isaacsim', 'omni', 'carb', 'pxr'} for n in sys.modules))

    def test_cli_forwards_only_explicit_operator_modes(self):
        cases = [([], []), (['--verify'], ['--verify']),
                 (['--prepare', '/inputs/fresh.json'], ['--prepare', '/inputs/fresh.json']),
                 (['--execute', '/inputs/sealed.json', '--output', '/outputs/fresh'],
                  ['--execute', '/inputs/sealed.json', '--output', '/outputs/fresh'])]
        for arguments, suffix in cases:
            with self.subTest(arguments=arguments), patch.object(launcher, 'main', return_value=0) as run:
                self.assertEqual(cli.main(['cloth', '/config/operator.json', '--task', 'long_task_18', *arguments]), 0)
                run.assert_called_once_with(['/config/operator.json', '--task', 'long_task_18', *suffix])

    def test_default_description_never_starts_child_or_sdk(self):
        with patch.object(launcher, 'read_operator', return_value={}), patch.object(
                launcher, 'describe', return_value={'native_started': False}), patch(
                'subprocess.run', side_effect=AssertionError('No child in default mode')), patch(
                'os.execve', side_effect=AssertionError('No execution without --execute')), redirect_stdout(StringIO()):
            self.assertEqual(cli.main(['cloth', '/config/operator.json', '--task', 'spread_tablecloth']), 0)

    def test_cli_invalid_configuration_does_not_leak_paths(self):
        with patch.object(launcher, 'read_operator', side_effect=ValueError('/private/operator-data')), redirect_stderr(StringIO()) as error:
            self.assertEqual(cli.main(['cloth', '/config/operator.json', '--task', 'collect_tablecloth']), 2)
        self.assertNotIn('/private/operator-data', error.getvalue())


if __name__ == '__main__':
    unittest.main()
