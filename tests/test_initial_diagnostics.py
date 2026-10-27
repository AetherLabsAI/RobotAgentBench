from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from robotagentbench.simulation.initial_diagnostics import (
    NAME, TARGET, accept_initial_with_diagnostics,
)


@dataclass(frozen=True)
class Pose:
    position_m: tuple
    quaternion_wxyz: tuple


class InitialFailureDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)
        self.error = RuntimeError('Tray initial rejected: unheld false')
        self.history = deque((dict(time_s=(180+i)/60, physics_dt_s=1/60,
            pose=Pose((-1.46, 5.53, .92), (1., 0., 0., 0.)),
            held=False, robot_contact=False, on_floor=False,
            supports=frozenset(['/World/Kitchen/Worktop'])) for i in range(61)), maxlen=62)
        sample = ((0., 0., 1.), (-1.46, 5.53, .92), (0., 0., .1))
        self.active = {
            'tray': ((TARGET+'/Asset/Object', '/World/Robot/hand'), 1, (1, 2), (sample,)),
            'unrelated': (('/World/Other', '/World/Robot/base'), 1, (3, 4), (sample,)),
            'zero': ((TARGET+'/Asset/Object', '/World/Other'), 1, (1, 3), ()),
        }
        self.forbidden = [Mock(side_effect=AssertionError('Must not update/read native state'))
                          for _ in range(7)]
        contacts = SimpleNamespace(active=self.active, pairs=self.forbidden[0], ingest=self.forbidden[1])
        self.port = SimpleNamespace(history=self.history, contacts=contacts,
            geometry=SimpleNamespace(target=TARGET, body=TARGET+'/Asset/Object',
                                     floors=['/World/Background/Shell/Floor']),
            config=SimpleNamespace(initial_history_s=.25), robot='/World/Robot',
            after_physics=self.forbidden[2], reduce_contacts=self.forbidden[3],
            detector=SimpleNamespace(observe=self.forbidden[4]),
            clock=SimpleNamespace(get_physics_dt=self.forbidden[5]))
        self.session = SimpleNamespace(ports={'meal_tray': self.port},
            accept_initial=Mock(side_effect=self.error), simulate=self.forbidden[6])

    def reject(self, report):
        with self.assertRaises(RuntimeError) as caught:
            accept_initial_with_diagnostics(self.session, self.output, report)
        self.assertIs(caught.exception, self.error)
        self.session.accept_initial.assert_called_once_with()
        for method in self.forbidden:
            method.assert_not_called()

    def test_failure_copies_exact_window_and_target_contacts_without_mutation(self):
        self.history[0]['held'] = True  # Old history must not enter the 0.25 s window.
        self.history[-4]['robot_contact'] = True
        before_history, before_contacts = deepcopy(self.history), deepcopy(self.active)
        report = {}
        self.reject(report)
        record = json.loads((self.output/NAME).read_text())
        self.assertEqual(record['requested_sample_count'], 16)
        self.assertEqual(record['recorded_sample_count'], 16)
        self.assertEqual(record['rows'][0]['time_s'], self.history[-16]['time_s'])
        self.assertEqual(record['any_in_window'], dict(held=False, robot_contact=True, on_floor=False))
        self.assertEqual(record['latest_target_pair_count'], 1)
        self.assertEqual(record['latest_target_contact_pairs'][0]['paths'], list(self.active['tray'][0]))
        self.assertEqual(self.history, before_history)
        self.assertEqual(self.active, before_contacts)
        self.assertEqual(report['initial_failure_diagnostic']['status'], 'written')
        self.assertTrue(record['cache_only'])
        self.assertFalse(record['native_state_modified'])

    def test_each_rejection_flag_is_preserved_separately(self):
        for name in ('held', 'robot_contact', 'on_floor'):
            with self.subTest(name=name):
                path = self.output/name; path.mkdir()
                for key in ('held', 'robot_contact', 'on_floor'):
                    self.history[-1][key] = key == name
                with self.assertRaises(RuntimeError) as caught:
                    accept_initial_with_diagnostics(self.session, path, {})
                self.assertIs(caught.exception, self.error)
                record = json.loads((path/NAME).read_text())
                self.assertEqual(record['any_in_window'],
                                 {key: key == name for key in ('held', 'robot_contact', 'on_floor')})
        for method in self.forbidden:
            method.assert_not_called()

    def test_success_returns_original_receipt_and_never_inspects_caches(self):
        receipt = {'accepted': True}
        self.session.accept_initial = Mock(return_value=receipt)
        with patch('robotagentbench.simulation.initial_diagnostics.meal_tray_cache',
                   side_effect=AssertionError('No success diagnostics')) as snapshot:
            self.assertIs(accept_initial_with_diagnostics(self.session, self.output, {}), receipt)
        snapshot.assert_not_called()
        self.session.accept_initial.assert_called_once_with()
        self.assertFalse((self.output/NAME).exists())

    def test_diagnostic_exception_cannot_replace_original_rejection(self):
        report = {}
        with patch('robotagentbench.simulation.initial_diagnostics.meal_tray_cache',
                   side_effect=KeyboardInterrupt('diagnostic failed')):
            self.reject(report)
        self.assertEqual(report['initial_failure_diagnostic'],
                         dict(status='unavailable', error_type='KeyboardInterrupt'))
        self.assertFalse((self.output/NAME).exists())

    def test_existing_sidecar_is_never_overwritten_and_original_error_survives(self):
        (self.output/NAME).write_text('older evidence')
        report = {}
        self.reject(report)
        self.assertEqual((self.output/NAME).read_text(), 'older evidence')
        self.assertEqual(report['initial_failure_diagnostic']['error_type'], 'FileExistsError')

    def test_other_tasks_are_not_inspected(self):
        self.session.ports = {'other': object()}
        report = {}
        self.reject(report)
        self.assertEqual(report['initial_failure_diagnostic']['status'], 'not_applicable')
        self.assertFalse((self.output/NAME).exists())

    def test_malformed_cache_preserves_original_error_without_partial_sidecar(self):
        self.history[-1]['pose'] = Pose((float('nan'), 0, 0), (1, 0, 0, 0))
        report = {}
        self.reject(report)
        self.assertEqual(report['initial_failure_diagnostic']['error_type'], 'ValueError')
        self.assertFalse((self.output/NAME).exists())


if __name__ == '__main__':
    unittest.main()
