"""CPU raw operator tests; actual receipt verifier, mocked process/model I/O.

No real CLI, worker, model or GPU starts. Native-shaped fixtures are schema
tests only and do not claim physical execution or completed task evidence.
"""
import base64
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from robotagentbench.interfaces.action_contract import ORDER, specification
from robotagentbench.interfaces import robot_profiles
from robotagentbench.model_examples.codex import __main__ as entry
from robotagentbench.model_examples.codex import client_config, raw_operator


def observation(step=0):
    png = base64.b64encode(b'\x89PNG\r\n\x1a\n' + struct.pack('>I', 13) + b'IHDR'
                          + struct.pack('>II', 640, 480) + bytes(9)).decode()
    return dict(episode_id='mock-raw-episode', step=step, time_s=step / 20,
        instruction='Synthetic operator test, not an evaluation', image_step=step,
        joints=dict.fromkeys(robot_profiles.NATIVE_JOINTS, 0.),
        images=dict.fromkeys(('head', 'left', 'right'), png),
        base_state=dict(position_world_m=[0., 0., 0.], orientation_world_xyzw=[0., 0., 0., 1.],
            linear_velocity_world_m_s=[0., 0., 0.], angular_velocity_world_rad_s=[0., 0., 0.],
            yaw_world_unwrapped_rad=0.))


class FakeWorker:
    def __init__(self):
        self.calls = []
        self.closed = False
        self.process = SimpleNamespace(wait=Mock(return_value=0))
        self.owner = dict(status='finished', evaluation_valid=True, success=False, score=2.,
                          scores={'total': 2., 'maximum': 5.}, private_targets='OWNER ONLY')

    def read(self):
        return dict(op='ready', contract=specification(dict.fromkeys(ORDER[:20], [-4., 4.])),
                    observation=observation())

    def exchange(self, request):
        self.calls.append(deepcopy(request))
        if request['op'] == 'close':
            return dict(op='closed', result=self.owner)
        return dict(op='result', actions_executed=1, episode_finished=False, observation=observation(1))

    def close(self):
        self.closed = True
        return 0


class FakeServer:
    def __init__(self):
        self.sent, self.events, self.closed = [], 0, False

    def start_thread(self, model, instructions, tools):
        self.model, self.instructions, self.tools = model, instructions, deepcopy(tools)
        return dict(model=model, thread={'id': 'thread'})

    def request(self, name, args, **options):
        self.sent.append((name, deepcopy(args)))
        return {'turn': {'id': 'turn'}} if name == 'turn/start' else {}

    def next_event(self, *args):
        self.events += 1
        if self.events == 1:
            return dict(method='rawResponse/completed', params={'threadId': 'thread'})
        name, args = ('execute_actions', {'actions': [[0.] * 22]}) if self.events == 2 else ('done', {'reason': 'Stop synthetic test'})
        return dict(id=self.events, method='item/tool/call', params=dict(threadId='thread', turnId='turn',
                    namespace='arena', callId='call-' + str(self.events), tool=name, arguments=args))

    def send(self, value):
        self.sent.append(deepcopy(value))

    def close(self):
        self.closed = True


class OperatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='codex-raw-operator-cpu-')
        self.root = Path(self.temp.name).resolve()
        self.cli = self.root / 'fake-codex'
        self.cli.touch()
        self.value = dict(schema='robotagentbench.codex_example.v1', codex_executable=str(self.cli),
            expected_cli_version='mock-version', model='gpt-6-astra', effort='medium',
            max_model_responses=20, transport=None, prepare_argv=[], worker_argv=['mock-native-worker'],
            output=str(self.root / 'agent-output'), worker_mode='raw22', native_receipt=dict(
                family='atomic_coffee', task_id='mock-task', native_output=str(self.root / 'native-output'),
                read_argv=['mock-reader', '--native-output', str(self.root / 'native-output')]))
        self.bundle = dict(native_output=str(self.root / 'native-output'), files={'native_report.json': {'synthetic': True}})
        self.worker, self.server, self.verifier = FakeWorker(), FakeServer(), Mock(return_value=True)
        self.commands = []

    def tearDown(self):
        self.temp.cleanup()

    def config_file(self, value=None):
        path = self.root / 'operator.json'
        path.write_text(json.dumps(value if value is not None else self.value))
        return path

    def process(self, argv, **kwargs):
        self.commands.append((deepcopy(argv), kwargs))
        if argv == [str(self.cli), '--version']:
            return SimpleNamespace(stdout='mock-version\n')
        self.assertEqual(argv, self.value['native_receipt']['read_argv'])
        self.assertIs(kwargs['shell'], False)
        return SimpleNamespace(stdout=json.dumps(self.bundle), stderr='')

    def mocked(self, *, real_verifier=False):
        stack = ExitStack()
        stack.enter_context(patch('robotagentbench.model_examples.codex.protocol.Worker', return_value=self.worker))
        stack.enter_context(patch('robotagentbench.model_examples.codex.app_server.AppServer', return_value=self.server))
        stack.enter_context(patch('robotagentbench.model_examples.codex.raw_operator.subprocess.run', side_effect=self.process))
        if not real_verifier:
            stack.enter_context(patch.dict(sys.modules, {'robotagentbench.model_examples.native_receipts':
                                                       SimpleNamespace(verify=self.verifier)}))
        return stack

    def report(self):
        return json.loads((Path(self.value['output']) / 'operator_result.json').read_text())

    def test_old_configuration_remains_coding_default(self):
        old = {k: v for k, v in self.value.items() if k not in ('worker_mode', 'native_receipt')}
        self.assertEqual(entry.configuration(self.config_file(old)), old)
        explicit = dict(old, worker_mode='coding')
        self.assertEqual(entry.configuration(self.config_file(explicit)), explicit)

    def test_raw_configuration_requires_explicit_receipt_and_no_preparation(self):
        self.assertEqual(entry.configuration(self.config_file()), self.value)
        mutations = [lambda v: v.pop('native_receipt'), lambda v: v.update(prepare_argv=['generate-scene']),
                     lambda v: v.update(worker_mode='unknown'),
                     lambda v: v['native_receipt'].update(read_argv='shell command'),
                     lambda v: v['native_receipt'].update(family='unknown'),
                     lambda v: v['native_receipt'].update(native_output='/'),
                     lambda v: v['native_receipt'].update(native_output=v['output']),
                     lambda v: v['native_receipt'].update(extra='not allowed')]
        for mutate in mutations:
            value = deepcopy(self.value)
            mutate(value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                entry.configuration(self.config_file(value))

    def test_describe_never_starts_cli_worker_model_or_receipt_reader(self):
        with patch('subprocess.run') as process, patch('subprocess.Popen') as start, redirect_stdout(io.StringIO()) as text:
            self.assertEqual(entry.main([str(self.config_file())]), 0)
        process.assert_not_called()
        start.assert_not_called()
        value = json.loads(text.getvalue())
        self.assertEqual(value['worker_mode'], 'raw22')
        self.assertFalse(value['model_started'])
        self.assertFalse(value['native_started'])

    def test_client_generator_preserves_default_and_exposes_explicit_raw_mode(self):
        common = ['--codex-executable', str(self.cli), '--cli-version', 'mock-version',
                  '--worker-argv-json', '["mock-native-worker"]', '--output', self.value['output']]
        with patch('subprocess.run') as process, redirect_stdout(io.StringIO()):
            legacy = self.root / 'legacy.json'
            self.assertEqual(client_config.main(common + ['--destination', str(legacy)]), 0)
            self.assertNotIn('worker_mode', json.loads(legacy.read_text()))
            raw = self.root / 'raw.json'
            self.assertEqual(client_config.main(common + ['--destination', str(raw), '--worker-mode', 'raw22',
                '--native-receipt-json', json.dumps(self.value['native_receipt'])]), 0)
            self.assertEqual(entry.configuration(raw)['native_receipt'], self.value['native_receipt'])
            with self.assertRaises(ValueError):
                client_config.main(common + ['--destination', str(self.root / 'missing.json'), '--worker-mode', 'raw22'])
            with self.assertRaises(ValueError):
                client_config.main(common + ['--destination', str(self.root / 'wrong.json'),
                    '--native-receipt-json', json.dumps(self.value['native_receipt'])])
        process.assert_not_called()

    def test_raw_entry_uses_existing_session_and_checks_family_receipt(self):
        with self.mocked():
            self.assertEqual(entry.run(self.value), 0)
        report = self.report()
        self.assertEqual(report['status'], 'closed')
        self.assertEqual(report['worker_mode'], 'raw22')
        self.assertEqual(report['actions_executed'], 1)
        self.assertEqual(report['score'], 2.)
        self.assertFalse(report['success'])
        self.assertTrue(report['native_cleanup_verified'])
        self.assertEqual([x['op'] for x in self.worker.calls], ['execute_actions', 'close'])
        self.assertTrue(self.worker.closed)
        self.assertEqual([x['name'] for x in self.server.tools], ['execute_actions', 'observe', 'done'])
        self.verifier.assert_called_once_with('atomic_coffee', 'mock-task', 'mock-raw-episode', self.bundle,
                                             closed_ack=True, worker_exit_code=0)
        self.assertEqual(len(self.commands), 2)
        self.assertNotIn('OWNER ONLY', json.dumps(self.server.sent))
        self.assertNotIn('OWNER ONLY', self.server.instructions)
        self.assertNotIn('native_output', self.server.instructions)

    def test_ack_and_exit_without_valid_family_receipt_do_not_claim_cleanup_or_score(self):
        self.verifier.return_value = False
        with self.mocked():
            self.assertEqual(entry.run(self.value), 1)
        report = self.report()
        self.assertEqual(report['status'], 'cleanup_unverified')
        self.assertFalse(report['native_cleanup_verified'])
        self.assertIsNone(report['score'])
        self.assertIsNone(report['evaluation_valid'])

    def test_wrong_receipt_output_rejected_without_fallback_or_second_collection(self):
        self.bundle['native_output'] = str(self.root / 'old-native-output')
        with self.mocked(), self.assertRaises(ValueError):
            entry.run(self.value)
        self.verifier.assert_not_called()
        self.assertFalse(self.report()['native_cleanup_verified'])
        self.assertEqual(len(self.commands), 2)
        self.assertEqual(sum(row['op'] == 'close' for row in self.worker.calls), 1)

    def test_receipt_reader_failure_is_not_retried(self):
        original = self.process

        def fail(argv, **kwargs):
            if argv == self.value['native_receipt']['read_argv']:
                self.commands.append((deepcopy(argv), kwargs))
                raise subprocess.CalledProcessError(1, argv)
            return original(argv, **kwargs)

        self.process = fail
        with self.mocked(), self.assertRaises(subprocess.CalledProcessError):
            entry.run(self.value)
        self.assertEqual(len(self.commands), 2)
        self.assertFalse(self.report()['native_cleanup_verified'])
        self.assertIsNone(self.report()['score'])

    def test_uncertain_action_holds_worker_without_close_or_receipt_fetch(self):
        def fail(request):
            self.worker.calls.append(deepcopy(request))
            raise TimeoutError('synthetic unacknowledged robot action')

        self.worker.exchange = fail
        with self.mocked(), self.assertRaises(RuntimeError):
            entry.run(self.value)
        self.assertEqual([x['op'] for x in self.worker.calls], ['execute_actions'])
        self.worker.process.wait.assert_called_once_with()
        self.assertFalse(self.worker.closed)
        self.verifier.assert_not_called()
        self.assertEqual(len(self.commands), 1)
        report = self.report()
        self.assertEqual(report['status'], 'uncertain_worker_held_for_operator')
        self.assertTrue(report['in_flight_action'])
        self.assertIsNone(report['score'])

    def test_bad_ready_never_starts_model_or_sends_close(self):
        self.worker.read = lambda: dict(op='ready', contract={'schema': 'wrong'})
        with self.mocked(), self.assertRaises(ValueError):
            entry.run(self.value)
        self.assertEqual(self.worker.calls, [])
        self.worker.process.wait.assert_called_once_with()
        self.assertEqual(self.server.events, 0)
        self.verifier.assert_not_called()

    def test_model_error_with_no_action_inflight_can_close_once_but_not_score(self):
        def fail(*args):
            raise RuntimeError('synthetic model transport failure')

        self.server.next_event = fail
        with self.mocked(), self.assertRaises(RuntimeError):
            entry.run(self.value)
        self.assertEqual(self.worker.calls, [{'op': 'close'}])
        self.assertTrue(self.report()['native_cleanup_verified'])
        self.assertFalse(self.report()['evaluation_valid'])
        self.assertIsNone(self.report()['score'])
        self.assertEqual(len(self.commands), 2)

    def test_version_mismatch_never_starts_worker_or_model(self):
        self.value['expected_cli_version'] = 'different'
        with self.mocked() as stack, self.assertRaises(ValueError):
            entry.run(self.value)
        self.assertEqual(self.worker.calls, [])
        self.assertEqual(self.server.events, 0)
        self.verifier.assert_not_called()
        self.assertEqual(self.report()['status'], 'invalid_before_start')

    def test_real_verifier_accepts_each_dedicated_owner_from_raw_entry(self):
        from test_native_receipts import receipt_case
        cases = [
            ('atomic_coffee', task) for task in ('coffee_bag_interference',
                'coffee_machine_push_interference', 'coffee_bag_then_machine_interference',
                'coffee_button_lighting_interference')]
        cases += [('failure_recovery', 'failure_recovery_' + variant) for variant in
                  ('shelf_drink', 'coffee_bag', 'sofa', 'table_waste_empty_grasp', 'table_waste_tipped_cup')]
        cases += [('cloth', task) for task in ('spread_tablecloth', 'collect_tablecloth', 'long_task_18')]
        for family, task in cases:
            with self.subTest(family=family, task=task):
                self.value['output'] = str(self.root / task / 'agent-output')
                binding = self.value['native_receipt']
                binding.update(family=family, task_id=task, native_output=str(self.root / task / 'native'))
                binding['read_argv'] = ['mock-reader', '--native-output', binding['native_output']]
                self.bundle = receipt_case(family, task, 'mock-raw-episode', binding['native_output'])
                self.worker, self.server = FakeWorker(), FakeServer()
                files = self.bundle['files']
                if family == 'cloth':
                    owner_result = files['formal_owner_report.json']['final_result']
                else:
                    native = files['native_report.json']
                    owner_result = native.get('result', native.get('task_result', native.get('closed_result')))
                self.worker.owner = deepcopy(owner_result)
                with self.mocked(real_verifier=True):
                    self.assertEqual(entry.run(self.value), 0)
                self.assertTrue(self.report()['native_cleanup_verified'])
                self.assertEqual(self.report()['status'], 'closed')
                self.assertEqual(self.report()['actions_executed'], 1)
                self.assertNotIn('source_evaluation', json.dumps(self.server.sent))
                self.assertNotIn('scoring_status', json.dumps(self.server.sent))

    def test_real_verifier_rejects_wrong_episode_after_transport_close(self):
        from test_native_receipts import receipt_case
        self.value['native_receipt']['task_id'] = 'coffee_bag_interference'
        self.bundle = receipt_case('atomic_coffee', 'coffee_bag_interference', 'old-episode',
                                   self.value['native_receipt']['native_output'])
        with self.mocked(real_verifier=True):
            self.assertEqual(entry.run(self.value), 1)
        self.assertEqual(self.report()['status'], 'cleanup_unverified')
        self.assertFalse(self.report()['native_cleanup_verified'])
        self.assertIsNone(self.report()['score'])
        self.assertEqual([x['op'] for x in self.worker.calls], ['execute_actions', 'close'])


if __name__ == '__main__':
    unittest.main()
