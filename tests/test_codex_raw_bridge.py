"""Synthetic CPU-only raw JSONL bridge checks; no model, GPU, or native owner."""
import base64
from copy import deepcopy
import json
from pathlib import Path
import struct
import tempfile
import threading
import unittest
from uuid import UUID

from robotagentbench.interfaces import robot_profiles
from robotagentbench.interfaces.action_contract import ORDER, specification
from robotagentbench.interfaces.observation_extras import MAP_SCHEMA, MAP_SOURCE
from robotagentbench.model_examples.codex import raw_bridge
from robotagentbench.model_examples.codex.raw_tools import TOOLS as RAW_TOOLS
from robotagentbench.model_examples.codex.tools import TOOLS as CODING_TOOLS


def png(width, height):
    # Enough bytes for the existing public PNG-header checker. Not an observation
    # of a real environment and never used as a model or physical test fixture.
    return base64.b64encode(b'\x89PNG\r\n\x1a\n' + struct.pack('>I', 13) + b'IHDR'
                           + struct.pack('>II', width, height) + bytes(9)).decode()


def observation(step=0, sizes=((640, 480),) * 3):
    return dict(episode_id='synthetic-raw-episode', step=step, time_s=step / 20,
                instruction='Synthetic bridge fixture; not a benchmark episode',
                context='Public synthetic context', image_step=step,
                joints=dict.fromkeys(robot_profiles.NATIVE_JOINTS, 0.),
                images={camera: png(*size) for camera, size in zip(('head', 'left', 'right'), sizes)},
                base_state=dict(position_world_m=[0., 0., 0.], orientation_world_xyzw=[0., 0., 0., 1.],
                                linear_velocity_world_m_s=[0., 0., 0.],
                                angular_velocity_world_rad_s=[0., 0., 0.], yaw_world_unwrapped_rad=0.))


def ready():
    return dict(op='ready', contract=specification(dict.fromkeys(ORDER[:20], [-4., 4.])),
                observation=observation())


def result(step=0, count=0, **extra):
    return dict(op='result', actions_executed=count, episode_finished=False,
                observation=observation(step), **extra)


class FakeWorker:
    def __init__(self, initial=None, replies=(), exit_code=0):
        self.initial = deepcopy(ready() if initial is None else initial)
        self.replies = list(replies)
        self.calls = []
        self.reads = self.closes = 0
        self.exit_code = exit_code

    def read(self):
        self.reads += 1
        return deepcopy(self.initial)

    def exchange(self, request):
        self.calls.append(deepcopy(request))
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return deepcopy(reply)

    def close(self):
        self.closes += 1
        if isinstance(self.exit_code, BaseException):
            raise self.exit_code
        return self.exit_code


class RawBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='raw-bridge-cpu-')
        self.root = Path(self.temp.name)
        self.output_index = 0

    def tearDown(self):
        self.temp.cleanup()

    def bridge(self, worker=None, **options):
        self.output_index += 1
        worker = worker or FakeWorker()
        return raw_bridge.RawRobotBridge(worker, self.root / str(self.output_index), **options)

    def test_published_tool_schemas_only(self):
        expected = deepcopy(RAW_TOOLS) + [t for t in CODING_TOOLS
                                        if t['function']['name'] in ('observe', 'done')]
        self.assertEqual(raw_bridge.TOOLS, expected)
        definitions = raw_bridge.tool_specs()
        self.assertEqual([t['name'] for t in definitions], ['execute_actions', 'observe', 'done'])
        definitions[0]['inputSchema']['properties'].clear()
        self.assertEqual(raw_bridge.TOOLS, expected)

    def test_native_camera_profiles_state_context_and_no_privileged_fields(self):
        for sizes in (((640, 480),) * 3, ((1280, 800), (848, 480), (848, 480))):
            with self.subTest(sizes=sizes):
                initial = ready()
                initial['observation'] = observation(sizes=sizes)
                initial['observation'].update(score=999, target_pose=[9, 9, 9], tcp={'left': 'private'})
                initial['private_targets'] = {'secret': 'private marker'}
                worker = FakeWorker(initial)
                bridge = self.bridge(worker)
                content = json.dumps(bridge.initial_items())
                for forbidden in ('score', 'target_pose', 'private marker', 'tcp'):
                    self.assertNotIn(forbidden, content)
                self.assertEqual(worker.reads, 1)
                self.assertEqual(bridge.observed['images'], initial['observation']['images'])
                self.assertEqual(len(bridge.observed['joints']), 34)
                self.assertIn('Public synthetic context', content)
                self.assertNotIn('explored_map', content)
                self.assertNotIn('supplied explored map', bridge.instructions)
                self.assertNotIn('move_to(', bridge.instructions)
                self.assertNotIn('move_base', bridge.instructions)

    def test_public_map_is_preserved_without_creating_one(self):
        initial = ready()
        initial['observation']['navigation'] = dict(schema=MAP_SCHEMA, step=0, map_png=png(3, 3),
            origin_m=[0., 0.], resolution_m=.04, shape=[1, 1], explored_fraction=.1,
            map_source=MAP_SOURCE, premapped=False, task_labels=False, body_footprint_radius_m=.3)
        bridge = self.bridge(FakeWorker(initial))
        self.assertEqual(bridge.observed['navigation'], initial['observation']['navigation'])
        self.assertEqual(sum(item['type'] == 'inputImage' for item in bridge.initial_items()), 4)
        self.assertIn('supplied explored map', bridge.instructions)

    def test_coding_or_unknown_contract_and_resume_rejected(self):
        changes = [lambda row: row.update(policy_schema='short_tasks.coding_robot22.v1'),
                   lambda row: row['contract'].update(control_hz=60),
                   lambda row: row['contract'].update(extra='private'),
                   lambda row: row['contract'].update(observation_fields=['score']),
                   lambda row: row['contract'].update(max_tool_steps=True),
                   lambda row: row['contract'].update(joint_units={}),
                   lambda row: row.update(observation=observation(1)),
                   lambda row: row.update(native_owner_constraints={'fixed_base': False})]
        for change in changes:
            initial = ready()
            change(initial)
            with self.subTest(initial=initial.get('contract')), self.assertRaises(ValueError):
                self.bridge(FakeWorker(initial))

    def test_exact_model_rows_episode_step_and_fresh_uuid(self):
        worker = FakeWorker(replies=[result(2, 2), result(3, 1)])
        bridge = self.bridge(worker)
        rows = [[index / 100 for index in range(22)], [index / 200 for index in range(22)]]
        self.assertTrue(bridge.dispatch('first', 'execute_actions', {'actions': rows})['success'])
        sent = worker.calls[0]
        self.assertEqual(sent['actions'], rows)
        self.assertEqual(sent['expected_step'], 0)
        self.assertEqual(sent['episode_id'], 'synthetic-raw-episode')
        UUID(sent['request_id'])
        bridge.dispatch('second', 'execute_actions', {'actions': rows[:1]})
        self.assertEqual(worker.calls[1]['expected_step'], 2)
        self.assertNotEqual(worker.calls[0]['request_id'], worker.calls[1]['request_id'])
        self.assertEqual(bridge.observed['step'], 3)
        self.assertIsNone(bridge.pending)

    def test_fixed_base_owner_constraints_do_not_rewrite_model_command(self):
        initial = ready()
        initial['native_owner_constraints'] = deepcopy(raw_bridge.FIXED_OWNER_CONSTRAINTS)
        worker = FakeWorker(initial, [result(status='rejected')])
        bridge = self.bridge(worker)
        row = [0.] * 22
        row[-2:] = [.02, .01]
        feedback = bridge.dispatch('move', 'execute_actions', {'actions': [row]})
        self.assertFalse(feedback['success'])
        self.assertEqual(worker.calls[0]['actions'], [row])
        self.assertIn('required_base_commands', bridge.instructions)
        self.assertIsNone(bridge.pending)

    def test_native_rejection_is_failed_tool_not_zero_score_or_silent_retry(self):
        worker = FakeWorker(replies=[result(status='rejected', score=0, error='PRIVATE_OWNER_EXCEPTION')])
        bridge = self.bridge(worker)
        args = {'actions': [[0.] * 22]}
        feedback = bridge.dispatch('rejected', 'execute_actions', args)
        self.assertFalse(feedback['success'])
        text = json.dumps(feedback)
        self.assertIn('Native worker rejected', text)
        self.assertNotIn('PRIVATE_OWNER_EXCEPTION', text)
        self.assertNotIn('score', text)
        self.assertEqual(feedback, bridge.dispatch('rejected', 'execute_actions', args))
        self.assertEqual(len(worker.calls), 1)
        self.assertFalse(bridge.completed)

    def test_unknown_extra_empty_and_late_malformed_rows_send_nothing(self):
        worker = FakeWorker()
        bridge = self.bridge(worker)
        examples = [('execute_python', {'code': 'not exposed'}), ('observe', {'extra': 1}),
                    ('execute_actions', {'actions': []}),
                    ('execute_actions', {'actions': [[0.] * 22, [0.] * 21]}),
                    ('execute_actions', {'actions': [[0.] * 22, [True] + [0.] * 21]}),
                    ('execute_actions', {'actions': [[0.] * 22] * 601}), ('done', {'reason': '  '})]
        for index, (name, args) in enumerate(examples):
            with self.subTest(name=name, index=index):
                self.assertFalse(bridge.dispatch(str(index), name, args)['success'])
        with self.assertRaises(ValueError):
            bridge.dispatch('nonfinite', 'execute_actions', {'actions': [[0.] * 22, [float('nan')] * 22]})
        self.assertEqual(worker.calls, [])

    def test_narrower_native_chunk_limit_respected(self):
        initial = ready()
        initial['contract']['max_tool_steps'] = 1
        worker = FakeWorker(initial)
        bridge = self.bridge(worker)
        self.assertFalse(bridge.dispatch('too_many', 'execute_actions', {'actions': [[0.] * 22] * 2})['success'])
        self.assertEqual(worker.calls, [])

    def test_observe_and_response_dedup_do_not_advance_simulation(self):
        worker = FakeWorker(replies=[result()])
        bridge = self.bridge(worker)
        first = bridge.dispatch('once', 'observe', {})
        self.assertEqual(first, bridge.dispatch('once', 'observe', {}))
        self.assertEqual(worker.calls, [{'op': 'observe'}])
        self.assertEqual(bridge.observed['step'], 0)
        with self.assertRaises(RuntimeError):
            bridge.dispatch('once', 'done', {'reason': 'different'})

    def test_partial_execution_only_accepted_when_native_episode_ends(self):
        reply = result(1, 1)
        reply['episode_finished'] = True
        bridge = self.bridge(FakeWorker(replies=[reply]))
        feedback = bridge.dispatch('partial', 'execute_actions', {'actions': [[0.] * 22] * 2})
        self.assertTrue(feedback['success'])
        self.assertTrue(bridge.completed)
        self.assertNotIn('score', json.dumps(feedback))
        self.assertFalse(bridge.dispatch('later', 'observe', {})['success'])

    def test_uncertain_ack_is_held_and_never_replayed_or_closed(self):
        changes = [lambda reply: reply.update(actions_executed=True),
                   lambda reply: reply.update(actions_executed=2),
                   lambda reply: reply.update(episode_finished='false'),
                   lambda reply: reply.update(status='unknown'),
                   lambda reply: reply.update(status='rejected'),
                   lambda reply: reply['observation'].update(episode_id='other'),
                   lambda reply: reply['observation'].update(time_s=.1),
                   lambda reply: reply['observation'].update(step=0, image_step=0),
                   lambda reply: reply.update(op='ready')]
        for change in changes:
            reply = result(1, 1)
            change(reply)
            worker = FakeWorker(replies=[reply])
            bridge = self.bridge(worker)
            with self.subTest(reply=reply), self.assertRaises((RuntimeError, ValueError)):
                bridge.dispatch('one', 'execute_actions', {'actions': [[0.] * 22]})
            self.assertIsNotNone(bridge.pending)
            for name, args in [('observe', {}), ('execute_actions', {'actions': [[0.] * 22]})]:
                with self.assertRaises(RuntimeError):
                    bridge.dispatch('next', name, args)
            with self.assertRaises(RuntimeError):
                bridge.close()
            self.assertEqual(len(worker.calls), 1)
            self.assertEqual(worker.closes, 0)

    def test_timeout_after_send_keeps_original_pending_and_no_retry(self):
        worker = FakeWorker(replies=[TimeoutError('synthetic timeout')])
        bridge = self.bridge(worker)
        with self.assertRaises(TimeoutError):
            bridge.dispatch('one', 'execute_actions', {'actions': [[0.] * 22]})
        saved = json.loads((bridge.output / 'pending_action.json').read_text())
        self.assertEqual(saved['request'], worker.calls[0])
        with self.assertRaises(RuntimeError):
            bridge.dispatch('one', 'execute_actions', {'actions': [[0.] * 22]})
        with self.assertRaises(RuntimeError):
            bridge.close()
        self.assertEqual(len(worker.calls), 1)

    def test_parallel_dispatch_and_close_blocked_while_exchange_waits(self):
        worker = FakeWorker()
        entered, finish = threading.Event(), threading.Event()
        errors = []

        def exchange(request):
            worker.calls.append(deepcopy(request))
            entered.set()
            if not finish.wait(3):
                raise TimeoutError('synthetic test thread timed out')
            return result()

        worker.exchange = exchange
        bridge = self.bridge(worker)

        def first():
            try:
                bridge.dispatch('first', 'observe', {})
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=first)
        thread.start()
        try:
            self.assertTrue(entered.wait(3))
            with self.assertRaises(RuntimeError):
                bridge.dispatch('second', 'observe', {})
            with self.assertRaises(RuntimeError):
                bridge.close()
        finally:
            finish.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(worker.calls), 1)

    def test_done_is_local_abandonment_not_completion_or_close(self):
        worker = FakeWorker()
        bridge = self.bridge(worker)
        self.assertTrue(bridge.dispatch('done', 'done', {'reason': 'stop the synthetic test'})['success'])
        self.assertTrue(bridge.abandoned)
        self.assertFalse(bridge.completed)
        self.assertFalse(bridge.closed)
        self.assertEqual(worker.calls, [])

    def test_read_only_and_budget_send_no_extra_rpc(self):
        worker = FakeWorker(replies=[result()])
        bridge = self.bridge(worker, read_only=True, max_model_responses=2)
        self.assertFalse(bridge.dispatch('move', 'execute_actions', {'actions': [[0.] * 22]})['success'])
        self.assertTrue(bridge.dispatch('read', 'observe', {})['success'])
        with self.assertRaises(RuntimeError):
            bridge.dispatch('over_budget', 'observe', {})
        self.assertEqual(worker.calls, [{'op': 'observe'}])

    def test_close_checks_ack_and_process_exit_not_native_cleanup(self):
        owner_result = {'score': 3, 'success': False, 'private_targets': 'operator only'}
        worker = FakeWorker(replies=[dict(op='closed', result=owner_result)])
        bridge = self.bridge(worker)
        before = bridge.initial_items()
        receipt = bridge.close()
        self.assertTrue(receipt['closed_ack'])
        self.assertTrue(receipt['worker_closed'])
        self.assertFalse(receipt['native_cleanup_verified'])
        self.assertTrue(receipt['owner_receipt_verification_required'])
        self.assertEqual(bridge.owner_result, owner_result)
        self.assertEqual(bridge.initial_items(), before)
        self.assertEqual(json.loads((bridge.output / 'closure.json').read_text()), receipt)
        self.assertEqual(receipt, bridge.close())
        self.assertEqual(worker.calls, [{'op': 'close'}])
        self.assertEqual(worker.closes, 1)
        self.assertFalse(bridge.dispatch('after_close', 'observe', {})['success'])

    def test_nonzero_exit_does_not_claim_closed_and_does_not_retry(self):
        worker = FakeWorker(replies=[dict(op='closed')], exit_code=7)
        bridge = self.bridge(worker)
        receipt = bridge.close()
        self.assertTrue(receipt['closed_ack'])
        self.assertFalse(receipt['worker_closed'])
        self.assertFalse(bridge.closed)
        self.assertEqual(bridge.close(), receipt)
        self.assertEqual(len(worker.calls), 1)

    def test_close_uncertainty_records_evidence_without_resending(self):
        cases = [(TimeoutError('close timeout'), 0),
                 (dict(op='result'), 0),
                 (dict(op='closed'), TimeoutError('exit timeout')),
                 (dict(op='closed'), None)]
        for reply, exit_code in cases:
            worker = FakeWorker(replies=[reply], exit_code=exit_code)
            bridge = self.bridge(worker)
            with self.subTest(reply=reply, exit_code=exit_code), self.assertRaises((RuntimeError, TimeoutError)):
                bridge.close()
            receipt = json.loads((bridge.output / 'closure.json').read_text())
            self.assertFalse(receipt['worker_closed'])
            self.assertFalse(receipt['native_cleanup_verified'])
            self.assertIn('error_type', receipt)
            with self.assertRaises(RuntimeError):
                bridge.close()
            self.assertEqual(len(worker.calls), 1)


if __name__ == '__main__':
    unittest.main()
