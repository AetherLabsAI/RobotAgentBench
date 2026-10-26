"""Control-contract tests with CPU doubles, not native manipulation validation."""
from copy import deepcopy
import json
from io import StringIO
import unittest

from robotagentbench.interfaces import action_contract, coding_contract
from robotagentbench.runtime import action_host, coding_action_host
from robotagentbench.runtime.action_driver import NativeDriver as ActionDriver
from robotagentbench.runtime.coding_driver import NativeDriver as CodingDriver
from robotagentbench.runtime.completion import CompletionStream
from native_fixtures import make_host, png, request

BRANCHES = ((action_host, ActionDriver), (coding_action_host, CodingDriver))


class NativeControlTests(unittest.TestCase):
    def test_whole_chunk_is_validated_before_any_native_write(self):
        for module, driver in BRANCHES:
            with self.subTest(branch=module.__name__):
                host, session = make_host(module.SessionHost, driver)
                before = deepcopy(session.trace)
                command = request(count=2)
                command['actions'][-1][-2] = .1  # Invalid acceleration from rest.
                with self.assertRaises(module.InputRejected):
                    host.execute(command)
                self.assertEqual(session.trace, before)
                self.assertFalse(session.faulted)
                self.assertEqual(session.control_index, 0)

    def test_20_hz_boundaries_and_distinct_actor_hook_behavior(self):
        for module, driver in BRANCHES:
            for hz in (60, 240):
                with self.subTest(branch=module.__name__, hz=hz):
                    host, session = make_host(module.SessionHost, driver, hz=hz)
                    reply = host.execute(request(count=2))
                    self.assertEqual(reply['actions_executed'], 2)
                    self.assertAlmostEqual(session.clock.current_time, .1)
                    self.assertEqual(session.control_index, 2)
                    steps = [row for row in session.trace if row[0] == 'simulate']
                    hooks = [row for row in session.trace if row[0] == 'before']
                    self.assertEqual(len(steps), hz // 10)
                    self.assertEqual(len(hooks), len(steps) if driver is ActionDriver else 0)
                    self.assertNotIn('opaque_task_result', reply['observation'])

    def test_terminal_session_stops_chunk_without_reinterpreting_result(self):
        for module, driver in BRANCHES:
            with self.subTest(branch=module.__name__):
                host, session = make_host(module.SessionHost, driver, terminal_after=1)
                reply = host.execute(request(count=3))
                self.assertEqual(reply['actions_executed'], 1)
                self.assertTrue(reply['episode_finished'])
                self.assertEqual(session.result()['opaque_task_result'], {'preserved': True})
                with self.assertRaises(module.InputRejected):
                    host.execute(request(step=1, request_id='00000000-0000-0000-0000-000000000002'))

    def test_duplicate_uuid_returns_detached_cached_reply_without_replay(self):
        for module, driver in BRANCHES:
            with self.subTest(branch=module.__name__):
                host, session = make_host(module.SessionHost, driver)
                command = request()
                first = host.execute(command)
                trace = deepcopy(session.trace)
                second = host.execute(command)
                self.assertEqual(first, second)
                self.assertEqual(session.trace, trace)
                second['observation']['joints'].clear()
                self.assertEqual(first, host.execute(command))
                altered = deepcopy(command)
                altered['actions'][0][0] = .01
                with self.assertRaises(module.InputRejected):
                    host.execute(altered)

    def test_expired_uuid_and_wrong_step_are_rejected_without_replay(self):
        for module, driver in BRANCHES:
            with self.subTest(branch=module.__name__):
                host, session = make_host(module.SessionHost, driver)
                first = request()
                host.execute(first)
                host.execute(request(step=1, request_id='00000000-0000-0000-0000-000000000002'))
                trace = deepcopy(session.trace)
                for command in (first, request(step=0, request_id='00000000-0000-0000-0000-000000000003')):
                    with self.assertRaises(module.InputRejected):
                        host.execute(command)
                    self.assertEqual(session.trace, trace)

    def test_uncertain_native_write_faults_and_never_retries(self):
        for module, driver in BRANCHES:
            with self.subTest(branch=module.__name__):
                host, session = make_host(module.SessionHost, driver)
                def broken(dt):
                    session.clock.current_time += dt
                    raise RuntimeError('native failure after write')
                host.driver.simulate_fetch = broken
                with self.assertRaises(RuntimeError):
                    host.execute(request())
                self.assertTrue(host.faulted)
                self.assertEqual(session.result()['status'], 'error')
                trace = deepcopy(session.trace)
                with self.assertRaises(RuntimeError):
                    host.execute(request())
                self.assertEqual(trace, session.trace)

    def test_robot_only_mode_does_not_render_or_reuse_other_mode_reply(self):
        host, session = make_host(coding_action_host.SessionHost, CodingDriver)
        before = len([row for row in session.trace if row[0] == 'capture'])
        reply = host.execute(request(), robot_only=True)
        self.assertNotIn('images', reply['observation'])
        self.assertEqual(before, len([row for row in session.trace if row[0] == 'capture']))
        with self.assertRaises(coding_action_host.InputRejected):
            host.execute(request(), robot_only=False)

    def test_observation_must_not_advance_physics(self):
        for module, driver in BRANCHES:
            with self.subTest(branch=module.__name__):
                host, session = make_host(module.SessionHost, driver)
                original = host.driver.capture_png
                def stepping_capture():
                    session.clock.current_time += .01
                    return original()
                host.driver.capture_png = stepping_capture
                with self.assertRaises(RuntimeError):
                    host.observe()
                self.assertTrue(session.faulted)

    def test_camera_profile_and_public_whitelist(self):
        host, _ = make_host(action_host.SessionHost, ActionDriver)
        observation = host.observe()
        observation['private_scene'] = {'scores': [100], 'target_pose': [1, 2, 3]}
        observation['base_state']['hidden'] = 'not public'
        filtered = action_contract.public_observation(observation)
        self.assertNotIn('private_scene', filtered)
        self.assertNotIn('hidden', filtered['base_state'])
        observation['images'] = {'head': png(1280, 800), 'left': png(848, 480), 'right': png(848, 480)}
        self.assertEqual(coding_contract.public_observation(observation)['images'], observation['images'])
        observation['images']['right'] = png(640, 480)
        with self.assertRaises(ValueError):
            action_contract.public_observation(observation)

    def test_rpc_rejection_is_sanitized_and_eof_closes_session(self):
        for module, driver in BRANCHES:
            with self.subTest(branch=module.__name__):
                host, session = make_host(module.SessionHost, driver)
                output = StringIO()
                module.serve(host, input_stream=StringIO('{bad json}\n'), output_stream=output)
                replies = [json.loads(line) for line in output.getvalue().splitlines()]
                self.assertEqual(replies[-1]['status'], 'rejected')
                self.assertEqual(replies[-1]['actions_executed'], 0)
                self.assertEqual(session.control_index, 0)
                self.assertEqual(session.close_count, 1)

    def test_close_acknowledgement_waits_for_native_cleanup(self):
        output = StringIO()
        stream = CompletionStream(output)
        closed = json.dumps({'op': 'closed', 'result': {'status': 'finished'}}) + '\n'
        stream.write(closed)
        stream.flush()
        self.assertEqual(output.getvalue(), '')
        with self.assertRaises(RuntimeError):
            stream.write(closed)
        stream.complete()
        self.assertEqual(output.getvalue(), closed)
        stream.complete()
        self.assertEqual(output.getvalue(), closed)


if __name__ == '__main__':
    unittest.main()
