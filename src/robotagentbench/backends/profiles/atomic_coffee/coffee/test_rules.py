import unittest
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from .trigger import TriggerFacts, PourApproachTrigger
from .moving_frame import MovingMachineFrame, mouth_above_opening
from .push_motion import proposal, away_from_mouth_proposal, route_candidates, select_admitted_direction, NoPushRoute, PalmPushMotion


def upright():
    value = np.eye(4)
    value[:3, :3] = np.column_stack(([1., 0., 0.], [0., 0., 1.], [0., -1., 0.]))
    return value


class TriggerTests(unittest.TestCase):
    def setUp(self):
        self.trigger = PourApproachTrigger()
        self.trigger.arm()
        self.facts = TriggerFacts(1., True, True, 45., True)

    def test_tilt_boundary_and_once(self):
        self.assertFalse(self.trigger.observe(replace(self.facts, bag_tilt_deg=44.999), route_clear=True)['start'])
        self.assertTrue(self.trigger.observe(replace(self.facts, time_s=2.), route_clear=True)['start'])
        self.assertFalse(self.trigger.observe(replace(self.facts, time_s=3.), route_clear=True)['start'])

    def test_current_hold_not_history(self):
        self.trigger.observe(replace(self.facts, mouth_above_opening=False), route_clear=True)
        self.assertFalse(self.trigger.observe(replace(self.facts, time_s=2., holding_bag=False), route_clear=True)['start'])

    def test_wrong_location(self):
        self.assertFalse(self.trigger.observe(replace(self.facts, mouth_above_opening=False), route_clear=True)['start'])

    def test_blocked_does_not_consume_trigger(self):
        self.assertFalse(self.trigger.observe(self.facts, route_clear=False)['start'])
        self.assertTrue(self.trigger.observe(replace(self.facts, time_s=2.), route_clear=True)['start'])

    def test_missing_contacts(self):
        with self.assertRaises(RuntimeError):
            self.trigger.observe(replace(self.facts, contacts_complete=False), route_clear=True)
        self.assertIsNone(self.trigger.last_time)

    def test_duplicate_and_invalid(self):
        self.trigger.observe(self.facts, route_clear=True)
        with self.assertRaises(ValueError):
            self.trigger.observe(self.facts, route_clear=True)
        with self.assertRaises(ValueError):
            self.trigger.observe(replace(self.facts, time_s=2., bag_tilt_deg=float('nan')), route_clear=True)

    def test_requires_admission(self):
        with self.assertRaises(RuntimeError):
            PourApproachTrigger().observe(self.facts, route_clear=True)
        with self.assertRaises(RuntimeError):
            self.trigger.arm()


class FrameTests(unittest.TestCase):
    def test_translated_rotated_body_offset(self):
        body_in_root = np.eye(4)
        body_in_root[:3, 3] = [.02, .01, 0.]
        frame = MovingMachineFrame(body_in_root)
        q = Rotation.from_euler('z', 37., degrees=True).as_quat()
        frame.update([1., 2., 3.], q[[3, 0, 1, 2]], 1.)
        points = np.array([[0., 0., 0.], [.1, .2, .3]])
        np.testing.assert_allclose(frame.local(frame.world(points, 1.), 1.), points, atol=1e-14)
        np.testing.assert_allclose(frame.world(body_in_root[:3, 3], 1.), [1., 2., 3.])

    def test_stale_and_scaled_rejected(self):
        with self.assertRaises(ValueError):
            MovingMachineFrame(np.diag([2., 1., 1., 1.]))
        frame = MovingMachineFrame(np.eye(4))
        frame.update([0, 0, 0], [1, 0, 0, 0], 1.)
        with self.assertRaises(ValueError):
            frame.current(2.)
        with self.assertRaises(ValueError):
            frame.update([0, 0, 0], [1, 0, 0, 0], 1.)

    def test_live_opening_moves_not_old_position(self):
        hopper = SimpleNamespace(rim_y=.3, center_z=0., half_width=.1, half_depth=.08, corner_radius=.02)
        matrix = upright()
        self.assertTrue(mouth_above_opening([0., 0., .4], matrix, hopper))
        self.assertFalse(mouth_above_opening([0., 0., .29], matrix, hopper))
        self.assertFalse(mouth_above_opening([.1, .08, .4], matrix, hopper))
        matrix[:3, 3] = [.2, 0., 0.]
        self.assertFalse(mouth_above_opening([0., 0., .4], matrix, hopper))
        self.assertTrue(mouth_above_opening([.2, 0., .4], matrix, hopper))

    def test_tilted_machine_vertical_projection(self):
        hopper = SimpleNamespace(rim_y=.3, center_z=0., half_width=.1, half_depth=.08, corner_radius=.02)
        matrix = upright()
        matrix[:3, :3] = Rotation.from_euler('y', 15., degrees=True).as_matrix() @ matrix[:3, :3]
        rim = matrix[:3, :3] @ [0., .3, 0.]
        self.assertTrue(mouth_above_opening(rim + [0., 0., .1], matrix, hopper))


class PushTests(unittest.TestCase):
    def candidates(self):
        hopper = SimpleNamespace(rim_y=.3, center_z=0., half_width=.086, half_depth=.072, corner_radius=.014)
        return route_candidates(2, upright(), [.04, 0., .4], hopper)

    def test_opposite_candidate_preserves_distance_and_honest_prediction(self):
        a, b, *_ = self.candidates()
        self.assertEqual((a['side'], b['side']), ('left', 'right'))
        np.testing.assert_allclose(b['direction_world_xy_unit'], -np.asarray(a['direction_world_xy_unit']))
        for p in (a, b):
            self.assertEqual((p['distance_m'], p['push_duration_s']), (.1, 1.))
            self.assertFalse(p['future_source_actions_used'])
        self.assertFalse(b['predicted_current_mouth_outside_after_push'])

    def test_preferred_admitted_does_not_check_other_side(self):
        calls = []
        def admit(p):
            calls.append(p['side'])
            return True, dict(proposal=p)
        clear, result = select_admitted_direction(self.candidates(), admit)
        self.assertTrue(clear)
        self.assertEqual(calls, ['left'])
        self.assertFalse(result['opposite_side_fallback_used'])

    def test_blocked_preferred_checks_other_side_same_trigger_frame(self):
        trigger = PourApproachTrigger()
        trigger.arm()
        calls = []
        def admit(p):
            calls.append(p['side'])
            return (True, dict(proposal=p)) if p['side'] == 'right' else (False, dict(reason='native_machine_sweep_blocked'))
        clear, result = select_admitted_direction(self.candidates(), admit)
        self.assertEqual(calls, ['left', 'right'])
        self.assertTrue(result['opposite_side_fallback_used'])
        self.assertEqual(result['proposal']['side'], 'right')
        self.assertTrue(trigger.observe(TriggerFacts(1., True, True, 45., True), route_clear=clear)['start'])

    def test_all_blocked_explicit_error_not_delayed_retry(self):
        trigger = PourApproachTrigger()
        trigger.arm()
        with self.assertRaises(NoPushRoute) as caught:
            select_admitted_direction(self.candidates(), lambda p: (False, dict(reason='native_palm_route_blocked')))
        self.assertEqual(len(caught.exception.attempts), 4)
        self.assertIn('late retry disabled', str(caught.exception))
        self.assertFalse(trigger.fired)

    def test_front_back_fallback_at_same_45_degree_frame(self):
        for selected in (2, 3):
            calls = []
            candidates = self.candidates()
            def admit(p):
                calls.append(p['side'])
                return p is candidates[selected], dict(proposal=p)
            clear, evidence = select_admitted_direction(candidates, admit)
            self.assertEqual(calls, [p['side'] for p in candidates[:selected+1]])
            self.assertTrue(evidence['front_back_fallback_used'])
            self.assertFalse(evidence['opposite_side_fallback_used'])
            trigger = PourApproachTrigger()
            trigger.arm()
            self.assertTrue(trigger.observe(TriggerFacts(1., True, True, 45., True), route_clear=clear)['start'])

    def test_four_distinct_horizontal_directions_preserve_stroke(self):
        candidates = self.candidates()
        self.assertEqual({p['side'] for p in candidates}, {'left', 'right', 'front', 'back'})
        for p in candidates:
            self.assertAlmostEqual(np.linalg.norm(p['direction_world_xy_unit']), 1.)
            self.assertEqual(p['direction_world_xy_unit'][2], 0.)
            self.assertEqual((p['distance_m'], p['push_duration_s']), (.1, 1.))
            self.assertFalse(p['future_source_actions_used'])
        self.assertAlmostEqual(np.dot(candidates[0]['direction_world_xy_unit'], candidates[2]['direction_world_xy_unit']), 0.)

    def test_wide_hopper_does_not_prevent_front_back_route_search(self):
        hopper = SimpleNamespace(rim_y=.3, center_z=0., half_width=.2, half_depth=.08, corner_radius=.02)
        candidates = route_candidates(2, upright(), [0., 0., .4], hopper)
        self.assertFalse(candidates[0]['predicted_current_mouth_outside_after_push'])
        self.assertTrue(candidates[2]['predicted_current_mouth_outside_after_push'])

    def test_native_owner_does_not_catch_blocked_routes_and_wait(self):
        from .native_push import NativePalmPush
        owner = NativePalmPush.__new__(NativePalmPush)
        owner.reader = SimpleNamespace(trigger_facts=lambda: TriggerFacts(1., True, True, 45., True))
        owner.pending = None
        owner.start = None
        def fail(poses):
            raise NoPushRoute([dict(side=p['side'], admitted=False) for p in self.candidates()])
        owner._admit = fail
        with self.assertRaises(NoPushRoute):
            owner.observe(robot_native_poses={})

    def test_preferred_fit_error_still_checks_opposite(self):
        def admit(p):
            if p['side'] == 'left':
                raise ValueError('No finite palm placement')
            return True, dict(proposal=p)
        clear, result = select_admitted_direction(self.candidates(), admit)
        self.assertTrue(clear)
        self.assertTrue(result['opposite_side_fallback_used'])
        self.assertIn('finite palm', result['direction_attempts'][0]['evidence']['reason'])

    def test_native_owner_wires_both_routes_to_same_current_frame(self):
        from .native_push import NativePalmPush
        from evaluation.tasks.coffee_machine.specs import BAG
        owner = NativePalmPush.__new__(NativePalmPush)
        owner.seed = 2
        times, calls = [], []
        def current(time_s):
            times.append(time_s)
            return upright()
        owner.reader = SimpleNamespace(
            latest=dict(time_s=16., poses={BAG: dict(position_m=[.04, 0., .4], quaternion_wxyz=[1., 0., 0., 0.])}),
            bag_mouth=np.zeros(3), live_machine=SimpleNamespace(current=current),
            contract=SimpleNamespace(hopper=SimpleNamespace(rim_y=.3, center_z=0., half_width=.086, half_depth=.072, corner_radius=.014)))
        robot = {'same_native_frame': object()}
        def admit(candidate, poses):
            self.assertIs(poses, robot)
            calls.append(candidate['side'])
            return candidate['side'] == 'right', dict(proposal=candidate)
        owner._admit_direction = admit
        clear, evidence = owner._admit(robot)
        self.assertTrue(clear)
        self.assertEqual(times, [16.])
        self.assertEqual(calls, ['left', 'right'])
        self.assertTrue(evidence['opposite_side_fallback_used'])

    def test_far_push_is_away_from_current_mouth_not_seeded_toward_it(self):
        hopper = SimpleNamespace(rim_y=.3, center_z=0., half_width=.086, half_depth=.072, corner_radius=.014)
        for x in (-.04, .04):
            for seed in range(10):
                p = away_from_mouth_proposal(seed, upright(), [x, 0., .4], hopper)
                self.assertEqual(p['distance_m'], .1)
                self.assertEqual(p['sign'], -1 if x > 0 else 1)
                target = upright()
                target[:3, 3] += np.array(p['direction_world_xy_unit']) * p['distance_m']
                self.assertFalse(mouth_above_opening([x, 0., .4], target, hopper))
                self.assertFalse(p['future_source_actions_used'])

    def test_no_silent_expansion_when_ten_cm_cannot_clear_opening(self):
        hopper = SimpleNamespace(rim_y=.3, center_z=0., half_width=.2, half_depth=.08, corner_radius=.02)
        with self.assertRaisesRegex(ValueError, '10 cm cannot clear'):
            away_from_mouth_proposal(2, upright(), [0., 0., .4], hopper)

    def test_seeded_sides_range(self):
        sides = set()
        for seed in range(100):
            p = proposal(seed, upright())
            self.assertEqual(p, proposal(seed, upright()))
            self.assertTrue(.05 <= p['distance_m'] <= .1)
            sides.add(p['side'])
        self.assertEqual(sides, {'left', 'right'})

    def test_rotated_axis_and_no_target_commands(self):
        matrix = upright()
        matrix[:3, :3] = Rotation.from_euler('z', 40., degrees=True).as_matrix() @ matrix[:3, :3]
        p = proposal(3, matrix)
        motion = PalmPushMotion([0., 0., 1.], p)
        first, last = motion.sample(0.), motion.sample(1.)
        self.assertEqual(first['wrist_position_m'], [0., 0., 1.])
        self.assertAlmostEqual(np.linalg.norm(np.array(last['wrist_position_m']) - [0., 0., 1.]), p['distance_m'])
        self.assertEqual(last['wrist_position_m'][2], 1.)
        self.assertEqual(last['closure'], 0.)
        self.assertFalse(last['machine_pose_commands'])
        self.assertEqual(last, motion.sample(2.))


if __name__ == '__main__':
    unittest.main()
