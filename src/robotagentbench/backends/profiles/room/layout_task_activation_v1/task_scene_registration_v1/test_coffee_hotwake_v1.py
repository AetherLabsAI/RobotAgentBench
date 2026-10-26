"""Synthetic clock/activation fixtures, not native navigation or task success."""
import unittest
from types import SimpleNamespace as NS
from coffee_hotwake_v1 import CoffeeHotwakeCycle, demand


class Clock:
    hz = 60
    current_time = 0.
    def __init__(self): self.calls = []; self.state = b'owned-native-state-fixture'
    def snapshot(self): return self.state
    def set_hz(self, hz): self.hz = hz; self.calls.append(hz)


class Tests(unittest.TestCase):
    def setUp(self):
        self.counts = dict(airborne=0)
        self.service = NS(consumer=None, controller=NS(dispensing=False), animation=NS(counts=lambda: self.counts))
        self.clock = Clock()
        self.owner = NS(service=self.service, closed=False)
        self.cycle = CoffeeHotwakeCycle(self.owner, self.clock)
        self.off = dict(state='DORMANT', active=False)
        self.on = dict(state='ACTIVE', active=True)

    def begin(self, activation=None, **kw):
        return self.cycle.begin_control(activation or self.off, command_vw=kw.pop('command_vw', [0, 0]),
                                        base_velocity=kw.pop('base_velocity', [0]*6), **kw)

    def finish(self, plan):
        for _ in range(plan['substeps']): self.clock.current_time += plan['physics_dt_s']
        return self.cycle.finish_control(completed_substeps=plan['substeps'])

    def test_60_240_60_no_reset_continuous_20hz(self):
        device = self.service.controller
        plans = [self.begin()]; self.finish(plans[-1])
        plans.append(self.begin(self.on)); self.finish(plans[-1])
        plans.append(self.begin()); self.finish(plans[-1])
        self.assertEqual([p['substeps'] for p in plans], [3, 12, 3])
        self.assertEqual(self.clock.calls, [240, 60])
        self.assertAlmostEqual(self.clock.current_time, .15)
        self.assertIs(self.service.controller, device)
        self.assertIsNone(self.service.consumer)

    def test_existing_retained_and_protected_outputs(self):
        for state in ('ACTIVE', 'PINNED'):
            self.assertEqual(demand(self.service, dict(state=state, active=True))['hz'], 240)
        for state in ('PRELOAD', 'DORMANT'):
            self.assertEqual(demand(self.service, dict(state=state, active=False))['hz'], 60)

    def test_leave_does_not_freeze_unfinished_score(self):
        score = NS(process=NS(ended=False))
        self.service.consumer = score
        plan = self.begin(); self.finish(plan)
        self.assertEqual(plan['physics_hz'], 240)
        self.assertIs(self.service.consumer, score)
        score.process.ended = True
        plan = self.begin(); self.finish(plan)
        self.assertEqual(plan['physics_hz'], 60)

    def test_dispensing_and_airborne_hold_rate(self):
        self.service.controller.dispensing = True
        self.assertEqual(demand(self.service, self.off)['hz'], 240)
        self.service.controller.dispensing = False; self.counts['airborne'] = 1
        self.assertEqual(demand(self.service, self.off)['hz'], 240)
        self.counts['airborne'] = 0
        self.assertEqual(demand(self.service, self.off)['hz'], 60)

    def test_moving_robot_defers_switch_until_stopped(self):
        plan = self.begin(self.on, command_vw=[.1, 0], base_velocity=[.1, 0, 0, 0, 0, 0])
        self.assertTrue(plan['switch_pending']); self.assertFalse(plan['coffee_rate_ready'])
        self.finish(plan)
        plan = self.begin(self.on); self.assertTrue(plan['coffee_rate_ready']); self.finish(plan)

    def test_other_region_cannot_be_downgraded(self):
        plan = self.begin(other_required_hz=240); self.finish(plan)
        self.assertEqual(plan['physics_hz'], 240)

    def test_invalid_or_proxy_activation_refused(self):
        for row in (dict(state='PROXY', active=False), dict(state='INVALID', active=True),
                    dict(state='ACTIVE', active=False), dict(state='ACTIVE', active=1)):
            with self.assertRaises(ValueError): self.begin(row)
        self.assertEqual(self.clock.calls, [])

    def test_duplicate_begin_and_wrong_substeps_refused(self):
        self.begin()
        with self.assertRaises(RuntimeError): self.begin()
        with self.assertRaises(RuntimeError): self.cycle.finish_control(completed_substeps=12)
        self.assertTrue(self.cycle.faulted)

    def test_mid_action_clock_change_refused(self):
        plan = self.begin(); self.clock.hz = 240
        with self.assertRaises(RuntimeError): self.finish(plan)

    def test_state_change_during_switch_fails_closed(self):
        def bad_set(hz): self.clock.hz = hz; self.clock.state = b'changed'
        self.clock.set_hz = bad_set
        with self.assertRaises(RuntimeError): self.begin(self.on)
        self.assertTrue(self.cycle.faulted)

    def test_active_score_cannot_wait_for_upgrade_at_60(self):
        self.service.consumer = NS(process=NS(ended=False))
        with self.assertRaises(RuntimeError): self.begin(command_vw=[.1, 0])


if __name__ == '__main__': unittest.main(verbosity=2)
