"""Synthetic CPU reader tests, explicitly NOT native physical execution."""
import io
import unittest
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from evaluation.tasks.coffee_machine.specs import BAG, CUP, BUTTON, MACHINE_BODY
from evaluation.tasks.coffee_machine.isaac_evidence import PHYSICS_DT
from .isaac_evidence import MovingCoffeeReader
from .moving_frame import MovingMachineFrame


class FakeView:
    def __init__(self, position, quaternion):
        self.position, self.quaternion = position, quaternion

    def get_world_pose(self):
        return self.position, self.quaternion


class ReaderTests(unittest.TestCase):
    def fixture(self):
        reader = object.__new__(MovingCoffeeReader)
        reader.closed = False
        reader.world = SimpleNamespace(current_time=PHYSICS_DT)
        reader.last_time = 0.
        reader.live_machine = MovingMachineFrame(np.eye(4))
        reader.contract = SimpleNamespace(cup=(0., .02, .067), outlet=(0., .158, .067))
        q = Rotation.from_euler('x', 90., degrees=True).as_quat()[[3, 0, 1, 2]]
        reader.views = {path: FakeView(np.array([0., 0., 0.]), q) for path in (BAG, CUP, BUTTON, MACHINE_BODY)}
        reader.views[MACHINE_BODY].position = np.array([.08, .02, .92])
        reader.views[BUTTON].position = np.array([.08, .02, .92])
        reader.interface = SimpleNamespace(get_contact_report=lambda: [])
        reader.contacts = SimpleNamespace(ingest=lambda value: None, pairs=lambda: [])
        reader.diag = None
        reader.objects = (CUP, BAG)
        reader.summarize = lambda: (
            {o: dict(robot=False, supports=set(), ground=False, machine=False) for o in reader.objects},
            {o: {} for o in reader.objects}, False, False, 0.)
        reader.detector = SimpleNamespace(observe=lambda now, poses, bilateral: ({o: False for o in reader.objects}, {}))
        reader.bean_input = None

        def update(dt, beans, displacement):
            reader.bean_input = beans
            return dict(bean_count=0, raw_bean_count=0, dispensing=False, event=None, button_pressed=False)

        reader.controller = SimpleNamespace(pressed=False, update=update)
        reader.beans = SimpleNamespace(positions=lambda: np.array([[.08, .02, 1.2], [0., 0., 1.2]]))
        reader.press_by_robot = False
        reader.reset_accumulators()
        reader.step = -1
        reader.history = []
        reader.trace = io.StringIO()
        reader.bag_mouth = np.array([0., .12, 0.])
        return reader

    def test_native_body_drives_anchors_beans_and_button(self):
        reader = self.fixture()
        reader.after_step()
        np.testing.assert_allclose(reader.anchor, [.08, -.047, .94], atol=1e-14)
        self.assertAlmostEqual(reader.latest['button_displacement_m'], 0.)
        # First bean directly above the moved machine; second at the OLD root.
        self.assertAlmostEqual(reader.bean_input[0][0], 0.)
        self.assertAlmostEqual(reader.bean_input[1][0], -.08)
        self.assertEqual(reader.latest['machine_frame_source'], 'native_dynamic_body')
        self.assertEqual(reader.last_time, reader.world.current_time)

    def test_missing_or_stale_physical_step_rejected(self):
        reader = self.fixture()
        reader.world.current_time = 2 * PHYSICS_DT
        with self.assertRaises(RuntimeError):
            reader.after_step()

    def test_actor_appearance_does_not_issue_second_start(self):
        from .native_push import NativePalmPush
        from .trigger import TriggerFacts
        session = object.__new__(NativePalmPush)
        session.reader = SimpleNamespace(trigger_facts=lambda: TriggerFacts(2., True, True, 45., True))
        session.pending = dict(queued_time_s=1., event=dict(start=True, reason='pour_approach_triggered'))
        session.events = []
        session.human = SimpleNamespace(activate_on_trigger=lambda: dict(visible=True))
        event = session.observe(robot_native_poses={})
        self.assertFalse(event['start'])
        self.assertEqual(event['reason'], 'human_appeared')
        self.assertEqual(session.start, 2.)
        self.assertIsNone(session.pending)


if __name__ == '__main__':
    unittest.main()
