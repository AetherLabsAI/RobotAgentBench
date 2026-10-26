import hashlib
import json
import math
import unittest
from arena_api import bimanual as b
from evaluation.tasks.coffee_machine.specs import EPISODES, episodes, episode_manifest
from .replay_actions import adapt


class AdaptedReplayTests(unittest.TestCase):
    def test_only_authorized_open_samples_change(self):
        episode = episodes('1-2')[0]
        manifest = episode_manifest('1-2', episode)
        initial = dict(role='source', source_episode=episode, robot_initialization=manifest['robot_initialization'])
        path = EPISODES / '1-2' / episode / 'producer_plan.json'
        raw = path.read_bytes()
        source = json.loads(raw)
        candidate = adapt(initial)
        self.assertEqual(len(candidate['actions']), len(source['steps']))
        arm = b.LEFT_ARM if candidate['active_arm'] == 'left' else b.RIGHT_ARM
        grip = b.GRIPPERS[0] if candidate['active_arm'] == 'left' else b.GRIPPERS[1]
        count = 0
        for old, new in zip(source['steps'], candidate['actions']):
            expected = dict(source['initial_targets'])
            expected.update(zip(arm, old['q']))
            opening = old['opening']
            if math.pi / 4 < opening <= .8:
                opening = math.pi / 4
                count += 1
            expected[grip] = opening
            self.assertEqual(new, expected)
        self.assertEqual(candidate['changed_commands_count'], count)
        self.assertEqual(path.read_bytes(), raw)
        self.assertEqual(candidate['source_plan_sha256'], hashlib.sha256(raw).hexdigest())

    def test_random_initial_cannot_replay_source(self):
        with self.assertRaises(ValueError):
            adapt(dict(role='random', source_episode=episodes('1-2')[0]))


if __name__ == '__main__':
    unittest.main()
