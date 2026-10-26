import hashlib
import json
from pathlib import Path
import unittest
from .preview_actions import approach
from arena_api import bimanual,robot_profiles
from evaluation.tasks.coffee_machine.initial_state import package
from evaluation.tasks.coffee_machine.specs import EPISODES,episodes


class PreviewActionsTests(unittest.TestCase):
    def test_separate_diagnostic_preserves_source_and_legal_reset_grippers(self):
        episode=episodes('1-2')[0]
        manifest,_,_=package('1-2',episode)
        path=EPISODES/'1-2'/episode/'producer_plan.json'
        before=hashlib.sha256(path.read_bytes()).hexdigest();source=json.loads(path.read_text())
        result=approach(dict(source_episode=episode,robot_initialization=manifest['robot_initialization']))
        self.assertEqual(len(result['actions']),200)
        for generated,original in zip(result['actions'],source['steps']):
            self.assertEqual([generated[n] for n in bimanual.RIGHT_ARM],original['q'])
            for n in bimanual.GRIPPERS:
                self.assertEqual(generated[n],manifest['robot_initialization']['hold_targets'][n])
            robot_profiles.values(generated,robot_profiles.JOINT20)
        self.assertEqual(before,hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertFalse(result['scored_episode'])
        self.assertTrue(result['original_gripper_commands_not_replayed'])


if __name__=='__main__':unittest.main()
