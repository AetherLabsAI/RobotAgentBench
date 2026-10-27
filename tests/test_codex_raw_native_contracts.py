"""CPU contract checks using the actual dedicated hosts, never an Isaac owner.

Only the pure public contract/host modules are imported. Drivers and native
observations are synthetic. No task, scorer, model, scene or physical process
is constructed, and these tests are not physical-acceptance evidence.
"""
from contextlib import ExitStack
from importlib import import_module
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from robotagentbench.model_examples.codex.raw_bridge import RawRobotBridge, FIXED_OWNER_CONSTRAINTS
from robotagentbench.interfaces.action_contract import ORDER
from test_codex_raw_operator import observation


PROFILES = Path(__file__).resolve().parents[1] / 'src/robotagentbench/backends/profiles'
RECOVERY = PROFILES / 'failure_recovery/generic/short_task_robustness/tasks/grasp_failure_recovery'


def namespace(name, path):
    module = ModuleType(name)
    module.__package__ = name
    module.__path__ = [str(path)]
    return module


class NativeContractTests(unittest.TestCase):
    def test_original_dedicated_ready_contracts_match_public_raw_bridge(self):
        # Namespace paths correspond to the shipped native owners' read-only
        # robot API imports. Avoid their native owner and deployment bootstrap.
        atomic = PROFILES / 'atomic_coffee/coffee/public_integration_v1'
        aliases = {
            '_raw_atomic': namespace('_raw_atomic', atomic),
            '_raw_atomic.robot_api': namespace('_raw_atomic.robot_api', PROFILES / 'atomic_coffee/generic/arena_api'),
            'arena_api': namespace('arena_api', PROFILES / 'failure_recovery/generic/arena_api'),
            '_raw_recovery': namespace('_raw_recovery', RECOVERY),
            '_raw_room_recovery': namespace('_raw_room_recovery', PROFILES / 'failure_recovery/generic/short_tasks/agent_probe_v1'),
            '_raw_cloth': namespace('_raw_cloth', PROFILES / 'cloth/owner'),
        }
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, aliases))
            directory = stack.enter_context(tempfile.TemporaryDirectory(prefix='raw-native-contract-cpu-'))
            classes = [
                ('atomic_coffee', import_module('_raw_atomic.host').SessionHost),
                ('recovery_shelf_and_bag', import_module('_raw_recovery.public_host').SessionHost),
                ('recovery_sofa_and_cups', import_module('_raw_room_recovery.host').SessionHost),
                ('cloth', import_module('_raw_cloth.source_rpc').FixedSourceHost),
            ]
            for name, host_class in classes:
                with self.subTest(family=name):
                    obs = observation()
                    session = SimpleNamespace(started=True, faulted=False, closed=False,
                        paused=False, pending=None, control_index=0, episode_id=obs['episode_id'],
                        plan={'instruction': obs['instruction']}, clock=SimpleNamespace(current_time=0.))
                    driver = SimpleNamespace(limits=dict.fromkeys(ORDER[:20], [-4., 4.]),
                                             observe=lambda: obs)
                    native_host = host_class(session, driver)
                    bridge = RawRobotBridge(SimpleNamespace(read=native_host.ready), Path(directory) / name)
                    self.assertEqual(bridge.contract['control_hz'], 20)
                    self.assertEqual(bridge.contract['action_order'], ORDER)
                    self.assertEqual(bridge.observed['episode_id'], obs['episode_id'])
                    self.assertEqual(bridge.owner_constraints, FIXED_OWNER_CONSTRAINTS if name == 'cloth' else {})
                    self.assertNotIn('score', bridge.observed)


if __name__ == '__main__':
    unittest.main()
