import numpy as np

from experiments.physics_fluid.tablecloth_native_actions import repair_approach


def test_approach_changes_only_left_arm_and_preserves_downstream_commands():
    class Robot:
        by_name = {}

        @staticmethod
        def arm_names(side):
            return ['left']

    q = np.column_stack((np.r_[0., np.linspace(.03, .3, 10), .4, .5], np.arange(13)))
    phases = np.asarray(['settle']+['L_friction_approach']*10+['touch', 'pull'])
    commands, audit = repair_approach(dict(joint_names=np.asarray(['left', 'other']),
                                         joint_target=q), phases, Robot(), 10)
    assert np.array_equal(commands['joint_target'][11:], q[11:])
    assert np.array_equal(commands['joint_target'][:, 1], q[:, 1])
    assert commands['joint_target'][10, 0] == q[10, 0]
    assert not np.array_equal(commands['joint_target'][1:10, 0], q[1:10, 0])
    assert audit['max_joint_speed_rad_s'] <= .65
    assert audit['eligible_for_production'] is False
