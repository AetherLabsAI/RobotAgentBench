"""Runs ONLY inside the no-network policy container; no credentials or host paths."""
import contextlib
import dataclasses
import io
import json
import sys
import numpy as np
from g1_model import G1ArmIK
from move_to import Observation, MoveConfig
from planned_motion import MoveToController

wire_in, wire_out = sys.stdin, sys.stdout


def send(row):
    wire_out.write(json.dumps(row, allow_nan=False) + '\n')
    wire_out.flush()


def read():
    line = wire_in.readline()
    if not line:
        raise EOFError('bridge disconnected')
    return json.loads(line)


initial = read()
latest = initial['observation']
if initial['control_hz'] != 20:
    raise ValueError('20 Hz required')
held_targets = initial['held_targets']


class EpisodeFinished(BaseException):
    """Unwind the pending motion when the environment ends, without revealing its score."""
    pass


class ActionRejected(ValueError):
    def __init__(self, detail):
        self.detail = detail
        super().__init__(detail['error'])


class Port:
    def __init__(self, side):
        self.side = side
        self.explicit_gripper = False

    def observe(self):
        short = 'l' if self.side == 'left' else 'r'
        name = 'idx41_gripper_l_outer_joint1' if short == 'l' else 'idx81_gripper_r_outer_joint1'
        # Raw measured joints remain unchanged; only the normalized display is bounded.
        aperture = float(np.clip(latest['joints'][name] / (np.pi / 4), 0, 1))
        return Observation(latest['episode_id'], latest['step'], latest['joints'], aperture)

    def command_joints(self, targets, *, episode_id, expected_step, gripper=None):
        global latest
        row = dict(op='step', side=self.side, episode_id=episode_id,
                   expected_step=expected_step, targets=targets)
        # Omitted gripper retains the last drive target, not the measured opening under load.
        if gripper is not None:
            row['gripper'] = gripper
        send(row)
        reply = read()
        if reply.get('op') == 'episode_finished':
            latest = reply['observation']
            if latest['episode_id'] != episode_id or latest['step'] != expected_step + 1:
                raise RuntimeError('Invalid terminal observation sequence')
            raise EpisodeFinished()
        if reply.get('op') == 'action_rejected':
            observed = reply['observation']
            if observed['episode_id'] != episode_id or observed['step'] != expected_step:
                raise RuntimeError('Invalid rejection observation sequence')
            latest = observed  # NOT the error envelope: subsequent observe/FK must work.
            raise ActionRejected(reply['error'])
        if 'error' in reply:
            raise RuntimeError(reply['error'])
        latest = reply
        return self.observe()


controllers = {}
for side in ('left', 'right'):
    ik = G1ArmIK(side, max_iters=20, position_tolerance=.0001, angle_tolerance=.001)
    # Approved free-space default; the policy should slow down near contact.
    cfg = MoveConfig(control_hz=20, linear_speed=.04, angular_speed=.1, max_steps=200,
        joint_speeds={n:.3 for n in ik.arm_joints},
        eef_bounds={**{n:(-3,3) for n in ('x','y','z')},
                    'yaw':(-np.pi,np.pi), 'pitch':(-1.4,1.4), 'roll':(-np.pi,np.pi)},
        resync_thresholds={n:.05 for n in ik.arm_joints},
        oscillation_deadbands={n:.001 for n in ik.arm_joints})
    ctrl = MoveToController(Port(side), ik, config=cfg)
    o = ctrl.port.observe()
    ctrl._start(o, ik.read_q(o))
    ctrl.previous = np.array([held_targets[n] for n in ik.arm_joints])
    grip_name = 'idx41_gripper_l_outer_joint1' if side == 'left' else 'idx81_gripper_r_outer_joint1'
    ctrl.previous_gripper = held_targets[grip_name] / (np.pi/4)
    controllers[side] = ctrl


def tcp():
    # Reporting an execution error must not throw the same FK error again.
    # Missing FK is explicit (null), never a stale or clipped pose.
    result = {}
    for side, controller in controllers.items():
        try:
            result[side] = controller.ik.fk(controller.ik.read_q(controller.port.observe())).tolist()
        except Exception:
            result[side] = None
    return result


def move_to(side, targets, mode='replace'):
    ctrl = controllers[side]
    ctrl.port.explicit_gripper = 'gripper' in targets
    answer = dataclasses.asdict(ctrl.move_to(targets, mode=mode))
    answer['actual_pose'] = answer['actual_pose'].tolist()
    return answer


send({'op':'ready', 'tcp':tcp()})
scope = {'controllers':controllers, 'move_to':move_to, 'np':np, 'json':json}
while True:
    row = read()
    try:
        if row['op'] == 'sync':
            latest = row['observation']
            if 'held_targets' in row:
                # Trusted broker sends committed raw targets, never measured replacements.
                for side, ctrl in controllers.items():
                    ctrl.previous = np.array([row['held_targets'][n] for n in ctrl.ik.arm_joints])
                    grip_name = 'idx41_gripper_l_outer_joint1' if side == 'left' else 'idx81_gripper_r_outer_joint1'
                    ctrl.previous_gripper = row['held_targets'][grip_name] / (np.pi/4)
                    ctrl.goal = ctrl.goal_updates = None
                    ctrl.last_step = latest['step']
            answer = {'synchronized': True}
        elif row['op'] == 'move_to':
            answer = move_to(row['side'], row['targets'], row.get('mode', 'replace'))
        elif row['op'] == 'execute_python':
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                exec(row['code'], scope)
            answer = {'stdout':stream.getvalue()[:16000]}
        elif row['op'] == 'observe':
            answer = {}
        else:
            raise ValueError('Unknown sandbox command')
        send({'op':'result', 'result':answer, 'tcp':tcp()})
    except EpisodeFinished:
        send({'op':'result','result':{'status':'terminated','reason':'environment_episode_finished'},'tcp':tcp()})
    except ActionRejected as error:
        send({'op':'result', 'result':error.detail, 'tcp':tcp()})
    except Exception as error:
        # A Python error may follow already accepted steps. Refresh robot-only
        # FK from latest measured joints instead of returning stale TCP values.
        send({'op':'result', 'result':{'error':type(error).__name__ + ': ' + str(error)[:500]},
              'tcp':tcp()})
