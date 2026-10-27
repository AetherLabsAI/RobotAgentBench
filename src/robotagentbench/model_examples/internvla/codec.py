"""Pure finite physical-unit mapping and whole-chunk validation, no projection."""
from copy import deepcopy
import math
from uuid import uuid4

from robotagentbench.interfaces.action_contract import ORDER, public_observation
from robotagentbench.interfaces import mobile_base, robot_profiles as rp

REQUEST = 'robotagentbench.internvla.infer.v1'
RESPONSE = 'robotagentbench.internvla.actions.v1'


def contract(value):
    if (value.get('schema') != 'short_tasks.gpt.robot22.v1'
            or value.get('robot_profile') != rp.MOBILE22
            or value.get('control_hz') != 20 or value.get('action_order') != ORDER
            or value.get('camera_order') != ['head','left','right']
            or value.get('base') != mobile_base.specification()
            or type(value.get('max_tool_steps'))is not int
            or not 1 <= value['max_tool_steps'] <= 600):
        raise ValueError('Worker does not expose the native22/20Hz contract')
    if value.get('joint_units') != rp.NATIVE_UNITS:
        raise ValueError('Worker native unit declaration changed')
    limits = value.get('joint_limits', {})
    if set(limits) != set(ORDER[:20]):
        raise ValueError('Measured worker joint limits required')
    for pair in limits.values():
        if (not isinstance(pair, (list,tuple)) or len(pair) != 2
                or any(type(v) not in (int,float) or not math.isfinite(v) for v in pair)
                or pair[0] >= pair[1]):
            raise ValueError('Invalid measured joint limits')
    return deepcopy(value)


def state(observation):
    row = public_observation(observation)
    return [row['joints'][n] for n in ORDER[:20]] + row['base_state']['position_world_m'][:2] + [
        row['base_state']['yaw_world_unwrapped_rad']]


def inference_request(worker_contract, observation):
    # This recursive public allowlist strips scores, targets and arbitrary metadata.
    return dict(schema=REQUEST, contract=contract(worker_contract),
                observation=public_observation(observation))


def actions_response(observation, actions):
    return dict(schema=RESPONSE, episode_id=observation['episode_id'], step=observation['step'],
                action_order=list(ORDER), control_hz=20,
                representation='physical_absolute_joint20_base_velocity2',
                normalized=False, actions=actions, navigation_consumed=False)


def action_request(prediction, observation, worker_contract, previous_base):
    c = contract(worker_contract)
    if (not isinstance(prediction, dict) or set(prediction) != {
            'schema','episode_id','step','action_order','control_hz','representation',
            'normalized','actions','navigation_consumed'}
            or prediction['schema'] != RESPONSE
            or prediction['episode_id'] != observation['episode_id']
            or prediction['step'] != observation['step']
            or prediction['action_order'] != ORDER or prediction['control_hz'] != 20
            or prediction['representation'] != 'physical_absolute_joint20_base_velocity2'
            or prediction['normalized'] is not False or prediction['navigation_consumed'] is not False):
        raise ValueError('Stale, normalized, reordered or unsupported model output')
    actions = prediction['actions']
    if not isinstance(actions,list) or not 1 <= len(actions) <= c['max_tool_steps']:
        raise ValueError('Explicit bounded model chunk required')
    previous = dict(previous_base)
    for row in actions:
        targets, base = mobile_base.split_action(row)
        for name, target in targets.items():
            if not c['joint_limits'][name][0] <= target <= c['joint_limits'][name][1]:
                raise ValueError('Model output outside measured native joint limits')
        previous = mobile_base.validate(base, previous)
    return dict(op='execute_actions', request_id=str(uuid4()),
                episode_id=observation['episode_id'], expected_step=observation['step'],
                actions=deepcopy(actions)), previous
