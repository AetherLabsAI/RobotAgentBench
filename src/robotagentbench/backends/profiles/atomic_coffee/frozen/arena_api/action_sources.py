"""Action providers only. No scene edits, physics, task predicates or scoring.

Both providers return named radian targets to the SAME HTTP episode loop.
Recorded commands are never interpolated, padded, clipped, or retargeted here.
"""
from dataclasses import dataclass
import time
from typing import Protocol
from uuid import uuid4

from . import bimanual as b, policy_execution as pe, robot_profiles as rp
from .policy_contract import digest, blackbox_specification, pack_measured, validate_chunk


@dataclass
class ActionBatch:
    actions: list  # (named radian targets, source mapping interventions)
    audit: dict


class ActionSource(Protocol):
    """Trusted client-side extension boundary; the source cannot advance Isaac."""
    kind: str
    cleanup_key: str

    def prepare(self, *, instruction, cycles, task_id, seed, randomization, task_parameters=None) -> dict: ...
    def start(self, episode, request, retries) -> None: ...
    def validate_initial(self, observation, metrics, limits) -> dict | None: ...
    def next_batch(self, observation, images, count, cycle, limits) -> ActionBatch | None: ...
    def close(self) -> None: ...


class ModelActionSource:
    kind = 'model'
    cleanup_key = 'policy'  # Preserve existing model trace contract.

    def __init__(self, policy):
        self.policy = policy
        self.initialized = False

    def prepare(self, *, instruction, cycles, task_id, seed, randomization, task_parameters=None):
        if not getattr(self.policy, 'blackbox_transport', False):
            raise ValueError('A binary WebSocket policy connection is required')
        reply = self.policy.get('/v1/metadata'); reply.raise_for_status()
        self.metadata = metadata = reply.json()
        if cycles > metadata.get('max_session_inferences', 100):
            raise ValueError('Model service inference budget is too small; update/configure the provider')
        self.robot_profile = rp.get(metadata["profile"]["robot_profile"]).id
        self.profile = profile = blackbox_specification(self.robot_profile)
        self.profile_hash = digest(profile)
        if metadata['profile_sha256'] != self.profile_hash or metadata['profile'] != profile:
            raise RuntimeError('Model profile handshake mismatch')
        if metadata.get('wire_protocol') != 'arena.blackbox.ws.v1':
            raise RuntimeError('Black-box protocol handshake mismatch')
        if metadata['model']['kind'] == 'feedback_test':
            raise RuntimeError('Test fixture policies are not supported by the production runner')
        self.reset_payload = dict(instruction=instruction, profile_id=profile['profile_id'],
            profile_sha256=self.profile_hash, seed=0, max_inferences=cycles)
        return dict(model_metadata=metadata, model_kind=metadata['model']['kind'],
            execution_mode='policy', evaluation_mode='remote_blackbox',
            model_identity_verified=False, action_origin_verified=False,
            provider_metadata_trust='self_reported', score_scope='service_behavior_only',
            policy_transport='websocket_msgpack')

    def start(self, episode, request, retries):
        self.session_path = f'/v1/sessions/{episode}'
        self.request, self.retries = request, retries
        self.initialized = True  # Close even after an ambiguous reset.
        ack = request(self.policy, 'PUT', self.session_path, self.reset_payload,
                      self.metadata['instance_id'], retries)
        if ack['instance_id'] != self.metadata['instance_id']:
            raise RuntimeError('Policy restarted during reset')

    def validate_initial(self, observation, metrics, limits):
        return None

    def next_batch(self, observation, images, count, cycle, limits):
        measured = rp.measured(observation, self.robot_profile)
        # Evaluation/ground truth never enters the model payload.
        payload = dict(episode_id=str(observation.episode_id), step_index=observation.step_index,
            time_s=observation.time_s, source=observation.source,
            joint_positions_rad={n:v for n,v in measured.items()
                                 if rp.get(self.robot_profile).units[n] == "rad"}, images=images)
        if self.robot_profile == rp.JOINT20:
            payload["joint_positions_m"] = {n:v for n,v in measured.items()
                                             if rp.get(self.robot_profile).units[n] == "m"}
        request = dict(request_id=str(uuid4()), expected_inference_index=cycle, observation=payload)
        started = time.monotonic()
        response = self.request(self.policy, 'POST', self.session_path + '/act', request,
                                self.metadata['instance_id'], self.retries)
        elapsed = time.monotonic() - started
        if (response['request_id'] != request['request_id'] or
            response['instance_id'] != self.metadata['instance_id'] or
            response['session_id'] != str(observation.episode_id) or response['inference_index'] != cycle or
            response['input_sha256'] != digest(request) or response['profile_sha256'] != self.profile_hash or
            response['observation_step_index'] != observation.step_index or
            response['observation_time_s'] != observation.time_s or
            response['packed_measured_state'] != pack_measured(measured, self.robot_profile) or
            response['image_sha256'] != {n: im['sha256'] for n, im in images.items()} or
            abs(response['dt_s'] - b.CONTROL_DT) > 1e-9):
            raise RuntimeError('Policy returned stale/mismatched observation or action contract')
        chunk = response['action_chunk']; validate_chunk(chunk, self.robot_profile)
        return ActionBatch([pe.map_action(row, limits, self.robot_profile) for row in chunk[:count]],
            dict(response=response, executed_steps=0, discarded_steps=len(chunk),
                 measured_policy_roundtrip_s=elapsed, provider_inference_time_trust='self_reported'))

    def close(self):
        if self.initialized:
            reply = self.policy.delete(self.session_path); reply.raise_for_status()


class TrajectoryActionSource:
    kind = 'trajectory'
    cleanup_key = 'action_source'

    def __init__(self, path):
        from .trajectory import load_trajectory
        self.data, self.file_sha256 = load_trajectory(path)
        self.robot_profile = self.data["execution_profile"]["robot_profile"]

    def prepare(self, *, instruction, cycles, task_id, seed, randomization, task_parameters=None):
        expected = self.data['reset']
        from .task_registry import default_registry
        task = default_registry().get(task_id)
        if (expected['task_id'] != task_id or expected['seed'] != seed or
                expected['randomization'] != randomization or
                task.resolve_parameters(expected.get('task_parameters', {})) != task.resolve_parameters(task_parameters or {})):
            raise ValueError('Trajectory task/seed/randomization/parameters does not match this run')
        self.offset = 0
        return dict(execution_mode='recorded_trajectory', evaluation_mode='recorded_actions',
            model_kind=None, policy_transport=None, action_origin_verified=False,
            score_scope='recorded_trajectory_only', trajectory_sha256=self.file_sha256,
            trajectory_control_steps=len(self.data['actions']),
            trajectory_provenance=self.data['provenance'], trajectory_initial_match=None,
            initial_joint_match_check_enabled=False, initial_joint_match_policy='record_only')

    def start(self, episode, request, retries):
        pass

    def validate_initial(self, observation, metrics, limits):
        expected = self.data['reset']
        if metrics.get('replay_reset_contract') != expected['task_contract']:
            raise ValueError('Trajectory reset geometry/assets differ; explicit recalibration is required')
        actual = rp.measured(observation, self.robot_profile)
        rp.values(actual, self.robot_profile, 'measured initial state')
        reference = expected['joint_positions_rad'] if self.robot_profile == rp.JOINT16 else expected['joint_positions_native']
        tolerances = ({n: expected['joint_tolerance_rad'] for n in actual} if self.robot_profile == rp.JOINT16
                      else expected['joint_tolerances_native'])
        errors = {n: abs(actual[n] - reference[n]) for n in actual}
        # User decision: old settled joint readings are diagnostics, not a
        # replay gate. Do not correct the pose or alter any recorded targets.
        exceeded = [n for n in actual if errors[n] > tolerances[n]]
        for row in self.data['actions']:
            pe.validate(row, limits, self.robot_profile)  # Validate entire input before the first motion.
        return dict(joint_errors_native=errors, joint_tolerances_native=tolerances,
                    joint_units=rp.get(self.robot_profile).units, task_contract_matched=True,
                    initial_joint_match_check_enabled=False, joint_match_policy='record_only',
                    recorded_reference_exceeded=exceeded,
                    joint_match_within_recorded_tolerance=not exceeded,
                    all_action_limits_validated=True)

    def next_batch(self, observation, images, count, cycle, limits):
        if observation.step_index != self.offset:
            raise RuntimeError('Trajectory cursor differs from acknowledged controls')
        rows = self.data['actions'][self.offset:self.offset + count]
        if not rows:
            return None  # EOF is not success and does not add a hold tail.
        first = self.offset
        self.offset += len(rows)
        return ActionBatch([(dict(row), []) for row in rows],
            dict(source_start_index=first, source_end_index=self.offset,
                 executed_steps=0, discarded_steps=len(rows)))

    def close(self):
        pass
