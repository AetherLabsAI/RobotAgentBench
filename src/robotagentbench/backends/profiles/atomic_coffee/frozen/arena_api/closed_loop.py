"""One observe -> action source -> HTTP step -> evaluate loop for every source.

No USD authoring, physics implementation or task predicates in this module.
The legacy ``run(env, policy, ...)`` call remains a model-source wrapper.
"""
import argparse
import base64
import hashlib
import math
import os
from pathlib import Path
import time
from uuid import uuid4

import httpx

from . import bimanual as b, robot_profiles as rp
from . import policy_execution as pe
from .client import ArenaClient
from .policy_contract import digest
from .action_sources import ModelActionSource
from .schemas import EpisodeConfig
from .trace_recording import TraceRecorder


def request_once_or_retry(http, method, path, payload, instance_id, retries=1, *, timeout=None):
    """Same request body/UUID only; process restart or uncertain failure stops the run."""
    for attempt in range(retries + 1):
        try:
            response = http.request(method, path, json=payload,
                                    **({} if timeout is None else {'timeout': timeout}))
            response.raise_for_status()
            return response.json()
        except (httpx.TimeoutException, httpx.NetworkError):
            if attempt == retries:
                raise
            health = http.get("/healthz")
            health.raise_for_status()
            if health.json().get("instance_id") != instance_id:
                raise RuntimeError("Server restarted; do not retry an uncertain action")


def run(env, policy=None, *, action_source=None, instruction, output, cycles=3, prefix_steps=4,
        deadline_s=600, max_retries=1, service_label=None, max_steps=None,
        seed=7, on_observation=None, environment_task_id=pe.TASK_ID, randomization=None, task_parameters=None,
        physics_rate_selector=None):
    """Execute one episode. A trusted evaluator may stop after any observation.

    max_steps selects an exact budget (including a partial final chunk).
    on_observation receives a copy of EpisodeState, never the policy connection.
    Its nonempty stop reason ends execution before any further policy action.
    physics_rate_selector(copy_of_state) returns 60/120/240 before an action,
    or None to retain the current whole-scene rate. Control stays at 20 Hz.
    """
    if (policy is None) == (action_source is None):
        raise ValueError('Provide exactly one policy connection or action_source')
    source = action_source if action_source is not None else ModelActionSource(policy)
    from .task_registry import default_registry
    registration = default_registry().get(environment_task_id)
    task_parameters = registration.resolve_parameters(task_parameters or {})
    instruction = registration.resolve_instruction(instruction, task_parameters)
    if type(prefix_steps) is not int or not 1 <= prefix_steps <= 30:
        raise ValueError("Invalid action prefix length")
    if max_steps is not None:
        if type(max_steps) is not int or not 1 <= max_steps <= registration.max_steps:
            raise ValueError("Invalid physical step budget")
        cycles = math.ceil(max_steps / prefix_steps)
    else:
        if type(cycles) is not int:
            raise ValueError("Invalid inference budget")
        max_steps = cycles * prefix_steps
    if not instruction.strip() or not 1 <= cycles <= registration.max_inferences or not 1 <= max_steps <= registration.max_steps:
        raise ValueError('Invalid instruction or registered task budget')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Invalid environment seed")
    if isinstance(deadline_s, bool) or not math.isfinite(deadline_s) or deadline_s <= 0:
        raise ValueError("Invalid wall-clock watchdog budget")
    if not 0 <= max_retries <= 2:
        raise ValueError("Invalid retry limit")
    if service_label is not None and (not service_label.strip() or len(service_label) > 128):
        raise ValueError("Invalid operator service label")
    output.mkdir(parents=True, exist_ok=False)
    from .schemas import Randomization
    randomization = Randomization.model_validate(randomization or {}).model_dump()
    caps = env.capabilities()
    task_id = environment_task_id
    if caps["backend"] != "isaac_physics" or task_id not in caps["task_action_specs"]:
        raise RuntimeError("Physics action contract mismatch")
    if max_steps > caps['task_max_episode_steps'][task_id]:
        raise ValueError('Environment task budget is too small')
    episode = uuid4()
    started = time.monotonic()
    transcript = {"episode_id": str(episode), "instruction": instruction,
                  "environment_capabilities": caps, "action_source": source.kind,
                  "action_mode": "strict", "cycles_requested": cycles, "prefix_steps": prefix_steps,
                  "max_steps": max_steps, "environment_seed": seed,
                  "calls": [], "steps": [], "evaluation_valid": False, "task_success": None,
                  "closed_loop_transport_pass": False}

    transcript.update(service_label=service_label,
                      benchmark_status="not_ready: task reset/randomization/scoring not connected")
    initialized = False
    source_initialized = False
    stop_reason = None
    recorder = TraceRecorder(output)

    def evaluate(state):
        if on_observation is None:
            return None
        reason = on_observation(state.model_copy(deep=True))
        if reason is not None and (not isinstance(reason, str) or not reason.strip()):
            raise ValueError("Evaluator must return a nonempty stop reason or None")
        return reason

    def record(observation):
        (output / f"step{observation.step_index:04d}.json").write_text(observation.model_dump_json(indent=2) + "\n")
        encoded = {}
        if set(observation.images) != {"head", "left", "right"}:
            raise RuntimeError("Missing camera image")
        for camera, url in observation.images.items():
            if url != f"/v1/episodes/{episode}/frames/{observation.step_index}/{camera}.png":
                raise RuntimeError("Unexpected cross-episode/frame URL")
            response = env.http.get(url)
            response.raise_for_status()
            data = response.content
            if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                raise RuntimeError("Invalid PNG response")
            (output / f"step{observation.step_index:04d}_{camera}.png").write_bytes(data)
            encoded[camera] = {"png_base64": base64.b64encode(data).decode(), "sha256": hashlib.sha256(data).hexdigest()}
        return encoded

    try:
        recorder.metadata(transcript)
        transcript.update(source.prepare(instruction=instruction, cycles=cycles, task_id=task_id,
                                         seed=seed, randomization=randomization, task_parameters=task_parameters))
        source_initialized = True
        robot_profile = rp.get(source.robot_profile).id
        execution_spec = pe.spec(robot_profile)
        if caps.get("action_specs", {}).get(robot_profile) != execution_spec:
            raise RuntimeError("Selected robot profile is unavailable or incompatible")
        transcript.update(robot_profile=robot_profile, execution_profile=execution_spec,
                          execution_profile_sha256=digest(execution_spec))
        keys = pe.evidence_keys(robot_profile)
        source.start(episode, request_once_or_retry, max_retries)
        # Same client UUID makes a reset retry safe; no new reset after ambiguity.
        from .schemas import EpisodeState
        config = EpisodeConfig(task_id=task_id, instruction=instruction, seed=seed,
                               max_steps=max_steps, randomization=randomization, task_parameters=task_parameters,
                               robot_profile=robot_profile)
        transcript['episode_config'] = config.model_dump(mode='json')
        initialized = True  # Close also on an ambiguous reset response.
        state = EpisodeState.model_validate(request_once_or_retry(
            env.http, "PUT", f"/v1/episodes/{episode}", config.model_dump(mode="json"), caps["instance_id"], max_retries,
            timeout=env.reset_timeout))
        result = env.result(episode)
        if (state.robot_profile != robot_profile or result.robot_profile != robot_profile or
                state.task_parameters != task_parameters or result.task_parameters != task_parameters):
            raise ValueError('Environment did not acknowledge the requested task parameters')
        limits = result.metrics["action_limits_rad" if robot_profile == rp.JOINT16 else "action_limits_native"]
        if result.metrics.get("action_spec") != execution_spec:
            raise RuntimeError("Worker execution profile mismatch")
        transcript["initial_observation"] = state.observation.model_dump(mode="json")
        # Keep the portable reset contract, not every static asset diagnostic,
        # in the client trace. Full metrics stay on server.
        transcript['initial_metrics'] = {key: result.metrics[key] for key in
            ('replay_reset_contract', 'action_spec', 'action_limits_rad', 'initial_action_targets_rad',
             'action_limits_native', 'initial_action_targets_native', 'native_hold_targets', 'joint_units')
            if key in result.metrics}
        images = record(state.observation)
        transcript['source_initial_validation'] = source.validate_initial(state.observation, result.metrics, limits)
        if source.kind == 'trajectory':
            # Report the observed comparison, not an unconditional successful
            # gate: joint differences are allowed for every recorded replay.
            validation = transcript['source_initial_validation'] or {}
            transcript['trajectory_initial_match'] = validation.get('joint_match_within_recorded_tolerance')
            transcript['initial_joint_match_check_enabled'] = validation.get('initial_joint_match_check_enabled')
        stop_reason = evaluate(state) or (state.stop_reason if state.terminated or state.truncated else None)
        recorder.metadata(transcript)
        from .physics_rate import DEFAULT_HZ, evidence_substeps
        active_physics_hz = DEFAULT_HZ
        for cycle in range(cycles):
            if stop_reason is not None:
                break
            if time.monotonic() - started > deadline_s:
                raise TimeoutError("Closed-loop wall budget exhausted")
            if state.status != "running":
                raise RuntimeError("Episode ended before scheduled inference")
            prefix_length = min(prefix_steps, max_steps - state.observation.step_index)
            batch = source.next_batch(state.observation, images, prefix_length, cycle, limits)
            if time.monotonic() - started > deadline_s:
                raise TimeoutError("Closed-loop wall budget exhausted after inference")
            if batch is None:
                stop_reason = 'source_exhausted'
                break
            if not 1 <= len(batch.actions) <= prefix_length:
                raise ValueError('Action source returned an invalid prefix')
            call = batch.audit
            transcript["calls"].append(call)
            recorder.batch(cycle, call)
            for index, (targets, interventions) in enumerate(batch.actions):
                if time.monotonic() - started > deadline_s:
                    raise TimeoutError("Closed-loop wall budget exhausted")
                pe.validate(targets, limits, robot_profile)
                before = state.observation.time_s
                selected_hz = None if physics_rate_selector is None else physics_rate_selector(state.model_copy(deep=True))
                step_request = (env.prepare_step(state, targets) if selected_hz is None else
                                env.prepare_step(state, targets, physics_hz=selected_hz))
                new_state = request_once_or_retry(env.http, "POST", f"/v1/episodes/{episode}/step",
                                                 step_request.model_dump(mode="json"), caps["instance_id"], max_retries)
                next_state = EpisodeState.model_validate(new_state)
                if next_state.observation.step_index != state.observation.step_index + 1 or abs(next_state.observation.time_s - before - b.CONTROL_DT) > 1e-6:
                    raise RuntimeError("Unexpected physical step or simulation time")
                state = next_state
                # Persist acknowledged motion even if evidence/image/evaluator
                # processing fails afterwards. Never report this as zero motion.
                call["executed_steps"] += 1
                call["discarded_steps"] -= 1
                transcript["steps"].append({"cycle": cycle, "chunk_index": index,
                    "request": step_request.model_dump(mode="json"), "interventions": interventions,
                    "execution": state.execution,
                    "observation": state.observation.model_dump(mode="json")})
                recorder.step(transcript["steps"][-1], call)
                execution = state.execution
                if selected_hz is not None:
                    active_physics_hz = selected_hz
                substeps = evidence_substeps(execution or {}, active_physics_hz)
                if (not execution or execution.get("profile_id") != pe.profile_id(robot_profile) or
                    execution.get("applied_substeps") != substeps or
                    execution.get(keys["requested_targets_rad"]) != targets or
                    any(len(execution.get(keys[key], [])) != substeps for key in (
                        "substep_targets_rad", "interpolated_substep_targets_rad",
                        "measured_before_substep_rad", "measured_after_substep_rad",
                        "tracking_error_after_substep_rad"))):
                    raise RuntimeError("Missing/mismatched physical controller evidence")
                images = record(state.observation)
                stop_reason = evaluate(state) or (state.stop_reason if state.terminated or state.truncated else None)
                if stop_reason is not None:
                    break
            print("LOOP", cycle + 1, "physical_step", state.observation.step_index,
                  "action_source", source.kind, flush=True)
        transcript["closed_loop_transport_pass"] = True
        transcript["termination_reason"] = stop_reason or "step_budget"
        transcript["total_executed_steps"] = len(transcript["steps"])
        transcript["total_interventions"] = sum(len(s["interventions"]) for s in transcript["steps"])
        transcript["total_controller_interventions"] = sum(len((s["execution"] or {}).get("interventions", [])) for s in transcript["steps"])
        transcript["result"] = env.result(episode).model_dump(mode="json")
        transcript.update(evaluation_valid=state.evaluation_valid, task_success=state.success)
        if state.evaluation is not None:
            transcript['benchmark_status'] = 'task_runtime_connected; see per-episode evidence'
        return transcript
    except BaseException as exc:
        transcript["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        transcript["total_executed_steps"] = len(transcript["steps"])
        transcript["total_interventions"] = sum(len(s["interventions"]) for s in transcript["steps"])
        transcript["total_controller_interventions"] = sum(len((s["execution"] or {}).get("interventions", [])) for s in transcript["steps"])
        cleanup = {}
        if initialized:
            try:
                env.close_episode(episode)
                cleanup["environment"] = "closed"
                transcript['result'] = env.result(episode).model_dump(mode='json')
                if 'error' not in transcript:
                    transcript.update(evaluation_valid=transcript['result']['evaluation_valid'],
                                      task_success=transcript['result']['success'])
            except Exception as exc:
                cleanup["environment"] = type(exc).__name__
        if source_initialized:
            try:
                source.close()
                cleanup[source.cleanup_key] = "closed"
            except Exception as exc:
                cleanup[source.cleanup_key] = type(exc).__name__
        transcript["cleanup"] = cleanup
        recorder.finish(transcript)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-url", default="http://127.0.0.1:8089")
    parser.add_argument("--policy-url", default="ws://127.0.0.1:8091/v1/blackbox/ws")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--prefix-steps", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--service-label", help="Operator's submission label; never treated as a verified model identity")
    args = parser.parse_args()
    token = os.environ.get("ARENA_POLICY_TOKEN")
    from .policy_ws import BlackboxClient
    policy_client = BlackboxClient(args.policy_url, token=token)
    with policy_client as policy, ArenaClient(args.env_url, token=os.environ.get("ARENA_API_TOKEN"), timeout=240) as env:
        result = run(env, policy, instruction=args.instruction, output=args.output, cycles=args.cycles,
                     prefix_steps=args.prefix_steps, max_retries=args.max_retries,
                     service_label=args.service_label)
        print("CLOSED_LOOP_FRAMEWORK_PASS", result["total_executed_steps"], "mode", result["evaluation_mode"], flush=True)


if __name__ == "__main__":
    main()
