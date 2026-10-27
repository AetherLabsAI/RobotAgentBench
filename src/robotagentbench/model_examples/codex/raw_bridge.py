"""Codex tool adapter for an already admitted native raw-22 JSONL worker.

This is not a coding-policy host or a robot controller. The trusted operator
supplies the existing read/exchange/close transport. No model, subprocess,
Isaac, TCP, IK, action projection, or automatic retry is started here.
"""
import base64
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import threading
from uuid import uuid4

from robotagentbench.interfaces.action_contract import public_observation, specification
from .model_response import ModelFormatError, validate_choice
from .raw_tools import TOOLS as RAW_TOOLS
from .tools import TOOLS as CODING_TOOLS


# Reuse the published raw action, observation and stop schemas, not the coding
# tools or their controller prompt. In particular, no execute_python or IK.
TOOLS = deepcopy(RAW_TOOLS) + [deepcopy(tool) for tool in CODING_TOOLS
                              if tool['function']['name'] in ('observe', 'done')]
PUBLIC_FIELDS = frozenset(('episode_id', 'step', 'time_s', 'instruction', 'joints',
                          'base_state', 'context', 'image_step', 'images', 'navigation'))
FIXED_OWNER_CONSTRAINTS = dict(
    fixed_base=True, required_base_commands=[0., 0.],
    articulated_commands='20 named native physical-unit channels',
    original_source_substeps_per_control=3)


def tool_specs():
    """App Server dynamic tools, using the existing public JSON schemas."""
    return [dict(type='function', name=tool['function']['name'],
                 description=tool['function']['description'],
                 inputSchema=deepcopy(tool['function']['parameters'])) for tool in TOOLS]


def text_item(value):
    return dict(type='inputText', text=value if isinstance(value, str)
                else json.dumps(value, ensure_ascii=False, allow_nan=False))


def _contract(value):
    """Do not expose arbitrary ready-message metadata as a model contract."""
    if not isinstance(value, dict):
        raise ValueError('Native raw-22 contract required')
    declared = deepcopy(value)
    fields = declared.pop('observation_fields', None)
    if fields is not None and (not isinstance(fields, list)
            or any(not isinstance(field, str) or field not in PUBLIC_FIELDS for field in fields)):
        raise ValueError('Worker declared a non-public observation field')
    expected = specification(declared.get('joint_limits', {}))
    maximum = declared.get('max_tool_steps')
    if type(maximum) is not int or not 1 <= maximum <= 600:
        raise ValueError('Invalid native raw-action chunk limit')
    expected['max_tool_steps'] = maximum
    if declared != expected:
        raise ValueError('Native raw-22/20 Hz contract mismatch')
    return expected


class RawRobotBridge:
    """Sequential public tools; uncertain requests remain blocked for the owner.

    ``close`` verifies only the JSONL acknowledgement and transport process exit.
    Native world/GPU/source cleanup still requires the task owner's own receipt.
    The operator may use this class with the existing Codex ``session.run_turn``.
    """

    def __init__(self, remote, output, *, read_only=False, max_model_responses=200):
        if type(max_model_responses) is not int or max_model_responses <= 0:
            raise ValueError('A positive model-response budget is required')
        self.remote = remote
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.max_model_responses = max_model_responses
        self.read_only = read_only
        self.completed = self.abandoned = self.closed = False
        self.pending = None
        self.calls = {}
        self.tool_count = self.model_responses = 0
        self._lock = threading.Lock()
        self._close_attempted = False
        self.closure = None
        self.owner_result = None
        ready = remote.read()
        if (not isinstance(ready, dict) or ready.get('op') != 'ready'
                or ready.get('policy_schema') not in (None, 'short_tasks.gpt.robot22.v1')):
            raise ValueError('Requires an admitted native raw-22 worker, not a coding-policy host')
        self.contract = _contract(ready.get('contract'))
        self.owner_constraints = {}
        if 'native_owner_constraints' in ready:
            constraints = ready['native_owner_constraints']
            if (not isinstance(constraints, dict) or constraints != FIXED_OWNER_CONSTRAINTS
                    or constraints.get('fixed_base') is not True
                    or type(constraints.get('original_source_substeps_per_control')) is not int
                    or any(type(v) not in (int, float) for v in constraints['required_base_commands'])):
                raise ValueError('Unrecognized native owner constraint declaration')
            self.owner_constraints = deepcopy(constraints)
        self.observed = public_observation(ready['observation'])
        if self.observed['step'] != 0:
            raise ValueError('A freshly admitted raw worker is required; no implicit resume')
        self.snapshot('initial_snapshot.json')

    def save(self, name, value):
        # Trusted runner filenames only; no model-visible filesystem operation.
        path = self.output / name
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        temporary.replace(path)

    def snapshot(self, name='last_snapshot.json'):
        images = {}
        for camera in self.contract['camera_order']:
            raw = base64.b64decode(self.observed['images'][camera], validate=True)
            filename = f'step{self.observed["step"]:06d}_{camera}.png'
            (self.output / filename).write_bytes(raw)
            images[camera] = dict(file=filename, sha256=hashlib.sha256(raw).hexdigest())
        self.save(name, dict(observation={k: v for k, v in self.observed.items() if k != 'images'},
                             images=images, in_flight_action=self.pending, physics_checkpoint=False))

    @property
    def instructions(self):
        prompt = f'''You control the simulated robot through the supplied Arena tools only.
Use the task instruction, public context, all three camera images, and measured robot state.
You have {self.max_model_responses} model turns. Submit one tool call at a time and inspect
its returned observations before another. Call the Arena tools directly, not through a
host shell or an orchestration wrapper. Host files, external tools, reset and task state
are unavailable. There are no Cartesian, IK, TCP or Python execution tools in this worker.

execute_actions submits explicit 22D rows in the contract's exact order at 20 Hz:
20 absolute native joint targets followed by base forward velocity and yaw rate.
Use the published native units and limits, including gripper radians, not normalized
opening. The bridge does not interpolate, clip, zero, repair or replay your commands.
The original native worker validates the entire chunk and owns all physical execution.
observe returns the current cameras/state without advancing simulation.
If a request is rejected, inspect its result and current observations before choosing
a different action. Never resend an uncertain request. Acknowledged action execution
is not evidence of a successful grasp or completed task. Verify physical outcomes
visually. done stops your attempt; the native owner, not the model, determines scoring.
Robot contract:\n{json.dumps(self.contract, ensure_ascii=False, allow_nan=False)}
'''
        if self.owner_constraints:
            prompt += ('\nOriginal native owner constraints (commands are not rewritten):\n'
                       + json.dumps(self.owner_constraints, ensure_ascii=False, allow_nan=False) + '\n')
        if 'navigation' in self.observed:
            prompt += '''\nThe supplied explored map is read-only, derived from static geometry and
revealed by the measured camera view. It contains no task labels or target poses.
White is known free floor, dark is a known static obstacle, grey is unknown;
red marks measured robot pose and blue marks the executed path, not a planned route.
Use the map together with camera images. No navigation helper or route planner is supplied.
'''
        return prompt

    def observation_items(self):
        items = [text_item({k: v for k, v in self.observed.items() if k not in ('images', 'navigation')})]
        for camera in self.contract['camera_order']:
            items += [text_item('Camera: ' + camera), dict(type='inputImage',
                       imageUrl='data:image/png;base64,' + self.observed['images'][camera])]
        if 'navigation' in self.observed:
            navigation = dict(self.observed['navigation'])
            png = navigation.pop('map_png')
            items += [text_item(dict(explored_map=navigation)),
                      dict(type='inputImage', imageUrl='data:image/png;base64,' + png)]
        return items

    def initial_items(self):
        return self.observation_items()

    @property
    def close_attempted(self):
        """Operator lifecycle state, not a model-visible task result."""
        return self._close_attempted

    def dispatch(self, call_id, name, args):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('A robot operation is already in flight; no parallel dispatch')
        try:
            return self._dispatch(call_id, name, args)
        finally:
            self._lock.release()

    def _dispatch(self, call_id, name, args):
        if not isinstance(call_id, str) or not call_id:
            raise ModelFormatError('A non-empty call ID is required')
        signature = json.dumps([name, args], sort_keys=True, allow_nan=False)
        # Block even an old cached read while another outcome is uncertain.
        if self.pending is not None:
            raise RuntimeError('Unacknowledged robot request: no retry or further dispatch')
        if call_id in self.calls:
            old, result = self.calls[call_id]
            if old != signature:
                raise RuntimeError('Call ID reused with different arguments')
            return deepcopy(result)  # Reuse response bytes; never re-send native RPC.
        if self.completed or self.abandoned or self._close_attempted:
            return dict(success=False, contentItems=[text_item('Episode stopped; no action executed.')])
        self.tool_count += 1
        if self.tool_count > self.max_model_responses:
            raise RuntimeError('Tool-call budget reached; hold environment')
        choice = dict(message=dict(role='assistant', tool_calls=[dict(type='function', id=call_id,
                      function=dict(name=name, arguments=json.dumps(args, allow_nan=False)))]))
        try:
            validate_choice(choice, TOOLS)
            if name == 'done' and not args['reason'].strip():
                raise ModelFormatError('done requires a non-empty reason')
            if name == 'execute_actions' and len(args['actions']) > self.contract['max_tool_steps']:
                raise ModelFormatError('Chunk exceeds the native worker contract')
        except ValueError:
            result = dict(success=False, contentItems=[text_item(dict(status='rejected',
                          actions_executed=0, hint='Malformed/unknown tool; check the published schema.'))])
        else:
            result = self._execute(name, args, call_id)
        self.calls[call_id] = (signature, deepcopy(result))
        self.save(f'tool_{self.tool_count:04d}.json', dict(call_id=call_id, name=name, arguments=args,
                  success=result['success'], step=self.observed['step'],
                  feedback=[item['text'] for item in result['contentItems'] if item['type'] == 'inputText']))
        return result

    def _execute(self, name, args, call_id):
        if self.read_only and name != 'observe':
            return dict(success=False, contentItems=[text_item('Read-only test: motion/done disabled; no action executed.')])
        if name == 'done':
            self.abandoned = True
            self.snapshot()
            return dict(success=True, contentItems=[text_item(dict(status='model_abandoned', actions_executed=0))])
        request = dict(op=name, **deepcopy(args))
        if name == 'execute_actions':
            request.update(request_id=str(uuid4()), episode_id=self.observed['episode_id'],
                           expected_step=self.observed['step'])
        self.pending = dict(call_id=call_id, request=deepcopy(request), expected_step=self.observed['step'])
        self.save('pending_action.json', self.pending)
        self.snapshot()
        reply = self.remote.exchange(request)  # Exactly one send; exceptions retain pending.
        if not isinstance(reply, dict) or reply.get('op') != 'result':
            raise RuntimeError('Uncertain robot acknowledgement; never replay')
        count, finished = reply.get('actions_executed'), reply.get('episode_finished')
        status = reply.get('status', 'executed')
        maximum = len(args['actions']) if name == 'execute_actions' else 0
        after = public_observation(reply['observation'])
        if (type(count) is not int or not 0 <= count <= maximum or type(finished) is not bool
                or status not in ('executed', 'rejected')
                or (status == 'rejected' and count != 0)
                or (status == 'executed' and count != maximum and not finished)
                or after['episode_id'] != self.observed['episode_id']
                or after['step'] != self.observed['step'] + count
                or not math.isclose(after['time_s'] - self.observed['time_s'], count / 20,
                                    rel_tol=0, abs_tol=1e-6)):
            raise RuntimeError('Uncertain robot acknowledgement; never replay')
        self.observed = after
        self.pending = None
        self.completed = finished  # Episode termination is not task success.
        self.save('pending_action.json', dict(acknowledged=True, call_id=call_id, step=after['step']))
        self.snapshot()
        feedback = dict(status=status, actions_executed=count)
        if status == 'rejected':
            # No arbitrary exception, score, object truth, or owner report goes to the model.
            feedback['hint'] = 'Native worker rejected this request; no action from this request executed.'
        return dict(success=status != 'rejected', contentItems=[text_item(feedback)] + self.observation_items())

    def close(self):
        """Trusted operator only; no retry or concurrent/uncertain-action closure."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('A robot operation is in flight; closure requires owner review')
        try:
            if self._close_attempted:
                if self.closure is not None:
                    return deepcopy(self.closure)
                raise RuntimeError('Closure outcome uncertain; never re-send close')
            if self.pending is not None:
                raise RuntimeError('Unacknowledged robot request; closure requires owner review')
            self._close_attempted = True
            receipt = dict(close_requested=True, closed_ack=False, transport_exit_code=None,
                           worker_closed=False, native_cleanup_verified=False,
                           owner_receipt_verification_required=True)
            self.pending = dict(request={'op': 'close'}, expected_step=self.observed['step'])
            self.save('closure.json', receipt)
            self.save('pending_action.json', self.pending)
            self.snapshot()
            try:
                reply = self.remote.exchange({'op': 'close'})
                if not isinstance(reply, dict) or reply.get('op') != 'closed':
                    raise RuntimeError('Missing native closed acknowledgement')
                receipt['closed_ack'] = True
                self.owner_result = deepcopy(reply.get('result'))
                # Operator-only native result; never included in model observations.
                self.save('close_ack.private.json', reply)
                self.pending = None
                self.save('pending_action.json', dict(acknowledged=True, op='close'))
                code = self.remote.close()  # Existing transport waits for actual process exit.
                if type(code) is not int:
                    raise RuntimeError('Transport did not provide a process exit code')
                receipt['transport_exit_code'] = code
                receipt['worker_closed'] = code == 0
                self.closed = receipt['worker_closed']
                self.closure = deepcopy(receipt)
                return deepcopy(receipt)
            except Exception as exc:
                receipt['error_type'] = type(exc).__name__
                raise
            finally:
                self.save('closure.json', receipt)
                self.snapshot()
        finally:
            self._lock.release()
