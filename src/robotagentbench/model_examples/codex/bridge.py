"""Semantic bridge: existing schemas/validator/robot RPC, not a new executor."""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from .tools import SYSTEM, TOOLS, validate_tool
from .model_response import validate_choice, ModelFormatError
from robotagentbench.interfaces.coding_contract import public_observation, specification
from .protocol import SCHEMA, tcp_public
from .map_guidance import GUIDANCE as MAP_GUIDANCE
REFERENCE_PROMPT = ''  # This example has no demonstration reader.
REFERENCE_TOOL = None


TRANSPORT_GUIDANCE = '''\nCODEX TRANSPORT ADAPTER (not a change to robot semantics):
Call arena tools DIRECTLY, not through functions.exec or another orchestration wrapper.
This preserves the tool result's native text and image content. Inspect all three returned
camera images before planning the next action. No parallel robot calls. Use only the supplied Arena tools;
host shell, collaboration, external tools and filesystem access are not part of this evaluation.
The execute_python robot sandbox remains available; it is not a host-terminal command.
'''


def tool_specs(reference=False):
    if reference: raise ValueError('Demonstration selection is not enabled in this example')
    source = deepcopy(TOOLS) + ([deepcopy(REFERENCE_TOOL)] if reference else [])
    return [dict(type='function', name=t['function']['name'],
                 description=t['function']['description'],
                 inputSchema=t['function']['parameters']) for t in source]


def text_item(value):
    return {'type': 'inputText', 'text': value if isinstance(value, str)
            else json.dumps(value, ensure_ascii=False, allow_nan=False)}


def reference_items(parts):
    result = []
    for part in parts:
        if part['type'] == 'text':
            result.append(text_item(part['text']))
        elif part['type'] == 'image_url':
            image = part['image_url']
            # Dynamic tool images have no detail field in CLI 0.160.0.
            if image.get('detail') not in (None, 'auto'):
                raise ValueError('Explicit image detail cannot be silently dropped')
            if not image['url'].startswith('data:image/'):
                raise ValueError('Only existing image bytes, not host paths/URLs')
            result.append({'type': 'inputImage', 'imageUrl': image['url']})
        else:
            raise ValueError('Unsupported reference content')
    return result


class RobotBridge:
    def __init__(self, remote, output, demonstration=None, read_only=False, *, max_model_responses=200):
        if demonstration is not None: raise ValueError('No demonstration source is selected')
        self.max_model_responses = max_model_responses
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.remote, self.demonstration, self.read_only = remote, demonstration, read_only
        self.completed = False
        self.abandoned = False
        self.pending = None
        self.calls = {}
        self.tool_count = 0
        self.model_responses = 0
        ready = remote.read()
        if ready.get('op') != 'ready' or ready.get('policy_schema') != SCHEMA:
            raise ValueError('Requires admitted native coding-policy host')
        self.contract = ready['contract']
        declared = dict(self.contract)
        fields=declared.pop('observation_fields', None)
        allowed={'episode_id','step','time_s','instruction','joints','base_state','context',
                 'image_step','images','tcp','tcp_frame','navigation'}
        if fields is not None and(not isinstance(fields,list)or any(name not in allowed for name in fields)):
            raise ValueError('Worker declared a non-public observation field')
        if declared != specification(self.contract['joint_limits']):
            raise ValueError('Native MOBILE22/20 Hz contract mismatch')
        self.observed = self.clean(ready['observation'])
        if demonstration:
            demonstration.bind(self.observed['instruction'])
        self.snapshot('initial_snapshot.json')

    @staticmethod
    def clean(value):
        clean = public_observation(value)
        clean.update(tcp=tcp_public(value['tcp']), tcp_frame='base_link')
        return clean

    def save(self, name, value):
        # Called by trusted runner only; model never chooses output paths.
        path = self.output/name
        tmp = path.with_suffix(path.suffix+'.tmp')
        tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
        tmp.replace(path)

    def snapshot(self, name='last_snapshot.json'):
        images = {}
        for camera in self.contract['camera_order']:
            data = base64.b64decode(self.observed['images'][camera], validate=True)
            filename = f'step{self.observed["step"]:06d}_{camera}.png'
            (self.output/filename).write_bytes(data)
            images[camera] = {'file': filename, 'sha256': hashlib.sha256(data).hexdigest()}
        self.save(name, {'observation': {k:v for k,v in self.observed.items() if k!='images'},
            'images': images, 'in_flight_action': self.pending, 'physics_checkpoint': False})

    @property
    def instructions(self):
        prompt = (SYSTEM + '\nRobot contract:\n' + json.dumps(self.contract) + TRANSPORT_GUIDANCE).replace('You have 30 model turns in this integration trial.',
            f'You have {self.max_model_responses} model turns in this integration trial.')
        prompt += '\nExisting60-second Python compute recovery can reset policy variables after verified physical steps. Inspect returned current observations; never replay acknowledged actions or an uncertain request.\n'
        return prompt + (MAP_GUIDANCE if 'navigation' in self.observed else '')

    def observation_items(self):
        public = {k:v for k,v in self.observed.items() if k not in ('images','navigation')}
        items = [text_item(public)]
        for name in self.contract['camera_order']:
            items += [text_item('Camera: '+name), {'type':'inputImage',
                'imageUrl':'data:image/png;base64,'+self.observed['images'][name]}]
        if 'navigation' in self.observed:
            nav = dict(self.observed['navigation'])
            png = nav.pop('map_png')
            items += [text_item({'explored_map':nav}),
                {'type':'inputImage','imageUrl':'data:image/png;base64,'+png}]
        return items

    def initial_items(self):
        items = reference_items(self.demonstration.overview()) if self.demonstration else []
        return items + self.observation_items()

    def dispatch(self, call_id, name, args):
        signature = json.dumps([name, args], sort_keys=True, allow_nan=False)
        if call_id in self.calls:
            old, result = self.calls[call_id]
            if old != signature:
                raise RuntimeError('Call ID reused with different arguments')
            return deepcopy(result)
        if self.pending:
            raise RuntimeError('Unacknowledged robot action: no retry or further dispatch')
        if self.completed or self.abandoned:
            return {'success': False, 'contentItems': [text_item('Episode stopped; no action executed.')]}
        self.tool_count += 1
        if self.tool_count > self.max_model_responses:
            raise RuntimeError('Tool-call budget reached; hold environment')
        tools = deepcopy(TOOLS)+([REFERENCE_TOOL] if self.demonstration else [])
        choice = {'message': {'role':'assistant','tool_calls':[{'type':'function','id':call_id,
            'function':{'name':name,'arguments':json.dumps(args, allow_nan=False)}}]}}
        try:
            validate_choice(choice, tools)
            if name not in ('done', 'view_demonstration'):
                validate_tool({'op':name, **args})
            if name == 'done' and not args['reason'].strip():
                raise ModelFormatError('done requires a non-empty reason')
        except ValueError:
            result = {'success': False, 'contentItems': [text_item({
                'status':'rejected','actions_executed':0,'hint':'Malformed/unknown tool; check the published schema.'})]}
        else:
            result = self._execute(name, args, call_id)
        self.calls[call_id] = (signature, deepcopy(result))
        self.save(f'tool_{self.tool_count:04d}.json', {'call_id':call_id,'name':name,
            'arguments':args,'success':result['success'],'step':self.observed['step'],
            'feedback':[x['text'] for x in result['contentItems'] if x['type']=='inputText']})
        return result

    def _execute(self, name, args, call_id):
        if self.read_only and name not in ('observe', 'view_demonstration'):
            return {'success':False,'contentItems':[text_item('Read-only integration test: motion/done disabled; no action executed.')]}
        if name == 'done':
            self.abandoned = True
            self.snapshot()
            return {'success':True,'contentItems':[text_item({'status':'model_abandoned','actions_executed':0})]}
        if name == 'view_demonstration':
            try:
                feedback, parts = self.demonstration.detail(args['times_s'])
                items = [text_item(feedback)]+reference_items(parts)
            except ValueError:
                items = [text_item('Invalid reference time; no action executed.')]
            return {'success':True,'contentItems':items+self.observation_items()}
        request = {'op':name, **args}
        if name == 'execute_actions':
            request.update(request_id=str(uuid4()), episode_id=self.observed['episode_id'],
                           expected_step=self.observed['step'])
        self.pending = {'call_id':call_id,'request':request,'expected_step':self.observed['step']}
        self.save('pending_action.json', self.pending)
        self.snapshot()
        reply = self.remote.exchange(request)
        count = reply.get('actions_executed')
        maximum = len(args['actions']) if name=='execute_actions' else 0 if name=='observe' else self.contract['max_tool_steps']
        after = self.clean(reply['observation'])
        if (reply.get('op')!='result' or type(count) is not int or not 0 <= count <= maximum
            or after['episode_id']!=self.observed['episode_id']
            or after['step']!=self.observed['step']+count):
            raise RuntimeError('Uncertain robot acknowledgement; never replay')
        self.observed = after
        self.pending = None
        self.save('pending_action.json', {'acknowledged':True,'call_id':call_id,'step':after['step']})
        self.completed = bool(reply.get('episode_finished'))
        self.snapshot()
        feedback = {'actions_executed':count,'status':reply.get('status','executed')}
        if 'motion' in reply:
            feedback['motion'] = reply['motion']  # Existing robot-only broker output.
        return {'success':True,'contentItems':[text_item(feedback)]+self.observation_items()}
