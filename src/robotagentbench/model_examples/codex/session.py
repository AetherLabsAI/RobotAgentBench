"""One Codex turn, potentially multiple sequential dynamic tool calls.

The trusted owner retains the robot transport on errors. No automatic native
RPC retry, no environment close, no evaluation score exposed to the model.
"""
import time


ALLOWED_ITEMS = {'userMessage', 'agentMessage', 'reasoning', 'dynamicToolCall',
                 'functionCallOutput', 'contextCompaction'}


def user_items(items):
    out = []
    for item in items:
        if item['type'] == 'inputText':
            out.append({'type':'text', 'text':item['text']})
        elif item['type'] == 'inputImage':
            out.append({'type':'image', 'url':item['imageUrl']})
        else:
            raise ValueError('Unsupported robot input')
    return out


def run_turn(server, thread_id, bridge, *, input_items, effort='medium', timeout_s=None):
    started = time.monotonic()
    receipt = {'method':'codex_agent_robot_tools_v1', 'status':'running',
               'evaluation_valid':False, 'tool_calls':[], 'event_types':{},
               'model_responses':0, 'prior_model_responses':bridge.model_responses, 'warnings':[],
               'usage':None, 'cost_usd':None, 'billing':'ChatGPT plan; dollar cost not supplied',
               'native_close_requested':False, 'image_detail':'App Server default (dynamic result has no detail field)'}
    turn_id = None
    try:
        turn = server.request('turn/start', {'threadId':thread_id,
            'input':user_items(input_items), 'environments':[],
            'runtimeWorkspaceRoots':[], 'effort':effort}, timeout=60)
        turn_id = turn['turn']['id']
        receipt.update(thread_id=thread_id, turn_id=turn_id)
        while True:
            remaining = None if timeout_s is None else timeout_s-(time.monotonic()-started)
            if remaining is not None and remaining <= 0:
                raise TimeoutError('Integration turn budget expired; native environment retained')
            event = server.next_event(300 if remaining is None else min(remaining, 300))
            method = event.get('method', '')
            receipt['event_types'][method] = receipt['event_types'].get(method,0)+1
            params = event.get('params', {})
            if params.get('threadId') not in (None, thread_id):
                raise RuntimeError('Foreign thread event')
            if method == 'item/tool/call':
                if params.get('turnId') != turn_id or params.get('namespace') != 'arena':
                    raise RuntimeError('Foreign/unknown tool namespace or turn')
                result = bridge.dispatch(params['callId'], params['tool'], params['arguments'])
                receipt['tool_calls'].append({'tool':params['tool'],'call_id':params['callId'],
                    'success':result['success'], 'step':bridge.observed['step']})
                server.send({'id':event['id'],'result':result})
                if bridge.completed or bridge.abandoned:
                    server.request('turn/interrupt', {'threadId':thread_id,'turnId':turn_id})
                    receipt.update(status='completed' if bridge.completed else 'abandoned',
                                   evaluation_valid=bridge.completed)
                    break
            elif 'id' in event and 'method' in event:
                server.send({'id':event['id'], 'error':{'code':-32601,
                    'message':'Only Arena dynamic robot tools are permitted in this adapter'}})
                raise RuntimeError('Unexpected host/permission request: '+method)
            elif method in ('item/started', 'item/completed'):
                item = params.get('item', {})
                if item.get('type') not in ALLOWED_ITEMS:
                    raise RuntimeError('Unexpected built-in tool/item: '+str(item.get('type')))
                if method == 'item/completed' and item.get('type') == 'agentMessage':
                    pass  # Do not persist model prose or private reasoning/events.
            elif method == 'thread/tokenUsage/updated':
                receipt['usage'] = params.get('tokenUsage')
            elif method == 'rawResponse/completed':
                receipt['model_responses'] += 1
                bridge.model_responses += 1
                if bridge.model_responses >= bridge.max_model_responses:
                    raise RuntimeError('Model-response budget reached; no success/zero-score claim')
            elif method == 'warning':
                receipt['warnings'].append(params.get('message', 'App Server warning'))
            elif method == 'turn/completed':
                completed = params['turn']
                if completed['id'] != turn_id:
                    raise RuntimeError('Wrong completed turn')
                receipt.update(status='turn_completed_environment_held' if completed['status']=='completed'
                               else 'model_turn_failed', turn_status=completed['status'],
                               error=completed.get('error'))
                break
            elif method == 'error':
                receipt['last_server_error'] = params.get('error')
            bridge.save('codex_report.json', receipt)
    except Exception as exc:
        receipt.update(status='paused_adapter_error', error_type=type(exc).__name__, error=str(exc))
        if turn_id:
            try:
                server.request('turn/interrupt', {'threadId':thread_id,'turnId':turn_id}, timeout=10)
            except Exception:
                pass
    finally:
        bridge.snapshot()
        receipt.update(wall_time_s=time.monotonic()-started, last_step=bridge.observed['step'],
                       in_flight_action=bridge.pending, total_model_responses=bridge.model_responses)
        bridge.save('codex_report.json', receipt)
    return receipt
