"""Use existing worker JSONL pipes; does not start, reset or own a simulator."""
import json

from robotagentbench.interfaces import mobile_base
from robotagentbench.interfaces.action_contract import public_observation
from .codec import contract, inference_request, action_request

MAX_LINE=40*1024*1024


def read_line(stream):
    line=stream.readline(MAX_LINE+1)
    if not line:raise EOFError('Worker closed its stream; action outcome may be uncertain')
    if len(line)>MAX_LINE:raise ValueError('Oversized worker message')
    return json.loads(line,parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite worker message')))


def emit(stream,value):
    stream.write(json.dumps(value,allow_nan=False)+'\n');stream.flush()


def run(worker_output,worker_input,client,*,max_calls):
    if type(max_calls)is not int or max_calls<=0:raise ValueError('Explicit positive model-call budget required')
    ready=read_line(worker_output)
    if ready.get('op')!='ready':raise ValueError('Existing worker ready message required')
    c=contract(ready['contract']);obs=public_observation(ready['observation'])
    previous=dict(mobile_base.ZERO)
    calls=0;completed=False
    try:
        while calls<max_calls and not completed:
            prediction=client.infer(inference_request(c,obs));calls+=1
            request,_=action_request(prediction,obs,c,previous)
            # Nothing is sent if any row is invalid. No projection or retry.
            emit(worker_input,request)
            reply=read_line(worker_output)
            if reply.get('op')!='result'or reply.get('status')=='rejected':
                raise RuntimeError('Worker rejected model actions; no automatic correction/retry')
            count=reply.get('actions_executed')
            if type(count)is not int or not 0<=count<=len(request['actions']):
                raise RuntimeError('Worker action count is missing or invalid')
            updated=public_observation(reply['observation'])
            if updated['episode_id']!=obs['episode_id']or updated['step']!=obs['step']+count:
                raise RuntimeError('Worker returned stale/cross-episode action observation')
            completed=reply.get('episode_finished')
            if type(completed)is not bool or (not completed and count!=len(request['actions'])):
                raise RuntimeError('Partial unfinalized chunk; outcome uncertain')
            if count:previous=dict(zip(mobile_base.NAMES,request['actions'][count-1][20:]))
            obs=updated
        emit(worker_input,{'op':'close'})
        final=read_line(worker_output)
        if final.get('op')!='closed':raise RuntimeError('Missing original worker closed acknowledgment')
        # Private score remains operator-side, not fed back to the model.
        return dict(model_calls=calls,worker_closed=True,episode_finished=completed,
                    result=final.get('result'))
    except BaseException:
        # On in-flight I/O failure do not retry actions, send a second action or
        # claim cleanup. The external owner/watchdog must settle exact ownership.
        raise
