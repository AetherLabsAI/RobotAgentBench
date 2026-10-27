"""Portable server-tested chunk/project/commit loop, keeping original projection."""
import uuid
from ._server_projection import ActionAdapter
from .codec import contract
from robotagentbench.interfaces.action_contract import public_observation


def validate_task_id(task_id):
    if not isinstance(task_id,str)or not task_id or any(c not in'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_:-'for c in task_id):
        raise ValueError('Public task identifier required')


def prompt_binding(task_id,observation,explicit_prompt=None):
    validate_task_id(task_id)
    # As in Discovery's real control.model_command: replace the training table
    # with one public instruction. The key is a routing label, not a GT target.
    prompt=observation['instruction']if explicit_prompt is None else explicit_prompt
    if not isinstance(prompt,str)or not prompt:raise ValueError('Explicit public instruction text required')
    return 'public:'+task_id,prompt


def rollout(owner,model,ready,*,task_id,max_calls,on_event=lambda *_:None,explicit_prompt=None):
    if type(max_calls)is not int or max_calls<=0:raise ValueError('Explicit positive model-call budget required')
    c=contract(ready['contract']);obs=public_observation(ready['observation'])
    key,prompt=prompt_binding(task_id,obs,explicit_prompt)
    adapter=ActionAdapter(c);calls=executed=0;finished=False
    while not finished and calls<max_calls:
        # Preserve the old server's model-side prompt binding, without exposing
        # evaluator bindings, answer keys, score or task results to the policy.
        model_obs=dict(obs,instruction=prompt)
        request=dict(op='infer',observation=model_obs,action_order=c['action_order'],training_task_index=key)
        prediction=model.exchange(request)
        if (prediction.get('op')!='actions'or prediction.get('prompt')!=prompt
                or prediction.get('episode_id')!=obs['episode_id']or prediction.get('step')!=obs['step']):
            raise ValueError('Model reply does not match original public input/frame')
        if not isinstance(prediction.get('actions'),list)or len(prediction['actions'])!=50:
            raise ValueError('Preserve the original 50-row model chunk')
        on_event('prediction',calls,prediction)
        actions,projection=adapter.project(prediction['actions'])
        on_event('projection',calls,projection)
        reply=owner.exchange(dict(op='execute_actions',request_id=str(uuid.uuid4()),
            episode_id=obs['episode_id'],expected_step=obs['step'],actions=actions))
        if reply.get('status')=='rejected':adapter.commit(0);raise RuntimeError('Worker rejected projected actions; no automatic retry')
        count=reply['actions_executed'];adapter.commit(count)
        updated=public_observation(reply['observation'])
        if updated['episode_id']!=obs['episode_id']or updated['step']!=obs['step']+count:
            raise RuntimeError('Worker frame acknowledgment differs from executed rows')
        finished=reply['episode_finished']
        if type(finished)is not bool:raise ValueError('Explicit worker completion required')
        calls+=1;executed+=count;obs=updated
        on_event('reply',calls-1,dict(reply,observation=dict(obs,images='omitted_from_trace')))
    return dict(model_calls=calls,actions_executed=executed,episode_finished=finished,
                navigation_consumed=False,separate_context_consumed=False,
                public_prompt=prompt,public_task_key=key)
