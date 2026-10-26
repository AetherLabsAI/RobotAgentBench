"""Pure shared contract helpers for the document's multi-target adapters."""
from copy import deepcopy
import math


def options(request, task_name, allowed=()):
    if not isinstance(request, dict) or request.get('task_name') != task_name:
        raise ValueError('Wrong Short Task identity')
    value = request.get('options', {})
    if not isinstance(value, dict) or set(value)-set(allowed):
        raise ValueError('Unsupported task options')
    return value


def targets(request, pool, *, default=None, minimum=1, maximum=None):
    rows = request.get('targets', default)
    if not isinstance(rows, list) or not rows:
        raise ValueError('Explicit nonempty target list required')
    result=[]
    for row in rows:
        if not isinstance(row, dict) or set(row)-{'category', 'quantity'} or 'category' not in row:
            raise ValueError('Targets require category and optional quantity')
        if row['category'] not in pool:
            raise ValueError('Unsupported asset category: '+str(row['category']))
        quantity=row.get('quantity', 1)
        if type(quantity) is not int or quantity < 1:
            raise ValueError('Quantity must be a positive integer')
        result.extend([row['category']]*quantity)
    if len(result)<minimum or (maximum is not None and len(result)>maximum):
        raise ValueError('Unsupported target count')
    return result


def role(request, key, expected):
    value=request.get(key)
    if value is not None and (not isinstance(value, dict) or value != {'role': expected}):
        raise ValueError('Unsupported '+key+'; expected role '+expected)


def proposal(request, tasks):
    return dict(schema='openend.scene_requirements.request.v2', seed=request.get('seed', 1),
                layout_id=request.get('layout_id'), tasks=tasks,
                extras=deepcopy(request.get('extras', [])))


def plan(task_name, rule_version, request, tasks, components, maximum, process, final,
         instruction, requirements, *, unavailable=None):
    out=dict(task_name=task_name, rule_version=rule_version, available=unavailable is None,
             scene_request=proposal(request, tasks), runtime_components=components,
             initial_requirements=requirements, instruction=instruction,
             score_policy=dict(maximum=maximum, process_maximum=process, final_maximum=final,
                               arithmetic='component_defined'),
             runtime_adapter='short_tasks.tasks.'+task_name+'.adapter',
             native_validation='not_run')
    if unavailable is not None:
        out['unavailable']=dict(kind='rules', reason=unavailable[0], message=unavailable[1])
    return out


def summaries(plan, values):
    expected={row['instance_id'] for row in plan['runtime_components']}
    if isinstance(values, list):
        result={}
        for value in values:
            identity=value.get('instance_id', value.get('task_instance_id'))
            if identity in result or identity is None:raise ValueError('Unique component identity required')
            result[identity]=value
    elif isinstance(values, dict):result=values
    else:raise ValueError('Component summaries must be a mapping or identified list')
    if set(result)!=expected:raise ValueError('Missing, duplicate or unexpected internal component')
    for identity,value in result.items():
        if not isinstance(value, dict):raise ValueError('Component summary must be an object')
        for key in ('instance_id', 'task_instance_id'):
            if key in value and value[key]!=identity:raise ValueError('Wrong component identity')
    return result


def number(value):
    if type(value) not in (int,float) or not math.isfinite(value) or value<0:
        raise ValueError('Finite nonnegative component score required')
    return value


def result(plan, rows):
    channels={key:sum(number(r[key]) for r in rows)
              for key in ('process', 'current_final', 'deductions')}
    # Preserve each approved component's clipping, including shelf pick's
    # independent stage clipping. Original gross awards/penalties stay visible.
    channels['total']=sum(number(r.get('component_total',max(0,r['process']+r['current_final']-r['deductions']))) for r in rows)
    channels['maximum']=plan['score_policy']['maximum']
    return dict(task_name=plan['task_name'], rule_version=plan['rule_version'],
                scores=channels, success=bool(rows) and all(r['success'] for r in rows),
                stages=[dict(instance_id=r['instance_id'], process_complete=r['process_complete'],
                             process_points=r['process']) for r in rows],
                targets=deepcopy(rows), evidence_status='component_summaries_not_native_verification')


def close_component(summary, public_reason, source_close):
    """Public episode termination does not fabricate a source process terminal."""
    if not isinstance(public_reason,str) or not public_reason:raise ValueError('Explicit public termination reason required')
    if public_reason=='success' and summary.get('success') is not True:
        raise ValueError('No observed current component success')
    out=deepcopy(source_close('client_closed'))
    out['public_termination_reason']=public_reason
    out['source_close_reason']='client_closed'
    return out
