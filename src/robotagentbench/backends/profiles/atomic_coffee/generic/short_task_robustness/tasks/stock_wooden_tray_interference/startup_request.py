"""Preserve the generated task's identity and distance selection at startup."""
from .distance_bands import checked_rule, require_manifest_band


def request_for_generated_room(plan, manifest):
    from .adapter import TASK_NAME, RULE_VERSION
    if plan.get('task_name') != TASK_NAME or plan.get('rule_version') != RULE_VERSION:
        raise ValueError('Expected this interference task and rule version')
    require_manifest_band(plan, manifest)
    rule = checked_rule(plan.get('relocation_distance'))
    if rule['mode'] == 'default':
        options = {}
    elif rule['mode'] == 'custom':
        options = {'relocation_distance': {k:rule[k] for k in ('min_cm','max_cm')}}
    else:
        options = {'relocation_distance': rule['mode']}
    product, = plan['target_categories']
    return dict(schema='openend.short_task.request.v1', task_name=TASK_NAME,
                rule_version=RULE_VERSION, seed=manifest['seed'],
                layout_id=manifest['layout_id'],
                targets=[dict(category=product,quantity=1)], options=options)
