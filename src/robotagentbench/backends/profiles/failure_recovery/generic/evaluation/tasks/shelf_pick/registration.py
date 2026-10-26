"""Lightweight shelf-pick registration (API 4-1 … 4-14); no USD/Isaac dependency at service startup."""
from arena_api.task_runtime import TaskRegistration
from arena_api.bimanual import CONTROL_DT
from .process_rules import TIME_LIMIT_S
from .targets import SPECS, DEFAULT, spec as target_spec


def runtime(context):
    from .api_runtime import ShelfPickRuntime
    return ShelfPickRuntime(context)


def validate_randomization(flags):
    """Since the 2026-09-23 rebinding, 'layout' means: item layout from a held-out
    accepted source episode plus the ported robot perturbation, seeded by the job
    seed. Appearance (light/tint) randomization is not wired up - the source code
    for it only affects rendering - so it is rejected instead of silently ignored."""
    if any(v and k not in ('layout','appearance') for k,v in flags.items()):
        raise ValueError('Unsupported shelf randomization flag')
    if flags.get('appearance'):
        raise ValueError('Appearance randomization is not part of the new-data binding')


def registration(api_id=DEFAULT.api_id):
    spec=target_spec(api_id)
    steps=round(TIME_LIMIT_S/CONTROL_DT)
    if abs(steps*CONTROL_DT-TIME_LIMIT_S)>1e-9:raise ValueError('Task time limit is not control-aligned')
    return TaskRegistration(task_id=spec.api_id,description=spec.description,
        instruction=spec.instruction,
        max_steps=steps,default_steps=steps,max_inferences=steps,
        evaluator_id='shelf_pick.v2',runtime_factory=runtime,
        randomization_flags=('layout','appearance'),validate_randomization=validate_randomization)


def registrations():
    return [registration(api_id) for api_id in SPECS]
