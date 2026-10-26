"""Lightweight declarations of API 1-1 … 1-4; USD/Isaac imports stay in the worker."""
from arena_api.bimanual import CONTROL_DT
from arena_api.task_runtime import TaskRegistration
from .specs import APIS, EVALUATOR_ID, spec, episodes
from .process_rules import SPECS as RULES

TASK_IDS = APIS
PARAMETER_SCHEMA = dict(type='object', properties={
    'source_episode': dict(type='string', description='Layout of one packaged source episode of this API; default: the fixed initial state'),
    'random_layout': dict(type='boolean', default=False,
                          description='Seeded layout from the episode seed; only APIs whose definition names a reproducible sampler (1-3)')},
    additionalProperties=False)


def normalizer(api):
    def normalize(parameters):
        if not isinstance(parameters, dict) or set(parameters) - {'source_episode', 'random_layout'}:
            raise ValueError('Only source_episode and random_layout are supported')
        known = episodes(api)
        episode = parameters.get('source_episode', known[0])
        if not isinstance(episode, str) or episode not in known:
            raise ValueError('Unknown source episode for ' + api)
        random_layout = parameters.get('random_layout', False)
        if type(random_layout) is not bool:
            raise ValueError('random_layout must be a boolean')
        if random_layout:
            from .randomization import SUPPORTED
            if api not in SUPPORTED:
                raise ValueError(api + ': the definition table has no reproducible sampler yet')
            if episode != known[0]:
                raise ValueError('A seeded layout starts from the fixed initial state')
        return dict(source_episode=episode, random_layout=random_layout)
    return normalize


def runtime(context):
    from .api_runtime import CoffeeMachineRuntime
    return CoffeeMachineRuntime(context)


def registration(api):
    sp, rule = spec(api), RULES[api]
    steps = round(rule.time_limit_s / CONTROL_DT)
    return TaskRegistration(
        task_id=api, instruction=rule.instruction,
        description=f'{sp.name}（任务 {sp.task_no}）：{rule.instruction} 总时限 {rule.time_limit_s:g} s，满分 {rule.maximum_score}；'
                    '启动时必须通过初态物理验收。',
        max_steps=steps, default_steps=steps, max_inferences=steps,
        evaluator_id=EVALUATOR_ID, runtime_factory=runtime, randomization_flags=(),
        parameter_schema=PARAMETER_SCHEMA, normalize_parameters=normalizer(api))


def registrations():
    return tuple(registration(api) for api in TASK_IDS)
