"""Lightweight registration; importing the HTTP app must not load USD/Isaac."""
from arena_api.task_runtime import TaskRegistration
from .tasks.microwave_take.termination import DEFAULT_EPISODE_TIMEOUT_S, CONTROL_HZ
from .tasks.microwave_take.parameters import PARAMETER_SCHEMA, normalize_parameters, instruction_for_parameters


def runtime(context):
    from .tasks.microwave_take.api_runtime import MicrowaveTakeRuntime
    return MicrowaveTakeRuntime(context)


def validate_randomization(flags):
    group = [flags.get(k, False) for k in ('appearance', 'layout', 'microwave_pose')]
    if any(group) and not all(group):
        raise ValueError('Use the unified randomization: appearance, layout and microwave_pose together')


def register_tasks(registry):
    from .tasks.tabletop_pick.book_cloth.registration import registration as book_cloth_registration
    registry.register(book_cloth_registration())
    from .tasks.basket_place.registration import registrations as basket_registrations
    for placement in basket_registrations():
        registry.register(placement)
    from .tasks.coffee_table_place.registration import registrations as coffee_place_registrations
    for placement in coffee_place_registrations():
        registry.register(placement)
    from .tasks.personal_item_pick.registration import registrations as personal_pick_registrations
    for entry in personal_pick_registrations():
        registry.register(entry)
    from .tasks.small_table_pick.registration import registrations as small_table_registrations
    for entry in small_table_registrations():
        registry.register(entry)
    from .tasks.tabletop_pick.registration import registrations as tabletop_registrations
    for entry in tabletop_registrations():
        registry.register(entry)
    from .tasks.trash_disposal.registration import registrations as trash_registrations
    for entry in trash_registrations():
        registry.register(entry)
    from .tasks.air_fryer.registration import registrations as air_fryer_registrations
    for entry in air_fryer_registrations():
        registry.register(entry)
    from .tasks.refrigerator_take.registration import registration as fridge_registration
    registry.register(fridge_registration())
    from .tasks.refrigerator_take.registration import variant_registrations as fridge_variants
    for entry in fridge_variants():
        registry.register(entry)
    from .tasks.dining_place.registration import registration
    registry.register(registration())
    from .tasks.shelf_pick.registration import registrations as shelf_registrations
    for entry in shelf_registrations():
        registry.register(entry)
    from .tasks.tidy_sofa.registration import registrations as livingroom_registrations
    for entry in livingroom_registrations():
        registry.register(entry)
    from .tasks.microwave_put.registration import registration as put_initial_probe_registration
    registry.register(put_initial_probe_registration())
    from .tasks.microwave_put.formal_registration import registration as put_scoring_registration
    registry.register(put_scoring_registration())
    registry.register(TaskRegistration(
        task_id='10.1', description='开门、夹盘取出牛排盒、关门、稳定持有 5 秒',
        instruction='打开微波炉，抓住盘子并抬起取出，保持牛排盒在盘上，关门后稳定持有至少五秒。',
        max_steps=int(DEFAULT_EPISODE_TIMEOUT_S * CONTROL_HZ),
        default_steps=int(DEFAULT_EPISODE_TIMEOUT_S * CONTROL_HZ), max_inferences=6000,
        evaluator_id='microwave_take.v1', runtime_factory=runtime,
        randomization_flags=('appearance', 'layout', 'microwave_pose'),
        validate_randomization=validate_randomization,
        parameter_schema=PARAMETER_SCHEMA, normalize_parameters=normalize_parameters,
        parameter_instruction=instruction_for_parameters))
