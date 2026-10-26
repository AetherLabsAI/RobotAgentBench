"""Fresh original scorers. No extra open-gripper points or relocation gate."""
from .common import CURRENT, COFFEE_SOURCE, require_source, validate_native_start


def shelf_scorer(episode_id, physics_dt_s, native_initial_checks):
    validate_native_start(native_initial_checks)
    # Same recoverable pick arithmetic used by the current shelf robustness task.
    from short_tasks.tasks.stock_wooden_tray import adapter
    require_source(adapter, CURRENT)
    return adapter.RecoverableShelfProcess(episode_id, physics_dt_s, task_id='4-3')


def coffee_scorer(native_initial_checks):
    validate_native_start(native_initial_checks)
    from evaluation.tasks.coffee_machine import process_rules
    require_source(process_rules, COFFEE_SOURCE)
    return process_rules.Process('1-2')
