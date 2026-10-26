"""Application composition root; new task packages register only here."""
from .task_runtime import RuntimeRegistry, TaskRegistration


def default_registry():
    registry = RuntimeRegistry()
    registry.register(TaskRegistration("microwave_open_001", "Legacy reset not implemented", ready=False))
    registry.register(TaskRegistration("isaac_policy_execution", "Generic physical execution; no task scoring"))
    from evaluation.registration import register_tasks
    register_tasks(registry)
    return registry
