"""Trusted local diagnostics, separate from public task/scoring registration."""
DIAGNOSTIC_IDS = ("refrigerator_initial.v1",)


def create_reader(identifier, stage, output):
    if identifier != "refrigerator_initial.v1":
        raise ValueError("Unknown trusted diagnostic ID")
    from .tasks.refrigerator_take.initial_evidence import InitialEvidenceReader
    from .tasks.refrigerator_take.initial_state import TASK_DIR
    if not output.resolve().is_relative_to(TASK_DIR):
        raise ValueError("Fridge diagnostic output must remain in its task directory")
    return InitialEvidenceReader(stage)
