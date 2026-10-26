"""Trusted binding boundary for the separately validated native FEM owner.

No scene/score implementation or task-selected imports live in this router.
Operator code supplies the reviewed admission and execution callables. Without
that binding these tasks remain unavailable, never a CPU or preview fallback.
"""
from dataclasses import dataclass
from typing import Callable

from .admission import AdmittedLaunch, verify_manifest

TASK_NAMES = frozenset(('spread_tablecloth', 'collect_tablecloth', 'long_task_18'))


@dataclass(frozen=True)
class NativeClothBinding:
    admit_launch: Callable
    execute: Callable

    def __post_init__(self):
        if not callable(self.admit_launch) or not callable(self.execute):
            raise ValueError('Reviewed native cloth admission and execution are required')

    def run(self, path, output, *, task_name, options, workspace):
        if task_name not in TASK_NAMES:
            raise ValueError('Task is outside the bound cloth family')
        admitted = self.admit_launch(path, output, task_name=task_name,
                                     options=options, workspace=workspace)
        if (not isinstance(admitted, AdmittedLaunch)
                or admitted.launch_path != path or admitted.output != output
                or admitted.owner_kind != 'native_cloth'
                or admitted.plan.get('task_name') != task_name
                or admitted.request.get('task_name') != task_name
                or admitted.plan.get('execution_mode') != 'native_cloth_scored.v1'
                or admitted.plan.get('available') is not True
                or admitted.plan.get('scoring_available') is not True):
            raise ValueError('Native cloth owner did not admit the exact scored task')
        admitted.verify_inputs()
        verify_manifest(admitted.manifest)
        self.execute(admitted, options=options, workspace=workspace)
        raise RuntimeError('Native cloth interpreter handoff returned unexpectedly')
