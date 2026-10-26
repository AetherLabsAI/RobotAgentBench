"""Test-only continuation past 9.1 success; no motion or physics setters.

The ordinary API still ends at success unless explicitly opted in by the
operator's environment. Diagnostic runs record the first formal success and
continue observing the original recording; their EOF score is NOT an official
9.1 result or a handoff success criterion.
"""
from dataclasses import asdict
import json
import os


def continue_to_recording_end(runtime, evidence, score):
    if os.environ.get('ARENA_FRIDGE_BEEF_GRIPMAT_EXPERIMENT') != '1':
        raise ValueError('Handoff diagnostic requires the approved GripMat experiment')
    root = runtime.context.output
    if not getattr(runtime, '_handoff_diagnostic_started', False):
        with (root / 'handoff_diagnostic.json').open('x') as stream:
            json.dump(dict(diagnostic_only=True, formal_task_result=False,
                           purpose='Observe unchanged recorded left-to-right handoff after normal task success',
                           stop='source EOF; original public budget/failure checks retained',
                           controller_changed=False, trajectory_changed=False,
                           physics_changed_from_gripmat_baseline=(
                               os.environ.get('ARENA_FRIDGE_ORIGINAL_BEEF_EXPERIMENT') == '1')),
                      stream, indent=2)
        runtime._handoff_diagnostic_started = True
    if runtime.scorer.result().success and not getattr(runtime, '_handoff_first_success', False):
        with (root / 'first_formal_success.json').open('x') as stream:
            json.dump(dict(step_index=evidence.step_index, time_s=evidence.time_s,
                           episode_time_s=evidence.data.time_s, score=asdict(score),
                           diagnostic_continues=True), stream, indent=2, allow_nan=False)
        runtime._handoff_first_success = True
    return False
