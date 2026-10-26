"""Task-owned traceability manifest; no source tree writes or installations."""
import argparse
import importlib.util
import json
from pathlib import Path
from .initial import HERE, digest, task_output

MODULES = (
    'arena_api.task_runtime', 'arena_api.policy_execution', 'arena_api.physics_defaults',
    'evaluation.tasks.coffee_machine.specs', 'evaluation.tasks.coffee_machine.registration',
    'evaluation.tasks.coffee_machine.api_runtime', 'evaluation.tasks.coffee_machine.initial_state',
    'evaluation.tasks.coffee_machine.randomization', 'evaluation.tasks.coffee_machine.isaac_evidence',
    'evaluation.tasks.coffee_machine.process_rules', 'evaluation.tasks.coffee_machine.facts',
    'evaluation.tasks.coffee_machine.visual_coffee',
)


def capture(path):
    path = task_output(path)
    if path.exists():
        raise ValueError('Fresh manifest required')
    rows = []
    for module in MODULES:
        spec = importlib.util.find_spec(module)
        source = Path(spec.origin).resolve(strict=True)
        rows.append(dict(module=module, file=str(source), sha256=digest(source)))
    value = dict(schema='arena.robustness.source_manifest.v1', task_key=HERE.name,
                 read_only_sources=rows, old_source_api='1-3', runtime_id=None,
                 global_registration=False)
    path.write_text(json.dumps(value, indent=2)+'\n')
    return value


def verify(path):
    path = task_output(path)
    value = json.loads(path.read_text())
    for row in value['read_only_sources']:
        actual = Path(importlib.util.find_spec(row['module']).origin).resolve(strict=True)
        if str(actual) != row['file'] or digest(actual) != row['sha256']:
            raise RuntimeError('Source module path/hash changed: '+row['module'])
    return dict(verified=True, source_files=len(value['read_only_sources']))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('capture','verify'))
    p.add_argument('--manifest', type=Path, required=True)
    a = p.parse_args()
    result = capture(a.manifest) if a.command == 'capture' else verify(a.manifest)
    print(json.dumps(result if a.command == 'verify' else dict(files=len(result['read_only_sources']))))
