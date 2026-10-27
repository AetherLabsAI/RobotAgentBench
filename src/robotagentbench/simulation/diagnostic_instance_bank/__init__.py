"""Saved diagnostic task instances: loading never resamples or starts simulation."""
import hashlib
import json
from pathlib import Path

SCHEMA = 'robotagentbench.diagnostic_instance_bank.v1'
INSTANCE = 'robotagentbench.diagnostic_instance.v1'
TASKS = {
    'task_discovery': ('prepare_dinner', 'host_guests', 'prepare_to_leave', 'tidy_room'),
    'select_drink': (), 'select_snack': (), 'select_trash': (), 'select_sofa': (),
    'failure_recovery_shelf_drink': (), 'failure_recovery_coffee_bag': (),
    'failure_recovery_sofa': (), 'failure_recovery_table_waste_empty_grasp': (),
    'failure_recovery_table_waste_tipped_cup': (),
    'stock_wooden_tray_interference': (), 'stock_wooden_tray_lighting': (),
    'coffee_bag_interference': (), 'coffee_machine_push_interference': (),
    'coffee_bag_then_machine_interference': (), 'coffee_button_lighting_interference': (),
}


def read(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('Duplicate JSON key')
            value[key] = item
        return value
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identity(task, seed, scenario=None):
    if task not in TASKS or type(seed) is not int or not 1 <= seed < 2**32:
        raise ValueError('Known diagnostic task and original positive uint32 seed required')
    if (TASKS[task] and scenario not in TASKS[task]) or (not TASKS[task] and scenario is not None):
        raise ValueError('Exact existing Discovery scenario required only for Discovery')
    return task + ('--' + scenario if scenario else '') + '--%02d' % seed


def _within(root, relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts or path.as_posix() != relative:
        raise ValueError('Canonical bank-relative file path required')
    result = root / path
    if result.resolve() != result or not result.is_file():
        raise ValueError('Saved bank file absent or symbolic: ' + relative)
    return result


def load(bank, task, seed, *, scenario=None):
    """Verify and read an already saved instance, without calling its generator.

    Private scoring bindings are returned to a trusted operator only. Use
    ``model_input`` for the exact separately saved public instruction/context.
    Failed or unresolved rows cannot be promoted to prepared launch instances.
    """
    root = Path(bank).resolve(strict=True)
    key = identity(task, seed, scenario)
    index = read(root / 'index.json')
    if index.get('schema') != SCHEMA:
        raise ValueError('Wrong saved-bank schema')
    rows = [r for r in index['entries'] if r['instance_id'] == key]
    if len(rows) != 1:
        raise ValueError('Saved instance is missing or ambiguous')
    row = rows[0]
    record = _within(root, row['record'])
    if digest(record) != row['record_sha256']:
        raise ValueError('Saved record changed')
    value = read(record)
    if (value.get('schema') != INSTANCE or value.get('instance_id') != key
            or value.get('task_name') != task or value.get('seed') != seed
            or value.get('scenario') != scenario):
        raise ValueError('Saved record identity changed')
    for relative, expected in value['files_sha256'].items():
        if digest(_within(root, relative)) != expected:
            raise ValueError('Saved artifact changed: ' + relative)
    if value['status'] not in ('prepared_cpu_not_native', 'resolved_only_not_prepared', 'blocked'):
        raise ValueError('Unknown preparation status')
    if value['status'] == 'prepared_cpu_not_native' and not value.get('launch'):
        raise ValueError('A prepared instance needs its real original launch')
    return value


def model_input(bank, task, seed, *, scenario=None):
    """Return only the saved public input, never the private answer/target list."""
    value = load(bank, task, seed, scenario=scenario)
    path = value.get('model_input')
    if path is None:
        raise ValueError('No public input was prepared for this instance')
    result = read(_within(Path(bank).resolve(strict=True), path))
    if set(result) != {'instruction', 'context'} or not all(isinstance(v, str) for v in result.values()):
        raise ValueError('Only original public instruction and context may reach a model')
    return result
