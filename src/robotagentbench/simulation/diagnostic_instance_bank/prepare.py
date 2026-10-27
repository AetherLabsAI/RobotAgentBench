"""One original CPU preparation per saved diagnostic request; never simulate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

from . import INSTANCE, TASKS, identity


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def public_input(task, plan):
    if task == 'select_trash':
        public = plan['selection']['public']
        return dict(instruction=public['instruction'], context=public['context'])
    if task in ('select_drink', 'select_snack', 'select_sofa'):
        public = plan['public_question']
        return dict(instruction=' '.join(public.get(k, '') for k in ('instruction', 'scenario')),
                    context=public['context'])
    return dict(instruction=plan['instruction'], context='')


def request_for(registry, task, seed):
    if task == 'stock_wooden_tray_interference':
        from .relocation_request import request_for as formal_relocation_request
        return formal_relocation_request(seed)
    if task in registry.FAILURE_RECOVERY_TASKS:
        request = registry.failure_recovery_adapter().request_for(task)
    elif task in registry.ATOMIC_COFFEE_TASKS:
        request = registry.atomic_coffee_adapter().request_for(task)
    elif task in registry.CONTEXT_TASKS:
        adapter = registry.adapter(task)
        request = dict(schema='openend.short_task.request.v1', task_name=task,
                       rule_version=adapter.RULE_VERSION)
        if task != 'select_trash':
            # Public question_id is an existing original option, not a hidden
            # target selector. Select by public order, never by private answer.
            if task == 'select_sofa':
                bank = json.loads((Path(adapter.__file__).parent/'question_bank_v1/public_questions.json').read_text())
                rows = bank['questions']
            else:
                from short_task_ContextReasoning.tasks.select_shelf.bank import BANK_DIR
                bank = json.loads((BANK_DIR/'public_questions.json').read_text())
                rows = [q for q in bank['questions'] if q['kind'] == ('drink' if task == 'select_drink' else 'snack')]
            ids = sorted(q['id'] for q in rows)
            if len(set(ids)) < 30:
                raise ValueError('Original public question bank contains fewer than 30 unique IDs')
            request['options'] = dict(question_id=ids[seed-1])
    else:
        from short_tasks.agent_probe_v1.probe_random_layout import request_for as original
        if task == 'stock_wooden_tray_lighting':
            # Existing lighting variant accepts the base product request, while
            # the old example-file helper has no lighting example of its own.
            request = original('stock_wooden_tray')
            request.update(task_name=task, rule_version=registry.adapter(task).RULE_VERSION)
            request.pop('destination', None)
        else:
            request = original(task)
    request['seed'] = seed
    if task not in (registry.FAILURE_RECOVERY_TASKS | registry.ATOMIC_COFFEE_TASKS):
        request.pop('layout_id', None)  # Existing seeded layout selection, not a fixed old room.
    return request


def prepare_one(config, task, seed):
    if task == 'task_discovery':
        raise ValueError('Discovery derives from shared newly prepared OpenEnd rooms, not this short-task preparer')
    key = identity(task, seed)
    output_key = key + '--' + config.get('source_output_suffix', 'attempt01')
    bank = Path(config['bank'])
    relative = Path(config.get('case_subdirectory', 'cases/'+key))
    if relative.is_absolute() or '..' in relative.parts or relative.parts[:2] != ('cases', key):
        raise ValueError('Exact bank-owned case/attempt directory required')
    case = bank/relative
    case.mkdir(parents=True, exist_ok=False)
    phase = 'resolve'
    record = dict(schema=INSTANCE, instance_id=key, task_name=task, seed=seed,
                  scenario=None, status='blocked', launch=None, model_input=None,
                  files_sha256={}, native_accepted=False, model_called=False,
                  physics_scoring_camera_clock_timeout_modified=False)
    try:
        from short_tasks import registry
        if task == 'stock_wooden_tray_interference' and config.get('original_request_template'):
            template = Path(config['original_request_template'])
            raw = template.read_bytes()
            expected = config['original_request_template_sha256']
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError('The source-pinned formal relocation request changed')
            request = json.loads(raw)
            if (request.get('schema') != 'openend.short_task.request.v1'
                    or request.get('task_name') != task):
                raise ValueError('Exact original formal relocation task request required')
            # This source-pinned request is the actual native task entry, not
            # probe_random_layout's old pending-actor Sprite fallback. Preserve
            # task, targets, layout and distance selection; replace only seed.
            request['seed'] = seed
            record['original_request_template_sha256'] = expected
            record['request_seed_override_only'] = True
            record['files_sha256'][str(template.relative_to(bank))] = expected
        elif task == 'stock_wooden_tray_lighting' and config.get('shared_shelf_launch'):
            original = json.loads(Path(config['shared_shelf_launch']).read_text())
            request = json.loads(Path(original['request']).read_text())
            if request['task_name'] != 'stock_wooden_tray' or request['seed'] != seed:
                raise ValueError('Exact original ordinary-shelf task and actual seed required')
            request.update(task_name=task, rule_version=registry.adapter(task).RULE_VERSION)
        else:
            request = request_for(registry, task, seed)
        write(case/'request.json', request)
        plan = registry.resolve(request)
        registry.verify_sources(plan)
        write(case/'resolved.private.json', plan)
        write(case/'model_input.json', public_input(task, plan))
        record['model_input'] = str((case/'model_input.json').relative_to(bank))
        record['status'] = 'resolved_only_not_prepared'
        phase = 'prepare'
        from short_tasks.agent_probe_v1.prepare_task import prepare
        if task == 'stock_wooden_tray_interference' and config.get('original_reserve_first') is True:
            from .relocation_prepare import prepare as reserve_first
            output = registry.task_directory(task)/'generated_rooms'/('diagnostic_relocation_v40_'+output_key)
            result = reserve_first(request,Path(config['sensor']),output)
        elif task in registry.FAILURE_RECOVERY_TASKS:
            adapter = registry.failure_recovery_adapter()
            from short_task_robustness.tasks.grasp_failure_recovery.common import HERE
            output = HERE/'private_validation/diagnostic_instance_bank_v31'/output_key
            result = adapter.prepare_registered(request, None, None, output)
        elif task in registry.ATOMIC_COFFEE_TASKS:
            adapter = registry.atomic_coffee_adapter()
            worker = __import__(adapter.PACKAGE+'.prepare_worker', fromlist=['prepare'])
            output = Path(config['output_parents'][task])/output_key
            result = worker.prepare(request, output)
        elif task == 'stock_wooden_tray_lighting' and config.get('shared_shelf_launch'):
            from short_tasks.tasks.stock_wooden_tray_lighting.prepare_existing import prepare as derive_lighting
            output = registry.task_directory(task)/'checks/diagnostic_instance_bank_v31'/output_key
            derive_lighting(Path(config['shared_shelf_launch']), output)
            result = dict(launch=str(output/'launch.json'),
                          preparation='original_prepare_existing',
                          source_launch=config['shared_shelf_launch'])
            from short_tasks.agent_probe_v1.native_owner import admit_launch
            admitted = admit_launch(json.loads((output/'launch.json').read_text()), output/'diagnostic_cpu_admit')
            write(case/'original_cpu_admission.private.json', dict(
                passed=True, returned_items=len(admitted), native_started=False,
                original_ordinary_shelf_reused=config['shared_shelf_row']))
        else:
            output = (registry.task_directory(task)/'checks/diagnostic_instance_bank_v31'/output_key
                      if task in registry.CONTEXT_TASKS else
                      Path(config['generic'])/'short_tasks/agent_probe_v1/runs/diagnostic_instance_bank_v31'/output_key)
            result = prepare(request, Path(config['ready']), Path(config['sensor']), output)
        launch = Path(result['launch']) if isinstance(result, dict) and isinstance(result.get('launch'), str) else output/'launch.json'
        if not launch.is_file():
            raise ValueError('Original preparer did not produce a launch')
        if task in registry.FAILURE_RECOVERY_TASKS:
            # Recovery's generator does not itself invoke its original admission.
            # CPU admission checks only: this is not a simulation owner start.
            admission = registry.failure_recovery_adapter().admit_launch(
                json.loads(launch.read_text()), output/'diagnostic_cpu_admit')
            write(case/'original_cpu_admission.private.json', dict(
                passed=True, returned_items=len(admission), native_started=False))
        record['launch'] = config['backing_map'](launch) if callable(config.get('backing_map')) else str(launch)
        record.update(status='prepared_cpu_not_native', original_preparer_result=result,
                      original_launch_sha256=hashlib.sha256(launch.read_bytes()).hexdigest())
        from .variation import physical_signature
        semantic = physical_signature(launch, plan)
        write(case/'variation.private.json', semantic)
        record['variation_sha256'] = hashlib.sha256(json.dumps(semantic, sort_keys=True).encode()).hexdigest()
        record['output_logical'] = str(output)
    except BaseException as error:
        record.update(status='blocked', failure_phase=phase,
                      error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    # Case-local metadata is sealed now. The outer task-owned namespace runner
    # maps original logical output files to saved backing paths and seals them.
    for path in sorted(case.rglob('*')):
        if path.is_file():
            record['files_sha256'][str(path.relative_to(bank))] = hashlib.sha256(path.read_bytes()).hexdigest()
    write(case/'record.pending.json', record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--task', choices=tuple(TASKS), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--scenario')
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    def forbid_native(event, values):
        if event == 'import' and values and str(values[0]).startswith(('omni.kit', 'isaacsim', 'omni.physx', 'torch')):
            raise RuntimeError('CPU saved-instance preparation cannot import a simulation owner')
    sys.addaudithook(forbid_native)
    sys.path[:0] = config['python_paths']
    os.environ['PYTHONPATH'] = os.pathsep.join(config['python_paths'])
    if config.get('usd_schema_plugin_roots'):
        from pxr import Plug
        for root in config['usd_schema_plugin_roots']:
            Plug.Registry().RegisterPlugins(root)
    if args.task == 'task_discovery':
        from .discovery import prepare as derive_discovery
        row = derive_discovery(config, args.scenario, args.seed)
    else:
        row = prepare_one(config, args.task, args.seed)
    print(json.dumps(dict(instance_id=row['instance_id'], status=row['status'], launch=row['launch'],
                          failure_phase=row.get('failure_phase'), error=row.get('error'))), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
