"""Prepare immutable short-task configurations with the unchanged original owner.

No simulator, model, action, scoring or parameter override is implemented here.
Run only against an independently copied writable Runtime, never the shared
installation. Original failed sampling/admission is retained, not resampled.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

TASKS = ('prepare_coffee', 'bake_bread', 'stock_wooden_tray', 'retrieve_from_drawer',
    'sofa_to_drawer', 'hang_cap', 'clear_table_trash', 'fridge_to_heated_box',
    'clear_tablecloth', 'pack_object', 'prepare_cutlery', 'carry_meal_tray',
    'sofa_to_coffee_table', 'table_to_storage_basket', 'microwave_heat_take', 'wipe_table')
SCHEMA = 'robotagentbench.short_instance_bank.v1'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def semantic_fingerprint(state):
    # A changed layout label alone is not physical variation. The saved actual
    # object matrices, bindings and measured initial base remain the evidence.
    return fingerprint({key:value for key,value in state.items()if key!='layout_id'})


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.new')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    temporary.replace(path)


def physical_configuration(launch):
    """Semantic identity excludes seed, timestamps, output paths and USD hash.

    Actual composed object transforms and semantic target bindings determine
    distinctness. A changed seed or filename alone cannot increase this count.
    """
    from pxr import Usd, UsdGeom
    manifest = read(launch['manifest'])
    stage = Usd.Stage.Open(manifest['scene'])
    if not stage:
        raise ValueError('Prepared saved scene does not reload on CPU')
    cache = UsdGeom.XformCache()
    objects = []
    for row in manifest.get('objects', []):
        prim = stage.GetPrimAtPath(row['path'])
        if not prim or not prim.IsActive():
            raise ValueError('Saved manifest object is absent from its actual scene')
        matrix = cache.GetLocalToWorldTransform(prim)
        objects.append(dict(id=row.get('id'), role=row.get('role'),
            task_id=row.get('task_id'), matrix=[round(float(matrix[i][j]), 8) for i in range(4) for j in range(4)]))
    bindings = []
    for row in manifest.get('tasks', []):
        bindings.append({k: row[k] for k in ('id', 'type', 'target_path', 'source', 'destination', 'parameters') if k in row})
    value = dict(layout_id=manifest.get('layout_id'), objects=objects, task_bindings=bindings,
                 initial_base_xyyaw=launch.get('initial_base_xyyaw'))
    if not objects:
        raise ValueError('No actual object-state evidence; do not count a seed-only variation')
    return value


def _case_output(root, task, seed, attempt):
    label = task + '_seed%02d' % seed
    if task in ('microwave_heat_take', 'wipe_table'):
        return root / 'short_tasks/tasks' / task / ('execution/short_instance_bank_'+attempt) / label
    return root / 'short_tasks/agent_probe_v1/runs' / ('short_instance_bank_'+attempt) / label


def _files(launch_path):
    launch = read(launch_path)
    manifest_path = Path(launch['manifest'])
    manifest = read(manifest_path)
    result = [Path(launch_path), manifest_path, Path(launch['request']), Path(manifest['scene'])]
    for name in ('ready_pose', 'spawn', 'preparation_receipt'):
        if name in launch:
            result.append(Path(launch[name]))
    return {str(path): sha(path) for path in dict.fromkeys(result)}


def load(index_path, task, seed, *, verify_scene=False):
    """Read-only reload; does not regenerate, admit or execute the task.

    Preserve the saved container namespace for original internal references.
    Missing namespace/dependencies are errors, not grounds to rewrite a launch.
    """
    index = read(index_path)
    if index.get('schema') != SCHEMA or task not in TASKS or type(seed) is not int or not 1 <= seed <= 90:
        raise ValueError('Explicit saved short-task/seed identity required')
    matches = [r for r in index['instances'] if r['task_name'] == task and r['seed'] == seed]
    if len(matches) != 1 or matches[0]['status'] != 'prepared_cpu_not_native_accepted':
        raise ValueError('This task/seed has no accepted saved CPU preparation')
    row = matches[0]
    for path, expected in row['output_hashes'].items():
        if sha(path) != expected:
            raise ValueError('Saved launch/configuration was modified: ' + path)
    launch = read(row['launch'])
    if launch['task_name'] != task or read(launch['request'])['seed'] != seed:
        raise ValueError('Saved launch task/seed differs')
    if verify_scene and fingerprint(physical_configuration(launch)) != row['physical_fingerprint']:
        raise ValueError('Actual saved object poses or bindings changed')
    return row


def prepare(runtime_path, *, tasks=TASKS, seeds=tuple(range(1,31)), attempt='v31', resume=False,reuse_indexes=(),target_count=None):
    from robotagentbench.example import runtime
    from robotagentbench.simulation.source_tree import NativeSourceTree,SourceDependencyError
    def cpu_only(event,args):
        if event=='import'and args[0]in ('isaacsim.simulation_app','omni.kit.app','omni.physx'):
            raise RuntimeError('Saved-instance CPU preparation cannot start Isaac or PhysX')
    sys.addaudithook(cpu_only)
    base, root, configuration = runtime(runtime_path)
    marker = read(base.parent / 'OWNER.json')
    if marker != {'owner': 'RobotAgentBench-short-instance-bank-v31'}:
        raise ValueError('Use the explicitly independent short-instance Runtime only')
    if any(name not in TASKS for name in tasks):
        raise ValueError('Only approved short tasks 1–16')
    if target_count is not None and (type(target_count)is not int or not 1<=target_count<=30):
        raise ValueError('A bounded target of 1..30 distinct static preparations is required')
    if not attempt.replace('_','').isalnum() or any(type(seed)is not int or not 1<=seed<=90 for seed in seeds):
        raise ValueError('Explicit output-attempt and bounded original static candidate seeds1..90 required')
    out = base.parent / ('bank' if attempt=='v31' else 'bank_'+attempt)
    index_path = out / 'index.json'
    if index_path.exists() and not resume:
        raise ValueError('Existing bank: explicitly resume unclaimed seeds, never regenerate')
    out.mkdir(exist_ok=True)
    (out/'intents').mkdir(exist_ok=True)
    index = read(index_path) if index_path.exists() else dict(schema=SCHEMA,
        runtime=str(base), source_root=str(root), seeds=list(seeds),
        native_accepted=False, model_called=False, scoring_or_physics_modified=False, instances=[])
    index['seeds']=sorted(set(index.get('seeds',[]))|set(seeds))
    existing={(r['task_name'],r['seed'])for r in index['instances']}
    for previous_path in reuse_indexes:
        previous_path=Path(previous_path).resolve(strict=True)
        if not previous_path.is_relative_to(base.parent):raise ValueError('Reuse only an index from this owned Runtime view')
        previous=read(previous_path)
        if previous.get('runtime')!=str(base)or previous.get('source_root')!=str(root):
            raise ValueError('Do not mix preparation source views')
        for old in previous['instances']:
            key=(old['task_name'],old['seed'])
            if old['task_name']not in tasks or old['status']!='prepared_cpu_not_native_accepted':continue
            if type(old['seed'])is not int or not 1<=old['seed']<=90:raise ValueError('Invalid saved candidate seed')
            if key in existing:continue
            row=dict(load(previous_path,*key,verify_scene=False))
            row['semantic_fingerprint']=semantic_fingerprint(row['physical_configuration'])
            row['reused_saved_index']=str(previous_path)
            index['instances'].append(row);existing.add(key)
    save(index_path,index)
    source = NativeSourceTree(root, dict(configuration['source_files']))
    with source.activated():
        preparer = source.import_module('short_tasks.agent_probe_v1.prepare_task')
        request_for = source.import_module('short_tasks.agent_probe_v1.probe_random_layout').request_for
        ready = root/'layout_task_activation_v1/task_scene_registration_v1/recorded_ready_pose_042_v1.json'
        sensor = base/'example-sensor.json'
        for task in tasks:
            existing = {(r['task_name'], r['seed']) for r in index['instances']}
            for seed in seeds:
                good={semantic_fingerprint(r['physical_configuration'])for r in index['instances']
                      if r['task_name']==task and r['status']=='prepared_cpu_not_native_accepted'}
                if target_count is not None and len(good)>=target_count:
                    break
                if (task,seed) in existing:
                    continue
                intent = out/'intents'/('%s_%02d.json' % (task,seed))
                if intent.exists():
                    raise ValueError('Unsettled previous preparation intent; inspect before any retry: '+str(intent))
                request = request_for(task)
                request.update(seed=seed, layout_id=None)
                target = _case_output(root,task,seed,attempt)
                save(intent,dict(task_name=task,seed=seed,output=str(target),request=request,
                    preparation_only=True,physical_execution_permitted=False))
                started = time.monotonic()
                row = dict(task_name=task, short_task_number=TASKS.index(task)+1, seed=seed,
                    original_request=request, request_fingerprint=fingerprint(request), output=str(target))
                infrastructure = False
                try:
                    result = preparer.prepare(request,ready,sensor,target)
                    launch_path = Path(result['launch'])
                    launch = read(launch_path)
                    state = physical_configuration(launch)
                    row.update(status='prepared_cpu_not_native_accepted', launch=str(launch_path),
                        launch_sha256=sha(launch_path), output_hashes=_files(launch_path),
                        physical_configuration=state, physical_fingerprint=fingerprint(state),
                        semantic_fingerprint=semantic_fingerprint(state),
                        dependencies=read(launch['manifest']).get('source_hashes',{}),
                        native_accepted=False, process_score=None)
                except Exception as exc:
                    row.update(status='original_preparation_failed',error=type(exc).__name__+': '+str(exc),
                        exception_module_name=getattr(exc,'name',None),
                        traceback=traceback.format_exc(),native_accepted=False,process_score=None)
                    infrastructure = (isinstance(exc,(FileNotFoundError,ModuleNotFoundError,ImportError,PermissionError,SyntaxError,SourceDependencyError))
                        or getattr(exc,'reason',None) in ('component_source_changed','component_source_missing','scene_planning_failed')
                        or any(text in str(exc)for text in('Pinned source changed:','Pinned source missing:',
                            'No such file or directory:','Required dependency has no reviewed pin',
                            'resource changed:','pinned component sources changed:')))
                    if 'evidence: 'in str(exc):
                        log=Path(str(exc).split('evidence: ',1)[1])
                        if log.is_file()and log.is_relative_to(root):
                            tail=log.read_text()[-6000:]
                            row['composition_error_tail']=tail
                            infrastructure=infrastructure or any(marker in tail for marker in(
                                'Failed to open layer','No such file or directory','ModuleNotFoundError'))
                row['cpu_wall_seconds']=round(time.monotonic()-started,3)
                index['instances'].append(row)
                valid=[r for r in index['instances'] if r['status']=='prepared_cpu_not_native_accepted']
                index['counts']={name:dict(prepared=sum(r['task_name']==name for r in valid),
                    distinct_physical=len({semantic_fingerprint(r['physical_configuration'])for r in valid if r['task_name']==name}),
                    failed=sum(r['task_name']==name and r['status']=='original_preparation_failed'for r in index['instances']))for name in TASKS}
                save(index_path,index)
                if row['status'] == 'prepared_cpu_not_native_accepted':
                    load(index_path,task,seed,verify_scene=False)
                    row['read_only_reload_passed']=True
                    save(index_path,index)
                print(json.dumps(dict(task_name=task,seed=seed,status=row['status'],counts=index['counts'][task],
                                      error=row.get('error'))),flush=True)
                if infrastructure:
                    raise RuntimeError('Stop after actual dependency failure; no repeated seed dispatch')
    return index


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--load',type=Path)
    p.add_argument('--runtime',type=Path);p.add_argument('--task',choices=TASKS,action='append')
    p.add_argument('--seed',type=int);p.add_argument('--attempt',default='v31');p.add_argument('--resume',action='store_true');p.add_argument('--verify-scene',action='store_true')
    p.add_argument('--reuse-index',type=Path,action='append',default=[])
    p.add_argument('--seed-start',type=int,default=1);p.add_argument('--seed-end',type=int,default=30)
    p.add_argument('--target-count',type=int)
    args=p.parse_args(argv)
    if args.prepare:
        if args.runtime is None:raise ValueError('Explicit independent Runtime required')
        result=prepare(args.runtime,tasks=tuple(args.task)if args.task else TASKS,
            seeds=(args.seed,)if args.seed is not None else tuple(range(args.seed_start,args.seed_end+1)),attempt=args.attempt,resume=args.resume,
            reuse_indexes=args.reuse_index,target_count=args.target_count)
        if any(row['status']=='original_preparation_failed'for row in result['instances']):return 1
    else:
        if not args.task or len(args.task)!=1 or args.seed is None:raise ValueError('One saved task/seed required')
        row=load(args.load,args.task[0],args.seed,verify_scene=args.verify_scene)
        print(json.dumps(dict(task_name=row['task_name'],seed=row['seed'],launch=row['launch'],read_only_reload=True)))
    return 0


if __name__=='__main__':raise SystemExit(main())
