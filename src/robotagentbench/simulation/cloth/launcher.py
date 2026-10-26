"""Configured trusted cloth operator: describe/verify/prepare by default.

Execute is an explicit dedicated-process operation; agent RPC never supplies
Python imports, roots, templates, score rules or GPU exceptions.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from . import config

OPERATOR_SCHEMA='robotagentbench.cloth_operator.v1'
WATCHDOG_SECONDS=600


def read_operator(path):
    operator=config.read(path)
    fields={'schema','public_root','workspace','native','gpu','launch_templates',
        'outer_lifecycle'}
    if not isinstance(operator,dict)or set(operator)!=fields or operator['schema']!=OPERATOR_SCHEMA:
        raise ValueError('Explicit reviewed cloth operator configuration required')
    if set(operator['launch_templates'])!=config.TASKS:
        raise ValueError('All three exact cloth task templates are required')
    if type(operator['gpu'])is not int or operator['gpu']<0:
        raise ValueError('Explicit admitted device index required')
    if set(operator['workspace'])!={'input_roots','output_root','lock_root'}:
        raise ValueError('Explicit workspace boundaries required')
    if set(operator['native'])!={'dependencies','profile'}:
        raise ValueError('Existing reviewed interpreter and SDK profiles required')
    if operator['outer_lifecycle']!={
        'validation_watchdog_s':WATCHDOG_SECONDS,'exclusive_gpu':True,
        'read_only_inputs':True,'no_automatic_retry':True,'owned_cleanup_required':True}:
        raise ValueError('Existing independent-job lifecycle contract required')
    public=Path(operator['public_root'])
    if public.resolve()!=public or public!=config.HERE.parents[2]:
        raise ValueError('Operator must use this installed, pinned public source root')
    for task,template in operator['launch_templates'].items():
        if not isinstance(template,dict)or set(template)!={'directories','files','hashes'}:
            raise ValueError('Only fixed reviewed input templates are allowed')
        if set(template['directories'])!=config.DIRS or set(template['files'])!=config.FILES:
            raise ValueError('Complete source/asset/state/owner/SDK roles required')
        if template['directories']['public_root']!=str(public)or template['directories']['formal_root']!=str(config.HERE):
            raise ValueError('Template cannot select another package implementation')
        config.selection(task)
    return operator


def workspace_for(operator):
    from robotagentbench.simulation.workspace import NativeWorkspace
    boundary=operator['workspace']
    return NativeWorkspace(tuple(map(Path,boundary['input_roots'])),
        Path(boundary['output_root']),Path(boundary['lock_root']))


def launch_for(operator,task):
    selected=config.selection(task)
    template=deepcopy(operator['launch_templates'][task])
    owner=dict(schema=config.INNER,source_episode=selected['episode'],**template)
    owner.update(aliases=config.alias_recipe(owner),compatibility_mounts=config.compatibility_recipe(owner))
    return dict(schema=config.OUTER,task_name=task,execution_mode='native_cloth_scored.v1',owner=owner)


def describe(operator,task):
    value=launch_for(operator,task)
    return dict(task_name=task,selection=config.selection(task),launch_schema=value['schema'],
        owner_schema=value['owner']['schema'],execution_mode=value['execution_mode'],
        fixed_base=True,control_hz=20,physics_hz=60,substeps_per_control=3,
        head_camera=[1280,800],wrist_cameras=[[848,480],[848,480]],
        native_started=False,gpu_queried=False,ssh_invoked=False,
        validation_watchdog_s=WATCHDOG_SECONDS,watchdog_owner='external independent-job supervisor',
        outer_lifecycle_enforced_by_this_in_container_operator=False,
        source_owner_external_and_hash_pinned=True)


def verify(operator,task,*,mounted=True):
    from robotagentbench.simulation.dependencies import RuntimeDependencies
    from robotagentbench.simulation.environment import NativeEnvironment
    workspace=workspace_for(operator)
    value=launch_for(operator,task);config.validate(value,workspace,mounted=mounted)
    environment=NativeEnvironment.from_config(RuntimeDependencies.from_config(operator['native']['dependencies']),operator['native']['profile'])
    environment.verify()  # Read-only binary/file/metadata paths, not probe/import.
    return value,workspace,environment


def prepare(operator,task,path):
    value,workspace,_=verify(operator,task,mounted=False)
    return config.prepare(path,value,workspace=workspace)


def execute(operator,launch,output):
    """Only an explicitly authorized fresh dedicated process may call this."""
    from robotagentbench.simulation.dependencies import RuntimeDependencies
    from robotagentbench.simulation.environment import NativeEnvironment
    from robotagentbench.simulation.source_tree import NativeSourceTree
    from robotagentbench.simulation.worker import NativeRunOptions
    from robotagentbench.simulation.original_backend import run_original_worker
    from .binding import binding
    workspace=workspace_for(operator);path=workspace.input(launch);value=config.read(path)
    expected=launch_for(operator,value['task_name'])
    if value!=expected:raise ValueError('Prepared launch differs from the trusted fixed template')
    config.validate(value,workspace,mounted=True)
    environment=NativeEnvironment.from_config(RuntimeDependencies.from_config(operator['native']['dependencies']),operator['native']['profile'])
    root=Path(value['owner']['directories']['original_root'])
    pins={str(Path(p).relative_to(root)):h for p,h in value['owner']['hashes'].items()if Path(p).is_relative_to(root)}
    sources=NativeSourceTree(root,pins)
    return run_original_worker(path,output,options=NativeRunOptions(gpu=operator['gpu']),
        sources=sources,workspace=workspace,native_cloth_owner=binding(environment))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operator',type=Path)
    parser.add_argument('--task',choices=sorted(config.TASKS),required=True)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--verify',action='store_true')
    mode.add_argument('--prepare',type=Path,metavar='FRESH_LAUNCH')
    mode.add_argument('--execute',type=Path,metavar='PREPARED_LAUNCH')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv);operator=read_operator(args.operator)
    if args.execute:
        if args.output is None:raise ValueError('Explicit fresh output required')
        if config.read(args.execute).get('task_name')!=args.task:raise ValueError('Launch/task differs')
        execute(operator,args.execute,args.output)
        raise RuntimeError('Original cloth handoff returned')
    if args.output is not None:raise ValueError('--output is only valid with --execute')
    if args.prepare:
        print(json.dumps(dict(prepared_launch=str(prepare(operator,args.task,args.prepare)),native_started=False)))
    elif args.verify:
        verify(operator,args.task);print(json.dumps(dict(verified=True,task_name=args.task,native_started=False,gpu_queried=False)))
    else:print(json.dumps(describe(operator,args.task),indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
