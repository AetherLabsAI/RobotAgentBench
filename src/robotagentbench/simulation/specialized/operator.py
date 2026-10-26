"""Trusted specialized native operator. Inspection never starts a simulator.

Execution is a one-shot fresh-interpreter handoff, not an orchestrator, model
client, source downloader, scene generator or scoring implementation.
"""
import argparse
import json
import os
from pathlib import Path
import sys

from . import config


def describe(operator, family, task):
    supplied, profile, selected = config.selection(operator, family, task)
    return dict(family=family, task_name=task,
        entry_module=config.MODULES[family], entry_function='exec_native',
        source_identity=profile['release_identity'], source_files=profile['source_file_count'],
        required_family_startup_resources=len(profile['required_startup_resources']),
        launch_resource=selected['launch'], native_started=False, gpu_queried=False,
        ssh_invoked=False, prepared_scene_generated=False, native_accepted=False,
        outer_lifecycle_enforced_by_this_operator=False,
        external_independent_supervisor_required=True)


def dispatch_value(operator, operator_path, family, task, verified):
    return dict(schema=config.DISPATCH, family=family, task_name=task,
        operator_sha256=config.sha(operator_path), source_identities_sha256=config.IDENTITIES_SHA,
        launch=str(verified['launch_path']), launch_sha256=config.sha(verified['launch_path']),
        input_hashes={str(p): h for p, h in sorted(verified['inputs'].items())},
        scene_generated=False, original_admission_executed=False, native_started=False)


def prepare(operator, operator_path, family, task, destination):
    """Seal existing original prepared inputs; do not rerun original preparation."""
    verified = config.verify(operator, family, task, mounted=False)
    workspace = verified['workspace']
    destination = Path(destination)
    if (not destination.is_absolute() or destination.resolve() != destination
            or destination.parent.resolve() != destination.parent or not destination.parent.is_dir()
            or not any(destination != root and destination.is_relative_to(root) for root in workspace.input_roots)):
        raise ValueError('Fresh dispatch descriptor must remain below an explicit trusted input root')
    value = dispatch_value(operator, operator_path, family, task, verified)
    with destination.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    return destination


def _raw_options(options, expected_gpu):
    if options.gpu != expected_gpu or any(getattr(options, key) for key in (
            'smoke', 'coding_policy', 'collision_guard_audit',
            'allow_idle_utilization', 'allow_shared_gpu')):
        raise ValueError('Preserve the original exclusive raw-action owner without diagnostic overrides')


def callback(operator, operator_path, task):
    """Trusted injection seam; never selected from launch/model module fields."""
    operator_path = Path(operator_path)
    operator_sha = config.sha(operator_path)
    def handoff(family, launch, output, *, options, workspace):
        _raw_options(options, operator['gpu'])
        if config.sha(operator_path) != operator_sha or config.read_operator(operator_path) != operator:
            raise ValueError('Trusted operator changed before handoff')
        verified = config.verify(operator, family, task, mounted=True)
        if verified['workspace'] != workspace or verified['launch_path'] != Path(launch):
            raise ValueError('Handoff cannot select another workspace or prepared launch')
        output = workspace.output(output)
        if output.parent != workspace.output_root:
            raise ValueError('Only a fresh direct child of this task-owned output parent is allowed')
        from robotagentbench.simulation.source_tree import NAMESPACES
        if any(name.split('.')[0] in NAMESPACES for name in sys.modules):
            raise ValueError('Specialized native handoff requires a fresh source interpreter')
        # The bootstrap module and entry function are fixed in this package.
        # Isolated Python ignores inherited cwd/PYTHONPATH; only operator-bound
        # installed code and the reviewed SDK paths enter the fresh child.
        bootstrap = ('import runpy,sys;sys.dont_write_bytecode=True;'
            'sys.path.insert(0,' + repr(operator['public_root']) + ');'
            'sys.argv=["robotagentbench.simulation.specialized.child",*sys.argv[1:]];'
            'runpy.run_module("robotagentbench.simulation.specialized.child",run_name="__main__")')
        environment = verified['environment']
        argv = [str(environment.launcher), '-I', '-S', '-B', '-c', bootstrap,
                str(operator_path), operator_sha, family, task, str(launch), str(output)]
        os.execve(environment.launcher, argv, environment.environment(os.environ))
        raise RuntimeError('Fresh specialized handoff unexpectedly returned')
    return handoff


def execute(operator, operator_path, family, task, launch, output, dispatch):
    """Only an explicitly authorized independently supervised process may call."""
    verified = config.verify(operator, family, task, mounted=True)
    if verified['launch_path'] != Path(launch):
        raise ValueError('Only the task launch pinned by the trusted operator is accepted')
    descriptor = verified['workspace'].input(dispatch)
    if config.read(descriptor) != dispatch_value(operator, operator_path, family, task, verified):
        raise ValueError('Prepared dispatch identity differs; no automatic reprepare/retry')
    from robotagentbench.simulation.original_backend import run_original_worker
    from robotagentbench.simulation.source_tree import NativeSourceTree
    from robotagentbench.simulation.worker import NativeRunOptions
    sources = NativeSourceTree(verified['root'],
        {p.lstrip('/'): sha for p, sha in verified['profile']['pins'].items()})
    return run_original_worker(launch, output, options=NativeRunOptions(gpu=operator['gpu']),
        sources=sources, workspace=verified['workspace'],
        specialized_owner=callback(operator, operator_path, task))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operator', type=Path)
    parser.add_argument('--family', choices=sorted(config.MODULES), required=True)
    parser.add_argument('--task', required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--verify', action='store_true')
    modes.add_argument('--prepare', type=Path, metavar='FRESH_DISPATCH_DESCRIPTOR')
    modes.add_argument('--execute', action='store_true')
    parser.add_argument('--launch', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--dispatch', type=Path)
    args = parser.parse_args(argv)
    operator_path = args.operator.resolve(strict=True)
    if operator_path != args.operator:
        raise ValueError('Canonical absolute trusted operator path required')
    operator = config.read_operator(operator_path)
    if args.execute:
        if any(p is None for p in (args.launch, args.output, args.dispatch)):
            raise ValueError('Execute requires explicit original launch, fresh output and prepared dispatch')
        execute(operator, operator_path, args.family, args.task, args.launch, args.output, args.dispatch)
        raise RuntimeError('Original specialized owner returned unexpectedly')
    if any(p is not None for p in (args.launch, args.output, args.dispatch)):
        raise ValueError('Launch/output/dispatch are execute-only inputs')
    if args.prepare:
        result = dict(prepared_dispatch=str(prepare(operator, operator_path, args.family, args.task, args.prepare)),
                      native_started=False, scene_generated=False)
    elif args.verify:
        config.verify(operator, args.family, args.task, mounted=False)
        result = dict(verified_files=True, task_name=args.task, native_started=False,
                      gpu_queried=False, mounted_namespace_verified=False)
    else:
        result = describe(operator, args.family, args.task)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
