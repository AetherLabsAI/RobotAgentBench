"""Single-host Docker setup for the existing room model clients.

Restoration and cache preparation use CPU-only containers. Only ``run
--execute`` starts a model/native episode. No scorer, sampler, physics or
policy implementation is replaced here; specialized owners remain separate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from robotagentbench import namespace
from robotagentbench.example import runtime
from robotagentbench.simulation import native_deployment as deployment

SCHEMA = 'robotagentbench.docker_model_run.v1'
PYTHON = '/usr/local/bin/python3.11'
MODULE = 'robotagentbench.model_examples.docker_workflow'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def endpoint(value):
    if not value.startswith('unix:///'):
        raise ValueError('Use a local Unix Docker socket on the Linux worker host')
    return value


def image_id(image, docker_host):
    result = subprocess.run(['docker', '--host', endpoint(docker_host), 'image',
                             'inspect', '--format', '{{.Id}}', image],
                            check=True, capture_output=True, text=True)
    return deployment.image_id(result.stdout.strip())


def mount(args, source, target=None, *, readonly=True):
    source = Path(source).resolve(strict=True)
    target = str(source) if target is None else str(target)
    if ',' in str(source) or ',' in target or not Path(target).is_absolute() or '..' in Path(target).parts:
        raise ValueError('Canonical mount paths without commas required')
    for i, item in enumerate(args[:-1]):
        if item == '--mount' and 'dst='+target in args[i+1].split(','):
            raise ValueError('Duplicate mount target: '+target)
    args.extend(['--mount', 'type=bind,src='+str(source)+',dst='+target+(',readonly' if readonly else '')])


def container_prefix(base, image, docker_host, *, name=None, gpu=None, locks=None):
    mode = 'plan' if gpu is None else 'run'
    args = namespace.command(base, image, mode, name=name, gpu=gpu, locks=locks,
                             native_deployment=gpu is not None)
    args = args[:args.index(image)]
    args[1:1] = ['--host', endpoint(docker_host)]
    # Preserve saved absolute paths and the coding sandbox's host-daemon binds.
    mount(args, base, readonly=False)
    return args


def restore(args):
    base, _, _ = runtime(args.runtime)
    release = args.release.resolve(strict=True)
    out = args.output.resolve()
    if out.exists() or not out.parent.is_dir() or out.parent in (Path('/'), Path.home(), base):
        raise ValueError('Choose a fresh selection directory under an existing dedicated parent')
    if out.is_relative_to(base) or base.is_relative_to(out.parent) or out.is_relative_to(release.parent):
        raise ValueError('Keep the selection receipt directory separate from Runtime and the configuration bank')
    # Fail before container creation for an unsupported dedicated task owner.
    from robotagentbench.simulation.saved_instance_release.restore import describe
    selected = describe(release, base, args.task, args.episode, out)
    if selected['mode'] in ('cloth', 'specialized'):
        raise ValueError('Use the dedicated cloth/specialized owner, not this room example')
    image = image_id(args.image, args.docker_host)
    cmd = container_prefix(base, image, args.docker_host)
    mount(cmd, release.parent)
    mount(cmd, out.parent, readonly=False)
    cmd += [image, PYTHON, '-B', '-m',
            'robotagentbench.simulation.saved_instance_release.restore',
            '--release', str(release), '--runtime', str(base), '--task', args.task,
            '--episode', str(args.episode), '--output', str(out),
            '--sdk-site', deployment.SDK_ALIAS, '--execute']
    subprocess.run(cmd, check=True)
    return dict(selection=str(out/'selection.json'), native_started=False, model_called=False)


def selection(path):
    path = Path(path).resolve(strict=True)
    row = read(path)
    if row.get('schema') != 'robotagentbench.restored_compact_initial_state.v1':
        raise ValueError('Select a restored ordinary room/short-task instance')
    base, source, metadata = runtime(row['runtime'])
    launch = Path(row['launch']).resolve(strict=True)
    if (Path(row['source_root']) != source or not launch.is_relative_to(source)
            or sha(launch) != row['launch_sha256']):
        raise ValueError('Saved selection Runtime/source/launch identity changed')
    if read(launch).get('schema') not in ('openend.short_task.launch.v1', 'openend.coding_room.launch.v1'):
        raise ValueError('This example requires the original room/short-task owner')
    if row.get('readonly_resource_aliases'):
        raise ValueError('This selected variant needs its dedicated resource namespace')
    # The generic restoration receipt has a TOP-LEVEL public_input_binding.
    binding = row.get('public_input_binding')
    if binding:
        from robotagentbench.simulation.public_input import load
        from robotagentbench.simulation.workspace import NativeWorkspace
        load(binding, launch, workspace=NativeWorkspace((source,), source, source))
    return path, row, base, source, metadata


def configure(args):
    selected, row, base, source, metadata = selection(args.selection)
    name = deployment.name(args.name)
    if args.gpu < 0:
        raise ValueError('Explicit native GPU index required')
    locks = args.lock_root.resolve(strict=True)
    if not locks.is_dir():
        raise ValueError('Create the host-wide shared GPU-lock directory first')
    dest = args.destination.resolve()
    if dest.exists() or not dest.parent.is_dir() or dest.is_relative_to(base) or base.is_relative_to(dest):
        raise ValueError('Use a fresh configuration directory outside Runtime, with an existing parent')
    output = source/'short_tasks/agent_probe_v1/cap_v1/model_runs'/name
    if output.exists() or (base/'native_deployments'/name).exists():
        raise ValueError('This run identity is already prepared or consumed; choose a new name')
    model = None
    if args.agent == 'internvla':
        if args.model_python is None or args.model_dir is None or args.model_gpu is None:
            raise ValueError('InternVLA requires --model-python, --model-dir and --model-gpu')
        if args.model_gpu < 0 or args.model_gpu == args.gpu or args.max_calls < 1:
            raise ValueError('Use distinct native/model GPUs and a positive model-call budget')
        from .internvla.server_policy import model_config
        weights = args.model_dir.resolve(strict=True)
        # Do not resolve a venv's Python symlink to the base interpreter.
        model, _ = model_config(dict(python=str(args.model_python.absolute()),
            checkpoint=str(weights), stats=str(weights/'stats.json'), stats_key='G1_omnipicker',
            control_schema=str(weights/'control_schema.json'), vlm=str(weights/'vlm')))
    else:
        if args.codex_executable is None:
            raise ValueError('Codex requires --codex-executable')
        from .codex.worker import selected_image
        from .codex.sandbox_build import inspect_image
        policy = selected_image(source, metadata['source_files'])
        if metadata.get('coding_policy_image') != policy or inspect_image(policy, args.docker_host) != policy:
            raise ValueError('Stage the coding Runtime with your local sandbox build first')
        supplied_version = getattr(args, 'codex_client_version', None)
        if supplied_version:
            # Remote-worker preparation: the CLI and credentials stay on the
            # client host. That client still verifies its actual binary before
            # starting an episode; this value is not an authentication check.
            cli = args.codex_executable
            if not cli.is_absolute() or '..' in cli.parts or not supplied_version.startswith('codex-cli '):
                raise ValueError('Require the absolute client CLI path and its actual codex --version output')
            version = supplied_version.strip()
        else:
            cli = args.codex_executable.resolve(strict=True)
            version = subprocess.check_output([str(cli), '--version'], text=True).strip()
        docker = shutil.which('docker')
        if not docker:
            raise ValueError('Docker CLI must be available on the native Linux host')
        from .codex.namespace import docker_socket_group
        docker_socket_group(endpoint(args.docker_host)[len('unix://'):])
    image = image_id(args.image, args.docker_host)
    dest.mkdir()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Existing image-owned cache preparation: CPU-only, no host SDK required.
    prepare = namespace.command(base, image, 'prepare-native', name=name)
    prepare[1:1] = ['--host', endpoint(args.docker_host)]
    subprocess.run(prepare, check=True, stdout=sys.stderr)
    deployment.validate(base, name, image=image, sdk_site=deployment.SDK_ALIAS)
    worker = dict(agent=args.agent, runtime=str(base), image=image, name=name, gpu=args.gpu,
        lock_root=str(locks), docker_host=args.docker_host, selection=str(selected),
        selection_sha256=sha(selected), output=str(output), launch=row['launch'],
        public_input_binding=row.get('public_input_binding'),
        docker_cli=str(Path(docker).resolve()) if args.agent == 'codex' else None)
    argv = [sys.executable, '-B', '-m', MODULE, 'worker', str(dest/'run.json')]
    if args.agent == 'internvla':
        from .internvla.operator import template
        operator = template()
        operator.update(model=model, model_gpu=args.model_gpu, worker_gpu=args.gpu,
            lock_root=str(locks), output_root=str(source), max_calls=args.max_calls,
            tasks={row['task_id']: dict(worker_argv=argv, native_report='{output}/native/native_report.json', public_prompt=None)})
    else:
        from .codex.worker import CODING_PREFIX
        native = dict(schema='robotagentbench.codex_worker.v1', runtime=str(base),
            sdk_site=deployment.SDK_ALIAS, launch=row['launch'], output=str(output/'native'),
            lock_root='/locks', gpu=args.gpu, coding_files=[n for n in metadata['source_files'] if n.startswith(CODING_PREFIX)],
            policy_image=policy, docker_host=args.docker_host, input_roots=[str(source)])
        if worker['public_input_binding']:
            native['public_input_binding'] = worker['public_input_binding']
        write(dest/'native.json', native)
        operator = dict(schema='robotagentbench.codex_example.v1', codex_executable=str(cli),
            expected_cli_version=version, model='gpt-6-astra', effort='medium', max_model_responses=200,
            transport=None, prepare_argv=[], worker_argv=argv, output=str(output))
    write(dest/'operator.json', operator)
    files = ['operator.json'] + (['native.json'] if args.agent == 'codex' else [])
    write(dest/'run.json', dict(schema=SCHEMA, task=row['task_id'], worker=worker,
        files_sha256={n: sha(dest/n) for n in files}))
    return dict(config=str(dest/'run.json'), output=str(output), native_started=False, model_called=False)


def checked(path):
    path = Path(path).resolve(strict=True)
    value = read(path)
    if value.get('schema') != SCHEMA:
        raise ValueError('Require a Docker model configuration')
    expected = {'operator.json'} | ({'native.json'} if value['worker']['agent'] == 'codex' else set())
    if set(value['files_sha256']) != expected:
        raise ValueError('Configuration file set changed')
    for filename, pin in value['files_sha256'].items():
        if sha(path.parent/filename) != pin:
            raise ValueError('Prepared operator configuration changed')
    worker = value['worker']
    if sha(worker['selection']) != worker['selection_sha256']:
        raise ValueError('Saved selection changed')
    _, row, base, source, _ = selection(worker['selection'])
    if (str(base) != worker['runtime'] or row['launch'] != worker['launch']
            or row.get('public_input_binding') != worker['public_input_binding']):
        raise ValueError('Worker differs from its saved selection')
    if not Path(worker['output']).is_relative_to(source/'short_tasks/agent_probe_v1/cap_v1/model_runs'):
        raise ValueError('Native output must remain in its task-owned workspace')
    return path, value


def worker_command(path, value):
    w = value['worker']
    cmd = container_prefix(Path(w['runtime']), w['image'], w['docker_host'],
                           name=w['name'], gpu=w['gpu'], locks=w['lock_root'])
    mount(cmd, path.parent)
    selected_parent = Path(w['selection']).parent
    if not selected_parent.is_relative_to(Path(w['runtime'])) and not selected_parent.is_relative_to(path.parent):
        mount(cmd, selected_parent)
    if w['agent'] == 'codex':
        from .codex.namespace import docker_socket_group
        socket = w['docker_host'][len('unix://'):]
        mount(cmd, w['docker_cli'], '/usr/local/bin/docker')
        mount(cmd, socket, readonly=False)
        group = docker_socket_group(socket)
        if group is not None:
            cmd += ['--group-add', str(group)]
        cmd += ['--env', 'DOCKER_HOST='+w['docker_host']]
    return cmd + [w['image'], PYTHON, '-B', '-u', '-m', MODULE, 'enter', str(path)]


def worker(path):
    path, value = checked(path)
    if Path(value['worker']['output'], 'native').exists():
        raise ValueError('Native run already exists; never redispatch')
    cmd = worker_command(path, value)
    write(path.parent/'dispatch.json', dict(config_sha256=sha(path),
        container_name=cmd[cmd.index('--name')+1], automatic_retry=False))
    return subprocess.run(cmd, check=False).returncode


def enter(path):
    path, value = checked(path)
    w = value['worker']
    verified = deployment.validate(Path(w['runtime']), w['name'], image=w['image'], sdk_site=deployment.SDK_ALIAS)
    baseline = deployment.verify_image_baseline(sdk_site=deployment.SDK_ALIAS, check_visible_sdk=True)
    if baseline != read(verified['root']/'image_baseline.json'):
        raise ValueError('Native image baseline differs from prepared caches')
    if w['agent'] == 'codex':
        from .codex.worker import main
        return main([str(path.parent/'native.json'), '--execute'])
    from .native_worker import execute
    _, source, _ = runtime(w['runtime'])
    return execute(w['runtime'], deployment.SDK_ALIAS, Path(w['launch']),
        Path(w['output'])/'native', Path('/locks'), w['gpu'], input_roots=(source,),
        public_input_binding=w['public_input_binding'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    restore_p = sub.add_parser('restore', help='Restore one saved scene in a CPU-only native container')
    for flag in ('runtime', 'release', 'output'):
        restore_p.add_argument('--'+flag, required=True, type=Path)
    restore_p.add_argument('--task', required=True)
    restore_p.add_argument('--episode', required=True, type=int)
    setup = sub.add_parser('configure', help='Prepare private native caches and both client/worker configurations; no inference')
    setup.add_argument('--agent', choices=('internvla', 'codex'), required=True)
    for flag in ('selection', 'lock-root', 'destination'):
        setup.add_argument('--'+flag, required=True, type=Path)
    setup.add_argument('--name', required=True)
    setup.add_argument('--gpu', required=True, type=int)
    setup.add_argument('--model-gpu', type=int)
    setup.add_argument('--model-python', type=Path)
    setup.add_argument('--model-dir', type=Path)
    setup.add_argument('--codex-executable', type=Path)
    setup.add_argument('--codex-client-version', help='Remote client only: actual codex --version output; keep CLI/login on client host')
    setup.add_argument('--max-calls', type=int, default=100)
    for p in (restore_p, setup):
        p.add_argument('--image', default='robotagentbench:native')
        p.add_argument('--docker-host', default='unix:///var/run/docker.sock')
    for mode in ('run', 'worker', 'enter'):
        p = sub.add_parser(mode, help='Inspect or start the prepared run' if mode == 'run'
                          else 'Internal handoff used by the configured runner')
        p.add_argument('config', type=Path)
        if mode == 'run':
            p.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'restore':
        result = restore(args)
    elif args.command == 'configure':
        result = configure(args)
    elif args.command in ('worker', 'enter'):
        return (worker if args.command == 'worker' else enter)(args.config)
    else:
        path, value = checked(args.config)
        if not args.execute:
            result = dict(agent=value['worker']['agent'], task=value['task'], output=value['worker']['output'],
                          native_started=False, model_called=False)
        elif value['worker']['agent'] == 'codex':
            from .codex.__main__ import main as run_codex
            return run_codex([str(path.parent/'operator.json'), '--execute'])
        else:
            from .internvla.operator import run
            result = run(path.parent/'operator.json', value['task'], Path(value['worker']['output']))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
