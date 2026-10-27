"""Exercise the real isolated Python-tool transport with synthetic robot state.

No Codex call, simulator, GPU, physical action or evaluation score is involved.
Creates a fresh owned workspace and removes only its own test container.
"""
import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from uuid import uuid4

from .sandbox_build import ROOT, inspect_image
from .server_profile import PREFIX, deployment_payloads


def _inspect_confirms_removed(probe, container):
    """Accept only an exact missing-ID response from this container inspect.

    Docker CLI versions vary the capitalization/prefix of the same error.
    Other daemon/permission/transport errors do not establish absence.
    """
    if (not isinstance(container,str) or re.fullmatch(r'[0-9a-f]{64}',container) is None
            or type(probe.returncode) is not int or probe.returncode == 0
            or not isinstance(probe.stdout,str) or not isinstance(probe.stderr,str)):
        return False
    if probe.stdout.strip():
        try:
            if json.loads(probe.stdout) != []:
                return False
        except ValueError:
            return False
    message=re.fullmatch(r'(?i:(?:error(?::| response from daemon:)\s*)?'
                         r'no such (?:object|container):)\s*(\S+)',probe.stderr.strip())
    return message is not None and message.group(1)==container


def check(image, docker_host, output):
    identity = inspect_image(image, docker_host)
    output = Path(output)
    if (not output.is_absolute() or output.resolve() != output
            or output.exists() or not output.parent.is_dir()):
        raise ValueError('Choose a fresh canonical output under an existing directory')
    if os.getuid() == 0:
        raise ValueError('Use the non-root Docker operator')
    output.mkdir(mode=0o700)
    native = output / 'cap_v1' / 'native_tools'
    native.mkdir(parents=True)
    workspace = output / 'cap_v1' / 'agent_workspace'
    workspace.mkdir()
    source = Path(__file__).resolve().parents[2] / 'backends' / 'profiles' / 'room' / PREFIX
    for name in ('sandbox.py', 'sandbox_boot.py', 'batch_storage.py'):
        (native / name).write_bytes((source / name).read_bytes())
    for relative, raw in deployment_payloads(identity).items():
        name = relative.removeprefix(PREFIX)
        destination = workspace / name.removeprefix('starter/') if name.startswith('starter/') else native / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    (native / '__init__.py').write_text('')
    contract = json.loads((workspace / 'robot' / 'robot_contract.json').read_text())
    joints = {row['name']: 0.0 for row in contract['state_joints']}
    initial = {'control_hz': 20, 'held_targets': {key: joints[key] for key in contract['command_order']},
               'observation': {'episode_id': 'synthetic-sandbox-smoke', 'step': 0, 'joints': joints}}
    name = '_rab_sandbox_check_' + uuid4().hex
    spec = importlib.util.spec_from_file_location(name, native / '__init__.py', submodule_search_locations=[str(native)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    old_endpoint = os.environ.get('DOCKER_HOST')
    os.environ['DOCKER_HOST'] = docker_host
    sandbox = None
    container = None
    result = None
    try:
        cls = importlib.import_module(name + '.sandbox').Sandbox
        sandbox = cls(initial, workspace)
        container = sandbox.cid
        expected = dict(line.split('==') for line in (ROOT / 'requirements.lock').read_text().splitlines())
        code = ('import importlib.metadata as md\n'
                'expected=' + repr(expected) + '\n'
                'assert all(md.version(k)==v for k,v in expected.items()), "Dependency lock differs"\n'
                'assert set(controllers)=={"left","right"}\n'
                'print("RAB_SANDBOX_OK")\n')
        sandbox.send({'op': 'execute_python', 'code': code})
        reply = sandbox.receive(30)
        if reply.get('op') != 'result' or reply.get('result', {}).get('stdout') != 'RAB_SANDBOX_OK\n':
            raise RuntimeError('Isolated code/dependency handshake failed: ' + repr(reply.get('result')))
        sandbox.send({'op': 'observe'})
        observed = sandbox.receive(30)
        if observed.get('op') != 'result' or any(observed.get('tcp', {}).get(side) is None for side in ('left', 'right')):
            raise RuntimeError('Robot FK observation handshake failed')
        result = {'image_id': identity, 'ready': True, 'execute_python': True,
                  'dependency_lock_verified': True, 'robot_fk': True,
                  'model_called': False, 'simulation_started': False, 'physical_actions': 0}
    finally:
        try:
            if sandbox is not None:
                sandbox.close()
        finally:
            if old_endpoint is None:
                os.environ.pop('DOCKER_HOST', None)
            else:
                os.environ['DOCKER_HOST'] = old_endpoint
            for key in list(sys.modules):
                if key == name or key.startswith(name + '.'):
                    del sys.modules[key]
    if container:
        probe = subprocess.run(['docker', '--host', docker_host, 'container', 'inspect', container],
                               capture_output=True, text=True, timeout=30)
        if not _inspect_confirms_removed(probe,container):
            raise RuntimeError('Owned sandbox cleanup not confirmed')
    result['owned_container_removed'] = True
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    parser.add_argument('--docker-host', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(check(args.image, args.docker_host, args.output), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
