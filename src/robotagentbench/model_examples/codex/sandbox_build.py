"""Build inputs and explicit local-image validation, never implicit Docker pulls."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent / 'sandbox'
LOCK_SHA256 = 'c74fdf44032127126f4467fad74db8e29eacb086415858243652e062fde5a822'
LABELS = {'org.robotagentbench.policy-sandbox': '1',
          'org.robotagentbench.policy-lock': LOCK_SHA256}


def image_id(value):
    if not isinstance(value, str) or re.fullmatch(r'sha256:[0-9a-f]{64}', value) is None:
        raise ValueError('A resolved immutable Docker image ID is required')
    return value


def inspect_image(reference, docker_host):
    if (not isinstance(reference, str) or not reference or reference.startswith('-')
            or any(c.isspace() or ord(c) < 32 for c in reference)):
        raise ValueError('Explicit local Docker image reference required')
    if (not isinstance(docker_host, str) or not docker_host.startswith('unix:///')
            or '\0' in docker_host):
        raise ValueError('Explicit local unix:/// Docker endpoint required')
    result = subprocess.run(['docker', '--host', docker_host, 'image', 'inspect', reference],
                            check=True, capture_output=True, text=True, timeout=30)
    rows = json.loads(result.stdout)
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError('Expected one local image')
    row = rows[0]
    identity = image_id(row.get('Id'))
    if reference.startswith('sha256:') and reference != identity:
        raise ValueError('Resolved image identity differs')
    if row.get('Os') != 'linux' or row.get('Architecture') != 'amd64':
        raise ValueError('The robot sandbox requires Linux amd64')
    config = row.get('Config') or {}
    if config.get('Volumes') or config.get('OnBuild'):
        raise ValueError('Sandbox images must not declare volumes or build hooks')
    labels = config.get('Labels') or {}
    if any(labels.get(key) != value for key, value in LABELS.items()):
        raise ValueError('Build the sandbox with the published Dockerfile and dependency lock')
    return identity


def export(destination):
    if hashlib.sha256((ROOT / 'requirements.lock').read_bytes()).hexdigest() != LOCK_SHA256:
        raise ValueError('Published dependency lock differs')
    target = Path(destination).resolve()
    target.mkdir(parents=True, exist_ok=False)
    for name in ('Dockerfile', 'requirements.lock'):
        (target / name).write_bytes((ROOT / name).read_bytes())
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', type=Path, help='Export build inputs to a new directory')
    parser.add_argument('--image', help='Existing user-built local image tag or ID')
    parser.add_argument('--docker-host', help='Same Docker socket used by the worker')
    args = parser.parse_args(argv)
    if bool(args.image) != bool(args.docker_host):
        parser.error('--image and --docker-host must be supplied together')
    result = {'container_started': False, 'model_called': False}
    if args.export:
        result['build_context'] = str(export(args.export))
    if args.image:
        result['image_id'] = inspect_image(args.image, args.docker_host)
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
