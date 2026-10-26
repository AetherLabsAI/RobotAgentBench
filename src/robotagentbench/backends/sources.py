"""Locate and verify embedded task code without importing a native backend."""
import hashlib
import json
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent


def _relative(value):
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or path.as_posix() != value
            or any(p in ('.', '..') for p in path.parts) or '\\' in value):
        raise ValueError('Canonical relative backend source path required')
    return path


def _manifest():
    value = json.loads((HERE / 'manifest.json').read_text(encoding='utf-8'))
    if value.get('schema') != 'robotagentbench.backend_sources.v1':
        raise ValueError('Unsupported embedded backend manifest')
    if set(value['profiles']) != {'room', 'atomic_coffee', 'failure_recovery', 'cloth'}:
        raise ValueError('Complete selected task backend profiles required')
    return value


def root(profile):
    if profile not in _manifest()['profiles']:
        raise ValueError('Unknown embedded backend profile')
    path = HERE / 'profiles' / profile
    if not path.is_dir() or path.resolve() != path:
        raise ValueError('Embedded task source directory is missing or aliased')
    return path


def describe():
    return {name: dict(files=len(row['files']), bytes=sum(f['bytes'] for f in row['files'].values()))
            for name, row in _manifest()['profiles'].items()}


def verify(profile):
    selected = _manifest()['profiles'].get(profile)
    if selected is None:
        raise ValueError('Unknown embedded backend profile')
    base = root(profile)
    for name, item in selected['files'].items():
        path = base / _relative(name)
        if path.resolve() != path or not path.is_file() or path.stat().st_size != item['bytes']:
            raise ValueError('Embedded backend source is missing or aliased')
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != item['sha256']:
            raise ValueError('Embedded backend source differs from its selected version')
    return dict(profile=profile, verified=True, files=len(selected['files']),
                task_code_imported=False, native_started=False)
