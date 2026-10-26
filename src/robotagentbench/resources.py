"""Version-pinned asset download and safe offline installation; no task execution."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
from urllib.parse import quote
from urllib.request import urlopen

SCHEMA = 'robotagentbench.asset_release.v1'


def _relative(value):
    if not isinstance(value, str) or not value or '\\' in value or '\x00' in value:
        raise ValueError('Invalid release member path')
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(p in ('.', '..') for p in path.parts):
        raise ValueError('Canonical relative release member path required')
    return value


def _digest(value):
    if not isinstance(value, str) or not re.fullmatch('[a-f0-9]{64}', value):
        raise ValueError('Pinned SHA-256 required')
    return value


def _size(value):
    if type(value) is not int or value < 0:
        raise ValueError('Nonnegative exact byte size required')
    return value


def read_manifest(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('Duplicate release manifest field')
            value[key] = item
        return value
    value = json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError('Explicit versioned asset release manifest required')
    fields={'schema', 'version', 'hosting', 'archives'}
    if value.get('schema') == 'robotagentbench.asset_release.v2':
        fields.add('index')
    elif value.get('schema') != SCHEMA:
        raise ValueError('Explicit versioned asset release manifest required')
    if set(value) != fields:
        raise ValueError('Explicit versioned asset release manifest required')
    if not isinstance(value['version'], str) or not value['version']:
        raise ValueError('Release version required')
    host = value['hosting']
    if host is not None:
        if (not isinstance(host, dict) or set(host) != {'provider', 'repository', 'revision'}
                or host['provider'] != 'huggingface'
                or not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', host['repository'])
                or not re.fullmatch(r'[a-f0-9]{40}', host['revision'])):
            raise ValueError('Hugging Face dataset and immutable commit required')
    archives = value['archives']
    if not isinstance(archives, dict) or not archives:
        raise ValueError('At least one exact release archive required')
    all_members = set()
    for name, item in archives.items():
        _relative(name)
        if (not isinstance(item, dict) or set(item) != {'sha256', 'bytes', 'members'}
                or not isinstance(item['members'], dict) or not item['members']):
            raise ValueError('Archive digest, size and complete member list required')
        _digest(item['sha256']); _size(item['bytes'])
        for member, record in item['members'].items():
            _relative(member)
            if member in all_members or not isinstance(record, dict) or set(record) != {'sha256', 'bytes'}:
                raise ValueError('Duplicate or invalid asset member')
            all_members.add(member)
            _digest(record['sha256']); _size(record['bytes'])
    if 'index' in value:
        _relative(value['index'])
        if value['index'] not in all_members:
            raise ValueError('Selected resource index must be a pinned archive member')
    return value


def _verify_file(path, record):
    if path.resolve() != path or not path.is_file() or path.stat().st_size != record['bytes']:
        raise ValueError('Release file is missing, aliased or has the wrong size')
    with path.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != record['sha256']:
            raise ValueError('Release file checksum mismatch')


def fetch(manifest, cache):
    """Download immutable public HF files. Never accepts or transmits credentials."""
    host = manifest['hosting']
    if host is None:
        raise ValueError('Asset hosting is not published yet; use offline archives')
    cache = Path(cache).absolute()
    if cache.resolve() != cache:
        raise ValueError('Canonical cache directory required')
    cache.mkdir(parents=True, exist_ok=True)
    for name, record in manifest['archives'].items():
        target = cache / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.resolve() != target:
            raise ValueError('Aliased download path')
        if target.exists():
            _verify_file(target, record)
            continue
        partial = target.with_name(target.name + '.partial')
        if partial.exists() or partial.is_symlink():
            raise ValueError('Previous partial download retained; inspect it before retrying')
        if shutil.disk_usage(cache).free < record['bytes'] + 512 * 1024**2:
            raise ValueError('Insufficient download reserve')
        url = ('https://huggingface.co/datasets/' + host['repository'] + '/resolve/'
               + host['revision'] + '/' + quote(name, safe='/'))
        digest, count = hashlib.sha256(), 0
        with urlopen(url, timeout=60) as source, partial.open('xb') as output:
            while chunk := source.read(1024**2):
                count += len(chunk)
                if count > record['bytes']:
                    raise ValueError('Download exceeds pinned size')
                digest.update(chunk); output.write(chunk)
        if count != record['bytes'] or digest.hexdigest() != record['sha256']:
            raise ValueError('Download differs from selected release; partial retained')
        os.rename(partial, target)
    return dict(version=manifest['version'], archives=len(manifest['archives']), verified=True)


def install(manifest, cache, destination):
    """Verify every archive/member before promoting a fresh asset installation."""
    cache, destination = Path(cache).absolute(), Path(destination).absolute()
    if cache.resolve() != cache or destination.parent.resolve() != destination.parent:
        raise ValueError('Canonical archive and destination paths required')
    if destination.exists() or destination.is_symlink():
        raise ValueError('Fresh asset destination required; existing files are never overwritten')
    total = sum(r['bytes'] for a in manifest['archives'].values() for r in a['members'].values())
    if shutil.disk_usage(destination.parent).free < total + 512 * 1024**2:
        raise ValueError('Insufficient extraction reserve')
    for name, record in manifest['archives'].items():
        _verify_file(cache / name, record)
    stage = Path(tempfile.mkdtemp(prefix=destination.name + '.partial-', dir=destination.parent))
    # Keep an interrupted/invalid staging tree for inspection; never delete an
    # unknown directory and never use tarfile.extract/extractall.
    for name, record in manifest['archives'].items():
        expected, seen = record['members'], set()
        with tarfile.open(cache / name, 'r|gz') as archive:
            for member in archive:
                key = _relative(member.name)
                if not member.isfile() or key not in expected or key in seen:
                    raise ValueError('Unexpected, duplicate or non-file archive member')
                seen.add(key); metadata = expected[key]
                if member.size != metadata['bytes']:
                    raise ValueError('Archive member has the wrong size')
                target = stage / key
                target.parent.mkdir(parents=True, exist_ok=True)
                digest, count = hashlib.sha256(), 0
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError('Archive member is unreadable')
                with source, target.open('xb') as output:
                    while chunk := source.read(1024**2):
                        count += len(chunk)
                        if count > metadata['bytes']:
                            raise ValueError('Archive member exceeds pinned size')
                        digest.update(chunk); output.write(chunk)
                if count != metadata['bytes'] or digest.hexdigest() != metadata['sha256']:
                    raise ValueError('Asset member checksum mismatch')
                target.chmod(0o644)
        if seen != set(expected):
            raise ValueError('Archive is missing required asset members')
    receipt = dict(schema='robotagentbench.asset_installation.v1', version=manifest['version'],
                   manifest_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
                   assets=str(destination), archives=len(manifest['archives']), files=sum(len(a['members']) for a in manifest['archives'].values()),
                   verified=True, native_started=False)
    with (stage / 'installation.json').open('x') as stream:
        json.dump(receipt, stream, indent=2); stream.write('\n')
    if destination.exists() or destination.is_symlink():
        raise ValueError('Destination appeared during installation; staging tree retained')
    os.rename(stage, destination)
    return receipt


def verify(manifest, destination):
    destination = Path(destination).absolute()
    if destination.resolve() != destination:
        raise ValueError('Canonical asset installation required')
    count = 0
    for archive in manifest['archives'].values():
        for name, record in archive['members'].items():
            _verify_file(destination / name, record)
            count += 1
    return dict(version=manifest['version'], verified=True, files=count, native_started=False)
