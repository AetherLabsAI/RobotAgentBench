"""Restore indexed original source bytes without USD, SDK, or scene sampling.

Recipes contain only bounded byte copies and literals. They do not replace an
original source pin: the reconstructed file must match that pin exactly before
the dedicated namespace can bind it read-only.
"""
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import re

RECIPE_SCHEMA = 'robotagentbench.exact_byte_reconstruction.v1'
PATCH_SCHEMA = 'robotagentbench.exact_byte_patch.v1'
MAX_SOURCE_BYTES = 64 * 1024**2
MAX_PATCH_BYTES = 4 * 1024**2
MAX_OPERATIONS = 10000
CHUNK_BYTES = 1024**2


def _digest(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('Exact source SHA256 required')
    return value


def _size(value, limit):
    if type(value) is not int or not 0 < value <= limit:
        raise ValueError('Exact source size exceeds the bounded format')
    return value


def _canonical(path):
    path = Path(path).absolute()
    if path.resolve(strict=True) != path:
        raise ValueError('Canonical exact-source path required')
    return path


def _sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _object(assets, objects, digest, limit):
    record = objects.get(_digest(digest))
    expected = 'objects/' + digest[:2] + '/' + digest
    if (not isinstance(record, dict) or set(record) != {'path', 'sha256', 'bytes'}
            or record['path'] != expected or record['sha256'] != digest):
        raise ValueError('Exact source recipe requires an installed content object')
    size = _size(record['bytes'], limit)
    path = _canonical(assets / expected)
    if not path.is_file() or path.stat().st_size != size or _sha(path) != digest:
        raise ValueError('Exact source object bytes differ')
    return path, size


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate exact-source patch key')
        result[key] = value
    return result


def reconstruct(index, assets, output, digest):
    """Return one fresh exact file, or None if no indexed recipe exists.

    The caller supplies a verified release index and its fresh owned plan
    directory. Missing, malformed or changed indexed dependencies fail closed.
    Existing target files are never reused or overwritten.
    """
    digest = _digest(digest)
    recipes = index.get('source_reconstruction', {})
    if not isinstance(recipes, dict):
        raise ValueError('Invalid exact-source reconstruction index')
    if digest not in recipes:
        return None
    row = recipes[digest]
    if (not isinstance(row, dict)
            or set(row) != {'schema', 'base_sha256', 'patch_sha256', 'target_bytes'}
            or row['schema'] != RECIPE_SCHEMA):
        raise ValueError('Unknown exact-source reconstruction recipe')
    objects = index.get('objects')
    if not isinstance(objects, dict) or digest in objects:
        raise ValueError('Reconstructed source must not masquerade as an installed object')
    assets, output = _canonical(assets), _canonical(output)
    if not assets.is_dir() or not output.is_dir() or output.is_relative_to(assets):
        raise ValueError('Exact source output must be an independent owned directory')
    target_size = _size(row['target_bytes'], MAX_SOURCE_BYTES)
    base, base_size = _object(assets, objects, row['base_sha256'], MAX_SOURCE_BYTES)
    patch, _ = _object(assets, objects, row['patch_sha256'], MAX_PATCH_BYTES)
    with gzip.open(patch, 'rb') as stream:
        raw = stream.read(MAX_PATCH_BYTES + 1)
    if len(raw) > MAX_PATCH_BYTES:
        raise ValueError('Expanded exact-source patch exceeds its bound')
    value = json.loads(raw, object_pairs_hook=_unique)
    if (not isinstance(value, dict)
            or set(value) != {'schema', 'base_sha256', 'target_sha256', 'target_bytes', 'operations'}
            or value['schema'] != PATCH_SCHEMA or value['base_sha256'] != row['base_sha256']
            or value['target_sha256'] != digest or type(value['target_bytes']) is not int
            or value['target_bytes'] != target_size):
        raise ValueError('Exact-source patch identity differs from its indexed recipe')
    operations = value['operations']
    if not isinstance(operations, list) or not 0 < len(operations) <= MAX_OPERATIONS:
        raise ValueError('Exact-source operation count exceeds its bound')
    # Validate the entire bounded recipe before creating output. Never evaluate
    # source text or interpret a path from the patch payload.
    validated, total = [], 0
    for operation in operations:
        if not isinstance(operation, list) or not operation:
            raise ValueError('Invalid exact-source operation')
        if operation[0] == 'copy' and len(operation) == 3:
            _, offset, length = operation
            if (type(offset) is not int or offset < 0 or type(length) is not int
                    or length <= 0 or offset + length > base_size):
                raise ValueError('Exact-source copy range is outside the original base')
            validated.append(('copy', offset, length))
        elif operation[0] == 'literal' and len(operation) == 2 and isinstance(operation[1], str):
            literal = base64.b64decode(operation[1], validate=True)
            length = len(literal)
            if not length:
                raise ValueError('Empty exact-source literal')
            validated.append(('literal', literal))
        else:
            raise ValueError('Unknown exact-source operation')
        total += length
        if total > target_size:
            raise ValueError('Exact-source operations exceed the declared output size')
    if total != target_size:
        raise ValueError('Exact-source operations do not cover the declared output size')
    directory = output / 'exact_source_inputs'
    directory.mkdir(exist_ok=True)
    _canonical(directory)
    target, staged = directory / digest, directory / (digest + '.partial')
    if target.exists() or target.is_symlink():
        raise ValueError('Exact-source destination must be fresh')
    checksum = hashlib.sha256()
    with base.open('rb') as source, staged.open('xb') as destination:
        for operation in validated:
            if operation[0] == 'literal':
                chunks = (operation[1],)
                for block in chunks:
                    destination.write(block); checksum.update(block)
            else:
                _, offset, remaining = operation
                source.seek(offset)
                while remaining:
                    block = source.read(min(CHUNK_BYTES, remaining))
                    if not block:
                        raise ValueError('Original exact-source base ended unexpectedly')
                    destination.write(block); checksum.update(block)
                    remaining -= len(block)
        destination.flush()
        os.fsync(destination.fileno())
    if staged.stat().st_size != target_size or checksum.hexdigest() != digest:
        raise ValueError('Reconstructed source differs from the original pinned SHA256')
    # Atomic, no-overwrite publication; a failed partial file is never bound.
    staged.chmod(0o444)
    os.link(staged, target)
    staged.unlink()
    return target
