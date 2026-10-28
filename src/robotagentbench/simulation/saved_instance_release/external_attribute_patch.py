"""Restore exact indexed attributes for original Sdf patch version 2.

Attributes are copied from hash-pinned installed resources without simulation or
changes to scoring or physical values. Version 1 does not use these helpers.
"""
from pathlib import Path, PurePosixPath
import hashlib
import json
import re

SCHEMA = 'robotagentbench.original_sdf_patch.v2'
LEGACY_FIELDS = frozenset(('schema', 'original_path', 'original_scene_sha256',
    'base_resource', 'base_sha256', 'delta_sha256', 'removed_specs',
    'changed_specs', 'borrowed_specs', 'root_order', 'root_fields',
    'child_orders', 'property_orders'))
EXTERNAL_FIELDS = frozenset(('target_path', 'resource', 'resource_sha256',
    'source_path', 'field_names', 'fields_sha256'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    if not isinstance(value, str) or not re.fullmatch('[a-f0-9]{64}', value):
        raise ValueError('An exact external attribute digest is required')
    return value


def relative(value):
    if (not isinstance(value, str) or not value or '\\' in value
            or '\x00' in value or PurePosixPath(value).is_absolute()
            or PurePosixPath(value).as_posix() != value
            or any(p in ('.', '..') for p in value.split('/'))):
        raise ValueError('Canonical indexed external resource required')
    return value


def checked(path, expected):
    path = Path(path).absolute()
    digest(expected)
    if path.resolve(strict=True) != path or not path.is_file() or sha(path) != expected:
        raise ValueError('External attribute resource pin differs')
    return path


def validate(recipe):
    if (not isinstance(recipe, dict) or recipe.get('schema') != SCHEMA
            or set(recipe) != LEGACY_FIELDS | {'asset_index', 'external_attribute_specs'}):
        raise ValueError('Unknown version-2 original patch field')
    index = recipe['asset_index']
    if (not isinstance(index, dict) or set(index) != {'path', 'sha256'}
            or index['path'] != 'resource-index-v17.json'):
        raise ValueError('Explicit Assets017 index required')
    digest(index['sha256'])
    rows = recipe['external_attribute_specs']
    if not isinstance(rows, list) or not rows:
        raise ValueError('An explicit external attribute list is required')
    targets = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != EXTERNAL_FIELDS:
            raise ValueError('Unknown external attribute field or override')
        target = row['target_path']
        if not isinstance(target, str) or target in targets:
            raise ValueError('Duplicate or invalid external attribute target')
        targets.add(target)
        if target not in recipe['changed_specs'] or target in recipe['borrowed_specs']:
            raise ValueError('External attribute target is not uniquely declared changed')
        if not relative(row['resource']).startswith('libraries/'):
            raise ValueError('Only indexed Runtime library resources may be borrowed')
        digest(row['resource_sha256']); digest(row['fields_sha256'])
        names = row['field_names']
        if (not isinstance(names, list) or not names
                or any(not isinstance(n, str) or not n for n in names)
                or names != sorted(set(names))):
            raise ValueError('Exact sorted authored field names required')


def property_path(value):
    from pxr import Sdf
    if not isinstance(value, str) or any(c in value for c in '{}[]<>'):
        raise ValueError('A canonical absolute attribute path is required')
    path = Sdf.Path(value)
    if not path.IsAbsolutePath() or not path.IsPropertyPath() or str(path) != value:
        raise ValueError('A canonical absolute attribute path is required')
    return path


def attribute_identity(spec):
    """Hash typed Sdf serialization of every authored field, without rounding.

    A fixed temporary namespace removes only the attribute's original name.
    CopySpec-remapped namespace-valued fields are restored before serialization.
    Full all-field/tree-order comparison against the original scene remains the
    exporter proof, independently of this per-attribute corruption guard.
    """
    from pxr import Sdf
    if not isinstance(spec, Sdf.AttributeSpec):
        raise ValueError('Only AttributeSpec may be borrowed')
    names = sorted(spec.ListInfoKeys())
    canonical = Sdf.Layer.CreateAnonymous('.usda')
    Sdf.CreatePrimInLayer(canonical, '/ExactAttribute')
    if not Sdf.CopySpec(spec.layer, spec.path, canonical, '/ExactAttribute.value'):
        raise ValueError('Exact attribute identity cannot be serialized')
    copied = canonical.GetObjectAtPath('/ExactAttribute.value')
    for key in names:
        copied.SetInfo(key, spec.GetInfo(key))
    return names, hashlib.sha256(canonical.ExportToString().encode('utf-8')).hexdigest()


def indexed_resource(resource, expected, *, runtime, metadata, index):
    """Require both the installed content-addressed index and Runtime binding."""
    relative(resource); digest(expected)
    runtime = Path(runtime).resolve(strict=True)
    if (index.get('schema') != 'robotagentbench.asset_index.v1'
            or index.get('version') != '0.1.7' or metadata.get('version') != '0.1.7'
            or index.get('locations', {}).get(resource) != expected
            or metadata.get('library_files', {}).get(resource) != expected):
        raise ValueError('External resource is not identically indexed and installed')
    record = index.get('objects', {}).get(expected)
    if (not isinstance(record, dict) or set(record) != {'bytes', 'sha256', 'path'}
            or record['sha256'] != expected or record['path'] != 'objects/'+expected[:2]+'/'+expected
            or type(record['bytes']) is not int or record['bytes'] < 0):
        raise ValueError('External resource content address differs')
    actual = metadata.get('resources', {}).get(resource)
    if actual is None:
        raise ValueError('External resource is absent from Runtime resources')
    actual = checked(actual, expected)
    if not actual.is_relative_to(runtime) or actual.stat().st_size != record['bytes']:
        raise ValueError('External resource leaves Runtime or has wrong size')
    assets = Path(metadata['assets']).resolve(strict=True)
    installed = checked(assets/record['path'], expected)
    if not installed.is_relative_to(assets) or installed.stat().st_size != record['bytes']:
        raise ValueError('External asset object differs')
    return actual


def sources(recipe, *, runtime, metadata, patch):
    from pxr import Sdf
    validate(recipe)
    if runtime is None or not isinstance(metadata, dict):
        raise ValueError('Indexed Runtime binding is required for version 2')
    assets = Path(metadata['assets']).resolve(strict=True)
    index_path = checked(assets/recipe['asset_index']['path'], recipe['asset_index']['sha256'])
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('Duplicate asset index field')
            value[key] = item
        return value
    index = json.loads(index_path.read_text(), object_pairs_hook=unique)
    result, layers = {}, {}
    for row in recipe['external_attribute_specs']:
        target = property_path(row['target_path']); origin = property_path(row['source_path'])
        if patch.GetObjectAtPath(target) is not None:
            raise ValueError('External attribute is overridden by delta')
        key = row['resource'], row['resource_sha256']
        if key not in layers:
            path = indexed_resource(*key, runtime=runtime, metadata=metadata, index=index)
            layers[key] = Sdf.Layer.FindOrOpen(str(path))
            if layers[key] is None:
                raise ValueError('External attribute source layer cannot open')
        layer = layers[key]; spec = layer.GetObjectAtPath(origin)
        if not isinstance(spec, Sdf.AttributeSpec):
            raise ValueError('External source is missing or is not AttributeSpec')
        names, signature = attribute_identity(spec)
        if names != row['field_names'] or signature != row['fields_sha256']:
            raise ValueError('External attribute authored fields differ')
        result[str(target)] = (layer, str(origin))
    return result
