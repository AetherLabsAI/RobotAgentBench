"""Restore original generated USD values from pinned shared layers.

This is a file reconstruction format, not a scene generator. It never samples,
simulates, modifies a material, or interprets task scoring. Large unchanged
prototype attributes are copied verbatim from a pinned original shared layer.
"""
from pathlib import Path, PurePosixPath
import hashlib
import json
import re
from . import external_attribute_patch as external_v2

SCHEMA = 'robotagentbench.original_sdf_patch.v1'
TASKS = frozenset(('failure_recovery_sofa',
                  'failure_recovery_table_waste_empty_grasp',
                  'failure_recovery_table_waste_tipped_cup'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked_file(path, digest):
    path = Path(path).absolute()
    if (not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest)
            or path.resolve(strict=True) != path or not path.is_file()
            or sha(path) != digest):
        raise ValueError('Pinned original scene member differs')
    return path


def validate_recipe(recipe):
    if isinstance(recipe, dict) and recipe.get('schema') == external_v2.SCHEMA:
        external_v2.validate(recipe)
        legacy = {key: value for key, value in recipe.items() if key in external_v2.LEGACY_FIELDS}
        legacy['schema'] = SCHEMA
        validate_recipe(legacy)
        return
    if not isinstance(recipe, dict) or recipe.get('schema') != SCHEMA:
        raise ValueError('An original Sdf patch recipe is required')
    for key in ('base_sha256', 'delta_sha256', 'original_scene_sha256'):
        if not isinstance(recipe.get(key), str) or not re.fullmatch('[0-9a-f]{64}', recipe[key]):
            raise ValueError('Original scene patch requires exact SHA-256 pins')
    for key in ('removed_specs', 'changed_specs', 'root_order', 'root_fields'):
        values = recipe.get(key)
        if not isinstance(values, list) or any(not isinstance(v, str) for v in values) or len(values) != len(set(values)):
            raise ValueError('Original scene patch has invalid or duplicate entries')
    for key in ('borrowed_specs', 'child_orders', 'property_orders'):
        if not isinstance(recipe.get(key), dict):
            raise ValueError('Original scene patch has an invalid mapping')
    if set(recipe['borrowed_specs']) - set(recipe['changed_specs']):
        raise ValueError('Borrowed specs must be explicitly changed specs')


def restore_layer(recipe, *, base, delta, output, runtime=None, metadata=None):
    """Reconstruct one pinned original layer without opening a USD Stage."""
    validate_recipe(recipe)
    base = checked_file(base, recipe['base_sha256'])
    delta = checked_file(delta, recipe['delta_sha256'])
    output = Path(output).absolute()
    if output.exists() or output.parent.resolve(strict=True) != output.parent or output.suffix != '.usdc':
        raise ValueError('A fresh canonical task-owned USD output is required')
    from pxr import Sdf
    original = Sdf.Layer.FindOrOpen(str(base))
    patch = Sdf.Layer.FindOrOpen(str(delta))
    if original is None or patch is None:
        raise ValueError('Original scene patch layer cannot be opened')
    external = (external_v2.sources(recipe, runtime=runtime, metadata=metadata, patch=patch)
                if recipe['schema'] == external_v2.SCHEMA else {})
    work = Sdf.Layer.CreateAnonymous('.usda')
    work.TransferContent(original)

    def remove(path):
        if work.GetObjectAtPath(path) is None:
            return
        edit = Sdf.BatchNamespaceEdit()
        edit.Add(Sdf.Path(path), Sdf.Path.emptyPath)
        if not work.Apply(edit):
            raise ValueError('Original scene patch cannot remove its specified spec')

    for path in sorted(recipe['removed_specs'], key=lambda p: (len(Sdf.Path(p).GetPrefixes()), p), reverse=True):
        remove(path)
    for path in sorted(recipe['changed_specs'], key=lambda p: (len(Sdf.Path(p).GetPrefixes()), p)):
        borrowed = path in recipe['borrowed_specs'] or path in external
        if path in external:
            source_layer, source_path = external[path]
        else:
            source_path = recipe['borrowed_specs'].get(path, path)
            source_layer = original if path in recipe['borrowed_specs'] else patch
        source = source_layer.GetObjectAtPath(source_path)
        if source is None or path == '/':
            raise ValueError('Original scene patch names an unavailable spec')
        values = {key: source.GetInfo(key) for key in source.ListInfoKeys()}
        if isinstance(source, Sdf.PrimSpec):
            if borrowed:
                raise ValueError('Only exact geometric properties may be borrowed')
            target = Sdf.CreatePrimInLayer(work, path)
            for key in set(target.ListInfoKeys()) - set(values):
                target.ClearInfo(key)
            for key, value in values.items():
                target.SetInfo(key, value)
        else:
            if borrowed and not isinstance(source, Sdf.AttributeSpec):
                raise ValueError('Only exact attribute specs may be borrowed')
            remove(path)
            if not Sdf.CopySpec(source_layer, source_path, work, path):
                raise ValueError('Original scene spec copy failed')
            # CopySpec can remap namespace-valued fields. Restore the exact
            # source values whose equality was proved by the release exporter.
            if borrowed:
                target = work.GetObjectAtPath(path)
                for key, value in values.items():
                    target.SetInfo(key, value)

    # Retain original root/child/property order as well as every authored field.
    pending = output.with_name(output.stem+'.pending.usdc')
    if pending.exists():
        raise ValueError('A previous partial scene output cannot be reused')
    result = Sdf.Layer.CreateNew(str(pending))

    def ordered_copy(path):
        prior = work.GetPrimAtPath(path)
        if prior is None:
            raise ValueError('Original scene order names a missing prim')
        fresh = Sdf.CreatePrimInLayer(result, path)
        for key in prior.ListInfoKeys():
            fresh.SetInfo(key, prior.GetInfo(key))
        properties = recipe['property_orders'].get(path, [str(p.path) for p in prior.properties.values()])
        children = recipe['child_orders'].get(path, [str(p.path) for p in prior.nameChildren.values()])
        if (len(properties) != len(set(properties)) or set(properties) != {str(p.path) for p in prior.properties.values()}
                or len(children) != len(set(children)) or set(children) != {str(p.path) for p in prior.nameChildren.values()}):
            raise ValueError('Original scene order changes the spec inventory')
        for prop in properties:
            if not Sdf.CopySpec(work, prop, result, prop):
                raise ValueError('Original property copy failed')
        for child in children:
            ordered_copy(child)

    if set(recipe['root_order']) != {str(p.path) for p in work.rootPrims}:
        raise ValueError('Original scene root order changes the prim inventory')
    for path in recipe['root_order']:
        ordered_copy(path)
    if set(recipe['root_fields']) != set(patch.pseudoRoot.ListInfoKeys()):
        raise ValueError('Original scene root metadata differs from its patch')
    # USD 0.24 cannot convert the generic subLayerOffsets vector to Python.
    # Copy its public typed sequence after creating the matching path entries.
    if 'subLayers' in recipe['root_fields']:
        result.subLayerPaths = list(patch.subLayerPaths)
    if 'subLayerOffsets' in recipe['root_fields']:
        if len(result.subLayerPaths) != len(patch.subLayerOffsets):
            raise ValueError('Original sublayer offset/path cardinality differs')
        for index, offset in enumerate(patch.subLayerOffsets):
            result.subLayerOffsets[index] = Sdf.LayerOffset(offset.offset, offset.scale)
    for key in recipe['root_fields']:
        if key not in ('subLayers', 'subLayerOffsets'):
            result.pseudoRoot.SetInfo(key, patch.pseudoRoot.GetInfo(key))
    result.Save()
    if output.exists():
        raise ValueError('Scene output appeared during reconstruction')
    pending.rename(output)
    return output


def _saved_scene_source_matches(logical, digest, source_layers, attachments):
    """Accept direct sources or one exact saved manifest/localization anchor."""
    if logical in source_layers:
        return source_layers[logical] == digest
    matches = []
    for anchor, anchor_digest in source_layers.items():
        parent = PurePosixPath(anchor).parent
        records = []
        for name in ('manifest.json', 'localization.json'):
            indexed = [a for a in attachments if a.get('role') == 'owner_json'
                       and a.get('original_path') == str(parent / name)]
            if len(indexed) != 1:
                records = []
                break
            record = indexed[0]
            path = checked_file(record['source'], record['sha256'])
            if path.stat().st_size != record['bytes']:
                raise ValueError('Saved scene source record size differs')
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                raise ValueError('Saved scene source record is not an object')
            records.append(value)
        if not records:
            continue
        manifest, localization = records
        source = manifest.get('source_hashes', {})
        localized = localization.get('original_layers_sha256', {})
        if not isinstance(source, dict) or not isinstance(localized, dict):
            raise ValueError('Saved scene source maps are not objects')
        if logical not in source and logical not in localized:
            continue
        if (manifest.get('schema') != 'openend.generated_room.preview.v1'
                or manifest.get('scene') != anchor
                or manifest.get('scene_sha256') != anchor_digest
                or source.get(logical) != digest
                or localized.get(logical) != digest
                or localization.get('physics_parameters_changed') is not False
                or localization.get('material_parameters_changed') is not False):
            raise ValueError('Saved nested scene source anchor differs')
        matches.append(anchor)
    if len(matches) > 1:
        raise ValueError('Saved nested scene source anchor is ambiguous')
    return len(matches) == 1


def restore_generated_layers(state, *, attachments, runtime, metadata, work):
    """Resolve shared layers only through installed, hash-pinned Runtime resources."""
    recipes = state.get('generated_scene_patches', [])
    if not isinstance(recipes, list):
        raise ValueError('Generated scene patches must be an indexed list')
    if not recipes:
        return {}, {}, []
    if state.get('source_profile') != 'failure_recovery' or state.get('task_id') not in TASKS:
        raise ValueError('Generated scene patches are supported only for the original room-recovery tasks')
    runtime = Path(runtime).resolve(strict=True)
    source_layers = state.get('saved_initialization', {}).get('initial', {}).get('source_layers', {})
    paths, hashes, files = {}, {}, []
    for index, recipe in enumerate(recipes):
        validate_recipe(recipe)
        logical = recipe.get('original_path')
        if logical in paths or not _saved_scene_source_matches(logical, recipe['original_scene_sha256'], source_layers, attachments):
            raise ValueError('Generated scene patch does not match the saved initial source layer')
        resource = recipe.get('base_resource')
        if not isinstance(resource, str) or PurePosixPath(resource).is_absolute() or '..' in PurePosixPath(resource).parts:
            raise ValueError('A canonical installed shared-scene resource is required')
        base = metadata.get('resources', {}).get(resource)
        if base is None or metadata.get('library_files', {}).get(resource) != recipe['base_sha256']:
            raise ValueError('The shared original scene is not installed with its pinned digest')
        base = checked_file(base, recipe['base_sha256'])
        if not base.is_relative_to(runtime):
            raise ValueError('Shared original scene leaves the installed Runtime')
        deltas = [a for a in attachments if a.get('role') == 'generated_scene_patch' and a.get('sha256') == recipe['delta_sha256']]
        if len(deltas) != 1:
            raise ValueError('One exact indexed original Sdf patch attachment is required')
        binding = dict(runtime=runtime, metadata=metadata) if recipe['schema'] == external_v2.SCHEMA else {}
        output = restore_layer(recipe, base=base, delta=deltas[0]['source'], output=Path(work)/('generated_scene_'+str(index)+'.usdc'), **binding)
        paths[logical], hashes[logical] = str(output), sha(output)
        files.append(output)
    return paths, hashes, files
