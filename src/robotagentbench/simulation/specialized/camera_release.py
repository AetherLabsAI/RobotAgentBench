"""Fixed reviewed source deltas, derived from two retained owner assemblies.

This is not a hash refresh facility. Only the reviewed file/path/old-hash/new-
hash pairs below may change. The original assembly remains a required resource.
Camera dimensions and Recovery furniture identity bindings are explicit fixed
release changes. No poses, lens parameters, scenes, scoring or physical settings
are derived. The original assemblies and every unchanged source remain required.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

POLICY_SHA = '16ac99961a66644ce145b935a01d4341143cbd89de7cbf25b32f89e9204be8ad'


def policy():
    data = Path(__file__).with_suffix('.json').read_bytes()
    if hashlib.sha256(data).hexdigest() != POLICY_SHA:
        raise ValueError('Reviewed camera-only release policy changed')
    return json.loads(data)


def profile(family):
    return policy()['profiles'][family]


def provenance(family):
    approved = policy()
    selected = approved['profiles'][family]
    result = dict(id=approved['id'], base_assembly_sha256=selected['base_assembly_sha256'],
                resolutions=approved['resolutions'], changed_sources=sorted(selected['files']),
                native_validation_claimed=False, prior_acceptance_scope='upstream_only')
    if 'source_binding_release' in selected:
        result['source_binding_release'] = deepcopy(selected['source_binding_release'])
    return result


def derive_assembly(path, family):
    """Return exact derived bytes; never mutate the original downloaded asset."""
    raw = Path(path).read_bytes()
    selected = profile(family)
    if hashlib.sha256(raw).hexdigest() != selected['base_assembly_sha256']:
        raise ValueError('Camera delta requires the exact retained original assembly')
    manifest = json.loads(raw)
    changes = selected['files']
    seen = set()
    for row in manifest['profiles'][family]['files']:
        change = changes.get(row['original'])
        if change is None:
            continue
        if (row['original'] in seen or row['sha256'] != change['before']['sha256']
                or row['bytes'] != change['before']['bytes']):
            raise ValueError('Camera delta source identity is not the reviewed original')
        row.update(change['after'])
        seen.add(row['original'])
    if seen != set(changes) or 'camera_release' in manifest:
        raise ValueError('Incomplete or already modified original camera source assembly')
    manifest['camera_release'] = provenance(family)
    result = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    if hashlib.sha256(result).hexdigest() != selected['derived_assembly_sha256']:
        raise ValueError('Camera-only derived assembly differs from its fixed release identity')
    return result


def rebind_saved_source_pins(value, family):
    """Update only explicit saved code pins, after verifying the new operator.

    The immutable saved bank is not rewritten. Each fresh input still passes
    the native owner's unchanged source verification and initial-state gates.
    Unknown pins are never refreshed from the filesystem.
    """
    changes = profile(family)['files']
    record_changes = {row['before']['sha256']: row['after']['sha256']
                      for path, row in changes.items() if path.endswith('component_sources.json')}
    evidence = []

    def walk(item):
        if isinstance(item, list):
            return [walk(child) for child in item]
        if not isinstance(item, dict):
            return item
        result = {key: walk(child) for key, child in item.items()}
        code = item.get('code_sources')
        if isinstance(code, dict):
            for path, old in code.items():
                row = changes.get(path)
                if row is None:
                    continue
                before, after = row['before']['sha256'], row['after']['sha256']
                if old not in (before, after):
                    raise ValueError('Saved camera source pin is not an approved predecessor')
                result['code_sources'][path] = after
                if old != after:
                    evidence.append(dict(field='code_sources', path=path, before=old, after=after))
        if 'source_record_sha256' in item:
            old = item['source_record_sha256']
            if old in record_changes:
                result['source_record_sha256'] = record_changes[old]
                evidence.append(dict(field='source_record_sha256', before=old, after=record_changes[old]))
            elif old not in record_changes.values():
                raise ValueError('Saved component record is not an approved camera release predecessor')
        return result

    return walk(deepcopy(value)), evidence
