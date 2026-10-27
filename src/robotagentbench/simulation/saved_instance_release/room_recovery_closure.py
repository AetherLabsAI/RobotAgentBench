"""Exact immutable dependencies of a saved room Recovery owner's nested launch.

Only indexed owner JSON is read. Paths are data, never executable instructions.
The original scene verifier and its hashes remain authoritative at restoration.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

from .original_scene_patch import TASKS

LEGACY_GENERIC = '/home/lingjun/robot_agent_arena/short_tasks/agent_probe_v1/deployment_generic_20261010/source'
ROOTS = ('/home/lingjun/robot_agent_arena',
         '/mnt/agent-bench-data/lingjun/releases/living50-candidate-20260919',
         '/workspace/lukun/arena_software/isaac51-py311/lib/python3.11/site-packages/isaacsim')
PIN_FIELDS = ('source_layers', 'code_sources', 'source_hashes')


def _path(value):
    if not isinstance(value, str):
        raise ValueError('Original scene dependency path must be a string')
    path = PurePosixPath(value)
    if (not path.is_absolute() or path.as_posix() != value or '..' in path.parts
            or '\\' in value or '\x00' in value
            or not any(path != PurePosixPath(root) and path.is_relative_to(root) for root in ROOTS)):
        raise ValueError('Original scene dependency leaves its declared source namespaces')
    return value


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate owner JSON key')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite owner JSON')))


def source_requirements(selected):
    """Return every static source pin in the selected owner's indexed JSON.

    Declared generated layers and owner sidecars are restored/rebound by the
    existing saved-instance owner, so they are not aliased to historical output.
    Immutable source/asset pins are never removed, refreshed or guessed.
    """
    state = selected['state']
    if state.get('task_id') not in TASKS:
        return {}
    from robotagentbench.simulation.specialized import camera_release
    attachments = selected['attachments']
    generated = set()
    documents = [state]
    originals = set()
    for row in attachments:
        original = row.get('original_path')
        if original is not None:
            _path(original)
            if original in originals:
                raise ValueError('Duplicate original owner attachment path')
            originals.add(original)
        if row.get('role') in ('owner_json', 'scene_root', 'scene_layer'):
            if original is None:
                raise ValueError('Original owner attachment identity is required')
            generated.add(original)
        if row.get('role') == 'owner_json':
            source = Path(row['source']).absolute()
            if (source.resolve(strict=True) != source or not source.is_file()
                    or _sha(source) != row['sha256']):
                raise ValueError('Indexed original owner JSON changed')
            documents.append(_read(source))
    for row in state.get('generated_scene_patches', []):
        generated.add(_path(row['original_path']))
    found = {}

    def visit(value):
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for field in PIN_FIELDS:
                if field not in value:
                    continue
                pins = value[field]
                if not isinstance(pins, dict):
                    raise ValueError('Original scene ' + field + ' must be a mapping')
                for path, digest in pins.items():
                    _path(path)
                    if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
                        raise ValueError('Original scene source requires an exact SHA-256')
                    previous = found.setdefault(path, digest)
                    if previous != digest:
                        raise ValueError('Conflicting original scene source hashes')
            for child in value.values():
                visit(child)

    for document in documents:
        rebound, _ = camera_release.rebind_saved_source_pins(document, 'failure_recovery')
        visit(rebound)
    return {path: digest for path, digest in found.items() if path not in generated}


def verify_original_launch(initial, *, source_root):
    """Run original initial-source and manifest verifiers in a CPU subprocess.

    The caller already verified this source root via the dedicated source
    contract. No SDK, stage, physics, scoring or model is initialized here.
    """
    original = initial.get('original_launch')
    if not isinstance(original, dict):
        raise ValueError('Saved room recovery requires its embedded original launch')
    initial_sources = {}
    for field in ('source_layers', 'code_sources'):
        if not isinstance(initial.get(field), dict):
            raise ValueError('Saved room recovery requires its original ' + field)
        initial_sources[field] = initial[field]
    paths = {}
    for name in ('manifest', 'request', 'preparation_receipt', 'ready_pose', 'scene', 'sensors', 'spawn'):
        value = original.get(name)
        if value is None:
            if name in ('manifest', 'request'):
                raise ValueError('Embedded original launch is missing ' + name)
            continue
        path = Path(value).absolute()
        if path.resolve(strict=True) != path or not path.is_file():
            raise ValueError('Embedded original launch dependency is unavailable: ' + name)
        expected = original.get(name + '_sha256')
        if expected is not None and _sha(path) != expected:
            raise ValueError('Embedded original launch dependency changed: ' + name)
        paths[name] = str(path)
    source = Path(source_root).absolute()
    original_common = source / 'short_task_robustness/tasks/grasp_failure_recovery/common.py'
    if (source.resolve(strict=True) != source or not (source / 'short_tasks/scene.py').is_file()
            or not original_common.is_file()):
        raise ValueError('Verified original room source root required')
    # Fixed import/program; only JSON and file paths cross the process boundary.
    program = ('import json,sys;from pathlib import Path;'
               'sys.path.insert(0,sys.argv[1]);'
               'from short_task_robustness.tasks.grasp_failure_recovery.common import verify_sources;'
               'initial=json.load(sys.stdin);'
               'verify_sources(initial["source_layers"]);verify_sources(initial["code_sources"]);'
               'from short_tasks.scene import verify_manifest;'
               'verify_manifest(json.loads(Path(sys.argv[2]).read_text()))')
    subprocess.run([sys.executable, '-B', '-c', program, str(source), paths['manifest']],
                   input=json.dumps(initial_sources, allow_nan=False),
                   check=True, capture_output=True, text=True)
    return dict(original_manifest=paths['manifest'], source_hashes_verified=True,
                initial_source_layers_verified=True, initial_code_sources_verified=True,
                initial_source_pins=sum(len(pins) for pins in initial_sources.values()),
                launch_dependencies_verified=len(paths), native_started=False)
