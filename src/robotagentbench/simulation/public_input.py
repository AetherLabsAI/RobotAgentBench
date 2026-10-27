"""Trusted saved public text binding, never an action/RPC or evaluator input.

The operator mounts the binding and public JSON below an explicit read-only
workspace input root. Both files and the exact unchanged launch are pinned.
No prompt is inferred from target bindings, scene manifests or score plans.
"""
import hashlib
import json
from pathlib import Path

SCHEMA = 'robotagentbench.public_input_binding.v1'


def _sha(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in '0123456789abcdef' for c in value)):
        raise ValueError('Explicit lowercase SHA256 required for public input')
    return value


def _json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate public input JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique)


def from_selection(selection):
    """Build operator metadata from unified select; do not write or resample.

    Save this returned object as a new JSON file inside the execution
    namespace's read-only input root, then pass that file to the worker.
    """
    if selection.get('schema') != 'robotagentbench.saved_instances.selection.v1':
        raise ValueError('A verified saved-instance selection is required')
    public = selection.get('model_input')
    binding = selection.get('model_input_binding', {})
    if (not isinstance(public, dict) or set(public) != {'instruction', 'context'}
            or not isinstance(public['instruction'], str) or not public['instruction'].strip()
            or not isinstance(public['context'], str) or binding.get('status') != 'saved_public_input'):
        raise ValueError('No separately saved public instruction/context is available')
    expected = _sha(selection['operator']['launch_sha256'])
    if binding.get('launch_sha256') != expected:
        raise ValueError('Saved public selection belongs to a different launch')
    source = Path(binding['source'])
    if not source.is_absolute() or '..' in source.parts or str(source) != binding['source']:
        raise ValueError('Explicit immutable public input path required')
    return dict(schema=SCHEMA, source=str(source), sha256=_sha(binding['sha256']), launch_sha256=expected)


def load(binding_path, launch_path, *, workspace):
    """Return only instruction/context and the immutable input pin additions."""
    binding_path = workspace.input(binding_path)
    launch_path = workspace.input(launch_path)
    raw_binding = binding_path.read_bytes()
    binding = _json(raw_binding)
    if (not isinstance(binding, dict)
            or set(binding) != {'schema', 'source', 'sha256', 'launch_sha256'}
            or binding['schema'] != SCHEMA):
        raise ValueError('Exact trusted public input binding required')
    if hashlib.sha256(launch_path.read_bytes()).hexdigest() != _sha(binding['launch_sha256']):
        raise ValueError('Public input belongs to a different saved launch')
    source = workspace.input(binding['source'])
    raw_public = source.read_bytes()
    if hashlib.sha256(raw_public).hexdigest() != _sha(binding['sha256']):
        raise ValueError('Saved public input changed')
    public = _json(raw_public)
    if not isinstance(public, dict) or set(public) not in (
            {'instruction', 'context'}, {'prompt', 'context'}):
        raise ValueError('Only explicit public instruction/context are model-visible')
    instruction = public.get('instruction', public.get('prompt'))
    if not isinstance(instruction, str) or not instruction.strip() or not isinstance(public['context'], str):
        raise ValueError('Saved public instruction and context must be text')
    pins = ((binding_path, hashlib.sha256(raw_binding).hexdigest()), (source, binding['sha256']),
            (launch_path, binding['launch_sha256']))
    return {'instruction': instruction, 'context': public['context']}, pins
