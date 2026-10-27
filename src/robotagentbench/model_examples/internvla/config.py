"""Explicit deployment inputs: never infer a G1 codec from a model name."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from robotagentbench.interfaces.action_contract import ORDER
from robotagentbench.interfaces import robot_profiles as rp

TEMPLATE = {
    'schema': 'robotagentbench.internvla.g1.v1',
    'checkpoint': '/absolute/path/to/G1_finetuned/pretrained_model',
    'stats': '/absolute/path/to/matching/stats.json',
    'control_schema': '/absolute/path/to/control_schema.json',
    'vlm': '/absolute/path/to/local/Qwen3.5-2B',
    'stats_key': 'G1_omnipicker',
    'device': 'cuda:0',
    'dtype': 'bfloat16',
    'endpoint': 'http://127.0.0.1:8765/infer',
    'request_timeout_s': 120,
    'artifacts_sha256': {
        'config.json': 'REPLACE_WITH_CHECKPOINT_CONFIG_SHA256',
        'model.safetensors': 'REPLACE_WITH_WEIGHT_FILE_SHA256',
        'stats': 'REPLACE_WITH_MATCHING_STATS_SHA256',
        'control_schema': 'REPLACE_WITH_CONTROL_SCHEMA_SHA256',
    },
    'codec': 'g1_absolute22_mean_std_eps1e-6.v1',
    'context_mode': 'official_language_memory',
    'navigation_mode': 'preserved_not_consumed',
}


def template():
    return deepcopy(TEMPLATE)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def load(path):
    value = read(path)
    if set(value) != set(TEMPLATE) or value['schema'] != TEMPLATE['schema']:
        raise ValueError('Exact example configuration schema required')
    for key in ('codec', 'context_mode', 'navigation_mode'):
        if value[key] != TEMPLATE[key]:
            raise ValueError('Unsupported model codec or input interpretation: '+key)
    if value['dtype'] not in ('float32', 'bfloat16') or not isinstance(value['device'], str):
        raise ValueError('Explicit supported dtype/device required')
    if type(value['request_timeout_s']) not in (float, int) or not 0 < value['request_timeout_s'] <= 600:
        raise ValueError('Finite bounded HTTP timeout required')
    for key in ('checkpoint', 'stats', 'control_schema', 'vlm'):
        if not isinstance(value[key], str) or not Path(value[key]).is_absolute():
            raise ValueError('Explicit local absolute artifact path required: '+key)
    if not Path(value['vlm']).is_dir() or not (Path(value['vlm'])/'config.json').is_file():
        raise ValueError('An existing local VLM directory/config is required; no automatic download')
    if set(value['artifacts_sha256']) != set(TEMPLATE['artifacts_sha256']):
        raise ValueError('Pin config, single-file weights, matching stats and control schema')
    paths = dict(stats=Path(value['stats']), control_schema=Path(value['control_schema']))
    paths.update({name: Path(value['checkpoint'])/name for name in ('config.json', 'model.safetensors')})
    for name, path in paths.items():
        expected = value['artifacts_sha256'][name]
        if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
            raise ValueError('Real lowercase SHA256 required: '+name)
        if sha(path) != expected:
            raise ValueError('Model artifact hash mismatch: '+name)
    schema = read(value['control_schema'])
    validate_control_schema(schema)
    cfg = read(Path(value['checkpoint'])/'config.json')
    if (cfg.get('type') != 'internvla_a1_5' or cfg.get('n_obs_steps') != 1
            or cfg.get('max_state_dim') != 32 or cfg.get('max_action_dim') != 32
            or cfg.get('chunk_size') != 50 or cfg.get('tokenize_state') is not True
            or cfg.get('normalization_mapping') != {'VISUAL':'IDENTITY','STATE':'IDENTITY','ACTION':'IDENTITY'}):
        raise ValueError('This example supports the audited padded32/50-step G1 fine-tuning recipe only')
    stats = read(value['stats'])
    selected = stats[value['stats_key']]
    for name, width in (('observation.state', 23), ('action', 22)):
        row = selected[name]
        for field in ('mean', 'std'):
            if (not isinstance(row[field], list) or len(row[field]) != width
                    or any(type(x) not in (float,int) or not math.isfinite(x) for x in row[field])):
                raise ValueError('Exact finite G1 normalization dimensions required')
        if any(x < 0 for x in row['std']):
            raise ValueError('Negative normalization scale')
    return value


def validate_control_schema(schema):
    if (schema.get('version') != rp.MOBILE22 or schema.get('action_dim') != 22
            or schema.get('state_dim') != 23 or schema.get('control_hz') != 20):
        raise ValueError('A G1 fine-tuning control schema is required; base/RoboTwin is not G1')
    channels = schema.get('action_channels', [])
    units = [rp.NATIVE_UNITS[n] for n in ORDER[:20]] + ['m/s', 'rad/s']
    representations = ['absolute_joint_position_target']*20+['body_forward_velocity','world_z_yaw_rate']
    if (len(channels) != 22 or [r['name'] for r in channels] != ORDER
            or [r['unit'] for r in channels] != units
            or [r['representation'] for r in channels] != representations):
        raise ValueError('No automatic action reorder, delta integration, gripper normalization or unit guessing')
    state = schema.get('state_channels', [])
    names = ORDER[:20]+['base_x_world_m','base_y_world_m','base_yaw_world_unwrapped_rad']
    if (len(state) != 23 or [r['name'] for r in state] != names
            or [r['unit'] for r in state] != units[:20]+['m','m','rad']
            or [r['representation'] for r in state] != ['measured_joint_position']*20+[
                'measured_world_position','measured_world_position','measured_world_unwrapped_yaw']):
        raise ValueError('Exact measured joint20 plus world XY/unwrapped-yaw state required')
