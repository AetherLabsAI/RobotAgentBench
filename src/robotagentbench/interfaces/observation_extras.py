"""Allowlisted public text and explored-map packets; no evaluator metadata."""
import base64
from copy import deepcopy
import math
import struct

MAP_SCHEMA = 'openend.native_map_only.v1'
MAP_SOURCE = 'static_scene_geometry_revealed_by_measured_camera_frustum_not_RGB_SLAM'
MAP_FIELDS = frozenset(('schema', 'step', 'map_png', 'origin_m', 'resolution_m',
    'shape', 'explored_fraction', 'map_source', 'premapped', 'task_labels',
    'body_footprint_radius_m'))


def public_context(value):
    if not isinstance(value, str):
        raise ValueError('Public task context must be text, never an evaluator dictionary')
    return value


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def public_navigation(value, step):
    if not isinstance(value, dict) or set(value) != MAP_FIELDS:
        raise ValueError('Require the original read-only explored-map packet')
    if (value['schema'] != MAP_SCHEMA or type(value['step']) is not int
            or value['step'] != step or value['premapped'] is not False
            or value['task_labels'] is not False or value['map_source'] != MAP_SOURCE):
        raise ValueError('Stale or privileged map packet')
    if (not isinstance(value['origin_m'], (list, tuple)) or len(value['origin_m']) != 2
            or not all(_number(v) for v in value['origin_m'])
            or not isinstance(value['shape'], (list, tuple)) or len(value['shape']) != 2
            or not all(type(v) is int and v > 0 for v in value['shape'])
            or not _number(value['resolution_m']) or value['resolution_m'] <= 0
            or not _number(value['body_footprint_radius_m']) or value['body_footprint_radius_m'] <= 0
            or not _number(value['explored_fraction']) or not 0 <= value['explored_fraction'] <= 1):
        raise ValueError('Invalid explored-map geometry metadata')
    png = value['map_png']
    if not isinstance(png, str):
        raise ValueError('Require an encoded map PNG')
    raw = base64.b64decode(png, validate=True)
    if (len(raw) < 33 or len(raw) > 2*1024*1024 or raw[:8] != b'\x89PNG\r\n\x1a\n'
            or raw[12:16] != b'IHDR'):
        raise ValueError('Invalid map PNG')
    # Original renderer uses three pixels per occupancy cell.
    if struct.unpack('>II', raw[16:24]) != (value['shape'][1]*3, value['shape'][0]*3):
        raise ValueError('Map PNG dimensions differ from its metric grid')
    return deepcopy(value)


def public_extras(row, *, include_navigation=True):
    result = {}
    if 'context' in row:
        result['context'] = public_context(row['context'])
    if include_navigation and 'navigation' in row:
        result['navigation'] = public_navigation(row['navigation'], row['step'])
    return result
