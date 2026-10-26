#!/usr/bin/env python3
"""Shared helpers for the coffee-task coverage survey (read-only on the datasets)."""
import hashlib
import json
import math
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

SCRATCH = '/tmp/claude-1006/-home-lingjun-robot-agent-arena-api-protocol/0e5432c5-b105-4446-a185-fa804be0f5c3/scratchpad'
OUT_DIR = os.path.join(SCRATCH, 'coverage_work')
DATA_LIST = '/home/lingjun/robot_agent_arena/api-protocol/data_list.json'
ROOT = '/mnt/agent-bench-data/lingjun/arena-data-review-20260920/datasets'
NOMINAL_MACHINE = (2.569096825, 0.166750471, 0.92)
# nominal bag position observed in 1.1 traces with bag_offset_m == 0 (see summary)
FFPROBE_CACHE = os.path.join(OUT_DIR, 'ffprobe_cache.json')

os.makedirs(OUT_DIR, exist_ok=True)


def load(path):
    with open(path) as f:
        return json.load(f)


def load_opt(path):
    if not os.path.exists(path):
        return None
    try:
        return load(path)
    except Exception as e:  # noqa
        return {'__error__': str(e)}


def records(task):
    return [r for r in load(DATA_LIST) if r.get('task') == task]


def episode_dir(rec, parent=False):
    d = os.path.dirname(rec['videos']['head'])
    return os.path.dirname(d) if parent else d


def norm_deg(a):
    """wrap to (-180, 180]"""
    a = (a + 180.0) % 360.0 - 180.0
    if a == -180.0:
        a = 180.0
    return a


def axes_from_matrix(m):
    """Return (x_axis, y_axis, z_axis, translation|None) as world vectors of the local axes.
    Handles 3x3 / 4x4, and both 'columns are axes' and 'rows are axes' storage by locating the
    up axis (local Y == world +Z for these Y-up assets)."""
    n = len(m)
    r = [[float(m[i][j]) for j in range(3)] for i in range(3)]
    col1 = [r[0][1], r[1][1], r[2][1]]
    row1 = r[1]
    col_score = abs(col1[2])
    row_score = abs(row1[2])
    trans = None
    if col_score >= row_score:
        conv = 'columns'
        ax = [[r[0][j], r[1][j], r[2][j]] for j in range(3)]
        if n == 4:
            trans = [float(m[0][3]), float(m[1][3]), float(m[2][3])]
    else:
        conv = 'rows'
        ax = [r[0], r[1], r[2]]
        if n == 4:
            trans = [float(m[3][0]), float(m[3][1]), float(m[3][2])]
    if n == 4 and trans is not None and all(abs(v) < 1e-12 for v in trans):
        # translation stored the other way round
        alt = [float(m[3][0]), float(m[3][1]), float(m[3][2])] if conv == 'columns' else [float(m[0][3]), float(m[1][3]), float(m[2][3])]
        if any(abs(v) > 1e-12 for v in alt):
            trans = alt
    return ax[0], ax[1], ax[2], trans, conv


def heading_deg(vec):
    return math.degrees(math.atan2(vec[1], vec[0]))


def machine_yaw_from_matrix(m):
    """Machine yaw relative to nominal (front = local +Z = world -X at yaw 0)."""
    _, _, z, trans, _ = axes_from_matrix(m)
    return norm_deg(heading_deg(z) - 180.0), trans


def local_z_heading(m):
    _, _, z, trans, _ = axes_from_matrix(m)
    return heading_deg(z), trans


def local_x_heading(m):
    x, _, _, trans, _ = axes_from_matrix(m)
    return heading_deg(x), trans


def tilt_deg(m):
    """angle between local up (local Y) and world Z"""
    _, y, _, _, _ = axes_from_matrix(m)
    nz = max(-1.0, min(1.0, y[2] / (math.sqrt(sum(v * v for v in y)) or 1.0)))
    return math.degrees(math.acos(nz))


def dist(a, b, n=3):
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(n)))


def sha_obj(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest()[:16]


def phases_of(steps, key='phase'):
    out = []
    for s in steps:
        p = s.get(key)
        if not out or out[-1][0] != p:
            out.append([p, 0])
        out[-1][1] += 1
    return out


# ---------------------------------------------------------------- ffprobe
_cache = None


def _load_cache():
    global _cache
    if _cache is None:
        _cache = load(FFPROBE_CACHE) if os.path.exists(FFPROBE_CACHE) else {}
    return _cache


def _probe(path):
    if not os.path.exists(path):
        return {'exists': False}
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
           'stream=codec_name,pix_fmt,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration',
           '-show_entries', 'format=duration,size', '-of', 'json', path]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        j = json.loads(p.stdout or '{}')
        st = (j.get('streams') or [{}])[0]
        fm = j.get('format') or {}
        num, den = (st.get('r_frame_rate') or '0/1').split('/')
        fps = float(num) / float(den) if float(den) else None
        return {
            'exists': True,
            'codec': st.get('codec_name'), 'pix_fmt': st.get('pix_fmt'),
            'width': st.get('width'), 'height': st.get('height'),
            'r_frame_rate': st.get('r_frame_rate'), 'avg_frame_rate': st.get('avg_frame_rate'),
            'fps': fps,
            'nb_frames': int(st['nb_frames']) if st.get('nb_frames') not in (None, 'N/A') else None,
            'duration_s': float(st['duration']) if st.get('duration') not in (None, 'N/A') else (float(fm['duration']) if fm.get('duration') else None),
            'bytes': int(fm['size']) if fm.get('size') else os.path.getsize(path),
            'stderr': p.stderr.strip()[:200] or None,
        }
    except Exception as e:  # noqa
        return {'exists': True, 'error': str(e)}


def ffprobe_many(paths):
    cache = _load_cache()
    todo = [p for p in paths if p not in cache or cache[p].get('error')]
    if todo:
        with ThreadPoolExecutor(max_workers=16) as ex:
            for p, r in zip(todo, ex.map(_probe, todo)):
                cache[p] = r
        with open(FFPROBE_CACHE, 'w') as f:
            json.dump(cache, f)
    return {p: cache[p] for p in paths}


# ---------------------------------------------------------------- appearance flattening
SKIP_APPEARANCE_KEYS = (
    'authored_property_paths', 'target_prim_paths', 'excluded_semantic_prims',
    'excluded_materials_untouched', 'randomization_registry', 'contract_source_revision',
)


def flatten(o, prefix='', out=None):
    """Flatten a JSON object to dotted keys. Lists of dicts carrying prim_path / furniture_id are keyed by that id."""
    if out is None:
        out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            if k in SKIP_APPEARANCE_KEYS:
                continue
            flatten(v, f'{prefix}.{k}' if prefix else k, out)
    elif isinstance(o, list):
        if o and all(isinstance(x, dict) for x in o):
            for i, x in enumerate(o):
                ident = x.get('furniture_id') or x.get('prim_path') or str(i)
                ident = str(ident).split('/')[-1]
                flatten({k: v for k, v in x.items() if k not in ('prim_path',)}, f'{prefix}[{ident}]', out)
        else:
            out[prefix] = json.dumps(o, ensure_ascii=False)
    else:
        out[prefix] = o
    return out


def appearance_compact(a):
    """Small per-episode appearance record (the semantically meaningful choices)."""
    if not a:
        return None
    plan = a.get('plan') or {}
    env = plan.get('environment') or {}
    km = plan.get('kitchen_materials') or {}
    lg = a.get('lighting_groups') or {}

    def grp(name):
        g = lg.get(name) or []
        return {
            'n': len(g),
            'intensity_scale': sorted(set(x.get('intensity_scale') for x in g)),
            'color_temperature_k': sorted(set(x.get('color_temperature_k') for x in g)),
        }
    dec = env.get('decor_selection') or {}
    lamp = dec.get('lamp_replacement') or {}
    return {
        'policy': a.get('policy'), 'version': a.get('version'), 'seed': a.get('seed'),
        'gen_seed': plan.get('gen_seed'),
        'palette': a.get('palette'), 'lighting_scheme': a.get('lighting_scheme'),
        'ceiling': grp('ceiling'), 'daylight': grp('daylight'),
        'materials': {m.get('furniture_id'): m.get('look') for m in (a.get('materials') or [])},
        'environment_style': env.get('style'), 'floor_texture_asset': env.get('floor_texture_asset'),
        'artwork': dec.get('artwork'), 'plants': dec.get('plants'),
        'lamp_mode': lamp.get('mode'), 'lamp': lamp.get('lamp'), 'lamp_slots': lamp.get('slots'),
        'central_table_option': (km.get('central_table') or {}).get('option_id'),
        'stove_rule': (km.get('stove') or {}).get('rule_id'),
        'stove_countertop': (km.get('stove') or {}).get('countertop_material_id'),
        'stove_lower': (km.get('stove') or {}).get('lower_material_id'),
        'stove_upper': (km.get('stove') or {}).get('upper_material_id'),
        'harmony_status': (plan.get('harmony') or {}).get('status'),
        'scope': a.get('scope'),
        'source_snapshot_sha256': a.get('source_snapshot_sha256'),
    }


PATHLIKE = re.compile(r'(scene_path|sidecar_path|episode_token|seed_u64|gen_seed|(^|\.)seed$)')


def varying_keys(flat_list):
    """flat_list: list of flattened dicts. Returns (varying, constant) summaries."""
    keys = []
    seen = set()
    for f in flat_list:
        for k in f:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    varying, constant = {}, {}
    for k in keys:
        vals = [f.get(k, '__absent__') for f in flat_list]
        distinct = []
        for v in vals:
            if v not in distinct:
                distinct.append(v)
        if len(distinct) == 1:
            constant[k] = distinct[0]
            continue
        nums = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
        rec = {'n_distinct': len(distinct), 'n_present': sum(1 for v in vals if v != '__absent__')}
        if nums and len(nums) == len([v for v in vals if v != '__absent__']):
            rec.update({'min': min(nums), 'max': max(nums), 'mean': sum(nums) / len(nums)})
            if len(distinct) <= 8:
                rec['values'] = sorted(distinct)
        else:
            cnt = {}
            for v in vals:
                cnt[str(v)] = cnt.get(str(v), 0) + 1
            if len(cnt) <= 20:
                rec['counts'] = dict(sorted(cnt.items(), key=lambda kv: -kv[1]))
            else:
                rec['example'] = str(distinct[0])[:120]
        rec['seed_or_path_like'] = bool(PATHLIKE.search(k))
        varying[k] = rec
    return varying, constant


def num_stats(vals):
    v = [float(x) for x in vals if x is not None]
    if not v:
        return None
    return {'n': len(v), 'min': min(v), 'max': max(v), 'mean': sum(v) / len(v)}


def capture_events_info(path):
    if not os.path.exists(path):
        return None
    n = 0
    first = last = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            e = json.loads(line)
            if first is None:
                first = e
            last = e

    def t(e):
        return e.get('simulation_time_s', e.get('simulation_time'))

    def st(e):
        return e.get('control_step', e.get('step'))
    return {'n_events': n, 'first_sim_time_s': t(first), 'last_sim_time_s': t(last),
            'first_step': st(first), 'last_step': st(last), 'first_phase': first.get('phase'), 'last_phase': last.get('phase')}


def video_block(paths):
    pr = ffprobe_many(list(paths.values()))
    return {k: pr[p] for k, p in paths.items()}


def save_rows(task, rows, extra=None):
    with open(os.path.join(OUT_DIR, f'rows_{task}.json'), 'w') as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    if extra is not None:
        with open(os.path.join(OUT_DIR, f'extra_{task}.json'), 'w') as f:
            json.dump(extra, f, ensure_ascii=False, indent=1)
