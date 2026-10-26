"""One authentic active20 medoid per ShortTask, independent of reset seed.

Task identity defines the cluster. No independently averaged coordinates,
asset-specific posture switching, stochastic selection, or geometry fallback.
"""
import copy
import hashlib
import json
import math
from pathlib import Path

from .postures import (ACTIVE_NAMES, FAMILIES, PERSONAL, MissingSourcePosture,
                       context, read_bank, source_hashes as recorded_hashes)


def task_indices(task_name):
    if task_name in FAMILIES:
        return FAMILIES[task_name]
    if task_name in PERSONAL:
        # Primary short-task records only; atomic pick extensions are not
        # interchangeable evidence for a complete short-task initial posture.
        return tuple(sorted({i for values in PERSONAL[task_name].values()
                             for i in values if i >= 34}))
    raise MissingSourcePosture('No authenticated short-task posture pool: ' + task_name)


def valid_targets(row):
    q = row.get('targets_rad_or_m', {})
    if set(q) != set(ACTIVE_NAMES) or not all(math.isfinite(v) for v in q.values()):
        raise ValueError('Incomplete/nonfinite recorded active20 posture')
    return q


def representative(task_name):
    bank, digest = read_bank()
    indices = task_indices(task_name)
    rows = [r for r in bank['rows'] if r['task_index'] in indices
            and r.get('targets_rad_or_m')]
    if task_name == 'carry_meal_tray':
        rows = [r for r in rows if r.get('variant') == 'empty_tray']
    if not rows:
        raise MissingSourcePosture('No authenticated full posture for ' + task_name)
    rows.sort(key=lambda r: (r['source_id'], r.get('episode_index', -1)))
    # Fit a common dimensionless metric to PRIMARY short-task initial states.
    # Global (not within-task) scales avoid amplifying a fixed joint's tiny
    # measured jitter; linear metres and angular radians are not mixed raw.
    reference = [valid_targets(r) for r in bank['rows']
                 if r['task_index'] >= 34 and r.get('targets_rad_or_m')]
    scales = {}
    for name in ACTIVE_NAMES:
        mean = math.fsum(q[name] for q in reference) / len(reference)
        variance = math.fsum((q[name] - mean)**2 for q in reference) / len(reference)
        scales[name] = math.sqrt(variance)
    values = [valid_targets(r) for r in rows]
    def distance(a, b):
        return math.sqrt(math.fsum(((a[n]-b[n])/scales[n])**2
                        for n in ACTIVE_NAMES if scales[n] > 0))
    totals = [math.fsum(distance(a, b) for b in values) for a in values]
    limit_path = Path(__file__).with_name('source_joint_limits.json')
    limits = json.loads(limit_path.read_text())['limits']
    if set(limits) != set(ACTIVE_NAMES):
        raise ValueError('Incomplete authenticated rig limits')
    eligible = [i for i, q in enumerate(values)
                if all(limits[n][0] <= q[n] <= limits[n][1] for n in ACTIVE_NAMES)]
    if not eligible:
        raise MissingSourcePosture('No unmodified whole source pose within rig limits: '+task_name)
    chosen = min(eligible, key=lambda i: (totals[i], rows[i]['source_id'],
                                                rows[i].get('episode_index', -1)))
    row = rows[chosen]
    return {
        'targets_rad': dict(values[chosen]),
        'posture_source': row['source_id'],
        'dataset_episode_index': row.get('episode_index'),
        'dataset_task_index': row['task_index'],
        'dataset_task_indices': list(indices),
        'source_evidence': row['posture_evidence'],
        'source_scope': 'same_short_task_pooled_parameter_variants',
        'source_bank_sha256': digest,
        'source_provenance_sha256': bank['provenance_sha256'],
        'source_records_used': len(rows),
        'admissible_source_records': len(eligible),
        'limit_rejected_episodes': [rows[i].get('episode_index') for i in range(len(rows))
                                    if i not in eligible],
        'selection_admissibility': 'unchanged_rig_limits_no_clipping',
        'source_selection': 'fixed_per_task_actual_sample_medoid_v3',
        'clustering': 'one_cluster_per_short_task_no_unsupervised_subclusters',
        'distance_metric': 'euclidean_global_primary_short_data_std_normalized_active20',
        'distance_scales_rad_or_m': scales,
        'distance_scale_records': len(reference),
        'medoid_mean_distance': totals[chosen] / len(rows),
        'single_record_only': len(rows) == 1,
        'complete_source_joint_tuple': True,
        'joint_coordinates_independently_sampled': False,
        'joint_coordinates_clipped': False,
        'active20_source_exact': True,
        'fixed_across_seeds_assets_sources': True,
        'native_physics_validated': False,
    }


def candidates(task_name, *, seed, asset=None, source=None, previous_targets=None):
    if type(seed) is not int or seed < 0:
        raise ValueError('Nonnegative integer seed required')
    if task_name in PERSONAL and asset not in PERSONAL[task_name]:
        raise MissingSourcePosture('Unsupported personal object: ' + str(asset))
    # Exactly one candidate: a collision must fail, not silently change posture.
    yield copy.deepcopy(representative(task_name))


def source_hashes():
    refs = recorded_hashes()
    for p in (Path(__file__), Path(__file__).with_name('source_joint_limits.json')):
        refs[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return refs
