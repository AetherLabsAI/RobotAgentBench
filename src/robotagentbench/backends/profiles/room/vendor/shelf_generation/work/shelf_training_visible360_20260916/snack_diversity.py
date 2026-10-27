"""Both-layer variety policy shared by sampling, selection and export checks."""
import collections
import json
from pathlib import Path

W = Path(__file__).resolve().parent
SNACKS = ('chomps_r', 'pring_bbq', 'oreo_gf', 'honeymaid', 'zbar_t', 'skit_littles')
DRINKS = ('coke_c3', 'cayman_jacked', 'good2grow', 'gatorade_t', 'gatorade_m', 'grapefruit_can', 'starbucks', 'sprite_c1')
POLICY = 'both_layers_diversity_first_v3_scene_first'
POSITION_POLICY = 'full_shelf_xy_v1'


def choose_without_early_repeats(rng, pool, count):
    """Shuffle each full round independently before selecting another copy."""
    pool = list(pool)
    assert pool and len(pool) == len(set(pool)) and count >= 0
    chosen = []
    while len(chosen) < count:
        chosen.extend(rng.permutation(pool).tolist()[:count - len(chosen)])
    return chosen


def inspect_layer(spec, support_id, kinds):
    counts = collections.Counter(o['product_family'] for o in spec['objects'] if o['support_id'] == support_id)
    assert counts and set(counts) <= set(kinds), (support_id, counts, kinds)
    count = sum(counts.values())
    possible = min(count, len(kinds))
    scene_first=spec.get('scene_sampling_policy') in ['scene_first_visible360_v1','targeted_quota_v1','target_first_random_neighborhood_v1']
    target = spec['product']['name'] if not scene_first and spec['product']['name'] in kinds else None
    eligible = [name for name in kinds if name != target]
    copies = [counts[name] for name in eligible]
    coverage = len(counts) / possible
    return dict(policy=POLICY, objects=count, unique_kinds=len(counts), possible_kinds=possible,
        counts=dict(counts), maximum_copies=max(counts.values()), coverage=coverage,
        strict_passed=len(counts) == possible and max(copies) - min(copies) <= 1
            and (target is None or counts[target] == 1),
        excessive_repetition=(max(counts.values()) >= 4
            or (max(counts.values()) >= 3 and max(counts.values()) / count >= .5)
            or coverage <= .5))


def inspect_snacks(spec):
    return inspect_layer(spec, 'tier2', SNACKS)


def inspect_drinks(spec):
    return inspect_layer(spec, 'tier3', DRINKS)


def allowed(spec):
    from legacy_import_policy import imported, counts_allowed
    if imported(spec):
        return (spec.get('target_visibility', {}).get('passed')
            and counts_allowed(spec)
            and not inspect_snacks(spec)['excessive_repetition']
            and not inspect_drinks(spec)['excessive_repetition'])
    return (spec.get('scene_sampling_policy') in ['scene_first_visible360_v1','targeted_quota_v1','target_first_random_neighborhood_v1']
        and spec.get('target_visibility',{}).get('passed')
        and spec.get('cpu_grasp_precheck',{}).get('passed')
        and inspect_snacks(spec)['strict_passed'] and inspect_drinks(spec)['strict_passed']
        and all(4<=sum(o['support_id']==sid for o in spec['objects'])<=10 for sid in ['tier3','tier2']))
