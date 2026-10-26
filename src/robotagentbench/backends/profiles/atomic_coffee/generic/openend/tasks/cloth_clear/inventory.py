"""Scene-generator hooks; do not spawn or move anything from a scorer."""
import random
from .scoring import Binding


def select_assets(seed, pool):
    """Uniform count 1..3, distinct library candidates, reproducible scene choice."""
    if type(seed) is not int or len(pool) != 20 or len(set(pool)) != 20:
        raise ValueError('Seed and complete twenty-object library required')
    if any(not p.startswith('cloth_clear_') for p in pool):
        raise ValueError('Explicit new-asset library required')
    rng = random.Random(seed)
    return rng.sample(sorted(pool), rng.randint(1, 3))


def bindings(episode_id, manifest, pool):
    """The unified scene explicitly declares targets and support roles."""
    group = manifest.get('cloth_clear')
    if not isinstance(group, dict) or group.get('version') != 1:
        raise ValueError('Explicit scene-owned cloth_clear inventory required')
    targets = group.get('target_paths')
    if not isinstance(targets, list) or not 1 <= len(targets) <= 4 or len(set(targets)) != len(targets):
        raise ValueError('One to four distinct target paths required')
    objects = manifest.get('objects', [])
    paths = [o['path'] for o in objects]
    if len(set(paths)) != len(paths):
        raise ValueError('Ambiguous scene inventory')
    by_path = {o['path']: o for o in objects}
    result = []
    for path in targets:
        obj = by_path.get(path)
        if obj is None or obj['id'] not in pool:
            raise ValueError('Unbound or wrong target asset')
        result.append(Binding(episode_id, path, path, group['cloth_path'], group['table_path']))
    return result
