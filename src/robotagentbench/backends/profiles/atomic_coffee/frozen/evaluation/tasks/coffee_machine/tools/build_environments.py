"""Pre-build every seeded appearance environment a batch needs, in the live Git tree.

The public appearance planner records Git provenance of its contract files, so environments are built here (where
.git exists) and travel with the frozen snapshot under runs/coffee_machine_environment/seed-<N>/; a worker only
verifies and uses them (api_runtime.initialize). One environment per appearance seed: the seeds of the packaged
source episodes of every API (api_runtime.appearance_seed) plus the random-layout seeds given on the command line.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
REPO = TASK.parents[2]
sys.path.insert(0, str(REPO))


def needed_seeds(random_seeds):
    from evaluation.tasks.coffee_machine.api_runtime import appearance_seed
    from evaluation.tasks.coffee_machine.specs import APIS, episodes
    seeds = {}
    for api in APIS:
        for episode in episodes(api):
            record = appearance_seed(api, episode, 'source', None)
            seeds.setdefault(record['seed'], []).append(f'{api}:{episode}')
    for seed in random_seeds:
        seeds.setdefault(int(seed), []).append('random layout seed')
    return seeds


def main():
    from evaluation.tasks.coffee_machine.api_runtime import ENVIRONMENTS, environment_dir
    from evaluation.tasks.coffee_machine.tools.build_environment import build
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--random-seeds', type=int, nargs='*', default=[7])
    p.add_argument('--rebuild', action='store_true', help='discard environments already built')
    a = p.parse_args()
    if not (REPO / '.git').exists():
        raise SystemExit('Environments must be built in the live Git tree (appearance provenance needs .git)')
    built = {}
    for seed, users in sorted(needed_seeds(a.random_seeds).items()):
        target = environment_dir(seed)
        if a.rebuild and target.exists():
            shutil.rmtree(target)
        if (target / 'scene_room.json').exists():
            report = json.loads((target / 'scene_room.json').read_text())
            if report.get('gen_seed') != 1_000_000 + seed:
                raise RuntimeError('Existing environment has another seed: ' + str(target))
            built[seed] = dict(reused=True, users=users, plan_sha256=report['plan_sha256'])
            continue
        report = build(target / 'scene_room.usda', seed)
        built[seed] = dict(reused=False, users=users, plan_sha256=report['plan_sha256'], style=report['style'], lighting_scheme=report['lighting_scheme'])
        print('built', seed, report['style'], report['lighting_scheme'], users, flush=True)
    index = dict(schema='arena.coffee_machine.environments.v1', root=str(ENVIRONMENTS), environments={str(k): v for k, v in built.items()})
    (ENVIRONMENTS / 'index.json').write_text(json.dumps(index, indent=1) + '\n')
    print(json.dumps(dict(environments=len(built), reused=sum(v['reused'] for v in built.values()))))


if __name__ == '__main__':
    main()
