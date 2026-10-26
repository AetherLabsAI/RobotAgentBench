"""Recompute the data producer's appearance plan from the episode seed (CPU only, needs pxr: run through
evaluation/tools/run_cpu_usd.sh). Read-only on the producer workspace: it imports the producer's adapter
bag_task/appearance_randomization.py (a vendored snapshot of this repo's appearance planner) and calls build_plan(seed).

1. Verification: for every 1-1 / 1-2 / 1-3 episode that has a recorded appearance (appearance.json or
   initialization.json["appearance"]), compare palette, lighting scheme and every light value with the recomputed plan.
2. Gap fill: 1-4 episodes have no appearance record on 202; their runner used seed = plan.randomization.seed, so the
   plan is recomputed from that seed and stored as the (derived, not recorded) appearance of the episode.
Writes references/data_side/appearance_plans/{verification.json, 1-4.json}."""
import json, sys
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
REPO = TASK.parents[2]
WB = Path('/home/lingjun/coffee-workspace/coffee_task_workbench_20260922')
sys.path.insert(0, str(WB / 'bag_task'))
import appearance_randomization as producer   # noqa: E402  (read-only import)

OUT = TASK / 'references/data_side/appearance_plans'


def summary(plan):
    k = plan.get('kitchen_materials') or {}
    return dict(palette=plan.get('style'), lighting_scheme=plan.get('lighting_scheme'),
                lights=[[l['prim_path'], round(l['intensity_scale'], 5), round(l['color_temperature_k'], 1)] for l in plan['lights']],
                central_table=(k.get('central_table') or {}).get('option_id'), stove=(k.get('stove') or {}).get('rule_id'),
                environment_style=(plan.get('environment') or {}).get('style'), floor=(plan.get('environment') or {}).get('floor_texture_asset'))


def recorded(ep_dir):
    p = ep_dir / 'appearance.json'
    if p.is_file(): return json.loads(p.read_text())
    p = ep_dir / 'initialization.json'
    if p.is_file(): return json.loads(p.read_text()).get('appearance')
    return None


def main():
    rows = [r for r in json.loads((REPO / 'data_list.json').read_text()) if r['task'] in ('1.1', '1.2', '1.3', '1.4')]
    API = {'1.1': '1-1', '1.2': '1-2', '1.3': '1-3', '1.4': '1-4'}
    ver = dict(checked=0, equal=0, mismatches=[]); derived = {}
    for r in rows:
        d = Path(r['videos']['head']).parent
        d = d.parent if d.name == 'motion' else d
        api = API[r['task']]
        if api == '1-4':
            seed = int(json.loads((d / 'plan.json').read_text())['randomization']['seed'])
            derived[d.name] = dict(seed=seed, kind='derived_from_seed_not_recorded', **summary(producer.build_plan(seed)))
            continue
        rec = recorded(d)
        if not rec: ver['mismatches'].append(dict(api=api, episode=d.name, problem='no recorded appearance')); continue
        plan = producer.build_plan(int(rec['seed']))
        a, b = summary(plan), summary(rec['plan'])
        ver['checked'] += 1
        if a == b: ver['equal'] += 1
        else: ver['mismatches'].append(dict(api=api, episode=d.name, seed=rec['seed'], diff=[k for k in a if a[k] != b[k]]))
    (OUT / 'verification.json').write_text(json.dumps(ver, indent=1, ensure_ascii=False) + '\n')
    (OUT / '1-4.json').write_text(json.dumps(dict(note='1-4 的外观记录不在 202；这里是用数据方同一份外观代码按 plan.randomization.seed 重算的计划，属于推导值而非来源记录。', episodes=derived), indent=1, ensure_ascii=False) + '\n')
    print('verified', ver['equal'], '/', ver['checked'], 'mismatches', len(ver['mismatches']), '| derived 1-4:', len(derived))


if __name__ == '__main__':
    main()
