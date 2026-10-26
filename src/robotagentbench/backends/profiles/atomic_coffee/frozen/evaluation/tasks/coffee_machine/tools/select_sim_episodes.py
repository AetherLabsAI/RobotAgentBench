"""Pick 5 source-rendered episodes per API (1-1 … 1-4 = data_list.json subtasks 1.1 … 1.4) for the review pages. CPU only, deterministic, read-only on the data.

The 200 episodes come from <repo>/data_list.json (task 1.1 … 1.4). Per-episode features are read from the batch manifests the
data producer wrote next to the episodes (dataset_manifest.json / manifest.json); appearance palette + lighting scheme are read
from each episode's appearance.json when it exists.

Selection = greedy farthest-point over range-normalised pose features, plus a bonus for every categorical value (arm, cup
style, group, palette, lighting, legacy start) not covered yet. First pick = episode nearest to the per-feature median.
It is a coverage heuristic for choosing what to look at, not a quality filter: producer status labels are copied unchanged.

Writes references/source_sim_videos/selection.json (consumed by tools/build_sim_media.py).
"""
import json, math, re
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
REPO = TASK.parents[2]
DATA = Path('/mnt/agent-bench-data/lingjun/arena-data-review-20260920/datasets')
OUT = TASK / 'references/source_sim_videos/selection.json'
N = 5
CAT_BONUS = 0.35
API = {'1.1': '1-1', '1.2': '1-2', '1.3': '1-3', '1.4': '1-4'}   # data_list.json 子任务编号（= 任务编号）-> API ID（用户 2026-09-29 裁定）


def appearance(ep_dir):
    for p in (ep_dir / 'appearance.json', ep_dir.parent / 'appearance.json'):
        if p.is_file():
            a = json.loads(p.read_text())
            kit = (a.get('kitchen_application') or {})
            return dict(palette=a.get('palette'), lighting=a.get('lighting_scheme'), appearance_seed=a.get('seed'),
                        central_table=(kit.get('central_table') or {}).get('option_id'))
    return {}


def episode_dir(rec):
    d = Path(rec['videos']['head']).parent
    return d.parent if d.name == 'motion' else d


def features():
    """page key -> {episode dir name: dict(feats=[…], feat_names=[…], cats={…}, info={…})}"""
    out = {}
    m = json.loads((DATA / 'coffee_cup_left15_right35_20260925/dataset_manifest.json').read_text())
    out['1.1'] = {s['sample_id']: dict(feat_names=['handle_yaw_deg', 'cup_x', 'cup_y', 'machine_x', 'machine_y'],
                                       feats=[s['angle'], s['cup_xyz'][0], s['cup_xyz'][1], s['machine_xyz'][0], s['machine_xyz'][1]],
                                       cats=dict(arm=s['arm'], cup_style=s['style'], group=s['group']),
                                       info=dict(producer_run=s['name'], producer_checks=s['checks'], manifest_sha256=s.get('files'))) for s in m['samples']}
    m = json.loads((DATA / 'coffee_pour_final50_20260924/manifest.json').read_text())
    out['1.2'] = {s['run']: dict(feat_names=['machine_dx', 'machine_dy', 'machine_yaw_deg', 'bag_dx', 'bag_dy', 'bag_yaw_deg'],
                                 feats=[*s['randomization']['machine_offset_m'][:2], s['randomization']['machine_yaw_deg'], *s['randomization']['bag_offset_m'][:2], s['randomization']['bag_yaw_deg']],
                                 cats=dict(group=s['group'], cup_style=s['randomization'].get('cup_style')),
                                 info=dict(producer_passed=s['passed'], bean_outcome=s.get('bean_outcome'), duration_s=s.get('duration'))) for s in m}
    m = json.loads((DATA / 'coffee_button_v2_20260925/manifest.json').read_text())
    out['1.3'] = {s['run']: dict(feat_names=['machine_dx', 'machine_dy', 'machine_dyaw_deg'],
                                 feats=[s['proposal']['machine_dx_m'], s['proposal']['machine_dy_m'], s['proposal']['machine_dyaw_deg']],
                                 cats={}, info=dict(seed=s['seed'], producer_status=s['status'], source_run=s.get('source_run'))) for s in m['episodes']}
    m = json.loads((DATA / 'coffee_retrieve_mixed_original11_20260926/dataset_manifest.json').read_text())
    legacy = set(m['legacy_initial_runs'])
    out['1.4'] = {s['id']: dict(feat_names=['machine_dx', 'machine_dy', 'machine_yaw_deg', 'handle_yaw_deg'],
                                feats=[*s['machine_offset_m'][:2], s['machine_yaw_deg'], s['initial_handle_world_yaw_deg']],
                                cats=dict(arm=s['arm'], cup_style=s['cup_style'], start='legacy_initial' if s['id'] in legacy else 'full_initial'),
                                info={}) for s in m['samples']}
    return out


def pick(rows):
    names = sorted(rows)
    k = len(rows[names[0]]['feats'])
    lo = [min(rows[n]['feats'][i] for n in names) for i in range(k)]
    hi = [max(rows[n]['feats'][i] for n in names) for i in range(k)]
    norm = {n: [(rows[n]['feats'][i] - lo[i]) / ((hi[i] - lo[i]) or 1.0) for i in range(k)] for n in names}
    med = [sorted(norm[n][i] for n in names)[len(names) // 2] for i in range(k)]
    dist = lambda a, b: math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    chosen = [min(names, key=lambda n: (dist(norm[n], med), n))]
    while len(chosen) < min(N, len(names)):
        covered = {(c, rows[n]['cats'][c]) for n in chosen for c in rows[n]['cats']}
        def score(n):
            return min(dist(norm[n], norm[c]) for c in chosen) + CAT_BONUS * sum(1 for c, v in rows[n]['cats'].items() if (c, v) not in covered)
        chosen.append(max((n for n in names if n not in chosen), key=lambda n: (round(score(n), 9), n)))
    return chosen, dict(zip(rows[names[0]]['feat_names'], zip(lo, hi)))


def main():
    records = [r for r in json.loads((REPO / 'data_list.json').read_text()) if r['task'] in ('1.1', '1.2', '1.3', '1.4')]
    feats = features()
    out = {}
    for sub, key in API.items():
        recs = {episode_dir(r).name: r for r in records if r['task'] == sub}
        rows = {}
        for name, r in recs.items():
            f = feats[sub].get(name)
            assert f, (key, name, 'episode of data_list.json missing from the batch manifest')
            a = appearance(episode_dir(r))
            f = dict(f, cats=dict(f['cats'], **{k: v for k, v in a.items() if k in ('palette', 'lighting') and v}), appearance=a)
            rows[name] = f
        assert len(rows) == 50, (key, len(rows))
        chosen, ranges = pick(rows)
        out[key] = dict(task_no=sub, pool=len(rows), method='确定性选择：位姿特征按范围归一化后做最远点贪心，每覆盖一个新的类别值（手臂、杯款、扰动档、配色、灯光、起点类型）加 %.2f；第一条取最接近中位数的 episode' % CAT_BONUS,
                        feature_ranges_in_batch={k: [round(v[0], 5), round(v[1], 5)] for k, v in ranges.items()}, episodes=[])
        for name in chosen:
            r = recs[name]; f = rows[name]
            out[key]['episodes'].append(dict(
                label=re.sub(r'[^A-Za-z0-9_.-]', '_', name), episode_dir=str(episode_dir(r)), videos=r['videos'],
                data_list=dict(id=r['id'], title=r['title'], collection=r['collection'], source_path=r['source_path'], source_exists_on_202=r['source_exists_on_202'],
                               source_status=r['source_status'], local_artifacts=r.get('local_artifacts')),
                features=dict(zip(f['feat_names'], [round(x, 5) for x in f['feats']])), categories=f['cats'], appearance=f['appearance'], producer=f['info']))
            print(key, name, f['cats'], [round(x, 3) for x in f['feats']])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
