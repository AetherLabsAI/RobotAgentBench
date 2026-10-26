"""Left-hand episode list of 1-1 / 1-4 for the left-hand replay batch (user 2026-10-01: replay only the data producer's
current left-hand episodes of 放杯子 and 取杯子). CPU only, deterministic, read-only on the data.

Every episode whose producer manifest says arm == left is listed, in the producer's manifest order; nothing is
filtered on status (producer labels are copied unchanged). Same record schema as references/source_sim_videos/selection.json
so tools/package_source_episode.py --selection can package them.

Writes references/left_hand_20261001/selection.json.
"""
import json
import re
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
REPO = TASK.parents[2]
DATA = Path('/mnt/agent-bench-data/lingjun/arena-data-review-20260920/datasets')
OUT = TASK / 'references/left_hand_20261001/selection.json'
API = {'1.1': '1-1', '1.4': '1-4'}


def episode_dir(rec):
    d = Path(rec['videos']['head']).parent
    return d.parent if d.name == 'motion' else d


def left_rows():
    out = {}
    m = json.loads((DATA / 'coffee_cup_left15_right35_20260925/dataset_manifest.json').read_text())
    out['1.1'] = [(s['sample_id'], dict(feat_names=['handle_yaw_deg', 'cup_x', 'cup_y', 'machine_x', 'machine_y'],
                                        feats=[s['angle'], s['cup_xyz'][0], s['cup_xyz'][1], s['machine_xyz'][0], s['machine_xyz'][1]],
                                        cats=dict(arm=s['arm'], cup_style=s['style'], group=s['group']),
                                        info=dict(producer_run=s['name'], producer_checks=s['checks'], manifest_sha256=s.get('files'))))
                  for s in m['samples'] if s['arm'] == 'left']
    m = json.loads((DATA / 'coffee_retrieve_mixed_original11_20260926/dataset_manifest.json').read_text())
    legacy = set(m['legacy_initial_runs'])
    out['1.4'] = [(s['id'], dict(feat_names=['machine_dx', 'machine_dy', 'machine_yaw_deg', 'handle_yaw_deg'],
                                 feats=[*s['machine_offset_m'][:2], s['machine_yaw_deg'], s['initial_handle_world_yaw_deg']],
                                 cats=dict(arm=s['arm'], cup_style=s['cup_style'], group=s['group'], start='legacy_initial' if s['id'] in legacy else 'full_initial'),
                                 info=dict(initial_state_status=s.get('initial_state_status'), group=s['group'])))
                  for s in m['samples'] if s['arm'] == 'left']
    return out


def main():
    records = [r for r in json.loads((REPO / 'data_list.json').read_text()) if r['task'] in API]
    rows = left_rows()
    out = {}
    for sub, key in API.items():
        recs = {episode_dir(r).name: r for r in records if r['task'] == sub}
        out[key] = dict(task_no=sub, pool=len(recs), method='数据方清单里 arm == left 的全部 episode，按清单顺序，不筛选', episodes=[])
        for name, f in rows[sub]:
            r = recs[name]
            out[key]['episodes'].append(dict(
                label=re.sub(r'[^A-Za-z0-9_.-]', '_', name), episode_dir=str(episode_dir(r)), videos=r['videos'],
                data_list=dict(id=r['id'], title=r['title'], collection=r['collection'], source_path=r['source_path'], source_exists_on_202=r['source_exists_on_202'],
                               source_status=r['source_status'], local_artifacts=r.get('local_artifacts')),
                features=dict(zip(f['feat_names'], [round(x, 5) for x in f['feats']])), categories=f['cats'], appearance={}, producer=f['info']))
        print(key, len(out[key]['episodes']), 'left-hand episodes')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
