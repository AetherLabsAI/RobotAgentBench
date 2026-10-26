"""Per-replay summary of a coffee machine batch for the left-hand investigation (CPU, read-only):
score / termination (run/physical-audit.json), the largest left-arm joint tracking error (run/episode/trace.jsonl:
commanded target vs measured position, per joint), and, when the reader ran with COFFEE_DIAG_ROBOT_CONTACTS=1,
every robot contact pair with the scene or itself (episodes/<id>/coffee_detection/robot_contacts.jsonl).

Usage: python -m evaluation.tasks.coffee_machine.tools.summarize_left_failures --batch artifacts/<batch> [--json out]
"""
import argparse
import collections
import json
from pathlib import Path

LEFT = ['idx21_arm_l_joint1', 'idx22_arm_l_joint2', 'idx23_arm_l_joint3', 'idx24_arm_l_joint4', 'idx25_arm_l_joint5',
        'idx26_arm_l_joint6', 'idx27_arm_l_joint7']


def tracking(run):
    errors = {j: (0.0, None) for j in LEFT}
    path = run / 'run/episode/trace.jsonl'
    if path.exists():
        stream = path.open()
    elif path.with_name('trace.jsonl.gz').exists():
        import gzip
        stream = gzip.open(path.with_name('trace.jsonl.gz'), 'rt', encoding='utf-8')
    else:
        return None
    for line in stream:
        row = json.loads(line)
        if row['kind'] != 'step':
            continue
        record = row['data']['record']
        target, measured = record['request']['action']['targets'], record['observation']['joint_positions_rad']
        for j in LEFT:
            e = abs(target[j] - measured[j])
            if e > errors[j][0]:
                errors[j] = (round(e, 3), record['observation']['step_index'])
    return {j[-6:]: v for j, v in errors.items()}


def contacts(run):
    files = list(run.glob('episodes/*/coffee_detection/robot_contacts.jsonl'))
    if not files:
        return None
    pairs = collections.defaultdict(lambda: dict(samples=0, max_impulse=0.0, first_step=None, last_step=None))
    for line in files[0].open():
        row = json.loads(line)
        for p in row['pairs']:
            key = p['a'].replace('/World/Robot/', '') + ' <-> ' + p['b'].replace('/World/Robot/', '')
            v = pairs[key]
            v['samples'] += 1
            v['max_impulse'] = max(v['max_impulse'], p['impulse'])
            v['first_step'] = row['step'] if v['first_step'] is None else v['first_step']
            v['last_step'] = row['step']
    return dict(sorted(pairs.items(), key=lambda kv: -kv[1]['samples']))


def summarize(batch):
    state = json.loads((batch / 'queue-state.json').read_text())
    out = {}
    for api, names in state['episodes'].items():
        for index, episode in enumerate(names, start=1):
            run = batch / api / f'source-run-{index}'
            row = dict(episode=episode)
            audit = run / 'run/physical-audit.json'
            if audit.exists():
                a = json.loads(audit.read_text())
                r = a['result']
                row.update(score=r['score'], maximum=r['maximum_score'], success=r['success'], failure=r['failure'],
                           termination=a['termination'], control_steps=a['control_steps'],
                           deductions=[d['kind'] for d in r['deductions']])
            else:
                row['status'] = (state['jobs'].get(f'{api}:source-run-{index}') or {}).get('status', 'pending')
            row['left_arm_max_error_rad'] = tracking(run)
            c = contacts(run)
            if c is not None:
                row['robot_contacts'] = c
                cam = {k: v for k, v in c.items() if '/Cam/' in k}
                row['camera_housing_contact'] = cam or None
            out[f'{api}:source-run-{index}'] = row
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch', type=Path, required=True)
    p.add_argument('--json', type=Path)
    a = p.parse_args()
    result = summarize(a.batch.resolve())
    if a.json:
        a.json.write_text(json.dumps(result, indent=1, ensure_ascii=False) + '\n')
    for key, row in result.items():
        err = row.get('left_arm_max_error_rad') or {}
        cam = row.get('camera_housing_contact')
        cam_text = ('; '.join(f"{k.split(' <-> ')[0].split('/')[0]} on {k.split(' <-> ')[1].split('/')[-1]} steps {v['first_step']}-{v['last_step']} max {v['max_impulse']:.1f}"
                              for k, v in cam.items()) if cam else ('no camera contact' if 'robot_contacts' in row else 'no contact log'))
        print(f"{key} {row['episode']}: {row.get('score', row.get('status'))}/{row.get('maximum', '')} {row.get('failure') or row.get('termination', '')}"
              f" | joint7 err {err.get('joint7')} joint1 {err.get('joint1')} | {cam_text}")


if __name__ == '__main__':
    main()
