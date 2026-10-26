"""Summarize a coffee machine batch (artifacts/<batch>/) for the review page.

Reads queue-state.json plus each job's report / CPU recheck / video manifest, links the three-camera videos into
preview/<api>/kitchen/ (symlinks, no copies), writes preview/kitchen-results.json and rebuilds the family page.
Pure CPU; never touches jobs or physics.

The batch runs from a frozen snapshot whose artifacts directory is the live one. The review page belongs to the live
tree (its definition tables may still be edited by review), so results and page are always written there:
ARENA_LIVE_TASK names the live task directory; without it this file's own task directory is used.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parents[1]
TASK = Path(os.environ.get('ARENA_LIVE_TASK', HERE)).resolve()
PREVIEW = TASK / 'preview'
SCENE = 'kitchen'          # must stay in step with tools/build_preview.py SCENE
FAILURES = {
    'cup_on_ground': '杯子接触地面', 'bag_on_ground': '豆袋接触地面', 'cup_toppled': '杯子倒下（倾斜 >41°）',
    'liquid_spilled': '洒出液体超过 5%', 'cup_overfilled': '杯内液体超过 60 ml', 'beans_below_70': '进仓不足 70 颗',
    'stage1_budget_exhausted': '阶段①操作预算用完', 'stage2_budget_exhausted': '阶段②操作预算用完',
    'stage3_budget_exhausted': '阶段③操作预算用完'}
TERMINATIONS = {'success': '成功结束', 'time_limit': '总时限到', 'client_closed': '来源轨迹播完（未达成成功条件）',
                'source_exhausted': '来源轨迹播完（未达成成功条件）'}
DEDUCTIONS = {'cup_touched_machine': '杯子碰机身', 'robot_touched_machine': '机器人碰机身', 'bag_disturbed': '豆袋倒下或偏离 >2 cm',
              'bag_displaced': '豆袋偏离 >2 cm', 'cup_displaced': '杯子偏离 >2 cm', 'extra_button_event': '多余的启动／停止'}


def read(p):
    return json.loads(Path(p).read_text())


def relative(path):
    path = Path(path)
    return os.path.relpath(path, TASK) if path.is_relative_to(TASK) else str(path)


def label(name, episode, total=5):
    if name == 'fixed-probe':
        return '固定初态短测'
    if name == 'random-probe':
        return '种子 7 随机初态短测'
    index = name.rsplit('-', 1)[1]
    if name.startswith('source-run-'):
        return f'来源轨迹回放 {index}/{total}'
    return f'第 {index} 条来源布局的初态验收（回放前置）'


def link(target, name, api):
    destination = PREVIEW / api / SCENE / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        destination.unlink()
    destination.symlink_to(os.path.relpath(target, destination.parent) if Path(target).is_relative_to(TASK) else str(target))
    return f'{api}/{SCENE}/{name}'


def deduction_text(d):
    name = DEDUCTIONS.get(d['kind'])
    if name is None:
        m = re.fullmatch(r'beans_(\d+)_(\d+)', d['kind'])
        name = f'进仓 {m.group(1)}–{m.group(2)} 颗' if m else d['kind']
    return f"{name} −{d['amount']}（{d['time_s']:.2f} s）"


def summarize(batch):
    state = read(batch / 'queue-state.json')
    notes = read(batch / 'failure-notes.json') if (batch / 'failure-notes.json').exists() else {}
    snapshot = read(batch / 'snapshot.json') if (batch / 'snapshot.json').exists() else {}
    out = dict(batch=relative(batch), scene=SCENE, notes=notes, pipeline=state.get('pipeline', {}), episodes=state.get('episodes', {}),
               blocked=state.get('blocked', {}), not_run=state.get('not_run', {}), max_parallel=state.get('max_parallel'),
               gpu_policy=state.get('gpu_policy'), snapshot=snapshot, apis={})
    for job_id, j in state['jobs'].items():
        api, _, name = job_id.partition(':')
        name = name.split(':preflight-')[0]
        if not re.fullmatch(r'fixed-probe|random-probe|layout-probe-\d+|source-run-\d+', name):
            continue
        rows = out['apis'].setdefault(api, {})
        row = rows.setdefault(name, dict(label=label(name, j.get('episode'), len(out['episodes'].get(api) or []) or 5), episode=j.get('episode'), attempts=[]))
        att = dict(kind=j.get('kind'), status=j.get('status'), gpu=j.get('gpu'), output=relative(j['output']) if j.get('output') else None,
                   started_at=j.get('started_at'), finished_at=j.get('finished_at'), note=j.get('note') or j.get('preflight_error'))
        if j.get('public_source_changed_during_run'):
            att['public_source_changed_during_run'] = j['public_source_changed_during_run']
        outp = Path(j['output']) if j.get('output') else None
        if outp and j.get('kind') == 'replay' and (outp / 'run/physical-audit.json').exists():
            audit = read(outp / 'run/physical-audit.json')
            r = audit['result']
            reason = FAILURES.get(r['failure']) if r['failure'] else TERMINATIONS.get(audit['termination'], audit['termination'])
            att.update(audit_passed=audit['passed'], control_steps=audit['control_steps'], score=r['score'], raw_score=r['raw_score'],
                       maximum_score=r['maximum_score'], success=r['success'], failure=r['failure'], termination=audit['termination'],
                       reason=reason, stage=r['stage'], stage_points=r['stage_points'], budgets=r['budgets'], hold=r['hold'],
                       deductions=[deduction_text(d) for d in r['deductions']], final_counts=r['final_counts'],
                       regressions=r['regressions'], end_time_s=r['end_time_s'], episode=audit['episode_id'])
        elif outp and j.get('kind') == 'replay' and (outp / 'job.json').exists():
            meta = read(outp / 'job.json')
            rep = meta.get('report') or {}
            att.update(replay_status=meta.get('status'), error=str(meta.get('error', ''))[:300] or None, score=rep.get('score'),
                       success=rep.get('success'), reason=rep.get('termination_reason'),
                       follow_up_error=(j.get('follow_up_error') or '')[-300:] or None)
        if outp and j.get('kind') == 'probe' and (outp / 'audit.json').exists():
            probe = read(outp / 'audit.json')
            manifest = read(outp / 'manifest.json') if (outp / 'manifest.json').exists() else {}
            acceptance = probe.get('acceptance') or {}
            att.update(audit_passed=probe['passed'], settle_steps=probe['settle_steps'], samples=probe['samples'], role=probe['role'],
                       acceptance_error=probe['acceptance_error'], measured=acceptance.get('measured'),
                       physics=(probe.get('episode_physics') or {}).get('profile'), layout_randomization=probe.get('layout_randomization'),
                       episode=manifest.get('episode_id'))
            rejected = [k for k, v in acceptance.get('checks', {}).items() if not v]
            if rejected:
                att['rejected_checks'] = rejected
        elif outp and j.get('kind') == 'probe':
            log = Path(str(outp) + '.log')
            manifest = read(outp / 'manifest.json') if (outp / 'manifest.json').exists() else {}
            text = manifest.get('error') or (log.read_text(errors='replace')[-600:] if log.exists() else '')
            att.update(error=str(text)[-300:] or None)
        if outp and j.get('kind') == 'probe' and (outp / 'settled_head.png').exists():
            att['initial_head'] = link(outp / 'settled_head.png', outp.name + '-settled_head.png', api)
        if outp and (outp / 'videos/manifest.json').exists():
            m = read(outp / 'videos/manifest.json')
            href = link(outp / 'videos', outp.name, api)
            att['videos'] = {v['camera']: dict(href=f'{href}/{v["file"]}', frames=v['frames'], fps=v['fps'], sha256=v['sha256']) for v in m['videos']}
            att['keyframe_steps'] = m.get('keyframe_steps')
            if (outp / 'keyframes.jpg').exists():
                att['keyframes'] = link(outp / 'keyframes.jpg', outp.name + '-keyframes.jpg', api)
        row['attempts'].append(att)
    for api, rows in out['apis'].items():
        order = state.get('pipeline', {}).get(api, sorted(rows))
        out['apis'][api] = {k: rows[k] for k in order if k in rows}
    out['batches'] = {api: out['batch'] for api in out['apis']}
    return out


def merge(result, previous):
    """APIs this batch does not run keep their last published results (e.g. the left-hand batch of 2026-10-01 runs only
    1-1 and 1-4; 1-2 / 1-3 stay as batch r4 published them). Per-API provenance is in `batches`."""
    if not previous:
        return result
    mine = set(result['apis']) | {k.split(':', 1)[0] for k in result['not_run']} | set(result['blocked'])
    batches = dict(previous.get('batches') or {api: previous['batch'] for api in previous.get('apis', {})})
    for api, rows in previous.get('apis', {}).items():
        if api in mine:
            continue
        result['apis'][api] = rows
        for field in ('notes', 'pipeline', 'episodes', 'blocked'):
            if api in (previous.get(field) or {}):
                result.setdefault(field, {})[api] = previous[field][api]
        for key, value in (previous.get('not_run') or {}).items():
            if key.startswith(api + ':'):
                result['not_run'][key] = value
        result['batches'][api] = batches.get(api, previous['batch'])
    result['previous_batches'] = sorted({b for b in batches.values() if b != result['batch']} | set(previous.get('previous_batches') or []))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--batch', type=Path, required=True)
    p.add_argument('--no-page', action='store_true')
    a = p.parse_args(argv)
    result = summarize(a.batch.resolve())
    PREVIEW.mkdir(parents=True, exist_ok=True)
    existing = PREVIEW / f'{SCENE}-results.json'
    result = merge(result, read(existing) if existing.exists() else None)
    tmp = PREVIEW / f'{SCENE}-results.json.tmp'
    tmp.write_text(json.dumps(result, indent=1, ensure_ascii=False) + '\n')
    tmp.replace(PREVIEW / f'{SCENE}-results.json')
    if not a.no_page:
        # the page builder is the one every review edit uses: system Python 3.12 (it is not part of any physics job)
        builder = '/usr/bin/python3' if Path('/usr/bin/python3').exists() else sys.executable
        subprocess.run([builder, str(TASK / 'tools/build_preview.py')], check=True, cwd=TASK.parents[2])
    print(json.dumps({api: {k: [(x.get('status'), x.get('score'), x.get('success')) for x in v['attempts']] for k, v in rows.items()}
                      for api, rows in result['apis'].items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
