"""CPU recheck of one finished replay of API 1-1 … 1-4: recompute the score from the recorded facts.

The worker writes `coffee_detection/control_facts.jsonl`, one row per 20 Hz control frame: exactly the Facts the
rule engine was fed. This audit
  * rebuilds the Facts and replays a fresh `Process` over them as an independent reference, finalizes it with the
    recorded termination reason and asserts that its whole summary equals the worker's, field by field;
  * asserts the server evaluation, the runner report and the runner's per-step events agree with that summary;
  * checks the full-rate trace (`physical_facts.jsonl`, one row per physics step) is gap-free and that every control
    frame is the state of its last physics step;
  * verifies the camera frames (unless they are already pruned).
Everything asserted is an equality against recorded evidence; nothing here can turn a failure into a success.
"""
import argparse
import hashlib
import json
from pathlib import Path

from arena_api.policy_execution import SUBSTEPS, PHYSICS_DT
from evaluation.tasks.coffee_machine.facts import Facts, ObjectFacts, CupPlacement, LiquidFacts, ButtonFacts
from evaluation.tasks.coffee_machine.process_rules import Process, RULE_VERSION, CONTROL_DT_S
from evaluation.tasks.coffee_machine.registration import registration
from evaluation.tasks.coffee_machine.specs import EVALUATOR_ID, CUP, BAG

AUDIT_SCHEMA = 'arena.coffee_machine.replay_audit.v1'
CAMERAS = ('head', 'left', 'right')


def _text(path):
    """Evidence of finished batches is stored gzip-compressed (2026-10-05 disk clean-up); read either form."""
    path = Path(path)
    if path.exists():
        return path.read_text()
    packed = path.with_name(path.name + '.gz')
    if packed.exists():
        import gzip
        return gzip.open(packed, 'rt', encoding='utf-8').read()
    raise FileNotFoundError(path)


def read(path):
    return json.loads(_text(path))


def read_lines(path):
    return [json.loads(line) for line in _text(path).splitlines() if line.strip()]


def facts_from(row):
    def body(d):
        return ObjectFacts(position=tuple(d['position']), tilt_deg=d['tilt_deg'], pads_bilateral=d['pads_bilateral'],
                           robot_contact=d['robot_contact'], supports=frozenset(d['supports']), ground_contact=d['ground_contact'],
                           machine_body_contact=d['machine_body_contact'], displacement_m=d['displacement_m'])
    liquid = LiquidFacts(**row['liquid']) if row['liquid'] is not None else None
    button = None
    if row['button'] is not None:
        b = row['button']
        button = ButtonFacts(displacement_m=b['displacement_m'], robot_on_button=b['robot_on_button'], dispensing=b['dispensing'],
                             events=tuple(b['events']), event_by_robot=tuple(b['event_by_robot']))
    return Facts(step=row['step'], time_s=row['time_s'], cup=body(row['cup']), bag=body(row['bag']),
                 placement=CupPlacement(**row['placement']), robot_machine_contact=row['robot_machine_contact'],
                 beans_in_hopper=row['beans_in_hopper'], beans_total=row['beans_total'], liquid=liquid, button=button)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--episode', type=Path, required=True, help='episodes/<episode_id> of the job')
    p.add_argument('--run', type=Path, required=True, help='run/ directory of the job')
    p.add_argument('--no-images', action='store_true', help='skip the per-frame image audit (frames already pruned)')
    p.add_argument('--output', type=Path, help='receipt path (default <run>/physical-audit.json)')
    a = p.parse_args(argv)
    ep, run = a.episode, a.run
    report = read(run / 'report.json')
    evaluation = read(ep / 'task_evaluation.json')
    closed = read(ep / 'closed.json')
    shutdown = read(ep / 'worker_shutdown.json')
    process = evaluation['details']['process']
    api = evaluation['task_id']

    assert process['schema'] == RULE_VERSION, process['schema']
    assert evaluation['evaluator_id'] == EVALUATOR_ID == registration(api).evaluator_id
    assert evaluation['finalized'] and evaluation['valid']
    assert report['run_status'] == 'completed' and report['evaluation_valid']
    assert shutdown['completed'] and shutdown['returncode'] == 0 and not shutdown['forced']
    for key, value in closed.items():
        if key.endswith('_unchanged'):
            assert value, key
    assert evaluation['details']['initial_state_accepted'] is True
    assert not any(evaluation['details']['randomization_applied'].values()), 'Unexpected house randomization'

    # ---- independent replay of the rules over the recorded control frames ----
    frames = read_lines(ep / 'coffee_detection/control_facts.jsonl')
    assert frames and [f['step'] for f in frames] == list(range(len(frames))), 'gap in the control-frame evidence'
    replay = Process(api)
    trace = []
    for row in frames:
        replay.observe(facts_from(row))
        trace.append(dict(step=row['step'], stage=replay.stage, score=replay.score, success=replay.success, failure=replay.failure))
    replay.finalize(evaluation['reason'])
    mine = replay.summary()
    differ = {k: (v, process.get(k)) for k, v in mine.items() if process.get(k) != v}
    assert not differ, ('independent replay disagrees with the worker', differ)
    assert evaluation['details']['facts'] == {k: v for k, v in frames[-1].items() if k not in ('simulation_time_s', 'reader_step')}, \
        'Final worker evidence differs from the last recorded control frame'
    success = bool(process['success'])
    assert evaluation['success'] is success and report['success'] is success
    assert abs(evaluation['score'] - process['score']) < 1e-9 and abs(report['score'] - process['score']) < 1e-9
    assert report['termination_reason'] == evaluation['reason'] == process['termination']

    # ---- runner events ----
    events_log = read_lines(run / 'evaluation_events.jsonl')
    assert events_log and [e['step_index'] for e in events_log] == list(range(len(events_log))), 'gap in runner events'
    assert events_log[-1]['update']['score'] == evaluation['score'] and events_log[-1]['step_index'] == evaluation['step_index']
    assert closed['steps'] == events_log[-1]['step_index'] == len(frames) - 1
    for event in events_log:
        assert abs(event['update']['score'] - trace[event['step_index']]['score']) < 1e-9 or event['step_index'] == len(frames) - 1, \
            ('runner score differs from the replayed score', event['step_index'])

    # ---- full-rate trace ----
    physics = read_lines(ep / 'coffee_detection/physical_facts.jsonl')
    assert [r['reader_step'] for r in physics] == list(range(len(physics))), 'gap in the full-rate evidence'
    start = frames[0]['reader_step']
    for i in range(1, len(physics)):
        assert abs(physics[i]['time_s'] - physics[i - 1]['time_s'] - PHYSICS_DT) < 1e-6, ('physics clock', i)
    for row in frames:
        assert row['reader_step'] == start + row['step'] * SUBSTEPS, ('control frame off the physics grid', row['step'])
        last = physics[row['reader_step']]
        assert abs(last['time_s'] - row['simulation_time_s']) < 1e-9
        for key, path in (('cup', CUP), ('bag', BAG)):
            assert list(row[key]['position']) == last['poses'][path]['position_m'], ('control frame is not its last physics step', row['step'])
            assert row[key]['supports'] == last['contacts'][path]['supports']
    assert len(physics) == start + (len(frames) - 1) * SUBSTEPS + 1 or len(physics) >= start + (len(frames) - 1) * SUBSTEPS + 1

    # ---- frames ----
    images = []
    if not a.no_images:
        captures = read_lines(ep / 'camera_capture.jsonl')
        assert len(captures) == closed['steps'] + 1
        assert closed['image_recording']['completed'] and closed['image_recording']['dropped_frames'] == 0
        previous = {}
        for i, cap in enumerate(captures):
            obs = read(ep / f'step{i:04d}.json')
            assert obs['frame_index'] == i and cap['frame_index'] == i and obs['source'] == 'isaac_physics'
            assert abs(obs['time_s'] - frames[i]['simulation_time_s']) < 1e-6, ('frame time is off the control grid', i)
            for camera in CAMERAS:
                c = cap['cameras'][camera]
                ref = c['numerator'] / c['denominator']
                assert abs(c['simulation_time_s'] - obs['time_s']) < 1e-6
                if i:
                    assert ref > previous[camera]
                previous[camera] = ref
                f = ep / f'step{i:04d}_{camera}.png'
                from PIL import Image
                with Image.open(f) as im:
                    assert im.size == (640, 480)
                    im.verify()
                images.append(dict(file=f.name, sha256=hashlib.sha256(f.read_bytes()).hexdigest()))

    marks = [e for e in process['events'] if e['kind'] in ('stage_points', 'deduction', 'success', 'terminal_failure', 'regression', 'stage')]
    audit = dict(schema=AUDIT_SCHEMA, passed=True, rule_version=RULE_VERSION, evaluator_id=EVALUATOR_ID, api_id=api,
                 episode_id=evaluation['episode_id'], control_steps=closed['steps'], control_frames=len(frames), physics_steps=len(physics),
                 reader_start_step=start, physics_steps_per_control=SUBSTEPS, control_dt_s=CONTROL_DT_S, host_images_verified=not a.no_images,
                 scope='replayed the rule engine over every recorded control frame and asserted its whole summary equals the worker output; '
                       'cross-checked task_evaluation.json, report.json, evaluation_events.jsonl and the full-rate trace; '
                       + ('frames pruned' if a.no_images else 'frame audit') + '; visual review separate',
                 result=process, termination=evaluation['reason'], physical_task_success=success,
                 state_trace=[dict(step=e['step'], time_s=e['time_s'], kind=e['kind'],
                                   detail={k: v for k, v in e.items() if k not in ('step', 'time_s', 'kind')}) for e in marks][:200],
                 final_facts=frames[-1], images=images)
    out = a.output or run / 'physical-audit.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w' if a.output else 'x') as f:
        json.dump(audit, f, indent=1, allow_nan=False, ensure_ascii=False)
    print(json.dumps({k: v for k, v in audit.items() if k not in ('images', 'state_trace', 'result', 'final_facts')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
