"""Build an isolated historical-layout diagnostic, never an OpenEnd acceptance.

Keep the source's effective physics and 60 Hz refrigerator helper. Only the
explicitly approved action clock is changed to 20 Hz. No source state replay.
"""
import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Source drift: expected one occurrence of {old!r}')
    return text.replace(old, new)


def adapt(source, terminal):
    # Omit the historical checkpoint and trajectory planner branches entirely.
    prefix = source.split('if a.checkpoint_recording:\n')[0]
    advance = source[source.index('def advance('):source.index('def base_state(')]
    constraint = source[source.index('def constrain_base('):source.index('def navigate_and_bridge(')]
    prefix = replace_once(prefix, "a=p.parse_args()", "a=p.parse_args()\nassert not any((a.historical_dock,a.correct_pick,a.checkpoint_recording,a.close_retreat,a.checkpoint_continue)), 'Diagnostic forbids docking, checkpoint resets and planner corrections'\nassert a.dense_replay, '20 Hz action file is required'")
    prefix = replace_once(prefix, "'control_hz':60", "'control_hz':20")
    prefix = replace_once(prefix, 'simulator time is tick/60', 'simulator time is tick/20')
    prefix = replace_once(prefix, 'base_command=np.zeros(3);', 'base_command=np.zeros(3);body_command=np.zeros(2);')
    advance = replace_once(advance, '    angle=door_rule()\n', '')
    start = advance.index('        applied_body_action=np.array(')
    end = advance.index("        nav_drives['drive']", start)
    advance = advance[:start] + "        applied_body_action=body_command.copy()\n        if abs(body_command[0])>1e-5 and abs(body_command[1])>1e-5:raise RuntimeError('Simultaneous drive and turn')\n" + advance[end:]
    advance = replace_once(advance, '    for _ in range(4):sim.step(render=False)', '    # Three unchanged helper ticks per single held 20 Hz action.\n    for helper_tick in range(3):\n        angle=door_rule()\n        for _ in range(4):sim.step(render=False)')
    for old,new in [('tick==120','tick==40'),('tick>=120:','tick>=40:'),('tick%60==0','tick%20==0'),('a.render and tick%3==0','a.render'),('(tick//3-1)%20','(tick-1)%20'),('f{tick//3-1:05d}','f{tick-1:05d}'),('(np.arange(len(rows))+1)/60','(np.arange(len(rows))+1)/20'),('recording_60hz.npz','recording_20hz.npz'),('control_hz=60','control_hz=20')]:
        advance = replace_once(advance,old,new)
    # The source dock branch is unreachable, and the assertion is explicit.
    advance = replace_once(advance, '    global tick,dock_active', '    global tick,dock_active\n    assert not dock_active')
    # Never anchor the base to a recorded pose; use its measured stopped pose.
    constraint = replace_once(constraint, '    if reference is not None:position=np.asarray(reference[:3]);quaternion=np.r_[reference[6],reference[3:6]]', "    assert reference is None, 'Recorded base poses are not actions'")
    replay = '''\ndef replay20():
    global phase,navigation_active,base_command,body_command
    log=np.load(a.dense_replay,allow_pickle=False)
    assert log['control_hz'].tolist()==[20]
    assert np.allclose(log['timestamp_start_s'],np.arange(len(log['action']))/20)
    action_names=log['action_names'].tolist()
    assert len(action_names)==22 and action_names[-2:]==['base_forward_m_s','base_yaw_rate_rad_s']
    assert np.isfinite(log['action']).all()
    for i,action in enumerate(log['action']):
        phase=str(log['phase'][i])
        if bool(log['base_navigation_active'][i]) and not navigation_active:
            stage.GetPrimAtPath('/World/Robot/base_link/FixedJoint').GetAttribute('physics:jointEnabled').Set(False)
            get_physx_simulation_interface().flush_changes();navigation_active=True
        mode=str(log['nav_constraint_mode'][i])
        if mode!=nav_constraint_mode:
            assert mode in ('hold','drive','turn')
            constrain_base(mode)
        body_command=action[-2:].copy()
        quat=robot.get_world_pose()[1]
        yaw=Rot.from_quat(np.r_[quat[1:],quat[0]]).as_euler('xyz')[2]
        base_command=np.array([math.cos(yaw)*body_command[0],math.sin(yaw)*body_command[0],body_command[1]])
        advance(action[:20],action_names[:20])
        if a.frames and i+1>=a.frames:
            save('bounded_20hz_diagnostic_not_full_episode');return
    audit=terminal.evaluate(rows,contacts)
    (OUT/'terminal_audit.json').write_text(json.dumps(audit,indent=2))
    if not audit['passed']:raise RuntimeError('20 Hz terminal audit failed: '+str(audit['checks']))
    save('historical_layout_20hz_terminal_passed_not_openend_acceptance')
try:
    replay20()
except Exception as exc:
    save('physical_trial_failed',str(exc));raise
finally:
    stream_rows.close()
    try:
        for writer in video_writers.values():writer.close()
    finally:app.close()
'''
    # Same two seconds and same thresholds, now represented by 40 control rows.
    for old,new in [('rows[-120:]','rows[-40:]'),("['tick']-120","['tick']-40"),('len(tail)==120','len(tail)==40')]:
        terminal=replace_once(terminal,old,new)
    runner=prefix+advance+constraint+replay
    ast.parse(runner);ast.parse(terminal)
    return runner,terminal


def prepare(root, output):
    source=root/'source'
    runner,terminal=adapt((source/'replay_code/run_trial.py').read_text(),(source/'replay_code/terminal_audit.py').read_text())
    output.mkdir(exist_ok=False)
    (output/'run_trial20.py').write_text(runner)
    (output/'terminal_audit.py').write_text(terminal)
    shutil.copy2(source/'replay_code/camera_stream.py',output/'camera_stream.py')
    shutil.copy2(source/'source_controls.npz',output/'source_controls.npz')
    job=json.loads((source/'job.json').read_text())
    job['scene']=str(source/'scene.usdc');job['control_hz']=20
    job['base_servo']['command_hz']=20
    job['status']='isolated_historical_layout_20hz_diagnostic_only'
    (output/'job.json').write_text(json.dumps(job,indent=2))
    receipt=dict(control_hz=20,physics_hz=240,fridge_helper_hz=60,
                 physics_substeps_per_action=12,action_steps=4880,
                 duration_s=244,terminal_hold_s=2,terminal_hold_samples=40,
                 historical_physics_recipe_preserved=True,current_room_modified=False,
                 recorded_states_used_as_actions=False,full_physical_replay_completed=False,
                 navigation_anchor='measured live pose at primitive change; never recorded pose',
                 task_success=None,source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (source/'replay_code').glob('*.py')})
    (output/'preparation.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();prepare(args.root.resolve(),args.output.resolve())
