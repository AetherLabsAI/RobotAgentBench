"""Native ShortTaskSession acceptance harness; never a model success claim."""
import numpy as np
from .reporting import report_json


def run(session, approach, robot, targets, names, indices, advance, capture, video_frame,
        record_video, output, report, checkpoint):
    if approach is None:
        return run_idle(session,robot,targets,indices,advance,capture,video_frame,
                        record_video,output,report,checkpoint)
    actor = session.environment_actor
    if 'robot_approach' in report:
        report['robot_approach']['production_session'] = bool(report.get('public_session_built'))
    actions, results = [], []
    triggered = False
    for frame in range(approach.last_frame+1):
        native = approach.native(frame)
        for name,value in native.items():
            targets[0,names.index(name)] = value
        from scipy.spatial.transform import Rotation
        root = robot.get_root_transforms()[0]
        robot_state = dict(xy=root[:2].tolist(),yaw=float(Rotation.from_quat(root[3:7]).as_euler('xyz',degrees=True)[2]))
        interval = session.begin_control(robot_state, command_vw=(0.,0.),
            base_velocity=tuple(map(float,robot.get_root_velocities()[0])))
        actions.append(dict(source_frame=frame,time_s=float(session.clock.current_time),native_drive_targets=native))
        for _ in range(interval['substeps']):
            session.before_physics(1/session.clock.hz)
            robot.set_dof_position_targets(targets,indices)
            session.after_physics(advance())
            if actor.proximity.fired and not triggered:
                capture('triggered')
                triggered = True
        result = session.finish_control(completed_substeps=interval['substeps'])
        results.append(result)
        if record_video:
            video_frame()
        report.update(status='runtime_session_running',control_frame=frame,
                      actor_phase=actor.audit.phase.value)
        if frame%20 == 0:
            checkpoint()
        if actor.started_at is not None and session.clock.current_time >= actor.started_at+5.:
            break  # Same 3-second intervention plus 2-second post-exit view.
        if actor.started_at is None and frame >= approach.trigger_last_frame:
            break
        if session.ledger.terminal is not None:
            break
    capture('final')
    (output/'robot_actions.json').write_text(report_json(actions)+'\n')
    (output/'session_results.json').write_text(report_json(results)+'\n')
    phase = actor.audit.phase.value
    source = next(iter(session.ports.values())).source
    report.update(status='motion_completed' if phase=='done' else 'waiting_no_trigger',
        runtime_backend_verified=phase=='done',full_task_verified=False,
        entry='public_build_session' if report.get('public_session_built') else 'real_ShortTaskSession_explicit_candidate_harness',
        public_availability_overridden_only_in_diagnostic=not report.get('public_session_built',False),
        continuous_action_clock_verified=[r['source_frame'] for r in actions]==list(range(len(actions))),
        collision_episodes=actor.ledger.contact_episodes,human_collision_deduction=actor.ledger.deduction,
        proximity_trigger=actor.proximity.latest,final_supported=bool(source.latest['contacts']['target_supported']),
        final_task_result=results[-1] if results else None,model_run=False)
    if phase=='done' and not report['final_supported']:
        raise RuntimeError('Native actor finished without target support')
    return report


def run_idle(session, robot, targets, indices, advance, capture, video_frame,
             record_video, output, report, checkpoint):
    """Existing five-second hold diagnostic through the public session.

    Deliberately does not make the robot approach or force the human trigger.
    It verifies initial loading/scoring and idle suppression, not recovery.
    """
    from scipy.spatial.transform import Rotation
    actor=session.environment_actor
    results=[]
    # Same 300 physical steps / 60 Hz idle window as the original task probe.
    for frame in range(100):
        root=robot.get_root_transforms()[0]
        state=dict(xy=root[:2].tolist(),yaw=float(Rotation.from_quat(root[3:7]).as_euler('xyz',degrees=True)[2]))
        interval=session.begin_control(state,command_vw=(0.,0.),
            base_velocity=tuple(map(float,robot.get_root_velocities()[0])))
        for _ in range(interval['substeps']):
            session.before_physics(1/session.clock.hz)
            robot.set_dof_position_targets(targets,indices)
            session.after_physics(advance())
        results.append(session.finish_control(completed_substeps=interval['substeps']))
        if record_video:video_frame()
        if actor.proximity.fired:
            raise RuntimeError('Idle probe unexpectedly triggered human; inspect initial proximity')
        if frame%20==0:
            report.update(status='idle_session_running',control_frame=frame)
            checkpoint()
    capture('final')
    (output/'session_results.json').write_text(report_json(results)+'\n')
    source=next(iter(session.ports.values())).source
    supported=bool(source.latest['contacts']['target_supported'])
    if not supported:raise RuntimeError('Idle target lost native shelf support')
    report.update(status='idle_completed',idle_session_verified=True,
        runtime_backend_verified=False,full_task_verified=False,
        entry='public_build_session' if report.get('public_session_built') else 'candidate_session',
        robot_motion_policy='idle_suppression_diagnostic',human_motion_started=False,
        idle_trigger_suppression_verified=True,control_frames=len(results),
        final_supported=supported,final_task_result=results[-1],model_run=False)
    return report
