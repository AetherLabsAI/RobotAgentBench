"""Unselected device maintenance, separate from detailed scoring readout.

The existing rail mechanism and coffee controller/animation still advance.
No physics disabling, pose restore, score update or fake full-fact sample.
"""
import math


def check_tick(clock, previous, helper):
    if any(helper[k] for k in ('events', 'docked_object_count', 'tray_dock_enabled')):
        raise RuntimeError('Manipulation assistance invalidates evidence')
    now, dt = float(clock.current_time), float(clock.get_physics_dt())
    if dt not in (1/60, 1/240) or not math.isfinite(now) or abs(now-previous-dt)>1e-6:
        raise RuntimeError('Missing device maintenance substep')
    return now, dt


def idle_fryer(port, helper):
    s=port.source
    now,dt=check_tick(port.clock,s.last_time,helper)
    if s.closed: raise RuntimeError('Closed fryer')
    report=helper['_contact_report'] if '_contact_report' in helper else s.interface.get_contact_report()
    s.contacts.ingest(report)
    contact,_,_=s.summarize()
    if port.composite.inherited_counter_support:
        c=contact[s.paths['other']]
        if c['robot'] or c['ground'] or c['basket'] or c['supports']!={'counter'}:
            port.composite.inherited_counter_support=False
    # AssetRail.step ignores its legacy opening argument and measures its own
    # tray/body views. Use the unchanged asset mechanism, even while unselected.
    if s.rail is not None: s.rail.step(None,s.step+1,now)
    for history in s.detector.history.values(): history.clear()
    s.step+=1;s.last_time=now;s.contiguous_240_steps=0;s.history.clear()
    s.reset_accumulators()
    port.inactive_substeps=getattr(port,'inactive_substeps',0)+1


def idle_coffee(port, helper):
    import numpy as np
    from pxr import Usd,UsdGeom
    from evaluation.tasks.coffee_machine import isaac_evidence as old
    s=port.owner.service;source=port.owner.source
    now,dt=check_tick(port.clock,source.last_time,helper)
    if s.closed or s.consumer is not None: raise RuntimeError('Unexpected coffee lifecycle')
    if s.last_time is not None and abs(now-s.last_time-dt)>1e-6:
        raise RuntimeError('Missing coffee device tick')
    report=helper['_contact_report'] if '_contact_report' in helper else source.interface.get_contact_report()
    source.contacts.ingest(report);pairs=tuple(source.contacts.pairs())
    # The controller needs only button, beans and cup (for visual liquid), not
    # bag pose, arm poses, relative-motion grasp or cup placement geometry.
    mp,mq=source.views[source.machine_body].get_world_pose()
    expected=np.asarray(source.expected_machine_body)
    if (not np.allclose(mp,expected[3,:3],rtol=0,atol=1e-6)
            or not np.allclose(old.rotation(mq).as_matrix().T,expected[:3,:3],rtol=0,atol=1e-6)
            or UsdGeom.XformCache().GetLocalToWorldTransform(s.stage.GetPrimAtPath(s.roles['machine']))!=s.machine):
        raise RuntimeError('Coffee machine anchor changed')
    button,_=source.views[s.roles['button']].get_world_pose()
    cup,cup_q=source.views[s.roles['cup']].get_world_pose()
    beans=source.beans.positions()
    if not all(np.isfinite(v).all() for v in (button,cup,cup_q,beans)):
        raise RuntimeError('Invalid device measurement')
    depth=float(s.machine_local(button)[1]);press=False
    for paths,samples in pairs:
        for i in (0,1):
            if (old.under(paths[i],old.ROBOT) and '/gripper_' in paths[i]
                    and old.under(paths[1-i],s.roles['button'])
                    and max(v for _,v in samples)>old.BUTTON_PRESS_IMPULSE):press=True
    before=s.controller.pressed
    device=s.controller.update(dt,dict(enumerate(map(tuple,s.machine_local(beans)))),depth)
    if depth<-.001 and press:s.press_by_robot=True
    if (before and not s.controller.pressed) or (depth>=-.001 and not press):s.press_by_robot=False
    with Usd.EditContext(s.stage,s.stage.GetSessionLayer()):
        s.animation.sync_cup(cup,cup_q)
        s.animation.update(dt,device['dispensing'],s.machine)
    if not getattr(port,'_reader_was_idle',False):s.detector=old.GraspDetector(s.objects)
    port._reader_was_idle=True
    # Preserve device state, not unselected scoring windows. latest stays a
    # timestamped last FULL sample and must never be consumed until refreshed.
    s._clear_window();s.sequence+=1;s.last_time=now;s.last_physics_dt=dt
    source.last_time=now;port.history.clear();port.consecutive_240=0
    port.inactive_substeps=getattr(port,'inactive_substeps',0)+1
