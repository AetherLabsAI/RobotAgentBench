"""Explicit inactive clocks and contact-only wakeup; never fabricated facts.

Inactive substeps carry time/contact evidence, not pose/placement/grasp scores.
Physics continues unchanged. Detailed sample indices count measured samples,
not inactive ticks. Missing ticks still fail closed across suspend/resume.
"""
import math
from .touch_wake import TouchWake


class SuspensibleEvidence:
    def evidence_time(self):
        last = None if self.last is None else self.last.time_s
        idle = getattr(self, '_inactive_time', None)
        if last is None: return idle
        return last if idle is None else max(last, idle)

    def suspend_substep(self, now, dt):
        if self.last is None: raise RuntimeError('Accept initial evidence before suspension')
        if self.ended: return
        previous = self.evidence_time()
        if (type(now) not in (int, float) or not math.isfinite(now)
                or dt not in (1/60, 1/240) or abs(now-previous-dt)>1e-6):
            raise ValueError('Missing inactive native substep')
        self.pick.pause()
        self.place.pause()
        self._inactive_time = now


def contact_report(port, helper):
    return helper['_contact_report'] if '_contact_report' in helper else port.interface.get_contact_report()


def observe_selection(port, row):
    kwargs={k: row[k] for k in ('finger_contact', 'supports', 'external_contact', 'on_floor')}
    if isinstance(port.contact_selection,TouchWake):kwargs['time_s']=row.get('time_s',port.clock.current_time)
    port.contact_selection.observe(**kwargs)
    # Do not lose a short genuine touch between two 20 Hz control boundaries.
    port._touch_pending = getattr(port, '_touch_pending', False) or row['finger_contact']
    port._selection_time = row.get('time_s', port.clock.current_time)


def selection_request(port):
    timestamp = getattr(port, '_selection_time',
                        None if port.latest is None else port.latest['time_s'])
    if timestamp is None or abs(timestamp-port.clock.current_time)>1e-6:
        raise RuntimeError('Stale contact selection evidence')
    if isinstance(port.contact_selection,TouchWake):
        return port.contact_selection.request(ended=port.ended,time_s=port.clock.current_time)
    request = port.contact_selection.request(ended=port.ended)
    if not port.ended and getattr(port, '_touch_pending', False): request['active'] = True
    port._touch_pending = False
    return request


def inactive_object(port, helper):
    if any(helper[k] for k in ('events', 'docked_object_count', 'tray_dock_enabled')):
        raise RuntimeError('Helper-modified physical evidence')
    now, dt = port.clock.current_time, 1/port.clock.hz
    if port.clock.hz not in (60, 240) or abs(now-port.last_time-dt)>1e-6:
        raise RuntimeError('Missing inactive native substep')
    if port.pending: raise RuntimeError('Unconsumed selected evidence before suspension')
    port.contacts.ingest(contact_report(port, helper))
    c = port.reduce_contacts()
    observe_selection(port, dict(c, time_s=now))
    # Clear only unfinished relative-motion evidence; scores/bindings survive.
    for history in port.detector.history.values(): history.clear()
    port.composite.suspend_substep(now, dt)
    port.last_time = now
    port.inactive_substeps = getattr(port, 'inactive_substeps', 0)+1
