"""User-approved 10 simulation seconds after the last real finger contact.

Wake eligibility only: touch/release does not award points, imply a grasp,
reset task state or extend the room's no-score-increase watchdog.
"""
import math

WAKE_SECONDS=10.


class TouchWake:
    def __init__(self):
        self.last_time=None;self.last_touch=None;self.touching=False

    def observe(self,*,time_s,finger_contact,supports,external_contact,on_floor):
        if (type(time_s) not in (int,float) or not math.isfinite(time_s) or time_s<0
                or (self.last_time is not None and time_s<self.last_time)):
            raise ValueError('Monotonic measured simulation time required')
        if any(type(v) is not bool for v in (finger_contact,external_contact,on_floor)):
            raise ValueError('Measured contact booleans required')
        self.last_time=time_s;self.touching=finger_contact
        if finger_contact:self.last_touch=time_s

    def request(self,*,ended,time_s):
        if (type(time_s) not in (int,float) or not math.isfinite(time_s)
                or (self.last_time is not None and time_s<self.last_time)):
            raise ValueError('Monotonic simulation clock required')
        active=not ended and self.last_touch is not None and time_s-self.last_touch<WAKE_SECONDS-1e-9
        return dict(active=active,locked=False,touching=active and self.touching,
                    last_touch_s=self.last_touch)
