"""Measured relative-motion confirmation of bilateral grasp, no commands."""
from collections import deque
from dataclasses import dataclass,asdict
import math
import numpy as np
from .geometry import rotation,HANDS


@dataclass(frozen=True)
class DetectionConfig:
    grasp_window_s: float=.1
    relative_position_tolerance_m: float=.003
    relative_angle_tolerance_rad: float=math.radians(5)
    support_face_tolerance_m: float=.003
    initial_history_s: float=.25
    initial_position_span_m: float=.002
    contact_release_debounce_s: float=0.


class GraspDetector:
    def __init__(self,objects,config=DetectionConfig()):
        self.config=config
        self.history={(obj,h):deque() for obj in objects for h in HANDS}
        self.last_time=None

    def observe(self,time_s,poses,bilateral):
        if not math.isfinite(time_s) or (self.last_time is not None and time_s<=self.last_time):
            raise ValueError('Stale grasp evidence')
        result={obj:False for obj,_ in self.history}
        details={}
        for (obj,hand),history in self.history.items():
            p=poses[obj];hp=poses[HANDS[hand]]
            hr=rotation(hp['quaternion_wxyz']);tr=rotation(p['quaternion_wxyz'])
            relative_pos=hr.inv().apply(np.asarray(p['position_m'])-np.asarray(hp['position_m']))
            relative_rot=hr.inv()*tr
            if not np.isfinite(relative_pos).all():raise ValueError('Nonfinite relative pose')
            if not bilateral[obj][hand]:
                history.clear()
            else:
                history.append((time_s,relative_pos,relative_rot))
                # Retain the sample bracketing the beginning of the window.
                while len(history)>1 and history[1][0]<=time_s-self.config.grasp_window_s+1e-9:
                    history.popleft()
            confirmed=False
            if history and time_s-history[0][0]+1e-9>=self.config.grasp_window_s:
                first=history[0]
                confirmed=all(np.linalg.norm(x[1]-first[1])<=self.config.relative_position_tolerance_m and
                    (first[2].inv()*x[2]).magnitude()<=self.config.relative_angle_tolerance_rad for x in history)
            result[obj]|=bool(confirmed)
            details[obj+':'+hand]=dict(confirmed=bool(confirmed),contact_span_s=0. if not history else time_s-history[0][0])
        self.last_time=time_s
        return result,details


def initial_window(history,window_s,physics_dt_s):
    """Select by contiguous physical samples, avoiding float32 cutoff loss."""
    count=math.ceil(window_s/physics_dt_s-1e-9)+1
    rows=list(history)[-count:]
    complete=len(rows)==count and rows[-1]['time_s']-rows[0]['time_s']+1e-6>=window_s
    return rows,complete
