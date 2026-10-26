"""Coffee-only 10 cm trigger, derived from the reviewed shelf rule.

Only the threshold differs. The shelf variant keeps its original 25 cm.
No global variable mutation, fallback evidence or pose writes.
"""
import math
import warnings
from short_task_robustness.tasks.stock_wooden_tray_interference.proximity import distance_to_box

TRIGGER_DISTANCE_M = .10


class ProximityTrigger:
    def __init__(self, target_path):
        if not isinstance(target_path,str) or not target_path:
            raise ValueError('Exact target identity required')
        self.target_path=target_path
        self.last_time=self.latest=None
        self.fired=self.robot_touched=False

    def observe(self,facts,*,route_clear):
        now=facts['simulation_time_s']
        if not math.isfinite(now) or now<0 or (self.last_time is not None and now<=self.last_time):
            raise ValueError('Fresh increasing native timestamp required')
        if facts['target_path']!=self.target_path:raise ValueError('Proximity target identity changed')
        for key in ('robot_contacting_target','robot_holding_target','target_supported','contacts_complete'):
            if type(facts[key]) is not bool:raise ValueError('Explicit boolean evidence required: '+key)
        if type(route_clear) is not bool:raise ValueError('Explicit route admission required')
        centers=facts['gripper_centers_target_local_m']
        if set(centers)!={'l','r'}:raise ValueError('Both gripper centers required')
        distances={hand:distance_to_box(point,facts['target_bounds_local_m']) for hand,point in centers.items()}
        self.last_time=now
        if not facts['contacts_complete']:
            warnings.warn('Incomplete coffee-bag trigger contacts; continuing '
                          'without authorizing interference.',RuntimeWarning)
            self.latest=dict(simulation_time_s=now,target_path=self.target_path,threshold_m=TRIGGER_DISTANCE_M,
                distances_m=distances,triggering_hand=None,start=False,fired=self.fired,
                reason='waiting_for_contact_evidence')
            return dict(self.latest)
        self.robot_touched|=facts['robot_contacting_target'] or facts['robot_holding_target']
        near=min(distances.values())<=TRIGGER_DISTANCE_M
        eligible=(not self.fired and not self.robot_touched and near and facts['target_supported'] and route_clear)
        reason=('already_triggered' if self.fired else 'robot_already_contacted' if self.robot_touched else
                'waiting_for_proximity' if not near else 'target_not_supported' if not facts['target_supported'] else
                'reserved_route_blocked' if not route_clear else 'triggered')
        if eligible:self.fired=True
        self.latest=dict(simulation_time_s=now,target_path=self.target_path,threshold_m=TRIGGER_DISTANCE_M,
            distances_m=distances,triggering_hand=min(distances,key=distances.get) if eligible else None,
            start=eligible,fired=self.fired,reason=reason)
        return dict(self.latest)
