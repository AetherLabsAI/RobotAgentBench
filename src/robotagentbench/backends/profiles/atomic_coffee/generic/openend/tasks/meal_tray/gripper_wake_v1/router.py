"""Task-local candidate: one gripper contact index, no scoring or native reads.

Input pairs must be the existing native contact stream's current pairs. This
module cannot turn a command, proximity, stale LOST contact or grasp claim into
a touch. Binding and physics observation remain host responsibilities.
"""
import math


def ancestors(path):
    if not isinstance(path,str) or not path.startswith('/World/'):
        raise ValueError('Absolute scene prim path required')
    yield path
    while '/' in path[1:]:
        path=path.rsplit('/',1)[0]
        yield path


class GripperWakeIndex:
    def __init__(self,bindings,fingers):
        self.index={};self.keys=tuple(bindings);self.fingers=frozenset(fingers)
        if not self.fingers or not self.keys:raise ValueError('Explicit fingers and tasks required')
        for f in self.fingers:
            tuple(ancestors(f))
            if not f.startswith('/World/Robot/'):raise ValueError('Robot finger root required')
        for key,roots in bindings.items():
            if not roots:raise ValueError('Empty task object binding')
            for root in roots:
                tuple(ancestors(root))
                if root=='/World/Robot' or root.startswith('/World/Robot/'):
                    raise ValueError('Task object cannot be the robot')
                self.index.setdefault(root,set()).add(key)
        self.time=None;self.armed=False;self.started={};self.last_touch={};self.touching=set()
        self.pairs_seen=0;self.gripper_pairs=0;self.target_lookups=0

    def arm_after_initial_acceptance(self,time_s):
        if self.armed:raise RuntimeError('No task reset on re-entry')
        self._time(time_s)
        self.armed=True;self.started.clear();self.last_touch.clear();self.touching.clear()

    def _time(self,t):
        if type(t) not in (int,float) or not math.isfinite(t) or t<0 or (self.time is not None and t<self.time):
            raise ValueError('Monotonic measured time required')
        self.time=t

    def observe(self,pairs,time_s):
        self._time(time_s)
        touched=set()
        for paths,samples in pairs:
            self.pairs_seen+=1
            if not samples:continue
            if len(paths)!=2:raise ValueError('Collider pair required')
            chains=[tuple(ancestors(p)) for p in paths]
            hands=[bool(self.fingers.intersection(chain)) for chain in chains]
            if hands[0]==hands[1]:continue  # no hand, or finger-to-finger
            self.gripper_pairs+=1
            other=chains[1 if hands[0] else 0]
            for root in other:
                self.target_lookups+=1
                touched.update(self.index.get(root,()))
        self.touching=touched if self.armed else set()
        if self.armed:
            for key in touched:
                self.started.setdefault(key,time_s)
                self.last_touch[key]=time_s

    def requests(self,ports,time_s):
        if not self.armed or self.time is None or abs(self.time-time_s)>1e-6:
            raise RuntimeError('Initial acceptance and fresh contact stream required')
        if set(ports)!=set(self.keys):raise ValueError('Task identities changed')
        # No port selection_request, contact reduction, pose or score calls.
        # Started processes persist through release, independently of focus/Hz.
        return {key:dict(active=key in self.started and not ports[key].ended,
            locked=False,touching=key in self.touching and not ports[key].ended,
            last_touch_s=self.last_touch.get(key)) for key in self.keys}

    def receipt(self):
        return dict(started=dict(self.started),pairs_seen=self.pairs_seen,
            gripper_pairs=self.gripper_pairs,target_lookups=self.target_lookups,
            scope='wake_only_no_grasp_or_score_inference')
