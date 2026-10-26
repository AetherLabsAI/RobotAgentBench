"""Current designated-tray position, gated by required process milestones.

Microwave requires completed illumination before any credit, including placement.
Every family also requires its process milestones. Scores are snapshots, not
sums of re-entry events. User approved 3 final-state points for all six families.
"""
from dataclasses import dataclass
import math

KINDS = ('coffee_to_meal_tray', 'bread_to_meal_tray',
         'microwave_box_to_meal_tray', 'salad_to_meal_tray',
         'cutlery_to_meal_tray', 'drink_to_meal_tray')
POINTS = dict.fromkeys(KINDS, 3)


def path(value):
    return (isinstance(value, str) and value.startswith('/World/') and
            all(x and x not in ('.', '..') for x in value.split('/')[1:]))


@dataclass(frozen=True)
class Binding:
    episode_id: str
    instance_id: str
    kind: str
    target: str
    source: str
    destination: str

    def __post_init__(self):
        if (not isinstance(self.episode_id, str) or not self.episode_id
                or not isinstance(self.instance_id, str) or not self.instance_id or self.kind not in KINDS):
            raise ValueError('Explicit meal task identity required')
        paths = (self.target, self.source, self.destination)
        if not all(path(x) for x in paths) or len(set(paths)) != 3:
            raise ValueError('Distinct canonical object paths required')
        if any(a.startswith(b+'/') or b.startswith(a+'/')
               for i,a in enumerate(paths) for b in paths[i+1:]):
            raise ValueError('Target/source/destination roles overlap')


@dataclass(frozen=True)
class PlacementFrame:
    binding: Binding
    step: int
    time_s: float
    supported_by_destination: bool
    whole_inside_destination: bool
    robot_contact: bool
    ground_contact: bool
    position_in_destination_m: tuple
    quaternion_in_destination_wxyz: tuple
    tilt_deg: float
    forbidden_plate_on_destination: bool | None = None
    illumination_completed: bool | None = None


class FinalState:
    def __init__(self, binding):
        if not isinstance(binding, Binding): raise ValueError('Scene binding required')
        self.binding=binding; self.last=None; self.met=False; self.ever_met=False
        self.first_met_step=None

    def observe(self, frame, *, process_complete=False):
        if type(process_complete) is not bool: raise ValueError('Measured process completion required')
        if not isinstance(frame, PlacementFrame) or frame.binding != self.binding:
            raise ValueError('Wrong final-state episode/task/target/tray')
        if (type(frame.step) is not int or frame.step < 0
                or (self.last is not None and frame.step <= self.last.step)
                or type(frame.time_s) not in (int,float) or not math.isfinite(frame.time_s)
                or abs(frame.time_s-frame.step*.05)>1e-6):
            raise ValueError('Ordered current 20 Hz evidence required')
        for key in ('supported_by_destination','whole_inside_destination','robot_contact','ground_contact'):
            if type(getattr(frame,key)) is not bool: raise ValueError('Measured boolean required: '+key)
        p,q=frame.position_in_destination_m,frame.quaternion_in_destination_wxyz
        if (not isinstance(p,tuple) or len(p)!=3 or not isinstance(q,tuple) or len(q)!=4
                or any(type(x) not in (int,float) or not math.isfinite(x) for x in (*p,*q,frame.tilt_deg))
                or abs(math.hypot(*q)-1)>1e-4 or not 0<=frame.tilt_deg<=180):
            raise ValueError('Finite measured tray-local pose and tilt required')
        forbidden=frame.forbidden_plate_on_destination
        if forbidden is not None and type(forbidden) is not bool: raise ValueError('Measured plate exclusion required')
        if self.binding.kind=='microwave_box_to_meal_tray' and forbidden is None:
            raise ValueError('Explicit small-plate presence/absence evidence required')
        met=process_complete and frame.supported_by_destination and frame.whole_inside_destination
        if self.binding.kind=='microwave_box_to_meal_tray':
            if type(frame.illumination_completed) is not bool:
                raise ValueError('Explicit completed illumination evidence required')
            met=met and not forbidden and frame.illumination_completed
        self.last=frame; self.met=bool(met)
        if met and not self.ever_met: self.first_met_step=frame.step
        self.ever_met |= met
        return self.summary()

    def qualify_process(self, complete):
        """Commit same-frame process completion after atomic evidence validation."""
        if type(complete) is not bool or self.last is None:
            raise ValueError('Validated current frame and process evidence required')
        f=self.last
        self.met=bool(complete and f.supported_by_destination and f.whole_inside_destination)
        if self.binding.kind=='microwave_box_to_meal_tray':
            self.met=self.met and f.illumination_completed and not f.forbidden_plate_on_destination
        if self.met and not self.ever_met:self.first_met_step=f.step
        self.ever_met |= self.met
        return self.summary()

    def invalidate(self):
        # Unobserved/inactive time must not be passed off as a current final state.
        self.met=False

    def summary(self):
        maximum=POINTS.get(self.binding.kind)
        return dict(schema='openend.meal_tray.final_state.v2',condition_met=self.met,
            score=(maximum if self.met else 0) if maximum is not None else None,
            maximum_score=maximum,points_approved=maximum is not None,
            ever_met=self.ever_met,first_met_step=self.first_met_step,
            last_observed_step=None if self.last is None else self.last.step,
            depends_on_operations=True,
            required_preparation=('completed_illumination' if self.binding.kind=='microwave_box_to_meal_tray' else None),hold_required_s=0,
            runtime_registered=False,native_physics_verified=False)
