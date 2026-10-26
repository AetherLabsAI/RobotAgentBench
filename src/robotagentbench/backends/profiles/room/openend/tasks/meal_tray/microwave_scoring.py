"""Approved 2 illumination + 8 extraction + 3 current destination points.

New typed facts explicitly distinguish box grasp from plate-supported carrying.
The old plate-only frame is never populated with invented box-as-plate facts.
"""
from dataclasses import dataclass
from copy import deepcopy
import math
from evaluation.tasks.microwave_take import process_rules as old
from .final_state import Binding, FinalState


@dataclass(frozen=True)
class ExtractionSample:
    binding: Binding
    time_s: float
    physics_dt_s: float
    illumination_completed: bool
    door_angle_deg: float
    box_front_clearance_m: float
    box_grasped: bool
    box_held: bool
    box_lifted: bool
    plate_present: bool
    plate_front_clearance_m: float | None
    plate_grasped: bool
    plate_held: bool
    plate_lifted: bool
    box_on_plate: bool
    contacts: tuple
    measurements_complete: bool


class MicrowaveTask:
    def __init__(self,binding,walls):
        if binding.kind!='microwave_box_to_meal_tray':raise ValueError('Microwave binding required')
        if len(walls)!=5 or {w.collider for w in walls}!=set(old.INNER_WALL_FACES):
            raise ValueError('All actual cavity walls required')
        self.binding=binding;self.walls={w.collider:w for w in walls}
        self.final=FinalState(binding);self.awards={};self.last_time=None
        self.collision_active=False;self.collisions=0
        self.hold_start=None;self.hold_satisfied=False;self.hold_deduction=0
        self.operation_closed=False;self.ended=False;self.events=[];self.illumination=False

    def _sample(self,s):
        if not isinstance(s,ExtractionSample) or s.binding!=self.binding or s.measurements_complete is not True:
            raise ValueError('Complete measured extraction facts required')
        numbers=[s.time_s,s.physics_dt_s,s.door_angle_deg,s.box_front_clearance_m]
        if s.plate_present:numbers.append(s.plate_front_clearance_m)
        if any(type(x) not in (int,float) or not math.isfinite(x) for x in numbers):
            raise ValueError('Finite physical measurements required')
        if (s.time_s<0 or not any(abs(s.physics_dt_s-1/h)<1e-10 for h in (60,240))
                or self.last_time is not None and s.time_s<=self.last_time):
            raise ValueError('Ordered real 60/240 Hz evidence required')
        flags=(s.illumination_completed,s.box_grasped,s.box_held,s.box_lifted,
               s.plate_present,s.plate_grasped,s.plate_held,s.plate_lifted,s.box_on_plate)
        if any(type(v) is not bool for v in flags):raise ValueError('Measured boolean required')
        if not s.plate_present and (s.plate_front_clearance_m is not None or any(
                (s.plate_grasped,s.plate_held,s.plate_lifted,s.box_on_plate))):
            raise ValueError('Absent plate cannot provide grasp or support evidence')
        if not isinstance(s.contacts,tuple):raise ValueError('Complete contact stream required')
        for p in s.contacts:
            if not isinstance(p,old.ContactPoint):raise ValueError('Actual cavity contact points required')
            p.validate()
        if not self.operation_closed and self.last_time is not None and abs(s.time_s-self.last_time-s.physics_dt_s)>1e-6:
            self.hold_start=None;self.hold_satisfied=False
        self.last_time=s.time_s;self.illumination=s.illumination_completed
        # No rewards, deferred stage flags or positive progress before lamp-off.
        if not s.illumination_completed:
            self.hold_start=None;self.hold_satisfied=False
            return
        hits=[p for p in s.contacts if old.interior_contact(p,self.walls,self.binding.target)]
        impact=any(p.impulse_norm_ns>old.IMPULSE_EPSILON_NS for p in hits)
        if impact and not self.collision_active:
            self.collisions+=1;self.collision_active=True
        elif not hits:self.collision_active=False
        if self.operation_closed:return
        direct=s.box_held and s.box_lifted
        via_plate=s.plate_present and s.plate_held and s.plate_lifted and s.box_on_plate
        outside=s.box_front_clearance_m>0
        carrying=outside and (direct or via_plate and s.plate_front_clearance_m>0)
        candidates=(('open',s.door_angle_deg>=old.OPEN_DEG,3),
            ('grasp',s.box_grasped or s.plate_present and s.plate_grasped and s.box_on_plate,1),
            ('take_out',carrying,3),('close',carrying and abs(s.door_angle_deg)<=2.,1))
        for name,met,points in candidates:
            if name in self.awards:continue
            if met:
                self.awards[name]=points
                self.events.append(dict(stage=name,time_s=s.time_s,points=points))
            break
        if 'close' in self.awards and abs(s.door_angle_deg)<=2. and carrying:
            if self.hold_start is None:self.hold_start=s.time_s
            self.hold_satisfied=s.time_s-self.hold_start>=old.HOLD_SECONDS-1e-9
        else:self.hold_start=None;self.hold_satisfied=False
        # This is the extraction component's actual successful handoff. Later
        # correct placement must not retroactively fail its finished hold.
        if self.hold_satisfied:self._close_operations()

    def observe(self,placement,samples=()):
        if self.ended:raise RuntimeError('Task already settled')
        # No partial score commit after malformed final/physics evidence.
        other=deepcopy(self)
        other.final.observe(placement,process_complete=other.hold_satisfied)
        if not isinstance(samples,tuple):raise ValueError('Dense sample tuple required')
        if samples:
            dt=samples[0].physics_dt_s
            if type(dt) not in (float,int) or not any(abs(dt-1/h)<1e-10 for h in (60,240)):
                raise ValueError('Unsupported sample dt')
            if len(samples)!=round(.05/dt):raise ValueError('Full selected control interval required')
            for n,s in enumerate(samples):
                if abs(s.physics_dt_s-dt)>1e-10 or abs(s.time_s-(placement.time_s-.05+(n+1)*dt))>1e-6:
                    raise ValueError('Dense samples must match the current final-state frame')
                # Correct placement is not an old held-in-air failure. Settle
                # the inherited incomplete-hold deduction once before release.
                other._sample(s)
            if samples[-1].illumination_completed!=placement.illumination_completed:
                raise ValueError('Mismatched lamp and placement evidence')
        elif placement.step!=0:
            # Settlement reads a new current placement, not new operation data.
            if placement.illumination_completed!=other.illumination:
                raise ValueError('Settlement cannot invent completed illumination')
        other.illumination=placement.illumination_completed
        # Re-evaluate the same validated boundary after this interval's process.
        other.final.qualify_process(bool(other.hold_satisfied))
        self.__dict__.update(other.__dict__)
        return self.summary()

    def _close_operations(self):
        if self.operation_closed:return
        if 'close' in self.awards and not self.hold_satisfied:self.hold_deduction=old.HOLD_FAILURE_PENALTY
        self.operation_closed=True

    def settle_current(self,placement,gate):
        # Device mechanics may finish while the scorer is asleep. Settlement
        # must use that real current gate, not the last selected lamp flag.
        from .microwave_illumination import IlluminationGate
        if not isinstance(gate,IlluminationGate):raise ValueError('Real illumination gate required')
        frame=gate.bind_frame(placement)
        other=deepcopy(self);other.final.observe(frame,process_complete=other.hold_satisfied)
        other.illumination=frame.illumination_completed
        if other.final.met:other._close_operations()
        self.__dict__.update(other.__dict__)
        return self.summary()

    def pause(self):
        self.final.invalidate()
        if not self.operation_closed:self.hold_start=None;self.hold_satisfied=False

    def finalize(self,reason):
        if not self.ended:self._close_operations();self.ended=True
        return self.summary()

    def summary(self):
        operation=max(0,sum(self.awards.values())-min(self.collisions,old.COLLISION_CAP)-self.hold_deduction)
        final=self.final.summary()
        illumination=2 if self.illumination else 0
        return dict(score=operation+illumination+final['score'] if self.illumination else 0,maximum_score=13,
            operation_score=operation+illumination if self.illumination else 0,final_state=final,
            illumination_score=illumination,extraction_score=operation if self.illumination else 0,
            operation_complete=bool(self.illumination and self.hold_satisfied),
            illumination_completed=self.illumination,awards=dict(self.awards),
            hold_satisfied=self.hold_satisfied,hold_deduction=self.hold_deduction,
            collision_deduction=min(self.collisions,old.COLLISION_CAP),
            success=bool(self.illumination and len(self.awards)==4 and self.hold_satisfied and self.final.met),
            ended=self.ended,runtime_registered=False,native_physics_verified=False)
