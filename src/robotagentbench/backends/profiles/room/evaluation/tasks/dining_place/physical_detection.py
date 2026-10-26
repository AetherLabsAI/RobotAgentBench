"""Measured-fact state machine for 10.2; no USD/control/physics writes."""
from dataclasses import asdict, dataclass
import math
from .process_rules import ProcessFrame, ProcessScorer
from arena_api.physics_defaults import PHYSICS_DT


@dataclass(frozen=True)
class DetectionConfig:
    # Reader debounce/resolution, not extra rewards or a 5-second task goal.
    support_confirm_s: float = .10
    unsupported_confirm_s: float = .025
    fall_distance_m: float = .003
    physics_dt_s: float = PHYSICS_DT

    def __post_init__(self):
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v <= 0
               for v in asdict(self).values()):
            raise ValueError('Positive finite detection settings required')


@dataclass(frozen=True)
class PhysicalFacts:
    step: int
    time_s: float
    plate_z_m: float
    payload_z_m: float
    payload_z_in_plate_m: float
    plate_grasped: bool
    plate_on_table: bool
    plate_in_region: bool
    payload_on_plate: bool
    payload_inside_plate: bool
    payload_above_plate: bool
    payload_grasped: bool
    payload_on_other_support: bool

    def __post_init__(self):
        if type(self.step) is not int or self.step < 0:
            raise ValueError('Invalid physical step')
        for key,value in asdict(self).items():
            if key in ('time_s','plate_z_m','payload_z_m','payload_z_in_plate_m'):
                if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value):
                    raise ValueError('Invalid physical measurement: '+key)
            elif key != 'step' and type(value) is not bool:
                raise ValueError('Missing physical fact: '+key)
        if self.time_s < 0:
            raise ValueError('Negative simulation time')


class PhysicalDetector:
    def __init__(self, episode_id, config=DetectionConfig()):
        self.config = config
        self.scorer = ProcessScorer(episode_id)
        self.previous = None
        self.plate_fall = self.payload_fall = None
        self.support_since = None
        self.events = []
        self.last = None
        self._closed = False

    def observe(self, f: PhysicalFacts):
        if self._closed:
            raise RuntimeError('Detector has been finalized')
        if not isinstance(f,PhysicalFacts):
            raise TypeError('Measured PhysicalFacts required')
        p = self.previous
        if f.step != (0 if p is None else p.step+1):
            raise ValueError('Missing/repeated physical sample')
        if p and abs(f.time_s-p.time_s-self.config.physics_dt_s)>1e-6:
            raise ValueError('Physical time gap; do not score incomplete evidence')
        plate_drop = box_drop = False
        # Pending flight starts at the last supported/grasped height, not the
        # first airborne point. Evaluate displacement BEFORE clearing on landing.
        if self.plate_fall is None and not f.plate_grasped and not (p and p.plate_on_table):
            if p and p.plate_grasped:
                self.plate_fall = (p.time_s,p.plate_z_m)
        if self.plate_fall is not None:
            started,z = self.plate_fall
            plate_drop = (z-f.plate_z_m >= self.config.fall_distance_m
                          and (f.time_s-started >= self.config.unsupported_confirm_s or f.plate_on_table))
            if plate_drop:
                self.events.append(dict(kind='plate_drop',step=f.step,time_s=f.time_s,fall_m=z-f.plate_z_m))
            if plate_drop or f.plate_grasped or f.plate_on_table:
                self.plate_fall = None

        if self.payload_fall is None and not f.payload_on_plate and not f.payload_grasped:
            if p and p.payload_on_plate:
                self.payload_fall = (p.time_s,p.payload_z_m,p.payload_z_in_plate_m)
        if self.payload_fall is not None:
            started,z,local_z = self.payload_fall
            off_plate = (not f.payload_inside_plate or not f.payload_above_plate or f.payload_on_other_support)
            falling = max(z-f.payload_z_m,local_z-f.payload_z_in_plate_m) >= self.config.fall_distance_m
            box_drop = (not f.payload_grasped and off_plate
                        and (falling or f.payload_on_other_support)
                        and (f.time_s-started >= self.config.unsupported_confirm_s or f.payload_on_other_support))
            if box_drop:
                self.events.append(dict(kind='payload_drop',step=f.step,time_s=f.time_s,fall_m=z-f.payload_z_m))
            if box_drop or f.payload_on_plate or f.payload_grasped:
                self.payload_fall = None

        bearing = f.plate_on_table and f.plate_in_region and not f.plate_grasped
        if not bearing:
            self.support_since = None
        elif self.support_since is None:
            self.support_since = f.time_s
        confirmed = bool(bearing and f.time_s-self.support_since+1e-9 >= self.config.support_confirm_s)
        self.scorer.observe(ProcessFrame(self.scorer.episode_id,f.step,f.time_s,
                            f.plate_in_region,confirmed,plate_drop,box_drop,True))
        self.previous = f
        self.last = dict(facts=asdict(f),table_support_confirmed=confirmed,
                         plate_drop_event=plate_drop,payload_drop_event=box_drop)
        return self.last

    def finalize(self):
        result = self.scorer.finalize().to_dict()
        self._closed = True
        return dict(result, detection_config=asdict(self.config),events=list(self.events),
                    task_success=None,automatic_termination=False)
