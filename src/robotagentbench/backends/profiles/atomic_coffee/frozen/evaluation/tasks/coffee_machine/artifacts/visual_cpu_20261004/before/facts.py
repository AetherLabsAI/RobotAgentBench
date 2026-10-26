"""Physical facts of one control frame: the only input of process_rules. No Isaac/USD imports.

The Isaac reader (isaac_evidence.py) fills these from live poses, PhysX contact reports, the button joint and the
liquid particles. Rules never read the simulator; the reader never scores. Units: metres, seconds, degrees, millilitres.

"Supported" always means an actual contact that pushes the object upward (contact normal pointing up on the object and
non-zero impulse), never a height or footprint test. "Grasped" means both finger pads of ONE gripper touch the object
and nothing outside the robot supports it.
"""
from dataclasses import dataclass, field, asdict
import math

# support surfaces a task object can rest on
COUNTER, TRAY, GROUND, MACHINE_BODY, OTHER = 'counter', 'tray', 'ground', 'machine_body', 'other'


@dataclass(frozen=True)
class ObjectFacts:
    """One rigid task object (cup or bean bag)."""
    position: tuple                 # world origin (x, y, z)
    tilt_deg: float                 # angle between the object's up axis and world up
    pads_bilateral: bool            # both finger pads of one gripper touch it
    robot_contact: bool             # any robot part touches it
    supports: frozenset = frozenset()    # subset of {COUNTER, TRAY, GROUND, MACHINE_BODY, OTHER} that push it upward now
    ground_contact: bool = False    # any contact with the ground, supporting or not
    machine_body_contact: bool = False   # contact with the coffee machine other than the normal bottom-on-tray support
    displacement_m: float = 0.      # distance of the origin from its accepted initial position

    @property
    def grasped(self):
        return self.pads_bilateral and not self.supports

    @property
    def released(self):
        return not self.robot_contact


@dataclass(frozen=True)
class CupPlacement:
    """Cup relative to the machine (only meaningful for the cup)."""
    xy_error_m: float               # horizontal distance cup origin -> CupPlacement anchor
    outlet_over_opening: bool       # vertical projection of the outlet falls inside the cup opening
    outside_bay: bool               # the whole cup collision geometry (body + handle) is beyond the bay exit plane


@dataclass(frozen=True)
class LiquidFacts:
    """Particle counts; one particle = ML_PER_PARTICLE millilitres."""
    emitted: int                    # particles that exist (1-3: emitted so far; 1-4: the pre-filled amount)
    in_cup: int                     # inside the cup cavity now
    airborne: int                   # outside the cavity and not yet settled on anything
    spilled: int                    # cumulative unique particles that settled outside the cup
    initial_in_cup: int = 0         # in the cup at initial acceptance (1-4 denominator)


@dataclass(frozen=True)
class ButtonFacts:
    displacement_m: float           # 0 = released, negative = pressed
    robot_on_button: bool           # a robot link touches the button collider now
    dispensing: bool                # machine state after this frame
    events: tuple = ()              # machine events since the previous control frame, in order:
                                    # 'started', 'stopped_by_button', 'stopped_no_beans', 'blocked_no_beans'
    event_by_robot: tuple = ()      # same length as events: the press that caused it was made by the robot


@dataclass(frozen=True)
class Facts:
    step: int
    time_s: float                   # simulation time since initial acceptance
    cup: ObjectFacts
    bag: ObjectFacts
    placement: CupPlacement
    robot_machine_contact: bool     # robot touches the machine; for 1-3 contact with the button itself is excluded
    beans_in_hopper: int = 0        # debounced (each bean continuously inside for 0.10 s)
    beans_total: int = 0
    liquid: LiquidFacts = None
    button: ButtonFacts = None

    def as_dict(self):
        d = asdict(self)
        for key in ('cup', 'bag'):
            d[key]['supports'] = sorted(d[key]['supports'])
        return d


ML_PER_PARTICLE = 0.004096


def finite(*values):
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)
