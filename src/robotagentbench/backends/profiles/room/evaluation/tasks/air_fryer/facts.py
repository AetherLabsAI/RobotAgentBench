"""Physical facts of one control frame: the only input of process_rules. No Isaac/USD imports.

The Isaac reader (isaac_evidence.py) fills these from live poses and PhysX contact reports. Rules never read the
simulator; the reader never scores. Units: metres, seconds, degrees.

"Supported by X" always means an actual contact with X that pushes the object upward (contact normal on the object
pointing up, non-zero impulse), never a height or footprint test. "Released" means no robot part touches the object.
"Grasped" means both finger pads of one gripper touch the object with opposing normals and the object has kept its pose
relative to that hand for 0.1 s (the public GraspDetector); nothing else needs to be true.
"""
from dataclasses import dataclass, asdict
import math

# support surfaces
BASKET_FLOOR, BASKET_WALL, PLATE, COUNTER, GROUND, FRYER_BODY, OTHER = 'basket_floor', 'basket_wall', 'plate', 'counter', 'ground', 'fryer_body', 'other'
SURFACES = frozenset({BASKET_FLOOR, BASKET_WALL, PLATE, COUNTER, GROUND, FRYER_BODY, OTHER})


@dataclass(frozen=True)
class CakeFacts:
    """One cake (the task cake or the other cake)."""
    position: tuple                 # world origin (x, y, z); the origin is on the cake's bottom face
    quaternion: tuple               # world orientation (w, x, y, z)
    tilt_deg: float                 # angle between the cake's up axis and world up
    pads_bilateral: bool            # both finger pads of one gripper touch it now
    grasped: bool                   # pads_bilateral held for 0.1 s with a stable pose relative to that hand
    robot_contact: bool             # any robot part touches it
    supports: frozenset = frozenset()   # subset of SURFACES that push it upward now
    ground_contact: bool = False    # any contact with the ground, supporting or not
    basket_contact: bool = False    # any contact with the basket (floor or walls), supporting or not
    inside_basket: bool = False     # whole collision shape inside the inner basket contour, between floor and wall top
    outside_basket: bool = False    # horizontal projection of the whole cake clear of the whole basket outline (handle included)
    displacement_m: float = 0.      # distance of the origin from its accepted initial position

    @property
    def released(self):
        return not self.robot_contact


@dataclass(frozen=True)
class TrayFacts:
    """The basket (drawer tray)."""
    position: tuple
    quaternion: tuple
    tilt_deg: float
    opening_m: float                # displacement along the drawer axis, in the fryer body frame (joint frame); 0 = closed
    body_contact: bool              # touches the fryer body (any contact)
    counter_supported: bool         # pushed upward by the counter top (contact normal z > 0.7)
    robot_contact: bool
    pads_bilateral: bool
    ground_contact: bool = False
    joint_released: bool = False    # 2.3: the rail joint has been released (stroke reached the release threshold)
    displacement_m: float = 0.


@dataclass(frozen=True)
class Facts:
    step: int
    time_s: float                   # simulation time since initial acceptance
    target: CakeFacts               # the task cake (target of 2.1 / 2.2; the basket cake of 2.3 / 2.5)
    other: CakeFacts                # the other cake (distractor of 2.1 / 2.2; the plate cake of 2.3 / 2.5)
    tray: TrayFacts

    def as_dict(self):
        d = asdict(self)
        for key in ('target', 'other'):
            d[key]['supports'] = sorted(d[key]['supports'])
        return d


def finite(*values):
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values)
