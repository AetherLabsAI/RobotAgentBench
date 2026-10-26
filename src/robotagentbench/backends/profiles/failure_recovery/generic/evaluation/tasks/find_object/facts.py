"""Physical facts of one control frame: the only input of process_rules. No Isaac/USD imports.

The Isaac reader (isaac_evidence.py) fills these from live poses and PhysX contact reports; rules never read the simulator; the reader
never scores. Units: metres, seconds, degrees. DEFINITION_TABLE 实现绑定: "supported by X" = an actual contact with X whose contact force on
the object points upward (world z > 0, non-zero impulse) at a point not above the object's collision centre; never a height or footprint
test. "Grasped" = both finger pads of one gripper touch the object with opposing normals and the object kept its pose relative to that
hand for 0.10 s within 3 mm / 5 deg (public GraspDetector). "Released" = no robot part touches the object.
"""
from dataclasses import dataclass, asdict, field
import math

SUPPORT, DRAWER_FLOOR, DRAWER_WALL, DESTINATION, GROUND, FURNITURE, OBJECT, ROBOT = ('support', 'drawer_floor', 'drawer_wall', 'destination', 'ground', 'furniture', 'object', 'robot')
SURFACES = frozenset({SUPPORT, DRAWER_FLOOR, DRAWER_WALL, DESTINATION, GROUND, FURNITURE, OBJECT, ROBOT})


@dataclass(frozen=True)
class TargetFacts:
    position: tuple                   # world origin of the rigid body (x, y, z)
    quaternion: tuple                 # (w, x, y, z)
    pads_bilateral: bool              # both pads of one gripper touch it now with opposing normals
    grasped: bool                     # pads_bilateral held 0.10 s with a stable pose relative to that hand
    robot_contact: bool               # any robot part touches it
    supports: frozenset = frozenset() # SURFACES that push it upward now (ROBOT = resting on a robot link, not a grasp)
    ground_contact: bool = False      # any contact with the floor (sticky within the control frame)
    above_rim: bool = True            # drawer tasks: lowest point of the collision geometry above the drawer's current rim top
    in_destination: bool = False      # place: whole horizontal projection inside the destination top outline
    hull_min_z: float = 0.            # lowest collision vertex, world z

    @property
    def released(self):
        return not self.robot_contact


@dataclass(frozen=True)
class DistractorFacts:
    robot_touched: bool               # the robot touched it at any time since the start (sticky)
    ground_contact: bool              # it touches the floor now (sticky within the frame)


@dataclass(frozen=True)
class Facts:
    step: int
    time_s: float                     # simulation time since initial acceptance (control clock)
    target: TargetFacts
    distractors: dict = field(default_factory=dict)   # name -> DistractorFacts (initial-state movable non-target instances)
    tv_contact: bool = False          # 4.17: robot touches the television now
    drawer_rim_z: float = 0.          # drawer tasks: current rim top, world z

    def as_dict(self):
        d = asdict(self)
        d['target']['supports'] = sorted(d['target']['supports'])
        return d


def finite(*values):
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values)
