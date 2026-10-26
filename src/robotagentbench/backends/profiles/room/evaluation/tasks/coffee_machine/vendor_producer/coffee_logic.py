"""Device logic, independent of Isaac, task scoring and robot control (metres)."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Hopper:
    half_width: float = .1013
    half_depth: float = .0853
    corner_radius: float = .0168
    center_z: float = -.027
    floor_y: float = .315
    rim_y: float = .4445

    def contains(self, point):
        """Bean centre in the open rounded cavity, expressed in machine Y-up frame."""
        x, y, z = point
        if not all(math.isfinite(v) for v in point):
            return False
        if not self.floor_y < y < self.rim_y:
            return False
        dx = max(abs(x) - (self.half_width-self.corner_radius), 0.)
        dz = max(abs(z-self.center_z) - (self.half_depth-self.corner_radius), 0.)
        return dx*dx + dz*dz < self.corner_radius**2


class CoffeeController:
    def __init__(self, hopper, presence_seconds=.10):
        self.hopper = hopper
        self.presence_seconds = presence_seconds
        self.inside_seconds = {}
        self.pressed = False
        self.dispensing = False
        self.elapsed = 0.

    def update(self, dt, bean_positions_local, button_displacement_m):
        """Input mapping only contains registered real bean rigid bodies.

        Debounce each bean independently; loss of all beans stops immediately.
        Press below -3 mm, release above -1 mm; one event per full press.
        No bean teleportation, deletion, spawning or consumption occurs here.
        """
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be finite and positive')
        self.elapsed += dt
        raw = {key for key, pos in bean_positions_local.items() if self.hopper.contains(pos)}
        self.inside_seconds = {key: self.inside_seconds.get(key, 0.) + dt for key in raw}
        count = sum(t + 1e-9 >= self.presence_seconds for t in self.inside_seconds.values())
        edge = False
        if not math.isfinite(button_displacement_m):
            self.dispensing = False
            event = 'invalid_button_pose'
        else:
            if not self.pressed and button_displacement_m <= -.003:
                self.pressed = True
                edge = True
            elif self.pressed and button_displacement_m >= -.001:
                self.pressed = False
            event = None
            if self.dispensing and count == 0:
                self.dispensing = False
                event = 'stopped_no_beans'
            # A stop caused by bean loss cannot turn into a start on this tick.
            if edge and event is None:
                if self.dispensing:
                    self.dispensing = False
                    event = 'stopped_by_button'
                elif count:
                    self.dispensing = True
                    event = 'started'
                else:
                    event = 'blocked_no_beans'
        return dict(time_s=self.elapsed, raw_bean_count=len(raw), bean_count=count,
                    button_pressed=self.pressed, button_displacement_m=button_displacement_m,
                    dispensing=self.dispensing, event=event)
