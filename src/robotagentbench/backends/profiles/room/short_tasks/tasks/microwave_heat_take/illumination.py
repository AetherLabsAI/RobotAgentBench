"""Reuse the original native illumination gate with standalone task identity."""
import math
from openend.tasks.meal_tray.microwave_illumination import IlluminationGate
from openend.tasks.meal_tray.rotary_component.rotary_return_runtime import RotaryReturnRuntime
from .registration import HeatBinding


class HeatGate(IlluminationGate):
    def __init__(self, binding, runtime, *, initial_time_s):
        if not isinstance(binding, HeatBinding) or not isinstance(runtime, RotaryReturnRuntime):
            raise ValueError("Standalone binding and original measured rotary runtime required")
        if (runtime.cycle.state != "idle" or runtime.cycle.events or runtime.cycle.last_time != -1
                or runtime.lamp is None or not runtime.cycle.require_return_home):
            raise ValueError("Fresh, uncompleted original mechanism required")
        if type(initial_time_s) not in (float, int) or not math.isfinite(initial_time_s) or initial_time_s < 0:
            raise ValueError("Actual native initial time required")
        self.binding, self.runtime = binding, runtime
        self.origin = self.last_time = float(initial_time_s)
        self.seen_running_lit = self.completed = self.failed = False
        self.last_result = None
        self._check_light(False)

    def bind_frame(self, frame):
        raise ValueError("This standalone gate has no meal-tray placement frame")
