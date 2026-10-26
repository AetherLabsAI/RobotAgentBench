"""Task9 composition rules (room/physics dispatch not yet connected)."""

from ..process_rules import PutProgressTracker
from .fridge_rules import ProcessScorer as RefrigeratorScorer


class MicrowaveProgressTracker(PutProgressTracker):
    """Consume actual room facts without a closed-door/right-hand reset gate."""

    def __init__(self, *, episode_id, task_id, payload_variant,
                 payload_instance, control_dt_s):
        super().__init__(episode_id=episode_id, task_id=task_id,
            payload_variant=payload_variant, payload_instance=payload_instance,
            control_dt_s=control_dt_s, scene_managed_initial=True)
