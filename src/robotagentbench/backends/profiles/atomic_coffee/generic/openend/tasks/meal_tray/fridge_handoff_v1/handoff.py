"""Read-only same-instance handoff and illumination eligibility.

The old refrigerator composite retains its own scores and success criteria.
No successful score or movement is synthesized by this adapter.
"""
import math
from dataclasses import replace
from ..microwave_illumination import IlluminationGate


class FridgeHandoff:
    def __init__(self, parent, *, episode_id, target, variant, parent_id):
        from fridge_to_microwave_v1.composite import Composite
        from layout_task_activation_v1.task_scene_registration_v1.fridge_to_microwave_v1.composite import Composite as NativeComposite
        if not isinstance(parent,(Composite,NativeComposite)):
            raise ValueError('Actual fridge composite required, not a completion flag')
        if (parent.episode_id,parent.target_prim,parent.variant)!=(episode_id,target,variant):
            raise ValueError('Fridge handoff must retain episode, target and variant')
        self.parent=parent;self.parent_id=parent_id
        self.time_s=None;self.parent_step=None;self.supports=None

    @property
    def ready(self):return self.time_s is not None

    def accept(self,now,*,inside_oven,supports,door_closed):
        from fridge_to_microwave_v1.composite import Sample
        from layout_task_activation_v1.task_scene_registration_v1.fridge_to_microwave_v1.composite import Sample as NativeSample
        if type(now) not in (float,int) or not math.isfinite(now) or now<0:
            raise ValueError('Actual finite simulation time required')
        if self.ready:return False
        p=self.parent
        if not p.success:return False
        if (not isinstance(p.last,(Sample,NativeSample)) or p.last.source!='isaac_physics'
                or p.last.episode_id!=p.episode_id or p.last.target_prim!=p.target_prim
                or p.last.time_s>now+1e-6 or p.phase!='microwave'
                or p.put_result is None):
            raise ValueError('Fridge handoff lacks trusted completed put evidence')
        if type(inside_oven) is not bool or type(door_closed) is not bool or not isinstance(supports,frozenset):
            raise ValueError('Current measured cavity/support facts required')
        if not (inside_oven and supports and door_closed):return False
        self.time_s=float(now);self.parent_step=p.last.step_index;self.supports=supports
        return True

    def permits_touch(self,request):
        last=request.get('last_touch_s')
        return bool(self.ready and last is not None and last>self.time_s+1e-9)

    def summary(self):
        return dict(status='accepted' if self.ready else 'waiting_for_fridge_to_microwave',
            prerequisite_instance_id=self.parent_id,target_prim=self.parent.target_prim,
            accepted_at_simulation_s=self.time_s,parent_step=self.parent_step,
            scene_reset=False,object_recreated=False)


class HandoffIlluminationGate(IlluminationGate):
    """Preserve device operation; qualify only cycles begun after real handoff."""
    def __init__(self,binding,runtime,*,initial_time_s,handoff):
        super().__init__(binding,runtime,initial_time_s=initial_time_s)
        self.handoff=handoff

    @property
    def eligible_completed(self):
        c=self.runtime.cycle;h=self.handoff
        return bool(self.completed and not self.failed and h.ready
            and c.contact_since is not None and c.contact_since>=h.time_s
            and c.started_at is not None and c.started_at>=h.time_s)

    def bind_frame(self,frame):
        frame=super().bind_frame(frame)
        return replace(frame,illumination_completed=self.eligible_completed)

    def summary(self):
        result=super().summary()
        result.update(physical_cycle_completed=self.completed,
            downstream_scoring_enabled=self.eligible_completed,fridge_handoff=self.handoff.summary())
        return result
