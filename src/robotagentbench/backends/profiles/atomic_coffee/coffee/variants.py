"""Approved coffee interference presets and sequencing, not a physics runner.

Component actors remain owned by their existing tasks. A future native owner
must call begin/complete with current, validated actor evidence; this module
does not synthesize contact, holding, motion, robot commands or score events.
"""
from dataclasses import asdict, dataclass
import math

BAG_MOVE = 'bag_move'
MACHINE_PUSH = 'machine_push'


@dataclass(frozen=True)
class Variant:
    name: str
    label: str
    component_order: tuple[str, ...]
    initial_owner: str
    process_score_instances: int = 1
    source_atomic_api: str = '1-2'
    runtime_registered: bool = False
    native_combination_verified: bool = False


_VARIANTS = (
    Variant('bag_only', 'Coffee bag relocation', (BAG_MOVE,),
            'coffee_bag_interference.joint_initialization'),
    Variant('machine_only', 'Coffee machine push', (MACHINE_PUSH,),
            'coffee_machine_push_interference.build_initial'),
    Variant('bag_then_machine', 'Coffee bag relocation + machine push',
            (BAG_MOVE, MACHINE_PUSH),
            'coffee_bag_interference.joint_initialization; dynamic machine opinion from machine-push variant'),
)
VARIANT_NAMES = tuple(item.name for item in _VARIANTS)


def get_variant(name):
    for item in _VARIANTS:
        if type(name) is str and item.name == name:
            return item
    raise ValueError('Explicit coffee variant required: ' + ', '.join(VARIANT_NAMES))


def describe_variant(name):
    value = asdict(get_variant(name))
    value.update(
        status=('combined_native_approach_probe_available_production_native_owner_and_scoring_pending'
                if name == 'bag_then_machine' else
                'existing_independent_component_native_owner_and_scoring_pending'),
        robot_goal='Pick up the coffee bag, pour beans into the current hopper, and return the bag.',
        score_owner='one original atomic 1-2 process scorer, not one scorer per interference',
        component_definitions={
            BAG_MOVE: 'short_task_robustness.tasks.coffee_bag_interference/definition.json',
            MACHINE_PUSH: 'short_task_robustness.tasks.coffee_machine_push_interference/definition.json',
        },
    )
    return value


class InterferenceSequence:
    """One-shot actor-order gate; no replacement for either physical trigger.

    Bag modes only allow robot process credit after validated bag relocation
    and human retirement. Machine eligibility still requires CURRENT hold,
    mouth/hopper and tilt evidence in its unchanged native trigger.
    """
    def __init__(self, variant):
        self.variant = get_variant(variant)
        self.active = None
        self.completed = []
        self.invalid_reason = None
        self.last_event_time_s = None

    @property
    def eligible_component(self):
        if self.invalid_reason or self.active is not None:
            return None
        order = self.variant.component_order
        return order[len(self.completed)] if len(self.completed) < len(order) else None

    @property
    def robot_process_scoring_enabled(self):
        return (self.invalid_reason is None and
                (BAG_MOVE not in self.variant.component_order or BAG_MOVE in self.completed))

    def _time(self, now):
        if (type(now) not in (int, float) or not math.isfinite(now) or now < 0
                or (self.last_event_time_s is not None and now < self.last_event_time_s)):
            raise ValueError('Finite nondecreasing native simulation event time required')

    def begin(self, component, *, simulation_time_s):
        self._time(simulation_time_s)
        if self.eligible_component is None or component != self.eligible_component:
            raise ValueError('Component disabled, already active/completed, or out of order')
        self.active = component
        self.last_event_time_s = simulation_time_s

    def complete(self, component, *, simulation_time_s,
                 native_completion_confirmed, actor_retired):
        self._time(simulation_time_s)
        if (self.invalid_reason or component not in self.variant.component_order
                or self.active != component):
            raise ValueError('Only the active admitted component can complete')
        if any(type(value) is not bool for value in (native_completion_confirmed, actor_retired)):
            raise ValueError('Explicit validated native completion and retirement facts required')
        self.last_event_time_s = simulation_time_s
        if not (native_completion_confirmed and actor_retired):
            return False
        self.completed.append(component)
        self.active = None
        return True

    def invalidate(self, component, *, simulation_time_s, reason):
        self._time(simulation_time_s)
        if self.invalid_reason or component not in (self.active, self.eligible_component) or component is None:
            raise ValueError('Cannot invalidate an unrelated or completed component')
        if type(reason) is not str or not reason.strip():
            raise ValueError('Explicit intervention-invalid reason required')
        self.invalid_reason = reason
        self.last_event_time_s = simulation_time_s
        # This is invalid environment intervention, never a robot task penalty.
