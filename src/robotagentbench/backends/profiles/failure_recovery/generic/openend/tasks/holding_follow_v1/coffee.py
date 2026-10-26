"""Openend coffee adapter: separate position-follow evidence from raw contact."""
from dataclasses import dataclass, asdict, replace
from evaluation.tasks.coffee_machine.facts import ObjectFacts
from evaluation.tasks.coffee_machine.openended_composite_v1.scene_service import SceneCoffeeService
from .detection import GraspDetector


@dataclass(frozen=True)
class FollowObjectFacts(ObjectFacts):
    following_confirmed: bool = False

    @property
    def grasped(self):
        return self.following_confirmed

    @property
    def released(self):
        return not self.robot_contact and not self.following_confirmed


class FollowCoffeeService(SceneCoffeeService):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.detector = GraspDetector(self.objects)
        self.measured_bilateral = {}

    def _contacts(self, *args, **kwargs):
        result = super()._contacts(*args, **kwargs)
        self.measured_bilateral = result[1]
        return result

    def facts(self, index, phase):
        facts = super().facts(index, phase)
        converted = {}
        for role in ('cup', 'bag'):
            path = self.roles[role]
            fields = asdict(getattr(facts, role))
            # Keep the actual contact flags; do not relabel following as touch.
            fields['pads_bilateral'] = bool(any(self.measured_bilateral[path].values()))
            converted[role] = FollowObjectFacts(**fields,
                following_confirmed=bool(self.latest['grasped'][path]))
        return replace(facts, **converted)
