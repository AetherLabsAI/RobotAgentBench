"""Existing native reader/executor with bake_bread's task-owned scorer."""
from openend.room import FryerPort
from evaluation.tasks.air_fryer.continuous_openended_v1.composite import Frame
from .opening_progress import IncrementalFryerTask


class BakeBreadPort(FryerPort):
    def accept_initial(self):
        if getattr(self, "meal_preparation_kind", None) is not None:
            raise ValueError("This port is exclusively for the bake_bread short task")
        self.initial = self.source.accept_initial()
        self.composite = IncrementalFryerTask(self.binding, self.variant)
        self.index = 0
        self.score = self.composite.observe(Frame(self.binding, self.source.selected_facts(0)))
        return self.initial
