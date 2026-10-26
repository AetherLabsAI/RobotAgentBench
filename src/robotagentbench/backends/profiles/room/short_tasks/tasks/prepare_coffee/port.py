"""Existing native reader, with only the Short Task's preparation evaluator."""
import json
from openend.room import CoffeePort
from evaluation.tasks.coffee_machine.openended_composite_v1.composite import Frame
from .evaluator import PrepareCoffeeEvaluator


class CoffeeShortPort(CoffeePort):
    def __init__(self, *args, **kwargs):
        try:
            super().__init__(*args, **kwargs)
        except Exception:
            if getattr(self, "owner", None) is not None:
                self.owner.close()
            if getattr(self, "trace", None) is not None:
                self.trace.close()
            raise

    def accept_initial(self):
        receipt = super().accept_initial()
        self.composite = PrepareCoffeeEvaluator(self.binding, evidence_source="native_physx")
        self.score = self.composite.observe(Frame(self.binding, self.owner.service.facts(0, "1-1")))
        self.index = 0
        self.owner.service._clear_window()
        return receipt

    def observe(self):
        service = self.owner.service
        if service.latest["sample"].time_s != self.clock.current_time:
            raise RuntimeError("Stale native coffee evidence")
        self.index += 1
        facts = service.facts(self.index, self.composite.facts_phase)
        self.score = self.composite.observe(Frame(self.binding, facts))
        self.trace.write(json.dumps(dict(facts=facts.as_dict(), score=self.score,
                                        simulation_time_s=self.clock.current_time), default=list)+"\n")
        self.trace.flush()
        service._clear_window()
        return self.score

    def pause(self):
        self.composite.pause()
        self.owner.service._clear_window()

    def finalize(self, reason):
        self.score = self.composite.finalize(reason)
        return self.score
