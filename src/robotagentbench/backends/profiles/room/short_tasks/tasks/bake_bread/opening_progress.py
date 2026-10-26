"""Task-only redistribution of the first opening's existing two points.

No simulation writes; validated physical facts and existing phase predicates
remain authoritative. The inherited observe() commits evidence transactionally.
"""
from copy import deepcopy

from openend.tasks.scores import FryerTask

OPENING_POLICY = "short_tasks.bake_bread.first_opening_2cm_v1"
OPENING_THRESHOLDS_M = (.02, .04, .06, .08, .10)
POINTS_PER_LEVEL = .4


class IncrementalFryerTask(FryerTask):
    def __init__(self, binding, bread_variant):
        super().__init__(binding, bread_variant)
        self.opening_events = []
        self.maximum_first_opening_m = 0.

    def _observe(self, facts):
        first_phase = self.index == 0
        super()._observe(facts)
        if not first_phase or self.failure is not None:
            return
        self.maximum_first_opening_m = max(self.maximum_first_opening_m, facts.tray.opening_m)
        for level, threshold in enumerate(OPENING_THRESHOLDS_M, 1):
            if level > len(self.opening_events) and facts.tray.opening_m >= threshold:
                self.opening_events.append(dict(level=level, threshold_m=threshold,
                    points=POINTS_PER_LEVEL, step=facts.step, time_s=facts.time_s,
                    observed_opening_m=facts.tray.opening_m))

    def summary(self):
        result = deepcopy(super().summary())
        first = result["phase_results"][0]
        stage = first["stage_points"]["1"]
        earned = round(len(self.opening_events) * POINTS_PER_LEVEL, 10)
        delta = earned - (stage["points"] if stage["awarded"] else 0.)
        first["points_awarded"] += delta
        first["raw_score"] += delta
        first["score"] = max(0., first["raw_score"])
        stage["earned_points"] = earned
        first["opening_progress"] = dict(policy=OPENING_POLICY,
            thresholds_m=list(OPENING_THRESHOLDS_M), points_per_threshold=POINTS_PER_LEVEL,
            earned_points=earned, maximum_points=2.,
            maximum_observed_opening_m=self.maximum_first_opening_m,
            events=deepcopy(self.opening_events),
            phase_completion_predicate_unchanged=True)
        result["score"] = max(0., sum(p["raw_score"] for p in result["phase_results"]))
        return result
