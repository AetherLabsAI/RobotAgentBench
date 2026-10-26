"""Original physical wipe adapter plus exact brush interaction evidence."""
import json
import numpy as np

from openend.tasks.wipe_table.port import WipePort
from openend.tasks.wipe_table.process_rules import StepFacts
from openend.tasks.wipe_table import stain as st
from .interaction import BrushInteraction
from .process_rules import ShortWipeScorer


class ShortWipePort(WipePort):
    def __init__(self, stage, manifest, output, episode_id, instance_id, parameters):
        super().__init__(stage, manifest, output, episode_id, instance_id, parameters)
        try:
            self.interaction = BrushInteraction(stage, episode_id,
                                                 self.output/"interaction_progress.json")
            self._admitted = False
        except BaseException:
            self.close()
            raise

    def initialize(self, clock, view):
        super().initialize(clock, view)
        self.interaction.initialize(view)

    def after_physics(self, helper):
        from openend.tasks.all_objects_integration.selective_read import contact_report
        super().after_physics(helper)
        self.interaction.after_physics(contact_report(self, helper), admitted=self._admitted)

    def _summary(self):
        return dict(super()._summary(), timeout_owner="short_task")

    def accept_initial(self):
        receipt = super().accept_initial()
        source = self.composite
        sample, _ = self.interaction.measured_sample(self.clock.current_time, baseline=True)
        scorer = ShortWipeScorer(self.episode_id, self.latest["dt"],
                                 self.latest["xy"], self.latest["yaw"])
        scorer.observe_step(StepFacts((source.last,), 1., 1., False, 0), sample, 0)
        self.composite = scorer
        self._admitted = True
        self.score = self._summary()
        return receipt

    def consume_control(self, active):
        from pxr import Usd
        if type(active) is not bool or not self.pending:
            raise RuntimeError("One complete measured control interval required")
        # The sole task must observe approach before contact wakes it. This does
        # not activate wiping: only the original physical grasp latch can do so.
        rows, self.pending = self.pending, []
        if any(abs(r["dt"]-rows[0]["dt"]) > 1e-12 for r in rows):
            raise ValueError("Native rate changed inside a control interval")
        eligible = [r for r in rows if r["activated"]]
        newly = (self.stain.update(np.logical_or.reduce([r["covered"] for r in eligible]))
                 if eligible else np.empty(0, dtype=int))
        if len(newly):
            with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
                self.opacity.Set(self.stain.vertex_opacity(self.base_opacity).tolist())
        self.composite.configure_clock(rows[0]["dt"])
        sample, rebase = self.interaction.measured_sample(rows[-1]["time"])
        facts = StepFacts(tuple(self._frame(r) for r in rows), self.stain.coverage(),
            self.stain.initial_coverage,
            any(r["contact"] and r["height"] <= st.FACE_HEIGHT_TOLERANCE_M for r in eligible),
            int(len(newly)))
        self.index += 1
        self.composite.observe_step(facts, sample, self.index, rebase=rebase)
        self.score = self._summary()
        self.trace.write(json.dumps(dict(time_s=rows[-1]["time"],
            coverage=self.stain.coverage(), passes=self.stain.passes.tolist(),
            newly_passed_cells=newly.tolist(), score=self.score))+"\n")
        self.trace.flush()
        return self.score

    def discard_control(self):
        for row in self.pending:
            self.composite.suspend_substep(row["time"], row["dt"])
        self.pending = []
        self.interaction.pause()

    def pause(self):
        super().pause()
        self.interaction.pause()
