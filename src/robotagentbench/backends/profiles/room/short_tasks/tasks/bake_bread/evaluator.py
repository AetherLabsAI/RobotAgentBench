"""Approved mechanical fryer task: insert, extract basket, pick up cake.

Heating is outside this short task. All manipulation thresholds and deductions
come from the existing FryerTask, except the approved first-opening point split.
This adapter never writes simulation state.
"""
from copy import deepcopy
import math

from evaluation.tasks.air_fryer.continuous_openended_v1.composite import Binding, Frame
from openend.tasks.scores import interrupt_hold
from .opening_progress import IncrementalFryerTask

RULE_VERSION = "openend.short_task.bake_bread.manipulation.v2"
MAXIMUM_SCORE = 12


def score_channels(result):
    if result.get("schema") != "openend.fryer.v1" or result.get("maximum_score") != 12:
        raise ValueError("Expected the existing twelve-point mechanical composite")
    phases = result["phase_results"]
    keys = [p["key"] for p in phases]
    if keys not in (["2.1"], ["2.1", "2.3"], ["2.1", "2.3", "2.5"],
                    ["2.2"], ["2.2", "2.3"], ["2.2", "2.3", "2.5"]):
        raise ValueError("Invalid ordered mechanical stages")
    values = [p[k] for p in phases for k in ("points_awarded", "deduction_total", "raw_score")]
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError("Finite measured score channels required")
    if any(p["points_awarded"] < 0 or p["deduction_total"] < 0 or
           abs(p["raw_score"] - p["points_awarded"] + p["deduction_total"]) > 1e-6 for p in phases):
        raise ValueError("Inconsistent mechanical score arithmetic")
    gross = sum(p["points_awarded"] for p in phases)
    total = max(0., sum(p["raw_score"] for p in phases))
    if not 0 <= gross <= MAXIMUM_SCORE or result["score"] != total:
        raise ValueError("Invalid mechanical score total")
    if type(result.get("success")) is not bool:
        raise ValueError("Typed mechanical success required")
    if result["success"] and (len(phases) != 3 or not all(p["success"] for p in phases)):
        raise ValueError("All three manipulation stages must complete")
    return dict(process=gross, current_final=0, deductions=gross-total,
                total=total, maximum=MAXIMUM_SCORE)


class BakeBreadEvaluator:
    task_name = "bake_bread"
    rule_version = RULE_VERSION
    available = True

    def __init__(self, binding, bread_variant="no_sugar", *, finish_on_small_plate=False):
        if not isinstance(binding, Binding):
            raise ValueError("A scene-owned air-fryer Binding is required")
        if type(finish_on_small_plate) is not bool or finish_on_small_plate:
            raise ValueError("Return-to-small-plate is outside this task")
        self.binding = binding
        self.bread_variant = bread_variant
        self.finish_on_small_plate = False
        self.component = IncrementalFryerTask(binding, bread_variant)

    @property
    def instruction(self):
        color = "白色" if self.bread_variant == "sugar" else "黄色"
        return (f"打开空气炸锅的炸篮，把盘中的{color}蛋糕放进去并推回；"
                "取出同一个炸篮放稳，再拿出目标蛋糕并按原规则端稳。"
                "不要求加热，也不要求放回小盘。")

    def observe(self, frame):
        if not isinstance(frame, Frame):
            raise ValueError("Typed air-fryer Frame required")
        self.component.observe(frame)
        return self.summary()

    def pause(self):
        if self.component.termination is None:
            interrupt_hold(self.component)

    def finalize(self, reason):
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Termination reason required")
        if reason == "success" and not self.component.success:
            raise ValueError("Cannot skip unfinished manipulation stages")
        if self.component.termination is None:
            self.component.finalize(reason)
        return self.summary()

    def component_summary(self):
        return self.component.summary()

    def summary(self):
        result = self.component_summary()
        return dict(task_name=self.task_name, rule_version=RULE_VERSION,
            instruction=self.instruction,
            status="completed" if result["success"] else "ended" if result["termination"] else "running",
            success=result["success"], termination_reason=result["termination"],
            scores=score_channels(result), stages=deepcopy(result["phase_results"]),
            manipulation_component=result, heating_required=False,
            target_category="cake", target_variant="white" if self.bread_variant == "sugar" else "yellow",
            finish_on_small_plate=False)

    def result(self, *, episode_id, actual_layout_id, evidence_time_s):
        if episode_id != self.binding.episode_id:
            raise ValueError("Result identity differs from the bound episode")
        if type(actual_layout_id) is not int or not 1 <= actual_layout_id <= 50:
            raise ValueError("Actual layout must be an integer in 1..50")
        if (type(evidence_time_s) not in (int, float) or not math.isfinite(evidence_time_s)
                or evidence_time_s < 0):
            raise ValueError("Finite nonnegative evidence time required")
        result = self.summary()
        result.update(schema="openend.short_task.result.draft.v1", episode_id=episode_id,
            actual_layout_id=actual_layout_id, evidence_time_s=evidence_time_s,
            targets=[dict(category="cake", instance_id=dict(self.binding.objects)["target"],
                          variant=result["target_variant"])])
        return result
