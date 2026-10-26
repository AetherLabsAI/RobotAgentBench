"""One task's immutable result and simulation-time progress deadline.

The native scene owns physics and facts. This class never creates assets,
steps physics or infers success from a score total. An explicit evaluator
callback settles outstanding deductions before a terminal receipt is frozen.
"""
from copy import deepcopy
import math

NO_PROGRESS_TIMEOUT_S = 60.


class SingleTaskEpisode:
    def __init__(self, *, task_name, episode_id, actual_layout_id, rule_version,
                 start_time_s, target_roles, maximum_score=None, score_arithmetic="gross_minus_deductions"):
        from short_tasks.registry import TASK_NAMES
        if task_name not in TASK_NAMES:
            raise ValueError("Unknown Short Task")
        if maximum_score is None and task_name != "prepare_coffee":
            raise ValueError("Only resolved coffee rules can produce a scored episode; baking remains unavailable")
        maximum = 16 if maximum_score is None else maximum_score
        if type(maximum) not in (int, float) or not math.isfinite(maximum) or maximum <= 0:
            raise ValueError("A frozen adapter must supply its resolved maximum")
        if score_arithmetic not in ("gross_minus_deductions", "component_defined"):
            raise ValueError("Explicit supported component arithmetic required")
        self.maximum, self.score_arithmetic = maximum, score_arithmetic
        if not isinstance(episode_id, str) or not episode_id or not isinstance(rule_version, str) or not rule_version:
            raise ValueError("Episode identity and rule version required")
        if type(actual_layout_id) is not int or not 1 <= actual_layout_id <= 50:
            raise ValueError("Actual layout must be bound")
        self._time(start_time_s)
        self.task_name, self.episode_id = task_name, episode_id
        self.layout, self.rule_version = actual_layout_id, rule_version
        self.target_roles = deepcopy(target_roles)
        self.last_time = self.start_time = self.progress_time = float(start_time_s)
        self.best_net_score = 0.
        self.latest = None
        self.terminal = None

    @staticmethod
    def _time(value):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Finite nonnegative simulation time required")

    def _snapshot(self, snapshot, *, settlement=False):
        if not isinstance(snapshot, dict) or type(snapshot.get("success")) is not bool:
            raise ValueError("Bound evaluator summary with explicit success required")
        if snapshot.get("task_name") != self.task_name:
            raise ValueError("Only the episode's unique top-level task may score")
        for key, expected in (("episode_id", self.episode_id), ("rule_version", self.rule_version)):
            if (settlement or key in snapshot) and snapshot.get(key) != expected:
                raise ValueError("Evaluator summary changed bound " + key)
        scores = snapshot.get("scores")
        if not isinstance(scores, dict) or set(scores) != {"process", "current_final", "deductions", "total", "maximum"}:
            raise ValueError("Explicit component score channels required")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in scores.values()):
            raise ValueError("Unresolved rules cannot produce benchmark progress")
        if any(v < 0 for v in scores.values()) or scores["maximum"] != self.maximum:
            raise ValueError("Resolved nonnegative score channels and pinned maximum required")
        if scores["process"] + scores["current_final"] > scores["maximum"] + 1e-9:
            raise ValueError("Awarded points exceed the pinned task maximum")
        if scores["total"] > self.maximum + 1e-9:
            raise ValueError("Total exceeds the pinned task maximum")
        if self.score_arithmetic == "gross_minus_deductions" and not math.isclose(scores["total"], max(0., scores["process"] + scores["current_final"] - scores["deductions"]),
                            rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError("Score total differs from awarded points minus deductions")
        return scores

    def observe(self, snapshot, *, simulation_time_s, finalizer=None):
        if self.terminal is not None:
            raise RuntimeError("A finalized episode cannot accept another action")
        self._time(simulation_time_s)
        if simulation_time_s <= self.last_time:
            raise ValueError("Evidence must advance the simulation clock")
        scores = self._snapshot(snapshot)
        if finalizer is not None and not callable(finalizer):
            raise ValueError("Evaluator finalizer must be callable")
        now = float(simulation_time_s)
        # A frame on the deadline may supply genuinely new progress. Evidence
        # after it cannot retrospectively revive an already expired episode.
        expired_before_frame = now > self.progress_time + NO_PROGRESS_TIMEOUT_S + 1e-9
        previous = (self.last_time, self.latest, self.best_net_score, self.progress_time)
        self.last_time, self.latest = now, deepcopy(snapshot)
        try:
            if expired_before_frame:
                return self.finish("no_new_net_progress_timeout", finalizer=finalizer)
            # Re-gaining revoked terminal points never exceeds this watermark.
            # Deductions remain in net score; no stage switch clears the watermark.
            if scores["total"] > self.best_net_score:
                self.best_net_score = float(scores["total"])
                self.progress_time = now
            if snapshot["success"]:
                return self.finish("success", finalizer=finalizer)
            if now >= self.progress_time + NO_PROGRESS_TIMEOUT_S - 1e-9:
                return self.finish("no_new_net_progress_timeout", finalizer=finalizer)
            return self.result()
        except BaseException:
            # A rejected callback cannot consume the public clock or manufacture
            # progress. The native owner separately reports its execution error;
            # this rollback does not attempt to undo the evaluator callback.
            self.last_time, self.latest, self.best_net_score, self.progress_time = previous
            raise

    def result(self):
        if self.terminal is not None:
            return deepcopy(self.terminal)
        snapshot = self.latest or {}
        scores = snapshot.get("scores", dict(process=0., current_final=0., deductions=0., total=0., maximum=self.maximum))
        result = dict(schema="openend.short_task.result.draft.v1", task_name=self.task_name,
                    episode_id=self.episode_id, actual_layout_id=self.layout,
                    rule_version=self.rule_version, status="running",
                    stages=deepcopy(snapshot.get("stages", snapshot.get("phase_results", []))),
                    targets=deepcopy(snapshot.get("targets", [dict(roles=deepcopy(self.target_roles),
                                  current=deepcopy(snapshot.get("current_result", {})))])),
                    scores=deepcopy(scores), success=False,
                    termination_reason=None, evidence_time_s=self.last_time)
        if "deduction_policy" in snapshot:
            result["deduction_policy"] = deepcopy(snapshot["deduction_policy"])
        return result

    def timer(self):
        return dict(timeout_s=NO_PROGRESS_TIMEOUT_S, last_new_progress_s=self.progress_time,
                    best_net_score=self.best_net_score,
                    no_new_progress_s=self.last_time-self.progress_time,
                    remaining_s=max(0., self.progress_time+NO_PROGRESS_TIMEOUT_S-self.last_time),
                    policy="episode_high_watermark_net_score_v1")

    def finish(self, reason, *, finalizer=None):
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Episode termination reason required")
        if self.terminal is not None:
            if self.terminal["termination_reason"] != reason:
                raise ValueError("Contradictory terminal result")
            return self.result()
        if reason == "success" and (self.latest is None or not self.latest["success"]):
            raise ValueError("Success requires the latest evaluator's current evidence")
        if self.latest is None:
            raise ValueError("No evaluated evidence to finalize")
        if finalizer is not None:
            if not callable(finalizer):
                raise ValueError("Evaluator finalizer must be callable")
            final = finalizer(reason)
            settled = self._snapshot(final, settlement=True)
            previous = self.latest["scores"]
            # Settlement is a read-only evaluator closeout at the same measured
            # frame. It cannot inject a fresh action, milestone or score reward.
            for key in ("process", "current_final", "maximum"):
                if settled[key] != previous[key]:
                    raise ValueError("Final settlement cannot change awarded points or maximum")
            if settled["deductions"] < previous["deductions"] or settled["total"] > previous["total"] + 1e-9:
                raise ValueError("Final settlement cannot revoke deductions or add net points")
            for key in ("evidence_step", "evidence_time_s"):
                if key in self.latest and final.get(key) != self.latest[key]:
                    raise ValueError("Final settlement must retain the measured evidence clock")
            if final["success"] != (reason == "success"):
                raise ValueError("Final evaluator success differs from episode termination")
            if final.get("termination") != reason:
                raise ValueError("Evaluator did not finalize the requested episode termination")
            snapshot = deepcopy(final)
        else:
            snapshot = self.latest
        # The closeout cannot extend the progress deadline, even when source
        # bean-count settlement changes the displayed final deduction total.
        previous_latest = self.latest
        self.latest = snapshot
        try:
            result = self.result()
        except BaseException:
            self.latest = previous_latest
            raise
        result.update(status="finished", success=reason == "success", termination_reason=reason)
        self.terminal = deepcopy(result)
        return self.result()
