"""Public failure receipts for unsupported rules, inputs and scene bindings."""
from copy import deepcopy


class TaskError(ValueError):
    def __init__(self, message, *, kind="input", reason="invalid_request"):
        super().__init__(message)
        self.kind, self.reason = kind, reason


def failure_result(plan, *, episode_id=None, layout_id=None, time_s=None,
                   status="unavailable", kind=None, reason=None, message=None):
    detail = plan.get("unavailable") or {}
    return dict(schema="openend.short_task.result.draft.v1",
                task_name=plan["task_name"], episode_id=episode_id,
                actual_layout_id=layout_id, rule_version=plan["rule_version"],
                status=status, stages=[deepcopy(r) if isinstance(r, dict) else
                    dict(name="initial_requirement", description=r)
                    for r in plan.get("initial_requirements", [])],
                targets=[], scores={k: None for k in ("process", "current_final", "deductions", "total", "maximum")},
                success=None, termination_reason=reason or detail.get("reason", "task_unavailable"),
                evidence_time_s=time_s,
                error=dict(kind=kind or detail.get("kind", "rules"),
                           message=message or detail.get("message", "Task variant is unavailable")))
