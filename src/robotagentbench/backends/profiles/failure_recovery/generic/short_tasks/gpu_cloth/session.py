"""Unscored cloth preview over one caller-owned fixed-rate native World.

The public lifecycle matches ShortTaskSession. There is no benchmark success,
score ledger, progress timeout, action generator or second physics world.
"""
from copy import deepcopy
import json
import math
from numbers import Real
from pathlib import Path

from short_tasks.common.contracts import TaskError, failure_result
from .interface import EXECUTION_MODE, preview_result
from .native import NativeClothComponent, PHYSICS_HZ, PHYSICS_DT_S, author_component


class PreviewSession:
    def __init__(self, plan, manifest, component, *, episode_id):
        if (plan.get("execution_mode") != EXECUTION_MODE
                or plan.get("scoring_available") is not False or not plan.get("available")
                or not isinstance(episode_id, str) or not episode_id):
            raise TaskError("An explicitly unscored GPU cloth preview is required", kind="binding")
        self.plan, self.manifest = deepcopy(plan), deepcopy(manifest)
        self.component = self.native = component
        self.episode_id = episode_id
        self.clock = self.world = self.pending = self.terminal = None
        self.started = self.paused = self.closed = self.faulted = False
        self.control_index = self.substeps = 0
        self.last_boundary = self.last_control_boundary = None

    def _error(self, exc):
        if self.terminal is not None:
            return
        self.faulted = True
        current = None if self.world is None else float(self.world.current_time)
        value = failure_result(self.plan, episode_id=self.episode_id,
            layout_id=None, time_s=current, status="error",
            kind=getattr(exc, "kind", "execution"),
            reason=getattr(exc, "reason", "native_cloth_reader_failed"),
            message=str(exc) or type(exc).__name__)
        value.update(execution_mode=EXECUTION_MODE, scoring_available=False,
            layout_scope=self.plan.get("layout_scope", "isolated_component_preview"),
            native_state={}, evidence_step=self.control_index)
        self.terminal = value

    def _check_clock(self):
        if self.world is None or not math.isclose(self.world.get_physics_dt(), PHYSICS_DT_S,
                rel_tol=0, abs_tol=1e-12):
            raise TaskError("Preview physics remains fixed at 240 Hz", kind="binding", reason="gpu_cloth_backend_mismatch")
        if hasattr(self.clock, "hz") and self.clock.hz != PHYSICS_HZ:
            raise TaskError("Preview clock cannot switch CPU/GPU or rates", kind="binding", reason="gpu_cloth_backend_mismatch")

    def initialize(self, clock, view=None):
        if self.clock is not None or self.closed or self.faulted:
            raise TaskError("Initialize once after the caller's World reset", kind="execution")
        self.clock, self.world = clock, getattr(clock, "world", clock)
        try:
            self._check_clock()
            self.component.initialize(clock, view)
            self.last_boundary = self.component.last_boundary
        except Exception as exc:
            self._error(exc)
            raise

    def accept_initial(self):
        if self.started or self.clock is None or self.closed or self.faulted:
            raise TaskError("Accept the actual native initial state exactly once", kind="binding")
        try:
            receipt = self.component.accept_initial()
            if receipt.get("accepted") is not True:
                raise TaskError("A measured native cloth actor is required", kind="binding", reason="initial_state_unverified")
            self._running_result()
        except Exception as exc:
            self._error(exc)
            raise
        self.started = True
        self.last_control_boundary = self.last_boundary
        return dict(accepted=True, public_task_count=1, task_name=self.plan["task_name"],
            initialization_owner="gpu_cloth_scene_owner.v1", preview_admission_only=True,
            scoring_available=False, full_action_validation=False,
            internal_initial_evidence={"gpu_cloth":receipt})

    def begin_control(self, robot_state=None, *, command_vw=None, base_velocity=None):
        if (not self.started or self.closed or self.paused or self.faulted
                or self.pending is not None or self.terminal is not None):
            raise TaskError("No available preview control interval", kind="execution")
        if (command_vw is None) != (base_velocity is None):
            raise TaskError("Provide command and measured velocity together")
        if command_vw is not None and (len(command_vw) != 2 or len(base_velocity) != 6
                or any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)
                       for v in (*command_vw, *base_velocity))):
            raise TaskError("Finite actual commands and measured base velocity are required")
        try:
            self._check_clock()
            if self.component.boundary() != self.last_boundary:
                raise TaskError("The last native state must match the current World", kind="execution", reason="fresh_native_state_required")
        except Exception as exc:
            self._error(exc)
            raise
        self.substeps = 0
        self.pending = dict(start_time_s=self.last_boundary.time_s,
            start_physics_step=self.last_boundary.physics_step,
            physics_hz=PHYSICS_HZ, physics_dt_s=PHYSICS_DT_S,
            control_hz=20, substeps=12, execution_mode=EXECUTION_MODE)
        return deepcopy(self.pending)

    def after_physics(self, helper_report=None):
        if self.clock is None or self.closed or self.faulted or self.terminal is not None:
            raise TaskError("No initialized running native cloth preview", kind="execution")
        if self.started and self.pending is None and not self.paused:
            raise TaskError("Native physics needs an open preview control interval", kind="execution")
        try:
            self._check_clock()
            if helper_report is not None:
                if not isinstance(helper_report, dict):
                    raise TaskError("Physical helper report must be a mapping")
                if any(helper_report.get(key) for key in ("events", "docked_object_count", "tray_dock_enabled")):
                    raise TaskError("Assisted cloth motion is not native preview evidence", kind="execution", reason="assisted_physics_evidence")
            self.component.after_physics()
            self.last_boundary = self.component.last_boundary
            if self.pending is not None:
                self.substeps += 1
                if self.substeps > self.pending["substeps"]:
                    raise TaskError("Too many native substeps in one control interval", kind="execution", reason="incomplete_control_interval")
        except Exception as exc:
            self._error(exc)
            raise

    def end_control(self, *, completed_substeps=12):
        if self.pending is None or self.faulted or self.closed:
            raise TaskError("No valid preview interval to finish", kind="execution")
        try:
            self._check_clock()
            if (type(completed_substeps) is not int or completed_substeps != 12
                    or self.substeps != completed_substeps
                    or self.last_boundary.physics_step-self.pending["start_physics_step"] != 12
                    or not math.isclose(self.last_boundary.time_s-self.pending["start_time_s"], .05,
                                        # Native PhysX time uses its float32 dt.
                                        # Sum the already enforced 1 ns per-step
                                        # tolerance; step count remains exact.
                                        rel_tol=0, abs_tol=12e-9)
                    or self.component.boundary() != self.last_boundary):
                raise TaskError("A 20 Hz preview interval needs exactly 12 fresh native substeps", kind="execution", reason="incomplete_control_interval")
            self.control_index += 1
            value = self._running_result()
        except Exception as exc:
            self._error(exc)
            raise
        self.pending = None
        self.last_control_boundary = self.last_boundary
        return value

    def finish_control(self, *, completed_substeps=12):
        return self.end_control(completed_substeps=completed_substeps)

    def _running_result(self, *, status="running", reason=None):
        value = preview_result(self.plan, self.component.summary(), episode_id=self.episode_id,
            status=status, termination_reason=reason)
        value["evidence_step"] = self.control_index
        return value

    def result(self):
        if self.terminal is not None:
            return deepcopy(self.terminal)
        if not self.started:
            exc = TaskError("The scene-owned native initial state is not admitted", kind="binding", reason="initial_state_not_accepted")
            value = failure_result(self.plan, episode_id=self.episode_id, status="error",
                kind=exc.kind, reason=exc.reason, message=str(exc))
            value.update(execution_mode=EXECUTION_MODE, scoring_available=False,
                layout_scope=self.plan.get("layout_scope", "isolated_component_preview"),
                native_state={}, evidence_step=self.control_index)
            return value
        return self._running_result()

    def pause(self):
        if (not self.started or self.closed or self.faulted or self.terminal is not None
                or self.pending is not None or self.paused):
            raise TaskError("Pause only between complete preview intervals", kind="execution")
        self.paused = True

    def resume(self):
        if not self.paused or self.closed or self.faulted or self.terminal is not None:
            raise TaskError("No resumable preview", kind="execution")
        self.paused = False

    def finalize(self, reason="client_closed"):
        if self.terminal is not None:
            return deepcopy(self.terminal)
        if (not self.started or self.pending is not None or self.closed or self.faulted
                or not isinstance(reason, str) or not reason or reason == "success"):
            raise TaskError("Close an admitted complete preview; benchmark success is unavailable", kind="execution", reason="preview_finalization_rejected")
        self._check_clock()
        if (self.component.boundary() != self.last_boundary
                or self.last_boundary != self.last_control_boundary):
            raise TaskError("Resume and complete a fresh interval after paused World advancement", kind="execution", reason="fresh_final_state_required")
        self.terminal = self._running_result(status="finished", reason=reason)
        return deepcopy(self.terminal)

    def close(self):
        if self.closed:
            return
        try:
            if self.started and self.terminal is None and not self.faulted:
                self.finalize()
        except Exception as exc:
            self._error(exc)
            raise
        finally:
            self.component.close()
            self.closed = True


def build_preview_session(plan, stage, manifest, output, episode_id, settings=None):
    """Author before physics attachment; the caller subsequently resets its World."""
    from short_tasks import registry
    from short_tasks.scene import verify_manifest
    if not registry.is_gpu_cloth_preview(plan):
        raise TaskError("Only the explicitly versioned GPU cloth preview may use this builder", kind="binding")
    if plan.get("layout_scope") != "isolated_component_preview":
        raise TaskError("Shared-room cloth native admission has not been validated", kind="load", reason="shared_world_cloth_preview_not_validated")
    output = Path(output)
    if output.exists():
        raise TaskError("Native evidence output must be a fresh directory", reason="output_exists")
    bound = registry.bind(plan, manifest)
    bound["source_receipt"] = registry.verify_sources(bound)
    verify_manifest(manifest)
    layer = stage.GetEditTarget().GetLayer()
    before = layer.ExportToString()
    component = None
    try:
        receipt = author_component(stage, manifest, settings)
        component = NativeClothComponent(stage, manifest, settings,
            episode_id=episode_id, task_name=plan["task_name"])
        output.mkdir(parents=True, exist_ok=False)
        (output/"cloth_authoring.json").write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
        return PreviewSession(bound, manifest, component, episode_id=episode_id)
    except Exception:
        if component is not None:
            component.close()
        layer.ImportFromString(before)
        raise
