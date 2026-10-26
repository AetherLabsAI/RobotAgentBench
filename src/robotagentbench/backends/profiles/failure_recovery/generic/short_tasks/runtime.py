"""Single benchmark lifecycle over caller-owned native physics and actions.

This host does not drive the robot or own a second World. All internal readers
stay current; the public result and progress deadline belong to one task.
"""
from copy import deepcopy
import math
from numbers import Real
from pathlib import Path

from . import registry
from .common.contracts import TaskError, failure_result
from .common.episode import SingleTaskEpisode


class UnavailableEpisode:
    def __init__(self, plan):
        self.plan = deepcopy(plan)

    def result(self):
        value = failure_result(self.plan)
        if registry.is_gpu_cloth_preview(self.plan):
            value.update(execution_mode="gpu_cloth_preview", scoring_available=False,
                         layout_scope=self.plan.get("layout_scope", "isolated_component_preview"),
                         requested_layout_id=self.plan.get("request", {}).get("layout_id"),
                         native_state={}, evidence_step=None)
        return value

    def begin_control(self, *args, **kwargs):
        error = self.plan["unavailable"]
        raise TaskError(error["message"], kind=error["kind"], reason=error["reason"])

    def close(self):
        pass


class ShortTaskSession:
    def __init__(self, plan, manifest, ports, *, episode_id, workspace_selector=None, snapshot_pool=None, contact_routes=None, environment_actor=None, interaction_reader=None, navigation_reader=None):
        if not plan["available"]:
            raise TaskError("Unavailable task cannot start a session", kind="rules")
        expected = {c["instance_id"] for c in plan["runtime_components"]}
        if set(ports) != expected or not isinstance(episode_id, str) or not episode_id:
            raise TaskError("Every internal reader needs a unique episode-bound identity", kind="binding")
        self.plan, self.manifest = deepcopy(plan), deepcopy(manifest)
        self.ports, self.episode_id = dict(ports), episode_id
        self.selector, self.pool = workspace_selector, snapshot_pool
        self.routes = contact_routes
        self.interaction = interaction_reader
        self.navigation = navigation_reader
        if bool(plan.get('navigation_policy')) != (navigation_reader is not None):
            raise TaskError('Declared navigation requires native stage/pose evidence', kind='binding')
        if bool(plan.get('interaction_policy')) != (interaction_reader is not None):
            raise TaskError('Declared interaction scoring requires native evidence', kind='binding')
        if plan.get("environment_actor_contract") is not None and environment_actor is None:
            raise TaskError("Declared environment actor is missing", kind="binding")
        # Only trusted native-owner code supplies this object. Request JSON
        # cannot construct an actor or waive the assisted-evidence checks.
        if environment_actor is not None and (
                plan["task_name"] != "stock_wooden_tray_interference"
                or plan.get("environment_actor_contract") != "shelf_human_relocation.v1"):
            raise TaskError("Environment actor is not declared by this task", kind="binding")
        self.environment_actor = environment_actor
        self._actor_step_prepared = False
        self.error_receipt = None
        self.clock = self.ledger = self.pending = None
        self.started = self.closed = self.paused = self.faulted = False
        self.last_physics_time = None
        self.control_index = 0
        self.substeps = 0

    def initialize(self, clock, view):
        if self.clock is not None or self.closed:
            raise TaskError("Bind once; initialize is not a reset entry", kind="execution")
        if clock.hz != 60:
            raise TaskError("The scene must start at the approved 60 Hz", kind="binding")
        self.clock, self.last_physics_time = clock, float(clock.current_time)
        try:
            from openend.tasks.meal_tray.nonmotion_closeout_v1.step_snapshot import SnapshotViews, bind_contacts
            if self.pool is None:
                self.pool = SnapshotViews(view)
            for port in self.ports.values():
                port.initialize(clock, self.pool)
            bind_contacts(self.ports, self.pool, self.routes)
            if self.interaction is not None:
                self.interaction.initialize(self.pool)
            if self.navigation is not None:
                self.navigation.initialize(self.pool)
            if self.environment_actor is not None:
                self.environment_actor.initialize(clock, self.pool, self.ports)
        except Exception as exc:
            self._fault(exc)
            raise

    def _fault(self, exc):
        self.faulted = True
        self.error_receipt = failure_result(self.plan, episode_id=self.episode_id,
            layout_id=self.manifest["layout_id"], time_s=None if self.clock is None else float(self.clock.current_time),
            status="error", kind=getattr(exc, "kind", "execution"),
            reason=getattr(exc, "reason", "native_reader_failed"), message=str(exc) or type(exc).__name__)

    def before_physics(self, dt):
        if self.environment_actor is None:
            return
        if (not self.started or self.closed or self.faulted or self.paused
                or self.pending is None or self._actor_step_prepared):
            raise TaskError("Actor requires one admitted active physical step", kind="execution")
        if abs(dt - 1 / self.clock.hz) > 1e-12:
            raise TaskError("Actor timestep differs from native clock", kind="execution")
        try:
            now = self.clock.current_time
            self.environment_actor.before_physics(dt)
            if self.clock.current_time != now:
                raise TaskError("Actor cannot advance the simulation clock", kind="execution")
            self._actor_step_prepared = True
        except Exception as exc:
            self._fault(exc)
            raise

    def after_physics(self, helper_report):
        if self.clock is None or self.closed or self.faulted:
            raise TaskError("No initialized native episode", kind="execution")
        if self.started and self.pending is None and not self.paused:
            raise TaskError("Physical substeps need an open control interval", kind="execution")
        now, dt = float(self.clock.current_time), 1 / self.clock.hz
        if self.clock.hz not in (60, 240) or abs(now - self.last_physics_time - dt) > 1e-6:
            exc = TaskError("Missing or repeated native physics evidence", kind="execution", reason="physics_evidence_discontinuous")
            self._fault(exc)
            raise exc
        if any(helper_report.get(k) for k in ("events", "docked_object_count", "tray_dock_enabled")):
            exc = TaskError("Assisted physical evidence cannot score a Short Task", kind="execution", reason="assisted_physics_evidence")
            self._fault(exc)
            raise exc
        helper_report = {"events": [], "docked_object_count": 0, "tray_dock_enabled": False, **helper_report}
        if self.pool is not None:
            self.pool.begin()
        try:
            if self.pool is not None and "_contact_report" in helper_report:
                helper_report["_contact_report"] = self.pool.accept_contact_report(helper_report["_contact_report"])
            if self.environment_actor is not None and self.started and not self.paused:
                if not self._actor_step_prepared:
                    raise TaskError("Missing actor pre-step", kind="execution")
                # Actor reads native contacts and gates its target port; it
                # cannot strip helper events or silently permit robot assists.
                self.environment_actor.after_physics(helper_report)
                if self.clock.current_time != now:
                    raise TaskError("Actor cannot advance the simulation clock", kind="execution")
                self._actor_step_prepared = False
            if self.routes is not None:
                if "_contact_report" not in helper_report:
                    helper_report["_contact_report"] = self.routes.interface.get_contact_report()
                self.routes.after_physics(helper_report["_contact_report"], now)
            if self.interaction is not None:
                self.interaction.after_physics(helper_report['_contact_report'],
                    admitted=self.started and not self.paused)
            for port in self.ports.values():
                if self.paused and hasattr(port, "after_inactive"):
                    port.after_inactive(helper_report)
                else:
                    port.after_physics(helper_report)
                    if self.paused:
                        if hasattr(port, "discard_control"):
                            port.discard_control()
                        else:
                            port.pause()
            self.last_physics_time = now
            if self.pending is not None:
                self.substeps += 1
        except Exception as exc:
            self._fault(exc)
            raise
        finally:
            if self.pool is not None:
                self.pool.end()

    def _snapshot(self):
        value = registry.normalize(self.plan, {k: p.score for k, p in self.ports.items()})
        if self.environment_actor is not None:
            value = self.environment_actor.score_snapshot(value)
        if self.interaction is not None:
            value = self.interaction.progress.augment(value,
                arithmetic=self.plan['score_policy'].get('arithmetic','gross_minus_deductions'))
        from .tasks.no_deductions_v1.policy import apply
        value = apply(self.plan, value)
        if self.navigation is not None:
            value = self.navigation.augment(value)
        value.update(task_name=self.plan["task_name"], episode_id=self.episode_id,
                     rule_version=self.plan["rule_version"], evidence_step=self.control_index,
                     evidence_time_s=float(self.clock.current_time))
        return value

    def accept_initial(self):
        if self.started or self.clock is None or self.closed or self.faulted:
            raise TaskError("Scene-owned native initial admission occurs exactly once", kind="binding")
        try:
            receipts = {k: p.accept_initial() for k, p in self.ports.items()}
            if any(r.get("accepted") is not True for r in receipts.values()):
                raise TaskError("Every reader must provide measured initial acceptance", kind="binding", reason="initial_state_unverified")
            if self.interaction is not None:
                self.interaction.sample(float(self.clock.current_time), baseline=True)
            if self.navigation is not None:
                self.navigation.sample(float(self.clock.current_time), baseline=True)
            value = self._snapshot()
            if value["scores"]["process"] or value["scores"]["current_final"] or value["success"]:
                raise TaskError("Scene initial state cannot complete task operations", kind="binding")
            policy = self.plan["score_policy"]
            ledger_type = SingleTaskEpisode
            if self.plan.get('robustness_timing_policy'):
                from short_tasks.agent_probe_v1.cap_v1.native_tools.codex_adapter_v1.robustness6_v1.timing120_v1.integration import shelf_episode_type
                ledger_type = shelf_episode_type()
            if self.navigation is not None:
                from .tasks.navigation_progress_v1.episode import NavigationEpisode
                ledger_type = NavigationEpisode
            self.ledger = ledger_type(task_name=self.plan["task_name"], episode_id=self.episode_id,
                actual_layout_id=self.manifest["layout_id"], rule_version=self.plan["rule_version"],
                start_time_s=float(self.clock.current_time), target_roles=self.plan.get("target_bindings", {}),
                maximum_score=policy["maximum"], score_arithmetic=policy.get("arithmetic", "gross_minus_deductions"))
            if self.plan.get('robustness_timing_policy'):
                self.ledger.bind_actor(self.environment_actor)
            self.ledger._snapshot(value)
            self.ledger.latest = deepcopy(value)
            if self.routes is not None:
                self.routes.arm(float(self.clock.current_time))
            if self.environment_actor is not None:
                self.environment_actor.arm()
        except Exception as exc:
            self._fault(exc)
            raise
        self.started = True
        return dict(accepted=True, public_task_count=1, task_name=self.plan["task_name"],
                    internal_initial_evidence=receipts, initialization_owner="unified_scene",
                    full_action_validation=False)

    def begin_control(self, robot_state, *, command_vw, base_velocity):
        if not self.started or self.closed or self.paused or self.faulted or self.pending is not None or self.ledger.terminal is not None:
            raise TaskError("Episode cannot accept another control interval", kind="execution")
        if (len(command_vw) != 2 or len(base_velocity) != 6
                or any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)
                       for v in (*command_vw, *base_velocity))):
            raise TaskError("Actual robot command and measured base velocity required")
        selection = (self.selector.select(robot_state) if self.selector is not None else
                     dict(current_workspace=None, requested_physics_hz=60))
        requested = selection["requested_physics_hz"]
        if requested not in (60, 240):
            raise TaskError("Unknown physical workspace rate", kind="binding")
        stopped = (max(map(abs, command_vw)) < .001
                   and math.sqrt(sum(v*v for v in base_velocity[:3])) < .005
                   and math.sqrt(sum(v*v for v in base_velocity[3:])) < .01)
        if self.clock.hz != requested and stopped:
            state, now = self.clock.snapshot(), self.clock.current_time
            if not isinstance(state, bytes):
                raise TaskError("Clock switch requires immutable native World evidence", kind="binding")
            for port in self.ports.values():
                if hasattr(port, "validate_physics_hz"):
                    port.validate_physics_hz(requested)
            try:
                self.clock.set_hz(requested)
                if self.clock.current_time != now or self.clock.snapshot() != state or self.clock.hz != requested:
                    raise TaskError("A rate change altered World state/time", kind="execution")
                for port in self.ports.values():
                    if hasattr(port, "set_physics_hz"):
                        port.set_physics_hz(requested)
            except Exception as exc:
                self._fault(exc)
                raise
        for port in self.ports.values():
            if hasattr(port, "begin_selected_control"):
                port.begin_selected_control(True)
        if self.interaction is not None:
            self.interaction.begin_control()
        self.substeps = 0
        self.pending = dict(start_time_s=float(self.clock.current_time), physics_hz=self.clock.hz,
                            substeps=self.clock.hz//20, workspace=selection)
        return deepcopy(self.pending)

    def _finalize_snapshot(self, reason):
        for port in self.ports.values():
            port.score = port.finalize(reason)
        snapshot = self._snapshot()
        # Closing does not create another measured control frame.
        snapshot["evidence_step"] = self.ledger.latest["evidence_step"]
        snapshot.update(termination=reason, success=reason == "success" and snapshot["success"])
        return snapshot

    def finish_control(self, *, completed_substeps):
        if self.pending is None or self.faulted:
            raise TaskError("No valid control interval to finish", kind="execution")
        if (completed_substeps != self.pending["substeps"] or self.substeps != completed_substeps
                or self.clock.hz != self.pending["physics_hz"]
                or abs(self.clock.current_time-self.pending["start_time_s"]-.05) > 1e-6):
            exc = TaskError("Incomplete 20 Hz native interval", kind="execution", reason="incomplete_control_interval")
            self._fault(exc)
            raise exc
        try:
            active = (self.routes.requests(self.ports, float(self.clock.current_time))
                      if self.routes is not None else {k: {"active": True} for k in self.ports})
            # Read stage eligibility BEFORE any component advances this frame.
            if self.interaction is not None:
                self.interaction.sample(float(self.clock.current_time))
            for key, port in self.ports.items():
                if not port.ended:
                    if hasattr(port, "consume_control"):
                        port.score = port.consume_control(active[key]["active"])
                    elif active[key]["active"]:
                        port.score = port.observe()
                    else:
                        port.pause()
                        if hasattr(port, "summary_current"):
                            port.score = port.summary_current()
            if self.navigation is not None:
                self.navigation.sample(float(self.clock.current_time))
            value = self._snapshot()
            result = self.ledger.observe(value, simulation_time_s=float(self.clock.current_time),
                                         finalizer=self._finalize_snapshot)
        except Exception as exc:
            self._fault(exc)
            raise
        self.pending = None
        self.control_index += 1
        return result

    def result(self):
        if self.error_receipt is not None:
            return deepcopy(self.error_receipt)
        if self.ledger is None:
            return failure_result(self.plan, episode_id=self.episode_id, layout_id=self.manifest["layout_id"],
                                  status="error", kind="binding", reason="initial_state_not_accepted",
                                  message="Only measured scene initial acceptance may start this episode")
        return self.ledger.result()

    def pause(self):
        if not self.started or self.closed or self.faulted or self.pending is not None:
            raise TaskError("Pause only between complete control intervals", kind="execution")
        for port in self.ports.values():
            port.pause()
        if self.interaction is not None:
            self.interaction.pause()
        if self.navigation is not None:
            self.navigation.pause()
        self.paused = True

    def resume(self):
        if not self.paused or self.closed or self.faulted or self.ledger.terminal is not None:
            raise TaskError("No resumable episode", kind="execution")
        for port in self.ports.values():
            port.pause()
        self.paused = False

    def finalize(self, reason="client_closed"):
        if self.pending is not None or self.ledger is None:
            raise TaskError("Finalize only an admitted complete control boundary", kind="execution")
        if abs(float(self.clock.current_time)-self.ledger.last_time) > 1e-6:
            raise TaskError("Resume and complete a fresh interval before finalizing a World advanced while paused", kind="execution", reason="fresh_final_state_required")
        return self.ledger.finish(reason, finalizer=self._finalize_snapshot)

    def close(self):
        if self.closed:
            return
        try:
            if self.ledger is not None and self.ledger.terminal is None and not self.faulted and self.pending is None:
                self.finalize()
        except Exception as exc:
            self._fault(exc)
            raise
        finally:
            errors = []
            if self.environment_actor is not None:
                try:
                    self.environment_actor.close()
                except Exception as exc:
                    errors.append(str(exc))
            for port in self.ports.values():
                try:
                    port.close()
                except Exception as exc:
                    errors.append(str(exc))
            self.closed = True
            if errors:
                raise TaskError("Reader cleanup failed: " + "; ".join(errors), kind="execution")


def build_session(request, stage, manifest, output, *, episode_id, workspace_selector=None, sensor=None,
                  runtime_settings=None):
    """Build native readers BEFORE the caller attaches physics; no robot policy."""
    plan = registry.resolve(request)
    if not plan["available"]:
        return UnavailableEpisode(plan)
    if plan['task_name'] in registry.ATOMIC_COFFEE_TASKS:
        from short_task_robustness.tasks.coffee_machine_push_interference.public_integration_v1.session import build_session as atomic_session
        return atomic_session(plan, stage, manifest, output, episode_id=episode_id)
    output = Path(output)
    if output.exists():
        raise TaskError("Native evidence output must be a fresh directory", reason="output_exists")
    plan = registry.bind(plan, manifest)
    if registry.is_gpu_cloth_preview(plan):
        # A separate trusted profile selects native GPU physics before attach.
        # The caller still owns the one World, actions and simulation clock.
        # CPU workspace sensors/ports must never be imported into this branch.
        try:
            plan["source_receipt"] = registry.verify_sources(plan)
            from .gpu_cloth.session import build_preview_session
            return build_preview_session(plan, stage, manifest, output,
                                         episode_id=episode_id, settings=runtime_settings)
        except TaskError as exc:
            if exc.kind != "load":
                raise
            plan.update(available=False, unavailable=dict(kind=exc.kind, reason=exc.reason, message=str(exc)))
            return UnavailableEpisode(plan)
    if workspace_selector is None:
        if sensor is None:
            raise TaskError("Native loading requires the calibrated workspace sensor", kind="binding", reason="workspace_sensor_required")
        from openend.tasks.all_objects_integration.workspace_physics import from_scene
        workspace_selector = from_scene(stage, manifest, sensor)
    from .scene import prepare_runtime_scene, verify_scene_sources
    try:
        verify_scene_sources()
        source_receipt = registry.verify_sources(plan)
        manifest = prepare_runtime_scene(plan, stage, manifest)
    except TaskError as exc:
        if exc.kind != "load":
            raise
        plan.update(available=False, unavailable=dict(kind=exc.kind, reason=exc.reason, message=str(exc)))
        return UnavailableEpisode(plan)
    plan["source_receipt"] = source_receipt
    output.mkdir(parents=True, exist_ok=False)
    ports = {}
    try:
        for declaration in plan["runtime_components"]:
            cls = registry.native_factory(plan, declaration)
            identity = declaration["instance_id"]
            ports[identity] = cls(stage, manifest, output/identity, episode_id, identity, declaration["parameters"])
            ports[identity].workspace_rates = True
        from openend.tasks.meal_tray.gripper_wake_v1.host import bindings_for, UnifiedGripperRoutes
        from openend.tasks.meal_tray.task_contact_routing import roots_for
        bindings = {}
        for declaration in plan["runtime_components"]:
            identity = declaration["instance_id"]
            port = ports[identity]
            if declaration["type"] == "snack_drink_to_tray":
                resolved = {"path": next(r["target_path"] for r in plan["target_bindings"] if r["instance_id"] == identity)}
            else:
                resolved = {}
            bindings[identity] = bindings_for(declaration["type"], port, resolved, roots_for)
        routes = UnifiedGripperRoutes(stage, bindings)
        interaction = None
        if plan.get('interaction_policy'):
            from .tasks.interaction_progress_v1.native import NativeProgress
            interaction = NativeProgress(stage, plan, manifest, ports, episode_id,
                output/'interaction_progress.json', routes.native.fingers)
        navigation = None
        if plan.get('navigation_policy'):
            from .tasks.navigation_progress_v1.native import NativeNavigation
            navigation = NativeNavigation(stage, plan, ports, workspace_selector)
    except Exception:
        for port in ports.values():
            try:
                port.close()
            except Exception:
                pass  # Preserve construction failure; still release other readers.
        raise
    actor = None
    try:
        if plan.get("environment_actor_contract") is not None:
            module = registry.adapter(plan["task_name"])
            actor = module.create_environment_actor(stage, manifest, output, episode_id)
        return ShortTaskSession(plan, manifest, ports, episode_id=episode_id,
                                workspace_selector=workspace_selector, contact_routes=routes,
                                environment_actor=actor, interaction_reader=interaction, navigation_reader=navigation)
    except Exception:
        if actor is not None:
            try:
                actor.close()
            except Exception:
                pass
        for port in ports.values():
            try:
                port.close()
            except Exception:
                pass
        raise
