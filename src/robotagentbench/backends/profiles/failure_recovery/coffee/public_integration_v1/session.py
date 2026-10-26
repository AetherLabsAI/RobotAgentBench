"""Original atomic lifecycle behind the unchanged public robot22 host."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

from . import adapter
from .contact_award import VERSION, TASK_IDS, TargetContactAward, observe_stream
from short_tasks.agent_probe_v1.cap_v1.native_tools.codex_adapter_v1.robustness6_v1.timing120_v1.integration import TimedWorker, lifecycle_type


def make_runtime(context, initial, variant):
    from evaluation.tasks.coffee_machine.api_runtime import CoffeeMachineRuntime
    from evaluation.tasks.coffee_machine.process_rules import Process
    from evaluation.tasks.coffee_machine.specs import spec
    from evaluation.tasks.coffee_machine.registration import normalizer
    from evaluation.tasks.coffee_machine.initial_state import SETTLE_SECONDS
    from arena_api.physics_defaults import steps_for_seconds

    class AfterRelocationProcess(Process):
        """Gate stage awards only; original failures, deductions and clocks remain live."""
        def _pour_beans(self, facts, emitted):
            if not self.allowed():
                return None
            return super()._pour_beans(facts, emitted)

    class PreparedCoffeeRuntime(TimedWorker, CoffeeMachineRuntime):
        def __init__(self):
            self.context, self.plan, self.appearance = context, initial, initial['appearance']
            self.spec = spec(context.task_id)
            self.parameters = normalizer(context.task_id)(context.task_parameters)
            self.settle_steps = steps_for_seconds(SETTLE_SECONDS[context.task_id])
            self.requires_gpu_dynamics = False
            self.reader = self.pending = self.actor = None
            self.control_index = -1
            self.armed = False
            self.process = AfterRelocationProcess(context.task_id) if variant in ('bag_only', 'bag_then_machine') else Process(context.task_id)
            if isinstance(self.process, AfterRelocationProcess):
                self.process.allowed = self.score_gate
            self.configure_timing(variant)

        def initialize(self, source):
            return Path(self.plan['scene'])

        def attach(self, stage):
            if variant == 'bag_only':
                super().attach(stage)
                from ..native_bag import NativeBagRelocation
                self.actor = NativeBagRelocation(stage, initial)
            else:
                from ..isaac_evidence import MovingCoffeeReader
                self.stage = stage
                self.reader = MovingCoffeeReader(stage, Path(context.output)/'coffee_detection', context.episode_id)
                if variant == 'machine_only':
                    from ..native_push import NativePalmPush
                    self.actor = NativePalmPush(stage, initial)
                else:
                    from ..combined_session import CombinedNativeSession
                    self.actor = CombinedNativeSession(stage, initial)

        def native_poses(self):
            from short_task_robustness.tasks.stock_wooden_tray_interference.human_asset import pose_matrix
            return {'/World/Robot/'+name: pose_matrix(row[:3], row[3:]).T for name, row in
                zip(self.native_robot.shared_metatype.link_names, self.native_robot.get_link_transforms()[0])}

        def on_physics_ready(self, world):
            super().on_physics_ready(world)
            from isaacsim.core.simulation_manager import SimulationManager
            view = SimulationManager.get_physics_sim_view()
            view.initialize_kinematic_bodies()
            self.actor.bind(view, self.reader)
            self.native_robot = view.create_articulation_view('/World/Robot')

        def validate_initial(self, metadata):
            super().validate_initial(metadata)
            if variant == 'machine_only':
                self.actor.arm()
            else:
                self.actor.arm(robot_native_poses=self.native_poses())
            self.armed = True

        def before_physics(self):
            if self.armed:
                self.actor.command_before_step(float(self.world.current_time))

        def after_physics(self, report):
            super().after_physics(report)
            if self.armed:
                self.actor.observe(robot_native_poses=self.native_poses())
                self.capture_timing_completions()

        def score_gate(self):
            if not self.armed:
                return False
            if variant == 'bag_then_machine':
                return self.actor.robot_process_scoring_enabled
            return bool(self.actor.completed and self.actor.retired)

        def close(self):
            try:
                if self.actor is not None:
                    (Path(context.output)/'interference.json').write_text(json.dumps(
                        dict(variant=variant, events=self.actor.events if hasattr(self.actor,'events') else self.actor.sequence.snapshot()),
                        indent=2, default=lambda x: x.tolist())+'\n')
                    self.actor.close()
            finally:
                super().close()

    if variant == 'button_lighting':
        from coffee_button_lighting_interference.runtime import LightingButtonRuntime

        class PreparedLightingRuntime(TimedWorker, LightingButtonRuntime):
            def __init__(self, context):
                super().__init__(context)
                self.configure_timing(variant)

            def initialize(self, source):
                self.plan, self.appearance = initial, initial['appearance']
                return Path(initial['scene'])

        return PreparedLightingRuntime(context)
    return PreparedCoffeeRuntime()


class AtomicCoffeeSession:
    """Duck-typed ShortTaskSession; one source lifecycle/scorer and simulation clock."""
    def __init__(self, plan, stage, manifest, output, episode_id):
        from arena_api.task_runtime import WorkerContext, TaskRegistration
        from arena_api import l1_scene
        self.plan = deepcopy(plan)
        self.output = adapter.owned(plan['task_name'], output, fresh=True)
        self.output.mkdir(parents=True)
        initial = json.loads(Path(manifest['initial_file']).read_text())
        api = plan['source_api']
        # Registration/context retain a valid positive structural step field;
        # the task-owned lifecycle clock has no absolute cap. Only the 120 s
        # progress deadline can terminate for elapsed time.
        if plan.get('robustness_timing_policy') != 'robustness_no_progress_120_manual_completion_v1':
            raise ValueError('Approved robustness timing policy required')
        steps = 2400
        context = WorkerContext(episode_id, api, self.output, plan['request']['seed'], steps, {},
            dict(source_episode=initial['source_episode'], random_layout=initial['role']=='random'))
        self.worker = make_runtime(context, initial, plan['variant'])
        self.lifecycle = lifecycle_type()(self.worker, context,
            TaskRegistration(api, plan['instruction'], max_steps=steps, default_steps=steps,
                             evaluator_id='coffee_machine.process.v2'), .05)
        actual = self.lifecycle.prepare_scene(l1_scene.BASE)
        if actual != Path(manifest['scene']).resolve():
            raise ValueError('Prepared atomic scene identity changed')
        self.lifecycle.attach(stage)
        self.episode_id, self.control_index = episode_id, 0
        self.clock = SimpleNamespace(hz=240, current_time=None)
        self.started = self.faulted = self.closed = self.paused = False
        self.pending = None
        from evaluation.tasks.coffee_machine.specs import BAG, BUTTON
        self.target_contact=TargetContactAward(episode_id,TASK_IDS[plan['task_name']],
            BUTTON if plan['variant']=='button_lighting' else BAG)
        self.contact_sample=0

    def initialize(self, world):
        self.world = world
        self.lifecycle.initialize(world)
        self.clock.current_time = float(world.current_time)

    def before_physics(self, dt):
        if abs(dt-1/240) > 1e-12:
            raise ValueError('Original atomic timestep required')
        hook = getattr(self.worker, 'before_physics', None)
        if hook is not None:
            hook()

    def after_physics(self, report):
        now = float(self.world.current_time)
        if abs(now-self.clock.current_time-1/240) > 1e-6:
            raise ValueError('Missing/repeated atomic physical substep')
        self.lifecycle.after_physics(report)
        if self.started:
            actor=getattr(self.worker,'actor',None)
            bag_actor=getattr(actor,'bag',actor)
            relocated=bool(getattr(bag_actor,'completed',False) and getattr(bag_actor,'retired',False))
            observe_stream(self.target_contact,self.worker.reader.contacts,sample_index=self.contact_sample,
                time_s=now,admitted=True,relocation_complete=relocated)
            self.contact_sample+=1
        self.clock.current_time = now
        if self.pending is not None:
            self.pending += 1

    def accept_initial(self, metadata):
        self.lifecycle.accept_initial(metadata, self.clock.current_time)
        self.started = True

    def begin_control(self, robot_state, *, command_vw, base_velocity):
        if not self.started or self.pending is not None or self.result()['status'] != 'running':
            raise ValueError('Atomic episode is not ready for a new control frame')
        self.pending = 0
        return dict(physics_hz=240, substeps=12)

    def finish_control(self, *, completed_substeps):
        if completed_substeps != 12 or self.pending != 12:
            raise ValueError('Exactly twelve atomic substeps required per control interval')
        self.lifecycle.after_control(self.clock.current_time)
        self.control_index += 1
        self.pending = None
        return self.result()

    def _fault(self, exc):
        self.faulted = True
        self.error = type(exc).__name__+': '+str(exc)

    def result(self):
        snapshot = self.lifecycle.evaluation()
        contact=self.target_contact.snapshot()
        total=None
        valid=bool(snapshot and snapshot['valid'])
        if snapshot:
            # Keep the original atomic evaluation intact; this robustness
            # task composes its extra stage above it, before the zero floor.
            total=max(0.,snapshot['details']['process']['raw_score']+contact['points'])
            if not snapshot['details'].get('facts',{}).get('contacts_complete',True):
                valid=False
            actor=getattr(self.worker,'actor',None)
            if getattr(getattr(actor,'bag',actor),'intervention_evidence_unknown',False):
                valid=False
        status = 'invalid' if self.faulted else 'completed' if self.closed or self.lifecycle.reason else 'running'
        return dict(schema='short_tasks.atomic_coffee.result.v1', task_name=self.plan['task_name'],
            episode_id=self.episode_id, status=status, evaluation_valid=bool(valid and not self.faulted),
            scoring_status=('invalid' if self.faulted else 'waiting_for_contact_evidence'
                if snapshot and not valid else 'original_atomic_process_plus_target_contact_v2'),
            scoring_revision=VERSION,target_contact=contact,
            score=None if self.faulted or not snapshot else total,
            maximum_score=self.plan['score_policy']['maximum'],
            success=False if self.faulted else bool(snapshot and snapshot['success']),
            source_evaluation=snapshot, layout_scope='atomic_source')

    def close(self):
        if self.closed:
            return
        try:
            self.lifecycle.close(completed=not self.faulted)
        except BaseException as exc:
            self._fault(exc)
            raise
        finally:
            self.closed = True
            (self.output/'result.json').write_text(json.dumps(self.result(), indent=2, allow_nan=False)+'\n')


def build_session(plan, stage, manifest, output, *, episode_id):
    adapter.verify_sources(plan)
    adapter.bind(plan, manifest)
    return AtomicCoffeeSession(plan, stage, manifest, output, episode_id)
