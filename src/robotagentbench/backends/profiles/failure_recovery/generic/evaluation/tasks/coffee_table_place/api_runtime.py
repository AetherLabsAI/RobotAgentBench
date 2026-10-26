"""Shared placement adapter, with optional reset-only task-local jitter.

Shared WorkerTask handles both model and trajectory sources. No dedicated
execution loop. No guessed success/timeout/randomization policy.
"""
import json
import copy
from arena_api.task_runtime import WorkerTask, TaskEvidence, TaskScore
from arena_api.policy_execution import SUBSTEPS
from arena_api.physics_defaults import steps_for_seconds, PHYSICS_DT
from .source_initial import TASK
from .bindings import binding_for
from .reset_binding import load_accepted_reset
from .settled_acceptance import assess, HISTORY_STEPS, SETTLE_SECONDS, ACCEPTANCE_SCHEMA
from .settled_acceptance import HELD_POSITION_SPAN_MAX_M, PAYLOAD_RELATIVE_SPAN_LIMIT_M, DETECTION


class CoffeeTablePlaceRuntime(WorkerTask):
    settle_steps = steps_for_seconds(SETTLE_SECONDS)

    def __init__(self, context):
        from .registration import normalize_parameters
        self.local_randomization = normalize_parameters(context.task_parameters)['local_randomization']
        if any(context.randomization.values()):
            raise ValueError('Whole-house randomization is separate; use task_parameters.local_randomization')
        if not context.output.resolve().is_relative_to((TASK/'artifacts').resolve()):
            raise ValueError('Output must stay in coffee_table_place/artifacts')
        self.context = context
        self.binding = binding_for(context.task_id)
        self.reader = None
        self.control_index = -1
        self.pending = None
        self.finalized = False
        self.randomization_record = None

    def _load_reset(self):
        return load_accepted_reset(self.context.task_id)

    def initialize(self, source):
        self.scene, self.reset, self.manifest = self._load_reset()
        return self.scene

    def randomize(self, scene):
        if not self.local_randomization:
            return scene
        if self.randomization_record is not None or scene != self.scene:
            raise ValueError('Local jitter must be applied once to the packaged reset')
        from .randomization import prepare
        self.scene, self.randomization_record = prepare(self.context.task_id, scene,
            self.context.output/'local_randomization', seed=self.context.seed)
        return self.scene

    def _verify_dependencies(self):
        self._load_reset()
        if self.randomization_record is not None:
            from .randomization import verify_record
            verify_record(self.randomization_record)

    def attach(self, stage):
        if self.local_randomization and self.randomization_record is None:
            raise RuntimeError('Randomized task must pass through the randomize hook')
        # 外观随机化（用户 2026-09-23：要与数据方的全部随机化对齐，外观也算）。
        # 只写渲染量，写完自证物理指纹未变；失败就抛，不静默放行。
        from .appearance import apply as apply_appearance
        self.appearance_record = apply_appearance(stage, self.context.task_id, self.context.seed)
        (self.context.output/'appearance.json').write_text(
            json.dumps(self.appearance_record, indent=2, allow_nan=False)+'\n')
        from .isaac_evidence import PlacementReader
        self.reader = PlacementReader(stage, self.binding, self.context.episode_id, HISTORY_STEPS,
                                      self.manifest.get('tray_floor_source_z_m'),
                                      output=self.context.output/'placement_evidence')

    def robot_initialization(self):
        return self.reset

    def on_physics_ready(self, world):
        self.world = world
        self.reader.initialize(world)

    def after_physics(self, helper_report):
        if (helper_report['events'] or helper_report['docked_object_count'] or
                helper_report['tray_dock_enabled']):
            raise RuntimeError('Assisted manipulation invalidates evidence')
        self.reader.after_step()

    def validate_initial(self, metadata):
        rows = list(self.reader.history)
        receipt = assess(rows, self.binding.payload is not None)
        receipt['checks'].update(physics_valid=not metadata['physics_issues'],
            self_collision=metadata['self_collisions_enabled'] is True,
            full_settle_completed=self.reader.step + 1 >= self.settle_steps)
        receipt.update(schema=ACCEPTANCE_SCHEMA,
            settle_seconds=SETTLE_SECONDS, history_steps=HISTORY_STEPS,
            history_seconds=HISTORY_STEPS*PHYSICS_DT,
            held_position_span_max_m=HELD_POSITION_SPAN_MAX_M, held_position_comparison='<=',
            payload_relative_position_span_limit_m=PAYLOAD_RELATIVE_SPAN_LIMIT_M, payload_position_comparison='<',
            payload_unloaded_duration_limit_s=DETECTION.unsupported_confirm_s, payload_unloaded_comparison='<',
            payload_support_ratio_is_diagnostic_only=True,
            settle_counts_toward_task_budget=False)
        receipt['accepted'] = all(receipt['checks'].values())
        metadata['task_initial_acceptance'] = receipt
        if not receipt['accepted']:
            failed = [name for name, passed in receipt['checks'].items() if not passed]
            raise RuntimeError('Initial acceptance failed: '+', '.join(failed))
        try:
            self._verify_dependencies()
            self.reader.geometry.check_destination_unchanged()
        except Exception:
            receipt['accepted'] = False
            receipt['checks']['dependencies_and_destination_unchanged'] = False
            raise
        receipt['checks']['dependencies_and_destination_unchanged'] = True
        from .randomization import replay_contract
        metadata['task_local_randomization'] = copy.deepcopy(self.randomization_record)
        metadata['task_appearance'] = copy.deepcopy(getattr(self, 'appearance_record', None))
        metadata['replay_reset_contract'] = dict(schema='coffee_table_place.reset.v1',
            task_id=self.context.task_id, held_body=self.binding.held_body,
            destination=self.binding.destination, destination_kind=self.binding.destination_kind,
            payload=self.binding.payload,
            settle_steps=self.settle_steps, initial_acceptance_policy='stable_relative_payload_tail120_v5',
            robot_initialization=self.manifest['robot_initialization'],
            source_episode_uid=self.manifest['source_episode_uid'],
            source_content_sha256=sorted(self.manifest['dependencies_repo_relative'].values()),
            seed=self.context.seed, randomization=dict(self.context.randomization),
            local_randomization=replay_contract(self.randomization_record))
        settle = not self.binding.payload
        metadata['task_evaluation_scope'] = (
            'settle.v3; 60 s registered budget; success = placed in the tray floor, released, '
            'still for 1 s (user 2026-09-23)' if settle else
            'score_only; 180s registered budget; success definition pending')
        self.reader.begin_episode()

    def collect_evidence(self):
        delta = self.reader.step-self.reader.start_step
        if delta != (self.control_index+1)*SUBSTEPS:
            raise RuntimeError('Missing/repeated control-frame evidence')
        self.reader.geometry.check_destination_unchanged()
        self.control_index += 1
        self.pending = TaskEvidence(self.context.episode_id, self.context.task_id,
            self.control_index, float(self.world.current_time), copy.deepcopy(self.reader.latest))
        return self.pending

    def evaluate(self, evidence):
        if evidence is not self.pending:
            raise RuntimeError('Unknown/stale evidence')
        self.pending = None
        summary = self.reader.process.summary()
        settle = summary['schema'].endswith('.v3')
        return TaskScore(summary['score'], summary['maximum_score'], dict(process=summary,
            facts=evidence.data,
            evaluation_scope='settle.v3' if settle else 'score_only',
            task_success_criteria_status='defined' if settle else 'not_defined',
            automatic_task_termination=bool(summary.get('automatic_termination'))))

    def check_success(self, evidence, score):
        # 5-x: placed in the tray floor, gripper released, still for 1 s (user
        # 2026-09-23). 30-x keep the earlier behaviour and never claim success here.
        return bool(score.details['process'].get('task_success'))

    def check_failure(self, evidence, score):
        # Floor contact or leaving the operable region during (1)/(2); the state
        # machine already evaluates failure before success on the same step.
        return score.details['process'].get('failure')

    def finalize(self, evidence, score, outcome):
        summary = self.reader.process.summary(final=True)
        self.finalized = True
        return TaskScore(summary['score'], summary['maximum_score'], dict(score.details,
            process=summary, settlement_reason=outcome.reason, source_exhaustion_is_success=False))

    def close(self):
        if getattr(self, 'reader', None) is not None:
            self.reader.close()
        if hasattr(self, 'manifest'):
            self._verify_dependencies()
