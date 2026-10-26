"""Refrigerator task plugin; shared lifecycle owns stepping and termination."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from arena_api.task_runtime import WorkerTask, TaskEvidence, TaskScore
from arena_api import policy_execution as pe
from .initial_state import TASK_DIR, TARGET
from .process_rules import ProcessScorer, Frame, CLOSED_DEG
from .variants import variant_for

PREPARED = TASK_DIR / 'prepared/ep_beefw3_001_20260913_v4'


class RefrigeratorTakeRuntime(WorkerTask):
    # Inherit native common 60-step settle, as used by accepted replay.
    def __init__(self, context):
        from .random_reset import validate_flags
        validate_flags(context.randomization)
        if context.task_parameters:
            raise ValueError('No additional task parameters are supported')
        if not context.output.resolve().is_relative_to(TASK_DIR.resolve()):
            raise ValueError('This task currently requires task-local output')
        self.context = context
        self.variant = variant_for(context.task_id)
        self.reader = None
        self.scorer = None
        self.current = None
        self.origin = None
        self.step_index = -1
        self.interval_hold = True
        self.interval_collision = False
        self.assistance = None
        self.random_plan = None

    def initialize(self, source):
        # This first integration exposes one explicit fixed baseline, not a
        # silently randomized substitute. All source references stay read-only.
        scene = (PREPARED / 'scene.usda').resolve(strict=True)
        manifest = json.loads((PREPARED / 'manifest.json').read_text())
        if hashlib.sha256(scene.read_bytes()).hexdigest() != manifest['scene_sha256']:
            raise ValueError('Approved prepared scene changed')
        if self.variant.kind != 'beef':
            from .recorded_variant_initial import build
            scene, self.random_plan = build(scene, self.context.output/'initial',self.variant.kind)
        return scene

    def robot_initialization(self):
        if self.variant.kind != 'beef' and not self.context.randomization.get('layout'):
            from .recorded_variant_initial import robot_plan
            return robot_plan(self.variant.kind)
        return super().robot_initialization()

    def randomize(self, scene):
        if not self.context.randomization.get('layout'):
            return scene
        from .random_reset import build
        # Build from the immutable source, not a layer that already has boxes.
        source = PREPARED/'scene.usda' if self.variant.kind != 'beef' else scene
        output = self.context.output/'random' if self.variant.kind != 'beef' else self.context.output
        scene, self.random_plan = build(source, output, self.context.seed)
        return scene

    def attach(self, stage):
        import os
        if os.environ.get('ARENA_FRIDGE_ORIGINAL_BEEF_EXPERIMENT') == '1':
            raise ValueError('Original col_box experiment retired: unset '
                             'ARENA_FRIDGE_ORIGINAL_BEEF_EXPERIMENT; '
                             'the selected beef asset uses 876 convex pieces')
        if os.environ.get('ARENA_FRIDGE_BEEF_GRIPMAT_EXPERIMENT', '1') == '1':
            from .experiments.beef_gripmat_v1.binding import apply
            apply(stage, self.context.output)
        from .isaac_evidence import IsaacEvidenceReader
        if self.variant.kind == 'beef':
            self.reader = IsaacEvidenceReader(stage, self.random_plan)
        else:
            from arena_api.food_box_assets import verify_instance
            verify_instance(stage, self.variant.body,
                            'leftover_box' if self.variant.kind == 'food' else 'salad_box')
            self.reader = IsaacEvidenceReader(stage, self.random_plan, target_kind=self.variant.kind)

    def on_physics_ready(self, world):
        self.reader.initialize(world)

    def after_physics(self, helper_report):
        if helper_report.get('docked_object_count') or helper_report.get('tray_dock_enabled'):
            raise ValueError('Object attachment is not approved')
        if helper_report.get('policy_id') != 'house.l1_task95_original_assistance.v3':
            raise ValueError('Unexpected fridge assistance baseline')
        self.assistance = helper_report
        self.current = self.reader.sample()
        f = self.current
        self.interval_hold &= (f['target_held'] and f['target_fully_out']
                               and all(abs(a) <= CLOSED_DEG for a in f['door_angles_deg']))
        self.interval_collision |= f['non_operational_collision']

    def validate_initial(self, metadata):
        f = self.current
        (self.context.output / 'initial_evidence.json').write_text(
            json.dumps(f, indent=2, allow_nan=False) + '\n')
        if (f is None or f['target_held'] or f['target_fully_out']
                or any(abs(a) > CLOSED_DEG for a in f['door_angles_deg'])
                or not f['source_floor_contact']):
            raise ValueError('Initial box must be supported inside closed refrigerator, not held')
        if self.random_plan and (set(f['background_box_support']) != {'beef','salad','food'}-{self.variant.kind}
                                 or not all(f['background_box_support'].values())):
            raise ValueError('Random initial background boxes must have measured source-shelf support')
        self.origin = f['time_s']
        self.scorer = ProcessScorer(self.context.episode_id, self.variant.body, initial_state_accepted=True)
        self.interval_hold, self.interval_collision = False, False
        metadata['replay_reset_contract'] = {
            'schema': 'arena.refrigerator_take.reset.v1', 'source_episode': 'ep_beefw3_001',
            'prepared_scene_sha256': hashlib.sha256((PREPARED / 'scene.usda').read_bytes()).hexdigest(),
            'target': self.variant.body, 'randomization': 'fixed',
            'gripper_execution_mode': pe.DEFAULT_GRIPPER_MODE,
            'source_compartment': 'right_door/shelf4',
            'assistance_policy': self.assistance['policy_id']}
        if self.random_plan:
            metadata['replay_reset_contract'].update(
                randomization=self.random_plan,
                source_compartment='right_door/'+self.random_plan['boxes'][self.variant.kind]['source_compartment'])
            if self.context.randomization.get('layout'):
                metadata['random_initial_state_accepted'] = True
        if self.variant.kind != 'beef':
            from arena_api.food_box_assets import REVISION
            metadata['replay_reset_contract'].update(source_episode=self.random_plan.get('source_episode'),
                target_kind=self.variant.kind, shared_food_box_asset=REVISION,
                source_layout='L1 original recording frame zero' if not self.context.randomization.get('layout') else 'L1 random policy')

    def collect_evidence(self):
        if self.origin is None or self.current is None:
            raise ValueError('Initial state not accepted')
        self.step_index += 1
        f = self.current
        frame = Frame(self.context.episode_id, self.variant.body, self.step_index,
                      f['time_s'] - self.origin, f['door_angles_deg'][1],
                      tuple(f['door_angles_deg']), f['target_held'], f['target_fully_out'],
                      self.interval_collision, self.interval_hold, True)
        self.interval_collision, self.interval_hold = False, True
        return TaskEvidence(self.context.episode_id, self.context.task_id,
                            self.step_index, f['time_s'], frame)

    def _score(self):
        result = self.scorer.result()
        return TaskScore(result.score, result.maximum_score,
                         dict(process=asdict(result), assisted_environment=True,
                              assistance_policy=self.assistance['policy_id'],
                              current_facts=self.current,
                              evidence_scope='native_door_pose_full_box_geometry_and_branch_contacts'))

    def evaluate(self, evidence):
        self.scorer.observe(evidence.data)
        return self._score()

    def check_success(self, evidence, score):
        import os
        if os.environ.get('ARENA_FRIDGE_HANDOFF_DIAGNOSTIC') == '1':
            from .experiments.beef_gripmat_v1.handoff import continue_to_recording_end
            return continue_to_recording_end(self, evidence, score)
        return self.scorer.result().success

    def finalize(self, evidence, score, outcome):
        self.scorer.finalize()
        result = self._score()
        with (self.context.output / 'refrigerator_result.json').open('x') as f:
            json.dump(dict(score=asdict(result), reason=outcome.reason, success=outcome.success,
                           assisted_environment=True), f, indent=2, allow_nan=False)
        return result

    def close(self):
        if self.reader is not None:
            self.reader.close()
            self.reader = None
