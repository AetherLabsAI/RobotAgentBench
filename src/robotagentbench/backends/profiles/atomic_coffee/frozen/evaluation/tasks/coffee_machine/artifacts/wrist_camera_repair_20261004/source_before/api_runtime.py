"""Coffee CPU animation revision (2026-10-04). Physical rigid-body robot/object evidence;
visual-only coffee surfaces and stream. No PBD liquid is instantiated.
Historical particle-based sources and videos are retained separately."""
import copy
import hashlib
import json
from pathlib import Path
from uuid import UUID

import numpy as np

from arena_api.task_runtime import WorkerTask, TaskEvidence, TaskScore, RobotInitialization
from arena_api.policy_execution import SUBSTEPS, PHYSICS_DT
from arena_api.physics_defaults import steps_for_seconds
from .specs import TASK, REPO, CUP, BAG, RESET_SCHEMA, spec as api_spec
from .initial_state import prepare, package, SETTLE_SECONDS, FILL_PARTICLES
from .process_rules import Process, RULE_VERSION
from .facts import COUNTER, TRAY

INITIAL_TILT_DEG = 11.5            # "直立": the upright threshold of the same definition tables
INITIAL_CUP_XY_M = 0.006           # README 1-4: on the placement point within 6 mm
INITIAL_BAG_TILT_DEG = 10.0        # README 1-2: upright bag
INITIAL_LIQUID_PERCENT = 95        # README 1-4
BUTTON_RELEASED_M = 0.001          # README 1-3: button up
ENVIRONMENTS = REPO / 'runs/coffee_machine_environment'   # the public appearance tools accept only the runs/ tree


def environment_dir(seed):
    return ENVIRONMENTS / f'seed-{int(seed)}'


def appearance_seed(api, episode, role, seed):
    """The seed the public appearance planner is run with (gen_seed = 1e6 + seed, as the producer's episodes)."""
    if role == 'random':
        return dict(seed=seed, source='episode seed of the random layout')
    manifest = json.loads((TASK / 'resources/source_episodes' / api / episode / 'episode.json').read_text())
    if api == '1-4':
        derived = json.loads((TASK / 'references/data_side/appearance_plans/1-4.json').read_text())['episodes']
        if episode not in derived:
            raise ValueError('No derived appearance seed for ' + episode)
        return dict(seed=int(derived[episode]['seed']), source='derived: seed of the placement run this episode continued from (the producer kept no per-episode appearance record)')
    return dict(seed=int(manifest['seed']), source='episode seed (150/150 producer appearance plans recompute from it)')


class CoffeeMachineRuntime(WorkerTask):
    def __init__(self, context):
        from .registration import normalizer
        self.spec = api_spec(context.task_id)
        self.context = context
        self.parameters = normalizer(context.task_id)(context.task_parameters)
        if any(context.randomization.values()):
            raise ValueError('House randomization flags do not apply; use task_parameters.random_layout')
        output = Path(context.output).resolve()
        if not output.is_relative_to((TASK / 'artifacts').resolve()):
            raise ValueError('Output must stay in coffee_machine/artifacts')
        self.settle_steps = steps_for_seconds(SETTLE_SECONDS[self.spec.api])
        self.requires_gpu_dynamics = False  # visual coffee has no PhysX particles
        self.setup = TASK / 'artifacts/episodes' / str(UUID(context.episode_id)) / 'setup'
        self.process = Process(self.spec.api)
        self.reader = None
        self.control_index = -1
        self.pending = None
        self.plan = None

    # ---- scene ----
    def initialize(self, source):
        from arena_api import l1_scene
        self.base = Path(source).resolve(strict=True)
        if self.base != l1_scene.BASE.resolve():
            raise ValueError('The service scene must be the public L1 baseline; the environments are built from it')
        self.setup.parent.mkdir(parents=True, exist_ok=True)
        role = 'random' if self.parameters['random_layout'] else 'source'
        self.appearance = appearance_seed(self.spec.api, self.parameters['source_episode'], role, self.context.seed)
        # pre-built in the live Git tree by tools/build_environments.py and frozen with the snapshot; never built here
        environment = environment_dir(self.appearance['seed']) / 'scene_room.usda'
        if not environment.is_file() or not environment.with_suffix('.json').is_file():
            raise RuntimeError('Environment for appearance seed %d is not pre-built: run tools/build_environments.py in the live tree and re-freeze' % self.appearance['seed'])
        report = json.loads(environment.with_suffix('.json').read_text())
        if report['gen_seed'] != 1_000_000 + self.appearance['seed'] or report['physics_authored'] or report['layout_authored']:
            raise RuntimeError('Environment does not match the requested appearance seed or authored more than appearance')
        if hashlib.sha256(environment.read_bytes()).hexdigest() != report['scene_sha256']:
            raise RuntimeError('Environment scene changed after it was built')
        self.appearance.update(gen_seed=report['gen_seed'], style=report['style'], lighting_scheme=report['lighting_scheme'],
                               plan_sha256=report['plan_sha256'], environment=str(environment), report=str(environment.with_suffix('.json')),
                               scene_sha256=report['scene_sha256'])
        scene = prepare(environment, self.setup, api=self.spec.api, episode=self.parameters['source_episode'],
                        seed=self.context.seed if role == 'random' else None, role=role)
        self.plan = json.loads((self.setup / 'initial.json').read_text())
        return scene

    def attach(self, stage):
        from .isaac_evidence import CoffeeReader
        self.stage = stage
        self.reader = CoffeeReader(stage, Path(self.context.output) / 'coffee_detection', self.context.episode_id, self.spec.api)

    def robot_initialization(self):
        values = self.plan['robot_initialization']
        return RobotInitialization(dict(values['positions']), dict(values['hold_targets']))

    def on_physics_ready(self, world):
        self.world = world
        self.reader.initialize(world)

    def after_physics(self, helper_report):
        if helper_report['events'] or helper_report['docked_object_count'] or helper_report['tray_dock_enabled']:
            raise RuntimeError('Manipulation helper invalidates coffee machine evidence')
        self.reader.after_step()

    # ---- initial acceptance ----
    def validate_initial(self, metadata):
        reader, sp = self.reader, self.spec
        rows = list(reader.history)
        if not rows:
            raise RuntimeError('Missing initial-state evidence')
        window = rows
        now = reader.latest
        cup, bag = now['contacts'][CUP], now['contacts'][BAG]
        placement = reader.placement(np.asarray(now['poses'][CUP]['position_m']), now['poses'][CUP]['quaternion_wxyz'])
        span = lambda path: float(np.ptp([r['poses'][path]['position_m'] for r in window], axis=0).max())
        checks = dict(
            history_complete=len(rows) == reader.history.maxlen and reader.step + 1 >= self.settle_steps,
            physics_valid=not metadata['physics_issues'],
            self_collision_enabled=metadata['self_collisions_enabled'] is True,
            source_layers_unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in reader.hashes.items()),
            objects_at_rest=max(span(CUP), span(BAG)) <= 0.002,
            robot_touches_nothing=not any(r['contacts'][o]['robot'] for r in window for o in (CUP, BAG))
                                  and not any(r['robot_machine'] or r['robot_on_button'] for r in window),
            nothing_on_the_ground=not any(r['contacts'][o]['ground'] for r in window for o in (CUP, BAG)),
            cup_upright=now['tilt_deg'][CUP] <= INITIAL_TILT_DEG,
            bag_upright_on_counter=now['tilt_deg'][BAG] <= INITIAL_BAG_TILT_DEG and COUNTER in bag['supports'])
        measured = dict(cup_tilt_deg=now['tilt_deg'][CUP], bag_tilt_deg=now['tilt_deg'][BAG], cup_supports=cup['supports'],
                        bag_supports=bag['supports'], cup_xy_error_m=placement.xy_error_m, cup_span_m=span(CUP), bag_span_m=span(BAG),
                        button_displacement_m=now['button_displacement_m'])
        if sp.api == '1-1':
            checks.update(cup_on_counter=COUNTER in cup['supports'] and not cup['machine'])
        else:
            checks.update(cup_on_tray=TRAY in cup['supports'],
                          cup_on_placement_point=placement.xy_error_m <= INITIAL_CUP_XY_M and placement.outlet_over_opening)
        if sp.has_beans:
            in_bag, in_hopper = reader.beans_in_bag(), now['device']['bean_count']
            measured.update(beans_in_bag=in_bag, beans_in_hopper=in_hopper)
            if sp.api == '1-2':
                checks.update(beans_in_bag=in_bag == 100, hopper_empty=in_hopper == 0)
            else:
                checks.update(beans_in_hopper=in_hopper == 100)
        frame = reader.cup_in_head_frame()
        measured.update(cup_in_head_frame=frame['visible'], head_frame_corners=frame['corners'])
        if sp.has_device:
            # README 1-3: the producer's layout gate is part of the initial acceptance
            checks.update(cup_in_head_frame=frame['visible'])
            checks.update(machine_idle=not now['device']['dispensing'] and reader.fluid.count == 0,
                          button_up=-now['button_displacement_m'] <= BUTTON_RELEASED_M)
        if sp.api == '1-4':
            liquid = reader.liquid_now()
            measured.update(liquid_mode='visual_cpu_v1', visual_volume_units=liquid['particles'])
            checks.update(liquid_prefilled=liquid['particles'] == FILL_PARTICLES,
                          liquid_in_cup=liquid['particles'] > 0 and liquid['in_cup'] * 100 >= INITIAL_LIQUID_PERCENT * liquid['particles'])
        checks = {k: bool(v) for k, v in checks.items()}
        receipt = dict(schema='arena.coffee_machine.initial_acceptance.v1', api=sp.api, source_episode=self.plan['source_episode'],
                       role=self.plan['role'], cup_style=self.plan['cup_style'], appearance=self.appearance, layout_randomization=self.plan['randomization'],
                       checks=checks, accepted=all(checks.values()), measured=measured,
                       time_s=now['time_s'], reader_step=reader.step, settle_seconds=SETTLE_SECONDS[sp.api],
                       window_seconds=len(window) * PHYSICS_DT, settle_counts_toward_task_budget=False)
        Path(self.context.output, 'initial_acceptance.json').write_text(json.dumps(receipt, indent=2, allow_nan=False, default=list) + '\n')
        metadata['task_initial_acceptance'] = json.loads(json.dumps(receipt, default=list))
        if not receipt['accepted']:
            raise RuntimeError('Coffee machine initial acceptance failed: ' + ', '.join(k for k, v in checks.items() if not v))
        metadata['replay_reset_contract'] = dict(
            schema=RESET_SCHEMA, task_id=sp.api, rule_version=RULE_VERSION, parameters=dict(self.parameters),
            source_episode=self.plan['source_episode'], role=self.plan['role'], cup_style=self.plan['cup_style'],
            public_coffee_assets=self.plan['public_coffee_assets'], public_binding_sha256=self.plan['public_binding_sha256'],
            episode_manifest_sha256=self.plan['episode_manifest_sha256'], reference_sha256=self.plan['reference_sha256'],
            robot_initialization=self.plan['robot_initialization'], layout_randomization=self.plan['randomization'],
            appearance=dict(seed=self.appearance['seed'], gen_seed=self.appearance['gen_seed'], plan_sha256=self.appearance['plan_sha256']),
            liquid_prefill=(dict(code='visual_coffee.py', mode='visual_cpu_v1', volume_units=FILL_PARTICLES) if sp.api == '1-4' else None),
            seed=self.context.seed, randomization=dict(self.context.randomization), settle_steps=self.settle_steps,
            finger_material_policy=metadata['finger_pad_material_policy']['policy_id'])
        reader.begin_episode()

    # ---- evidence and rules ----
    def collect_evidence(self):
        delta = self.reader.step - self.reader.start_step
        if delta % SUBSTEPS or delta // SUBSTEPS != self.control_index + 1:
            raise RuntimeError('Missing, repeated or misaligned control evidence')
        self.control_index += 1
        self.pending = TaskEvidence(self.context.episode_id, self.context.task_id, self.control_index,
                                    float(self.world.current_time), self.reader.frame())
        return self.pending

    def evaluate(self, evidence):
        if evidence is not self.pending:
            raise RuntimeError('Unknown or repeated task evidence')
        self.pending = None
        self.process.observe(evidence.data)
        summary = self.process.summary()
        return TaskScore(summary['score'], summary['maximum_score'], dict(process=summary, facts=evidence.data.as_dict()))

    def check_success(self, evidence, score):
        return bool(score.details['process']['success'])

    def check_failure(self, evidence, score):
        return score.details['process']['failure']

    def finalize(self, evidence, score, outcome):
        self.process.finalize(outcome.reason)
        summary = self.process.summary()
        return TaskScore(summary['score'], summary['maximum_score'],
                         dict(score.details, process=summary, settlement_reason=outcome.reason, source_exhaustion_is_success=False))

    def close(self):
        if self.reader is not None:
            self.reader.close()
        if self.plan is not None:
            package(self.spec.api, self.plan['source_episode'])     # the packaged layout must not change under a run
