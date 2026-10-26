"""Shelf-pick WorkerTask (API 4-1 … 4-14): trusted physical evidence and approved arithmetic only."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from arena_api.task_runtime import WorkerTask,TaskEvidence,TaskScore,RobotInitialization
from arena_api.policy_execution import SUBSTEPS,PHYSICS_DT
from .initial_state import TASK,REPO,ROBOT,BASE,BASELINE,snapshot_path,NEWDATA
from .targets import spec as target_spec
from .process_rules import ProcessScorer,MAXIMUM_SCORE
from .shelf_collision import RULE_VERSION


class ShelfPickRuntime(WorkerTask):
    def __init__(self,context):
        self.spec=target_spec(context.task_id)
        self.context=context;self.reader=None;self.control_index=-1;self.pending=None
        self.scorer=ProcessScorer(context.episode_id,PHYSICS_DT,self.spec.api_id)

    def initialize(self,source):
        """Fixed initial state = one accepted source episode (user decision 2026-09-23)."""
        self.base=Path(source).resolve(strict=True)
        self.snapshot=NEWDATA/self.spec.api_id/'fixed.json'
        output=TASK/'artifacts/episodes'/self.context.episode_id/'setup/fixed'
        output.parent.mkdir(parents=True,exist_ok=True)
        with (output.parent/'fixed.log').open('x') as log:
            subprocess.run(['bash',str(TASK/'agent_probe_v1/run_cpu_usd.sh'),'-m',
                'evaluation.tasks.shelf_pick.initial_state','--api',self.spec.api_id,'--output',str(output),
                '--base',str(self.base),'--snapshot',str(self.snapshot)],
                cwd=REPO,env={**os.environ,'ARENA_PYTHON':sys.executable},stdout=log,stderr=subprocess.STDOUT,check=True)
        self.plan=json.loads((output/'initial-state.json').read_text())
        if self.plan['api_id']!=self.spec.api_id or self.plan['target']!=self.spec.target:raise RuntimeError('Initial layer identity mismatch')
        return output/'initial.usda'

    def randomize(self,scene):
        """Randomized initial state (user decision 2026-09-23): the item layout comes
        from a held-out accepted source episode of this API, and the robot pose comes
        from the ported perturbation policy (base XY +/-5 cm, yaw +/-5 deg, arm joints
        +/-4 deg, waist and head fixed) seeded by this job's seed. Nothing here touches
        the trajectory, the assets or any physics parameter."""
        from .registration import validate_randomization
        from .episode_randomization import perturb
        validate_randomization(self.context.randomization)
        if not self.context.randomization.get('layout'):
            return scene
        if self.base!=BASE:raise RuntimeError('Randomization requires the task scene '+str(BASE))
        output=TASK/'artifacts/episodes'/self.context.episode_id/'setup/random'
        output.parent.mkdir(parents=True,exist_ok=True)
        held_out=json.loads((NEWDATA/self.spec.api_id/'random.json').read_text())
        perturbed,receipt=perturb(held_out,self.context.seed)
        self.snapshot=output.parent/'random-snapshot.json'
        self.snapshot.write_text(json.dumps(perturbed,indent=2,allow_nan=False)+'\n')
        (output.parent/'episode-randomization.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
        with (output.parent/'random-state.log').open('x') as log:
            subprocess.run(['bash',str(TASK/'agent_probe_v1/run_cpu_usd.sh'),'-m',
                'evaluation.tasks.shelf_pick.initial_state','--api',self.spec.api_id,'--output',str(output),
                '--base',str(self.base),'--snapshot',str(self.snapshot)],
                cwd=REPO,env={**os.environ,'ARENA_PYTHON':sys.executable},stdout=log,stderr=subprocess.STDOUT,check=True)
        self.plan=json.loads((output/'initial-state.json').read_text())
        self.plan['episode_randomization']=receipt
        if self.plan['api_id']!=self.spec.api_id or self.plan['target']!=self.spec.target:raise RuntimeError('Random layer identity mismatch')
        return output/'initial.usda'

    def attach(self,stage):
        from .isaac_detection import IsaacShelfReader
        self.stage=stage
        self.reader=IsaacShelfReader(stage,self.context.output/'shelf_detection',self.context.episode_id,self.plan['neighbors'],
                                     spec=self.spec,targets=self.plan['targets'],stock_roots=self.plan.get('stock_roots'),
                                     object_supports=(self.plan.get('source_episode') or {}).get('item_supports'))

    def robot_initialization(self):
        from pxr import UsdPhysics
        hold={}
        for name,value in self.plan['robot_initial_positions'].items():
            p=self.stage.GetPrimAtPath(ROBOT+'/joints/'+name)
            axis='angular' if p.IsA(UsdPhysics.RevoluteJoint) else 'linear'
            if p.HasAPI(UsdPhysics.DriveAPI,axis):hold[name]=value
        return RobotInitialization(dict(self.plan['robot_initial_positions']),hold)

    def on_physics_ready(self,world):
        self.world=world;self.reader.initialize(world)

    def after_physics(self,helper_report):
        if helper_report['events'] or helper_report['docked_object_count'] or helper_report['tray_dock_enabled']:
            raise RuntimeError('Manipulation helper invalidates shelf task evidence')
        self.reader.after_step()

    def validate_initial(self,metadata):
        reader=self.reader;cfg=reader.config
        if not reader.history:raise RuntimeError('Missing initial-state evidence')
        now=reader.latest['time_s']
        from .physical_detection import initial_window
        rows,enough=initial_window(reader.history,cfg.initial_history_s,PHYSICS_DT)
        objects=(self.spec.target,*self.plan['neighbors'])
        checks=dict(history_complete=enough,physics_valid=not metadata['physics_issues'],
            self_collision_enabled=metadata['self_collisions_enabled'] is True,
            all_stock_supported=all(all(r['contacts']['stock_supported'][p] for p in objects) for r in rows),
            no_robot_stock_contact=not any(r['contacts']['robot_contacts'] for r in rows),
            no_grasp=not any(any(r['grasped'].values()) for r in rows),
            no_shelf_collision=not any(r['contacts']['shelf_contact'] for r in rows),
            no_neighbor_collision=not any(r['contacts']['neighbor_contacts'] for r in rows),
            no_floor_or_workspace_loss=not any(r['contacts']['target_on_floor'] or r['outside_workspace'] for r in rows),
            # Each object is measured against the board it stands on: a source episode
            # fills both tiers, so one shared support top would reject the other tier.
            no_stock_penetration=all(reader.geometry.bottom_height(p,r['poses'][p])
                                     >=reader.geometry.support_tops.get(reader.object_supports.get(p,reader.geometry.support),reader.geometry.top)
                                     -cfg.support_face_tolerance_m for r in rows for p in objects),
            stable_stock=bool(rows) and all(np.ptp([r['poses'][p]['position_m'] for r in rows],axis=0).max()<=cfg.initial_position_span_m for p in objects),
            source_layers_unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in reader.hashes.items()))
        receipt=dict(schema='arena.shelf_pick.initial_acceptance.v1',api_id=self.spec.api_id,checks=checks,
                     accepted=all(checks.values()),time_s=now,reader_step=reader.step,
                     numerical_tolerances=asdict(cfg),geometry=reader.geometry.description())
        (self.context.output/'initial_acceptance.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
        if not receipt['accepted']:raise RuntimeError('Shelf initial acceptance failed: '+str(checks))
        metadata['task_initial_acceptance']=receipt
        metadata['replay_reset_contract']=dict(schema='arena.shelf_pick.reset.v2',task_id=self.spec.api_id,rule_version=self.scorer.rule_version,
            initial_snapshot_sha256=hashlib.sha256(Path(self.snapshot).read_bytes()).hexdigest(),
            source_episode=self.plan.get('source_episode'),episode_randomization=self.plan.get('episode_randomization'),
            source_content_sha256=sorted(set(self.plan['dependency_sha256'].values())),
            joint_positions=self.plan['robot_initial_positions'],target=self.plan['target'],acceptable_targets=list(self.plan['targets']),
            support=self.plan['support'],neighbors=self.plan['neighbors'],seed=self.context.seed,
            randomization=dict(self.context.randomization),detector_config=asdict(cfg),
            geometry=reader.geometry.description(),shelf_collision=reader.shelf_binding,settle_steps=self.settle_steps,
            finger_material_policy=metadata['finger_pad_material_policy']['policy_id'])
        reader.begin_episode();self.start_reader_step=reader.step

    def collect_evidence(self):
        delta=self.reader.step-self.start_reader_step
        if delta%SUBSTEPS or delta//SUBSTEPS!=self.control_index+1:
            raise RuntimeError('Missing, repeated or misaligned control evidence')
        self.control_index+=1
        self.pending=TaskEvidence(self.context.episode_id,self.context.task_id,self.control_index,
                                  float(self.world.current_time),self.reader.take_batch())
        return self.pending

    def evaluate(self,evidence):
        if evidence is not self.pending:raise RuntimeError('Unknown or repeated task evidence')
        result=self.scorer.observe_batch(evidence.data);self.pending=None
        return TaskScore(result['score'],MAXIMUM_SCORE,dict(process=result,events=list(self.scorer.events)))

    def check_success(self,evidence,score):
        return score.details['process']['success']

    def check_failure(self,evidence,score):
        return score.details['process']['failure']

    def finalize(self,evidence,score,outcome):
        result=self.scorer.finalize(outcome.reason)
        return TaskScore(result['score'],MAXIMUM_SCORE,dict(score.details,process=result,
            events=list(self.scorer.events),settlement_reason=outcome.reason))

    def close(self):
        if self.reader is not None:self.reader.close()
