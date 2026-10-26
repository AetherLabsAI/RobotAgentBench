"""Default shelf/bag failure resets from task configuration and CPU IK only."""
import json
import os
from copy import deepcopy
from pathlib import Path
from .common import HERE, CURRENT, owned, digest, physics, finish, write_json, require_source


def compose_shelf(output, seed, config):
    from pxr import Usd, UsdGeom, UsdPhysics
    from evaluation.tasks.shelf_pick import initial_state as source, episode_randomization
    from .catalog import generator, definition
    from .poses import place, trace_pose
    from .shelf_checkpoint import target_hand
    frozen = config['shelf']
    layouts = frozen['layouts']
    index = generator('shelf_drink',seed,'source-layout').randrange(len(layouts))
    snapshot = deepcopy(layouts[index])
    snapshot['joint_positions'].update(frozen['ready_active20'])
    snapshot, draw = episode_randomization.perturb(snapshot,seed)
    hand, side = target_hand(snapshot['target'],snapshot['robot'])
    out = owned(output,fresh=True); out.mkdir(parents=True)
    stage = Usd.Stage.CreateNew(str(out/'initial.usda'))
    stage.GetRootLayer().subLayerPaths = [os.path.relpath(source.BASE,out)]
    baseline = Usd.Stage.Open(str(source.BASE))
    for key in ('metersPerUnit','upAxis','timeCodesPerSecond','framesPerSecond'):
        if baseline.HasAuthoredMetadata(key):
            stage.SetMetadata(key,baseline.GetMetadata(key))
    stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
    target = snapshot['target_path']
    paths = {target,source.ROBOT,*(i['prim_path'] for i in snapshot['items'])}
    for root in source.STOCK_ROOTS:
        for child in stage.GetPrimAtPath(root).GetAllChildren():
            child.SetActive(str(child.GetPath()) in paths)
    before = physics(stage)
    for path, pose in [(source.ROBOT,snapshot['robot']),(target,snapshot['target'])]+[
            (i['prim_path'],i) for i in snapshot['items']]:
        p = stage.GetPrimAtPath(path)
        if not p or not p.IsActive():
            raise ValueError('Frozen layout asset missing: '+path)
        place(stage,path,trace_pose(dict(position=pose['position_m'],quaternion=pose['quaternion_wxyz'])))
        if path != source.ROBOT:
            body = UsdPhysics.RigidBodyAPI(p)
            if not body or body.GetKinematicEnabledAttr().Get() or not body.GetRigidBodyEnabledAttr().Get():
                raise ValueError('Target/background is not the original dynamic body')
    case = definition('shelf_drink')
    record = dict(variant='shelf_drink',source_api='4-3',seed=seed,target=target,
        neighbors=[i['prim_path'] for i in snapshot['items']],
        active_hand=hand,hand_selection=side,randomization=draw,
        robot_initialization=dict(positions=snapshot['joint_positions'],
            hold_targets={n:v for n,v in snapshot['joint_positions'].items() if
                '_arm_' in n or '_body_' in n or '_head_' in n or n in
                ('idx41_gripper_l_outer_joint1','idx81_gripper_r_outer_joint1')}),
        source_robot_base=snapshot['robot'],
        maximum_score=case['maximum_score'],completion_policy=case['endpoint'],instruction=case['instruction'],
        source_layout_selection=dict(candidate_index=index,candidate_count=len(layouts),
            selection='seeded_frozen_layout_configuration',resampled=False),
        object_pose_policy='unchanged_approved_layout_pool_from_task_owned_configuration',
        code_sources={**require_source(source,CURRENT),**require_source(episode_randomization,CURRENT)})
    # Runtime support map has a fixed per-tier identity, not a training lookup.
    from evaluation.tasks.shelf_pick.targets import TIER3,TIER2
    record['support'] = TIER3 if snapshot['target_tier']=='tier3' else TIER2
    write_json(out/'source_snapshot.json',snapshot)
    return stage,record,before


def build(output, *, variant, seed):
    if variant not in ('shelf_drink','coffee_bag'):
        raise ValueError('This procedural path supports only shelf and bag')
    from .catalog import generator
    generator(variant,seed,'validate')
    config_path = HERE/'procedural_config.json'
    config = json.loads(config_path.read_text())
    if config['schema'] != 'grasp_failure_recovery.procedural_configuration.v1':
        raise ValueError('Unknown fixed configuration')
    config['config_path'] = str(config_path)
    if variant == 'coffee_bag':
        from .procedural_coffee import compose
        stage,record,before = compose(output,seed,config)
        from evaluation.tasks.coffee_machine.specs import MACHINE,CUP,COUNTER
        record['ik_obstacles'] = [MACHINE,CUP,COUNTER]
    else:
        stage,record,before = compose_shelf(output,seed,config)
    from . import procedural_grasp
    from . import shelf_checkpoint,render_initials,poses,common,catalog
    from . import visual_preload,visual_instance_compare
    out = Path(output).resolve()
    initialization,closure,receipt = procedural_grasp.solve(stage,record,config)
    record.update(robot_initialization=initialization,gripper_closure=closure,
        initialization_policy='target_relative_empty_closed_grasp.v1',
        procedural_grasp=receipt,generation_reads_recorded_trajectory=False,
        source_checkpoint=None,configuration_sha256=digest(config_path),
        preparation_accepted=receipt['accepted_cpu'],
        active_arm_distribution='target_relative_IK_replaces_old_arm_checkpoint_or_jitter',
        fixed_ready_remaining_joints=True)
    record['code_sources'].update({str(config_path):digest(config_path),
        str(Path(__file__).resolve()):digest(__file__),
        str(Path(procedural_grasp.__file__).resolve()):digest(procedural_grasp.__file__)})
    for module in (shelf_checkpoint,render_initials,poses,common,catalog,
                   visual_preload,visual_instance_compare):
        record['code_sources'].update(require_source(module,HERE))
    write_json(out/'procedural_grasp.json',receipt)
    result = finish(stage,out,record,before)
    if not receipt['accepted_cpu']:
        raise ValueError('Target-relative proposal rejected; retained scene and diagnostics; no seed retry: '+
                         str(out/'procedural_grasp.json'))
    return visual_preload.apply(stage,out,result)
