"""Coffee-bag layout composer from frozen configuration, never trajectories.

This module only restores the fixed station and invokes the existing atomic
1-2 object sampler. The caller owns target-relative IK, closure, and finish().
No source episode directory, control plan, or recorded robot frame is opened.
"""
from copy import deepcopy
import json
import os
from pathlib import Path

from .common import (HERE, COFFEE_SOURCE, CURRENT, digest, owned, physics,
                     require_source)


def compose(output, seed, config):
    """Return ``(stage, record, physical_baseline)`` for one sampled bag scene.

    ``config['coffee']`` supplies frozen ``manifest``, ``binding``,
    ``ready_initialization``, ``episode``, and ``active_hand``. Optional
    ``config['config_path']`` identifies the task-owned configuration file.
    Fixed tool orientation is intentionally interpreted by the caller, not
    substituted with a recorded or nearest-frame orientation here.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom
    from arena_api import l1_scene
    from evaluation.tasks.coffee_machine import initial_state as source
    from evaluation.tasks.coffee_machine import randomization, specs

    if type(seed) is not int or not 1 <= seed <= 4294967295:
        raise ValueError('Positive uint32 seed required')
    frozen = config['coffee']
    manifest = deepcopy(frozen['manifest'])
    binding = deepcopy(frozen['binding'])
    initialization = deepcopy(frozen['ready_initialization'])
    episode = frozen['episode']
    provenance_path=HERE/'procedural_coffee_provenance.json'
    provenance=json.loads(provenance_path.read_text())
    if provenance['episode']!=episode:
        raise ValueError('Frozen coffee provenance belongs to a different layout')
    hand = frozen['active_hand']
    if hand != 'r':
        raise ValueError('Frozen coffee-bag task uses the original right arm')
    if (manifest.get('api') != '1-2' or binding.get('api') != '1-2'
            or manifest.get('episode') != episode
            or binding.get('episode') != episode):
        raise ValueError('Frozen coffee layout identity mismatch')
    if binding.get('coffee_objects_sha256') != manifest.get('coffee_objects_sha256'):
        raise ValueError('Frozen coffee layout and object binding disagree')
    if binding.get('cup_style') not in specs.CUP_STYLES:
        raise ValueError('Unknown frozen coffee cup style')
    if len(binding.get('beans') or []) != 100:
        raise ValueError('Coffee-bag initial state requires all 100 beans')
    if not initialization.get('positions') or not initialization.get('hold_targets'):
        raise ValueError('Frozen coffee ready initialization is incomplete')

    l1_root = (COFFEE_SOURCE if Path(l1_scene.__file__).resolve().is_relative_to(
        COFFEE_SOURCE.resolve()) else CURRENT)
    code_sources = {
        str(provenance_path):digest(provenance_path),
        **require_source(source, COFFEE_SOURCE),
        **require_source(randomization, COFFEE_SOURCE),
        **require_source(specs, COFFEE_SOURCE),
        **require_source(l1_scene, l1_root),
        str(Path(__file__).resolve()): digest(__file__),
    }
    if config.get('config_path') is not None:
        config_path = owned(config['config_path'])
        code_sources[str(config_path)] = digest(config_path)

    out = owned(output, fresh=True)
    out.mkdir(parents=True)
    stage = Usd.Stage.CreateNew(str(out / 'initial.usda'))
    stage.GetRootLayer().subLayerPaths = [os.path.relpath(path, out) for path in
        (source.REFERENCE, *l1_scene.LAYERS, l1_scene.BASE)]
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
    stage.GetPrimAtPath('/World').CreateAttribute(
        l1_scene.MARKER, Sdf.ValueTypeNames.String).Set(l1_scene.POLICY)
    stage.OverridePrim(specs.STATION).SetActive(True)
    if UsdGeom.XformCache().GetLocalToWorldTransform(
            stage.GetPrimAtPath(specs.STATION)) != Gf.Matrix4d(1):
        raise ValueError('Original coffee station frame changed')

    style = binding['cup_style']

    def set_cup_style(value):
        nonlocal style
        if value not in specs.CUP_STYLES:
            raise ValueError('Unknown sampled cup style')
        style = value
        prim = stage.GetPrimAtPath(specs.CUP)
        prim.GetReferences().SetReferences([Sdf.Reference(os.path.relpath(
            specs.SHARED_ASSETS / f'cup_{value}.usda', out))])
        prim.SetCustomDataByKey(source.STYLE_KEY, value)

    set_cup_style(style)
    for path, key in ((specs.MACHINE, 'machine'), (specs.CUP, 'cup'),
                      (specs.BAG, 'bag')):
        source.place(stage, path, binding['world'][key])
    source.place(stage, specs.BUTTON, binding['button_local'])
    for index, matrix in enumerate(binding['beans']):
        source.place(stage, f'{specs.BEANS}/Bean_{index:03d}', matrix)
    draw = randomization.apply(stage, '1-2', manifest, seed,
                               set_cup_style=set_cup_style)
    # Chosen existing cup asset is part of the approved object sampler, not a
    # physical override. All following pose/IK authoring must preserve this.
    before = physics(stage)
    l1_scene.verify(stage)
    inspection = source.inspect_stage(stage, '1-2')

    from .catalog import definition
    case = definition('coffee_bag')
    record = dict(
        variant='coffee_bag', failure_kind=case['failure_kind'], api='1-2',
        source_api='1-2', seed=seed, role='random', source_episode=episode,
        robot_initialization=initialization, active_hand=hand,
        robot_base=deepcopy(manifest['robot_base']), randomization=draw,
        cup_style=style, public_coffee_assets=binding['public_revision'],
        public_binding_sha256=provenance['public_binding_sha256'],
        episode_manifest_sha256=provenance['episode_manifest_sha256'],
        reference_sha256=digest(source.REFERENCE),
        settle_seconds=source.SETTLE_SECONDS['1-2'], inspection=inspection,
        appearance=dict(mode='original_baseline', seed=None, gen_seed=None,
                        plan_sha256=None),
        target=specs.BAG, time_limit_s=specs.spec('1-2').time_limit_s,
        maximum_score=case['maximum_score'], completion_policy=case['endpoint'],
        instruction=case['instruction'], code_sources=code_sources,
        object_pose_policy='original_atomic_sampler_from_frozen_layout_configuration',
        configuration_policy='frozen_task_configuration_no_recorded_trajectory',
        generation_reads_recorded_trajectory=False,
    )
    return stage, record, before
