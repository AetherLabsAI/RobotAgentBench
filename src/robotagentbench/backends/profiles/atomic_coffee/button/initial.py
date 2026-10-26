"""Compose the existing atomic 1-3 baseline without changing its parameters.

Source helpers package/place/inspect/randomize are read-only. This task owns
the output layer; source prepare() is NOT called with an out-of-scope output.
No old task path guard is disabled, patched, or temporarily replaced.
"""
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def task_output(path):
    out = Path(path).resolve()
    if out == HERE or not out.is_relative_to(HERE):
        raise ValueError('Output must stay in this exact lighting task')
    return out


def physical_properties(stage):
    return {str(prop.GetPath()): repr(prop.Get() if hasattr(prop, 'Get') else prop.GetTargets())
            for prim in stage.Traverse() for prop in prim.GetProperties()
            if prop.GetName().startswith(('physics:', 'physx', 'drive:'))}


def build(output, *, seed, role, episode=None):
    from .source_audit import verify
    verify(HERE / 'component_sources.json')
    from pxr import Gf, Sdf, Usd, UsdGeom
    from arena_api import l1_scene
    from evaluation.tasks.coffee_machine import initial_state as src, randomization
    from evaluation.tasks.coffee_machine.specs import (
        STATION, MACHINE, CUP, BAG, BUTTON, BEANS, ROBOT, SHARED_ASSETS, episodes)
    if type(seed) is not int or seed < 0 or role not in ('source', 'random'):
        raise ValueError('Explicit nonnegative seed and source/random role required')
    episode = episode or episodes('1-3')[0]
    if episode not in episodes('1-3'):
        raise ValueError('Unknown atomic 1-3 source episode')
    if role == 'random' and episode != episodes('1-3')[0]:
        raise ValueError('Original randomizer starts from its fixed source episode')
    output = task_output(output)
    if output.exists():
        raise ValueError('Fresh initial output required')
    manifest, binding, _ = src.package('1-3', episode)
    output.mkdir(parents=True)
    scene = output / 'initial.usda'
    stage = Usd.Stage.CreateNew(str(scene))
    stage.GetRootLayer().subLayerPaths = [os.path.relpath(p, output) for p in
        (src.REFERENCE, *l1_scene.LAYERS, l1_scene.BASE)]
    stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    stage.GetPrimAtPath('/World').CreateAttribute(l1_scene.MARKER, Sdf.ValueTypeNames.String).Set(l1_scene.POLICY)
    stage.OverridePrim(STATION).SetActive(True)
    if UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(STATION)) != Gf.Matrix4d(1):
        raise ValueError('Original station world frame changed')
    cup = stage.GetPrimAtPath(CUP)
    style = binding['cup_style']

    def set_style(new_style):
        nonlocal style
        style = new_style
        cup.GetReferences().SetReferences([Sdf.Reference(os.path.relpath(
            SHARED_ASSETS / ('cup_' + style + '.usda'), output))])
        cup.SetCustomDataByKey(src.STYLE_KEY, style)

    set_style(style)
    for path, key in ((MACHINE, 'machine'), (CUP, 'cup'), (BAG, 'bag')):
        src.place(stage, path, binding['world'][key])
    src.place(stage, BUTTON, binding['button_local'])
    for index, matrix in enumerate(binding['beans']):
        src.place(stage, BEANS + '/Bean_%03d' % index, matrix)
    record = randomization.apply(stage, '1-3', manifest, seed, set_cup_style=set_style) if role == 'random' else None
    l1_scene.verify(stage)
    inspection = src.inspect_stage(stage, '1-3')
    base = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(ROBOT)).ExtractTranslation()
    if max(abs(a-b) for a, b in zip(base, manifest['robot_base']['root_xyz'])) > 1e-9:
        raise ValueError('Robot source station changed')
    hashes = {layer.realPath: digest(layer.realPath) for layer in stage.GetUsedLayers()
              if layer.realPath and layer.realPath != str(scene)}
    from short_task_robustness.tasks.coffee_bag_interference.build_preview_scene import visual_preload
    visual_receipt = visual_preload(stage, owner_root=HERE)
    stage.GetRootLayer().Save()
    plan = dict(scene=str(scene), api='1-3', source_api='1-3', seed=seed, role=role, source_episode=episode,
                robot_initialization=manifest['robot_initialization'], robot_base=manifest['robot_base'],
                randomization=record, cup_style=style, public_coffee_assets=binding['public_revision'],
                reference_sha256=digest(src.REFERENCE),
                public_binding_sha256=digest(src.EPISODES / '1-3' / episode / 'public_binding.json'),
                episode_manifest_sha256=digest(src.EPISODES / '1-3' / episode / 'episode.json'),
                settle_seconds=src.SETTLE_SECONDS['1-3'], inspection=inspection,
                appearance=dict(mode='original_baseline', seed=None, gen_seed=None, plan_sha256=None),
                source_layers=hashes, scene_sha256=digest(scene), visual_preload=visual_receipt,
                new_physical_parameter_overrides=False, physical_settle_verified=False,
                global_registration=False)
    (output / 'initial.json').write_text(json.dumps(plan, indent=2, allow_nan=False) + '\n')
    return plan
