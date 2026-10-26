"""Per-episode scene layer of the coffee machine family: public house scene (+ seeded appearance) + public shared coffee assets + layout.

The machine, bag, cups and beans are the PUBLIC shared assets (resources/shared_assets/coffee_v1, /World/Task75).
A source episode contributes only its layout (tools/bind_public_assets.py -> public_binding.json): object poses, the
cup style, and for 1-4 the liquid already in the cup. Robot, gripper, physics scene, solver and material parameters
stay the public unified ones; nothing of the producer's robot or physics scene is loaded.

Two roles:
  source  the layout of one packaged source episode; the first packaged episode of an API is its fixed initial state,
          and a recorded trajectory is only ever replayed on the layout it was recorded on;
  random  a seeded layout from randomization.py (the data producer's own samplers).
The base scene is either the public L1 baseline or an environment built from it by the public tools/build_l1_scene.py
(baseline + public layers + seeded appearance, physics untouched). 1-4's liquid is not authored here: the producer's
prefill (vendor_producer/filled_coffee.py) runs in the reader before physics starts.
"""
import hashlib
import json
import math
import os
from pathlib import Path

from .specs import (TASK, REPO, EPISODES, STATION, MACHINE, CUP, BAG, BEANS, LIQUID, ROBOT, MACHINE_BODY, BUTTON,
                    CUP_SUPPORT, CUP_STYLES, SHARED_ASSETS, spec as api_spec, episodes, episode_manifest)

REFERENCE = TASK / 'resources/reference/station.usda'
SCHEMA = 'arena.coffee_machine.initial_state.v3'
PREPARED_MARKER = 'arena:l1ScenePrepared'      # set by tools/build_l1_scene.py on an environment scene_room.usda
FILL_PARTICLES = 10304                        # vendor_producer/filled_coffee.py::fill on any cup style (16 levels x 644)
STYLE_KEY = 'arena:coffeeCupStyle'
# Approved definition tables: initial acceptance after 1 s at rest, for all four APIs.
# (The producer also settles 1.0 s for 1-1 / 1-3 / 1-4; its pour runner settles 0.5 s, run_pour.py:231.)
SETTLE_SECONDS = {'1-1': 1.0, '1-2': 1.0, '1-3': 1.0, '1-4': 1.0}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def package(api, episode):
    """Verified layout of one source episode: (manifest, public binding, liquid layer or None)."""
    from arena_api import coffee_assets
    manifest = episode_manifest(api, episode)
    folder = EPISODES / api / episode
    if manifest['api'] != api or manifest['episode'] != episode:
        raise ValueError('Source episode identity mismatch')
    binding = json.loads((folder / 'public_binding.json').read_text())
    if (binding['api'], binding['episode'], binding['coffee_objects_sha256']) != (api, episode, manifest['coffee_objects_sha256']):
        raise ValueError('Layout binding belongs to another source episode')
    if binding['public_revision'] != coffee_assets.verify_files()['revision']:
        raise ValueError('Public coffee assets changed after the layout was bound; run tools/bind_public_assets.py')
    if binding['cup_style'] not in CUP_STYLES:
        raise ValueError('Unknown cup style')
    matrices = [*binding['world'].values(), binding['button_local'], *(binding['beans'] or [])]
    if not all(len(m) == 16 and all(type(v) in (int, float) and math.isfinite(v) for v in m) for m in matrices):
        raise ValueError('Invalid layout matrix')
    liquid = None       # the packaged final-frame particles stay as reference material only; the run uses the producer's prefill
    values = manifest['robot_initialization']
    for field in ('positions', 'hold_targets'):
        if not values[field] or not all(type(v) in (int, float) and math.isfinite(v) for v in values[field].values()):
            raise ValueError('Invalid recorded reset coordinates')
    return manifest, binding, liquid


def inspect_stage(stage, api):
    """Structural check only; physical acceptance happens after the settle."""
    from pxr import UsdPhysics, UsdGeom
    sp = api_spec(api)
    required = [ROBOT, MACHINE_BODY, BUTTON, CUP_SUPPORT, CUP, BAG]
    if sp.has_beans:
        required += [BEANS + '/Bean_000', BEANS + '/Bean_099']
    missing = [p for p in required if not stage.GetPrimAtPath(p) or not stage.GetPrimAtPath(p).IsActive()]
    if missing:
        raise ValueError('Missing initial-state instances: ' + ', '.join(missing))
    for path in (CUP, BAG, MACHINE_BODY, BUTTON):
        if not stage.GetPrimAtPath(path).HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError('Rigid-body binding missing: ' + path)
    if stage.GetPrimAtPath(MACHINE_BODY).GetAttribute('physics:kinematicEnabled').Get() is not True:
        raise ValueError('The coffee machine body must stay the kinematic body of the public asset')
    group = stage.GetPrimAtPath(BEANS)
    beans = list(group.GetChildren()) if group and group.IsActive() else []
    if len(beans) != (100 if sp.has_beans else 0):
        raise ValueError('Unexpected number of active beans: ' + str(len(beans)))
    style = stage.GetPrimAtPath(CUP).GetCustomDataByKey(STYLE_KEY)
    if style not in CUP_STYLES:
        raise ValueError('Cup style not bound')
    if UsdGeom.GetStageUpAxis(stage) != 'Z' or not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.):
        raise ValueError('Expected Z-up metre scene')
    return dict(required_instances=required, beans=len(beans), cup_style=style, structural_check='passed',
                physical_settle_acceptance=None, task_success=None)


def place(stage, path, matrix):
    """Pose only: one matrix op replaces the asset's default placement; the body starts at rest."""
    from pxr import Gf, UsdGeom, UsdPhysics
    prim = stage.GetPrimAtPath(path)
    if not prim:
        raise ValueError('Missing public coffee prim: ' + path)
    UsdGeom.Xformable(prim).MakeMatrixXform().Set(Gf.Matrix4d(*matrix))
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        body = UsdPhysics.RigidBodyAPI(prim)
        body.CreateVelocityAttr().Set(Gf.Vec3f(0)); body.CreateAngularVelocityAttr().Set(Gf.Vec3f(0))


def prepare(source, output, *, api, episode=None, seed=None, role='source'):
    from pxr import Gf, Sdf, Usd, UsdGeom
    from arena_api import l1_scene
    sp = api_spec(api)
    output = Path(output).resolve()
    # in a frozen snapshot `artifacts` is a link to the live artifacts directory
    if not (output.is_relative_to(TASK.resolve()) or output.is_relative_to((TASK / 'artifacts').resolve())) or output == TASK.resolve():
        raise ValueError('Initial-state output must stay inside the coffee_machine task directory')
    source = Path(source).resolve(strict=True)
    environment = None
    if source != l1_scene.BASE.resolve():
        probe = Usd.Stage.Open(str(source))
        if (source.name != 'scene_room.usda' or not source.is_relative_to((REPO / 'runs').resolve())
                or probe.GetPrimAtPath('/World').GetAttribute(PREPARED_MARKER).Get() is not True):
            raise ValueError('The base scene must be the public L1 baseline or an environment built from it by tools/build_l1_scene.py')
        environment = source
    if role not in ('source', 'random'):
        raise ValueError('Unknown initial-state role')
    if episode is None:
        episode = episodes(api)[0]
    manifest, binding, liquid = package(api, episode)
    output.mkdir(parents=True, exist_ok=False)
    scene = output / 'initial.usda'
    stage = Usd.Stage.CreateNew(str(scene))
    layers = [REFERENCE, source] if environment else [REFERENCE, *l1_scene.LAYERS, source]     # an environment already carries the public layers
    stage.GetRootLayer().subLayerPaths = [os.path.relpath(p, output) for p in layers]
    stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    stage.GetPrimAtPath('/World').CreateAttribute(l1_scene.MARKER, Sdf.ValueTypeNames.String).Set(l1_scene.POLICY)
    stage.OverridePrim(STATION).SetActive(True)
    if UsdGeom.Xformable(stage.GetPrimAtPath(STATION)).ComputeLocalToWorldTransform(0) != Gf.Matrix4d(1):
        raise ValueError('The public coffee group is expected in the world frame')
    cup = stage.GetPrimAtPath(CUP)
    cup.GetReferences().AddReference(os.path.relpath(SHARED_ASSETS / f'cup_{binding["cup_style"]}.usda', output))
    cup.SetCustomDataByKey(STYLE_KEY, binding['cup_style'])
    place(stage, MACHINE, binding['world']['machine'])
    place(stage, BUTTON, binding['button_local'])
    place(stage, CUP, binding['world']['cup'])
    place(stage, BAG, binding['world']['bag'])
    if sp.has_beans:
        for i, matrix in enumerate(binding['beans']):
            place(stage, f'{BEANS}/Bean_{i:03d}', matrix)
    else:
        # the source episodes of this API have no beans anywhere in the scene
        stage.OverridePrim(BEANS).SetActive(False)
    l1_scene.verify(stage)
    random_record = None
    style = binding['cup_style']
    if role == 'random':
        from . import randomization

        def set_cup_style(new_style):
            nonlocal style
            style = new_style
            cup.GetReferences().SetReferences([Sdf.Reference(os.path.relpath(SHARED_ASSETS / f'cup_{new_style}.usda', output))])
            cup.SetCustomDataByKey(STYLE_KEY, new_style)

        random_record = randomization.apply(stage, api, manifest, seed, set_cup_style=set_cup_style)
    report = inspect_stage(stage, api)
    base = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT)).ComputeLocalToWorldTransform(0).ExtractTranslation()
    if max(abs(a - b) for a, b in zip(base, manifest['robot_base']['root_xyz'])) > 1e-9:
        raise ValueError('Robot station differs from the source episode')
    stage.GetRootLayer().Save()
    plan = dict(
        schema=SCHEMA, api=api, task_no=sp.task_no, role=role, source_episode=episode, seed=seed,
        producer_status_label=manifest['producer_status_label'], source_seed=manifest['seed'],
        robot_initialization=dict(positions=manifest['robot_initialization']['positions'],
                                  hold_targets=manifest['robot_initialization']['hold_targets']),
        robot_base=manifest['robot_base'], facts=manifest['facts'], randomization=random_record,
        cup_style=style, source_cup_style=binding['cup_style'], public_coffee_assets=binding['public_revision'],
        liquid=(dict(method='vendor_producer/filled_coffee.py::fill at the cup pose, authored before physics (data producer\'s post-brew fixture)',
                     particles_expected=FILL_PARTICLES, packaged_final_frame_reference=binding['liquid']) if sp.api == '1-4' else None),
        beans_active=sp.has_beans, environment=(str(environment) if environment else None),
        environment_sha256=(digest(environment) if environment else None), base_scene='environment' if environment else 'baseline',
        reference_sha256=digest(REFERENCE), coffee_objects_sha256=manifest['coffee_objects_sha256'],
        episode_manifest_sha256=digest(EPISODES / api / episode / 'episode.json'),
        public_binding_sha256=digest(EPISODES / api / episode / 'public_binding.json'),
        settle_seconds=SETTLE_SECONDS[api], inspection=report,
        task_objects='public shared coffee assets; layout of the source episode',
        physical_parameters_overridden=False, robot_or_physics_scene_from_producer=False)
    (output / 'initial.json').write_text(json.dumps(plan, indent=2, allow_nan=False) + '\n')
    return scene


if __name__ == '__main__':
    import argparse
    from arena_api import l1_scene
    parser = argparse.ArgumentParser()
    parser.add_argument('--api', required=True)
    parser.add_argument('--episode')
    parser.add_argument('--role', default='source')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(prepare(l1_scene.BASE, args.output, api=args.api, episode=args.episode, seed=args.seed, role=args.role))
