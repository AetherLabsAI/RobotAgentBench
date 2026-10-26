"""Build the public environment one coffee-machine episode runs in: L1 baseline + public layers + seeded appearance.

Same public functions tools/build_l1_scene.py uses for appearance (arena_api.l1_runtime.appearance_plan /
apply_appearance, restore_kitchen_bindings), nothing else: no layout topology, no persisted physics export, no shelf
policy authoring. The worker applies its runtime physics policies to this scene exactly as it does to the plain
baseline, so the physics of the environment is the baseline's (tools/check_environment_physics.py proves it).
Appearance is render-only (lights, materials, textures); gen_seed = 1e6 + seed, as in the data producer's episodes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

TASK = Path(__file__).resolve().parents[1]
REPO = TASK.parents[2]
sys.path.insert(0, str(REPO))
SCHEMA = 'arena.coffee_machine.environment.v1'


def build(output, seed, base=None):
    from pxr import Usd, Sdf, UsdGeom
    from arena_api import l1_scene, l1_runtime
    base = Path(base or l1_scene.BASE).resolve(strict=True)
    output = Path(output).resolve()
    if output.exists() or output.name != 'scene_room.usda' or not l1_runtime.is_scene_output_directory(output.parent):
        raise ValueError('Use a new scene_room.usda under the public runs/ tree')
    if base != l1_scene.BASE.resolve():
        raise ValueError('The environment is built from the public L1 baseline only')
    output.parent.mkdir(parents=True, exist_ok=True)
    source = Usd.Stage.Open(str(base))
    stage = Usd.Stage.CreateNew(str(output))
    for key in ('upAxis', 'metersPerUnit', 'timeCodesPerSecond', 'framesPerSecond'):
        stage.SetMetadata(key, source.GetMetadata(key))
    stage.GetRootLayer().subLayerPaths = [os.path.relpath(p, output.parent) for p in (*l1_scene.LAYERS, base)]
    stage.OverridePrim('/World').CreateAttribute(l1_scene.MARKER, Sdf.ValueTypeNames.String).Set(l1_scene.POLICY)
    stage.OverridePrim('/World').CreateAttribute('arena:l1ScenePrepared', Sdf.ValueTypeNames.Bool).Set(True)
    stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
    l1_scene.verify(stage)
    robot_before = UsdGeom.Xformable(stage.GetPrimAtPath('/World/Robot')).GetLocalTransformation()
    kitchen_bindings = l1_runtime.restore_kitchen_bindings(stage)      # render bindings only
    stage.GetRootLayer().Save()
    l1_runtime.open_session(output)
    plan = l1_runtime.appearance_plan(seed)
    applied = l1_runtime.apply_appearance(output, plan)
    if applied.get('physics_effect') != 'none' or applied['gen_seed'] != 1_000_000 + seed:
        raise RuntimeError('Appearance application reported a physics effect or a different seed')
    reopened = Usd.Stage.Open(str(output))
    l1_scene.verify(reopened)
    if UsdGeom.Xformable(reopened.GetPrimAtPath('/World/Robot')).GetLocalTransformation() != robot_before:
        raise RuntimeError('Appearance changed the robot station')
    result = dict(schema=SCHEMA, scene=str(output), baseline=str(base), seed=seed, gen_seed=applied['gen_seed'],
                  style=applied['style'], lighting_scheme=applied['lighting_scheme'],
                  plan_sha256=hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
                  appearance_plan=plan, appearance=applied, kitchen_bindings=kitchen_bindings,
                  physics_authored=False, layout_authored=False, scene_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    output.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    a = p.parse_args()
    r = build(a.output, a.seed)
    print(json.dumps({k: r[k] for k in ('scene', 'seed', 'gen_seed', 'style', 'lighting_scheme', 'plan_sha256')}))


if __name__ == '__main__':
    main()
