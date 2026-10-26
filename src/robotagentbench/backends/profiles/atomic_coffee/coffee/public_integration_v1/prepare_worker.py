"""Isolated original-source initializer, invoked by the shared prepare entry."""
import argparse
import json
from pathlib import Path
from . import adapter


def prepare(request, output):
    from short_tasks import registry
    plan = registry.resolve(request)
    registry.verify_sources(plan)
    out = adapter.owned(request['task_name'], output, fresh=True)
    out.mkdir(parents=True)
    initial_dir = out / 'initial'
    variant = plan['variant']
    if variant == 'bag_only':
        from short_task_robustness.tasks.coffee_bag_interference.build_preview_scene import build
    elif variant == 'bag_then_machine':
        from ..build_combined import build
    elif variant == 'machine_only':
        from ..build_initial import build
    else:
        from coffee_button_lighting_interference.initial import build
    initial = build(initial_dir, seed=request['seed'], role='random')
    if plan['source_api'] == '1-2':
        # The preview builders predate the lifecycle receipt. Add only source
        # identity fields; do not author any extra scene/physical properties.
        from evaluation.tasks.coffee_machine import initial_state as source
        _, binding, _ = source.package('1-2', initial['source_episode'])
        initial.update(api='1-2', source_api='1-2',
            cup_style=initial['inspection']['cup_style'] if 'cup_style' in initial['inspection'] else binding['cup_style'],
            public_coffee_assets=binding['public_revision'],
            public_binding_sha256=adapter.digest(source.EPISODES/'1-2'/initial['source_episode']/'public_binding.json'),
            episode_manifest_sha256=adapter.digest(source.EPISODES/'1-2'/initial['source_episode']/'episode.json'),
            reference_sha256=adapter.digest(source.REFERENCE),
            appearance=dict(mode='original_baseline', seed=None, gen_seed=None, plan_sha256=None))
        # Read the actual randomized cup style, rather than assuming the source style.
        from pxr import Usd
        from evaluation.tasks.coffee_machine.specs import CUP
        stage = Usd.Stage.Open(initial['scene'])
        initial['cup_style'] = stage.GetPrimAtPath(CUP).GetCustomDataByKey(source.STYLE_KEY)
        (initial_dir/'initial.json').write_text(json.dumps(initial, indent=2, allow_nan=False)+'\n')
    manifest = dict(schema='short_tasks.atomic_coffee.scene.v1', task_name=plan['task_name'],
        seed=request['seed'], source_api=plan['source_api'], variant=variant,
        layout_scope='atomic_source', layout_id=None, scene=initial['scene'],
        scene_sha256=adapter.digest(initial['scene']), initial_file=str(initial_dir/'initial.json'),
        initial_sha256=adapter.digest(initial_dir/'initial.json'))
    launch = dict(schema='openend.short_task.launch.v1', task_name=plan['task_name'],
        request=str(out/'request.json'), manifest=str(out/'manifest.json'), scene=initial['scene'],
        layout_scope='atomic_source', layout_id=None, physics_backend='cpu',
        physics_hz=240, startup_physics_hz=240, control_hz=20,
        frequency_owner='original_atomic_coffee', public_action_protocol='short_tasks.gpt.robot22.v1')
    for name, value in [('request.json',request), ('manifest.json',manifest), ('launch.json',launch)]:
        (out/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    adapter.admit_launch(launch, out/'native_smoke')
    return dict(launch=str(out/'launch.json'), manifest=str(out/'manifest.json'),
                task_name=plan['task_name'], seed=request['seed'], layout_scope='atomic_source',
                initialization_generated=True, physical_settle_verified=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request-json', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    print(json.dumps(prepare(json.loads(a.request_json), Path(a.output))))
