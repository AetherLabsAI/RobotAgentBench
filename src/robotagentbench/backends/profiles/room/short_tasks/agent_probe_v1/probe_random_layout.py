"""Exercise the real CPU/Node planner; no USD composition or physics claim."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from short_tasks import registry
from short_tasks.scene import plan_scene, verify_scene_sources

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def request_for(name):
    if name not in registry.TASK_NAMES:
        raise ValueError('Unknown registered task')
    path = ROOT / 'short_tasks/examples' / (name + '.json')
    own = ROOT / 'short_tasks/tasks' / name
    if (own/'request.json').is_file():
        path=own/'request.json'
    if name == 'bake_bread':
        path=own/'request_v2.json'
    if name == 'prepare_coffee':
        path = ROOT / 'short_tasks/tasks/prepare_coffee/request_v2.json'
    if name == 'stock_wooden_tray_interference' and not path.is_file():
        # Additional pending actor variant has no saved example, unlike the
        # 14 dataset tasks. An explicit singleton is sufficient for admission.
        return dict(schema='openend.short_task.request.v1',task_name=name,
            rule_version='openend.short_task.stock_wooden_tray_interference.v1',
            seed=47,layout_id=None,targets=[dict(category='sprite_c1',quantity=1)])
    return json.loads(path.read_text())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(HERE) or out.exists():
        raise ValueError('Fresh output inside bridge task required')
    out.mkdir(parents=True)
    (out / 'tmp').mkdir()
    tempfile.tempdir = str(out / 'tmp')
    report = dict(scope='real_planar_planner_only', scene_generated=False,
                  native_physics_validated=False, source_check=verify_scene_sources(), tasks=[])
    for name in registry.TASK_NAMES:
        request = request_for(name)
        task = dict(task_name=name, trials=[])
        report['tasks'].append(task)
        if not registry.resolve(request)['available']:
            task['status'] = 'definition_unavailable'
            continue
        task['source_check'] = registry.verify_sources(registry.resolve(request))
        task['status'] = 'tested'
        baseline = None
        for label, seed, layout in [('random_'+str(s), s, None) for s in range(1, 6)] + [
                ('repeat_seed1', 1, None), ('fixed4_seed1', 1, 4), ('fixed4_seed2', 2, 4)]:
            q = dict(request, seed=seed)
            if layout is None: q.pop('layout_id', None)
            else: q['layout_id'] = layout
            row = dict(label=label, seed=seed, requested_layout=layout)
            task['trials'].append(row)
            try:
                result = plan_scene(q)
                (out / (name+'_'+label+'.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
                recipe = result.get('recipe', {})
                row.update(status=result['scene_status'], layout=result.get('actual_layout_id'),
                           attempts=result.get('attempts'), recipe_sha256=fingerprint(recipe),
                           placements=recipe.get('plan', {}).get('placements', []))
                if label == 'random_1': baseline = row['recipe_sha256']
                if label == 'repeat_seed1': row['identical_to_seed1'] = row['recipe_sha256'] == baseline
                if layout is not None: row['explicit_layout_preserved'] = row['layout'] == layout
            except Exception as exc:
                row.update(status='error', error=str(exc))
            print(json.dumps(dict(task=name, **{k:v for k,v in row.items() if k!='placements'})), flush=True)
        task['unique_random_layouts'] = sorted({r['layout'] for r in task['trials']
            if r['label'].startswith('random_') and 'layout' in r})
        (out / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    (out / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')


if __name__ == '__main__':
    main()
