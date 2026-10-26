"""Run inventory-only examples and regression tests. Outputs stay in this task."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import unittest
from planner import compile_plan, RequirementError, KNOWN_ASSETS, CLOTH_POOL, PORTABLE

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--name', default='demo_v1')
    ap.add_argument('--project', type=Path, help='Read-only actual 202 project inventory audit')
    args = ap.parse_args()
    if not args.name.isidentifier():
        ap.error('A simple new directory name is required')
    output = HERE / args.name
    if output.exists():
        ap.error('Output already exists; preserve previous evidence')
    result_text = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(HERE), pattern='test_*.py')
    result = unittest.TextTestRunner(stream=result_text, verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError(result_text.getvalue())
    request = json.loads((HERE/'example_request.json').read_text())
    plan = compile_plan(request)
    exact = dict(schema=request['schema'], seed=42, tasks=[
        dict(id='trash', type='trash_transfer', source='coffee_table.objects',
             items=[dict(kind='paper', count=2), dict(kind='can')]),
        dict(id='books', type='book_transfer', source='sofa', count=3)])
    exact_plan = compile_plan(exact)
    ambiguous = json.loads(json.dumps(request))
    ambiguous['extras'] = [dict(id='background_cup', asset='paper_cup', source='coffee_table.objects')]
    try:
        compile_plan(ambiguous)
    except RequirementError as exc:
        rejection = str(exc)
    else:
        raise AssertionError('Ambiguous background cup was not rejected')
    source_audit = None
    if args.project:
        project = args.project.resolve()
        registry_path = project/'api-protocol/all_assets/registry.json'
        registry = json.loads(registry_path.read_text())
        ids = {r['id'] for r in registry['objects']}
        if not KNOWN_ASSETS <= ids:
            raise RuntimeError('Missing approved library IDs: ' + repr(sorted(KNOWN_ASSETS-ids)))
        if set(CLOTH_POOL) != {i for i in ids if i.startswith('cloth_clear_')}:
            raise RuntimeError('Cloth library changed; review source before using')
        # Audit source entry existence only, NOT USD dependency closure or geometry.
        entries = [project/'api-protocol/all_assets'/i/'asset.usda' for i in KNOWN_ASSETS]
        if any(not p.is_file() for p in entries):
            raise RuntimeError('Asset entry is absent')
        catalog_path = project/'openend/catalog.json'
        catalog = json.loads(catalog_path.read_text())
        text = json.dumps(catalog)
        if any('"'+t+'"' not in text for t in (*PORTABLE, 'trash_transfer', 'cloth_clear')):
            raise RuntimeError('Runtime catalog family changed')
        watched = [project/'openend/room.py', catalog_path,
                   project/'layout_task_activation_v1/task_scene_registration_v1/build_room_demo_v1.py',
                   registry_path, project/'openend/tasks/cloth_clear/inventory.py',
                   project/'openend/tasks/trash_transfer/registration.py']
        source_audit = dict(asset_ids_checked=len(KNOWN_ASSETS), asset_entries_exist=True,
                            usd_composition_checked=False, supported_family_names_found=True,
                            sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in watched})
    report = dict(status='inventory_prototype_tests_passed', tests=result.testsRun,
                  failures=len(result.failures), errors=len(result.errors),
                  random_seed_coverage=dict(trash=200, cloth=200, restricted_trash_categories=50),
                  example_counts=dict(Counter(o['asset_id'] for o in plan['objects'])),
                  ambiguous_example_rejected=rejection, source_audit=source_audit,
                  scene_generated=False, native_physics_run=False, shared_entry_connected=False,
                  changed_scope='openend/tasks/scene_requirements only',
                  source_files_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in HERE.iterdir() if p.is_file()})
    output.mkdir()
    for name, data in [('request.json', request), ('plan.json', plan), ('exact_request.json', exact),
                       ('exact_plan.json', exact_plan), ('rejected_request.json', ambiguous), ('verification.json', report)]:
        with (output/name).open('x') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write('\n')
    with (output/'tests.txt').open('x') as f:
        f.write(result_text.getvalue())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
