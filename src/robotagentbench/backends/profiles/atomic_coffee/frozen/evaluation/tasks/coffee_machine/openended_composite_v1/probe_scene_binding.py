"""CPU/USD read-only audit, outputs only under this task directory."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from pxr import Usd, UsdGeom
from .scene_binding import ReadOnlySceneBinding, IncompleteScene, resolve


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def probe(manifest_path):
    manifest_path = Path(manifest_path).resolve(strict=True)
    m = json.loads(manifest_path.read_text())
    scene = Path(m['scene']).resolve(strict=True)
    before = {str(p): digest(p) for p in (manifest_path, scene)}
    stage = Usd.Stage.Open(str(scene))
    if not stage: raise RuntimeError('Cannot open unified scene')
    session_before = stage.GetSessionLayer().ExportToString()
    dirty_before = {l.identifier: l.dirty for l in stage.GetUsedLayers()}
    units = dict(up_axis=str(UsdGeom.GetStageUpAxis(stage)),
                 meters_per_unit=UsdGeom.GetStageMetersPerUnit(stage),
                 authored_up_axis=stage.HasAuthoredMetadata('upAxis'),
                 authored_meters_per_unit=stage.HasAuthoredMetadata('metersPerUnit'))
    layer_units = [dict(path=l.realPath, up_axis=l.pseudoRoot.GetInfo('upAxis'),
                        meters_per_unit=l.pseudoRoot.GetInfo('metersPerUnit'))
                   for l in stage.GetLayerStack() if l.realPath]
    resolved = resolve(stage, m)
    tests = {'repeated_binding_identical': resolve(stage, m) == resolved}
    binding = None
    blockers = list(resolved['missing'])
    try:
        binding = ReadOnlySceneBinding(stage, m)
        a, b = binding.geometry_snapshot(), binding.geometry_snapshot()
        tests['repeated_read_identical'] = a == b
    except IncompleteScene as e:
        blockers.append(str(e))
        a = dict(source='geometry_read_refused_invalid_scene_contract',
                 scoring_evidence_available=False)
        tests['invalid_scene_prevents_geometry_scoring'] = True
    tests['no_scoring_from_geometry'] = not a['scoring_evidence_available']
    for name, mutate in (
        ('missing_cup_rejected', lambda x: x.update(objects=[r for r in x['objects'] if not r['id'].startswith('coffee_cup_')])),
        ('duplicate_cup_rejected', lambda x: x['objects'].append(next(r for r in x['objects'] if r['id'].startswith('coffee_cup_')))),
        ('missing_support_rejected', lambda x: x['supports']['coffee'].update(path='/World/NotExisting')),
    ):
        candidate = deepcopy(m); mutate(candidate)
        try:
            resolve(stage, candidate); tests[name] = False
        except IncompleteScene:
            tests[name] = True
    if binding is not None and binding.resolved['missing']:
        try:
            binding.task_binding('probe', 'coffee-probe'); tests['incomplete_scene_cannot_activate'] = False
        except IncompleteScene:
            tests['incomplete_scene_cannot_activate'] = True
    if binding is None:
        tests['incomplete_scene_cannot_activate'] = True
    tests['source_files_unchanged'] = all(digest(p) == v for p, v in before.items())
    tests['session_layer_unchanged'] = stage.GetSessionLayer().ExportToString() == session_before
    tests['no_new_dirty_layers'] = all(not l.dirty or dirty_before.get(l.identifier, False) for l in stage.GetUsedLayers())
    bean_candidates = [str(p.GetPath()) for p in stage.Traverse()
                       if p.GetName().startswith('Bean_') or p.GetName() in ('CoffeeBeans', 'PourBeans')]
    blockers += (binding.requirements() if binding else [
        'Unified scene must provide persistent coffee device and liquid accounting state',
        'Live PhysX contact/grasp/support collector and 20 Hz evidence bridge not connected',
        'Scene-owned physical settling acceptance not available in static preview',
        'Composite timeout/leave policy and explicit global registration still pending'])
    return dict(layout_id=m['layout_id'], scene=str(scene), manifest=str(manifest_path),
                checks=tests, probe_passed=all(tests.values()), source_sha256=before,
                effective_stage_units=units, layer_units=layer_units,
                binding=resolved, static_geometry=a, bean_candidate_paths=bean_candidates,
                blockers=list(dict.fromkeys(blockers)), scoring_ready=False, runtime_registered=False,
                scene_modified=False, native_physics_test=False,
                status='read_only_binding_partial_scene_not_ready')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', action='append', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    out = Path(args.output).resolve()
    if not out.is_relative_to(Path(__file__).resolve().parent):
        raise ValueError('Audit output must stay in this task directory')
    if out.exists(): raise ValueError('Use a new audit output')
    rows = [probe(x) for x in args.manifest]
    result = dict(schema='coffee.openended.scene_binding.audit.v1',
                  kind='actual_unified_scene_read_only_usd_probe', layouts=rows,
                  checks_passed=all(x['probe_passed'] for x in rows),
                  scoring_ready=False, physics_verified=False)
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'layouts'}))
    for row in rows:
        print(json.dumps(dict(layout=row['layout_id'], checks=row['checks'], roles=row['binding']['roles'], blockers=row['blockers'])))
    return 0 if result['checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
