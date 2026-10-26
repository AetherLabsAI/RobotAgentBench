"""Read-only per-recorded-frame arm clearance, not a physics acceptance test."""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom, UsdPhysics
from audit_coffee_workspace_v3 import geometry, ROOT, TASK
from audit_task2_workspace import sha, matrix

HERE = Path(__file__).resolve().parent
WEB = HERE.parent / 'web'


def obb_aabb_separation(centers, axes, half, lo, hi):
    """Positive separating-axis margin proves disjoint boxes; <=0 is a suspect."""
    n = len(centers)
    delta = centers - (lo + hi) / 2
    bh = (hi - lo) / 2
    candidates = [np.broadcast_to(v, (n, 3)) for v in np.eye(3)]
    candidates += [axes[:, :, i] for i in range(3)]
    candidates += [np.cross(axes[:, :, i], v) for i in range(3) for v in np.eye(3)]
    margin = np.full(n, -np.inf)
    for axis in candidates:
        norm = np.linalg.norm(axis, axis=1)
        unit = axis / np.maximum(norm[:, None], 1e-12)
        ra = (np.abs(np.einsum('nji,nj->ni', axes, unit)) * half).sum(axis=1)
        rb = (np.abs(unit) * bh).sum(axis=1)
        gap = np.abs((delta * unit).sum(axis=1)) - ra - rb
        margin = np.maximum(margin, np.where(norm > 1e-8, gap, -np.inf))
    return margin


def main():
    measured_fingers = '--measured-fingers' in sys.argv
    evidence = json.loads((WEB / 'coffee-motion-envelope-v3.json').read_text())
    policy = json.loads((WEB / 'coffee-standard-zone-v2.json').read_text())
    layouts = json.loads((WEB / 'random-layouts-v1.json').read_text())['layouts']
    stage, parents, points, radii, base, fk_error, layers = geometry()
    assert layers == evidence['geometry_layers'], 'Robot/source geometry changed since standard-frame audit'
    finger_names = []
    if measured_fingers:
        xf = UsdGeom.XformCache()
        for prim in Usd.PrimRange(stage.GetPrimAtPath(ROOT)):
            name = prim.GetName()
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI) or not name.startswith(('gripper_l_', 'gripper_r_')) or name.endswith(('_base_link', '_center_link')):
                continue
            path = str(prim.GetPath())
            body = np.asarray(xf.GetLocalToWorldTransform(prim)).T.copy()
            u, _, v = np.linalg.svd(body[:3, :3]); body[:3, :3] = u @ v
            inv_body = np.linalg.inv(body)
            vertices = []
            for mesh in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
                if not mesh.IsA(UsdGeom.Mesh) or UsdGeom.Imageable(mesh).ComputeVisibility() == 'invisible':
                    continue
                xyz = np.asarray(UsdGeom.Mesh(mesh).GetPointsAttr().Get(), float)
                if not len(xyz):
                    continue
                vertices.append((inv_body @ np.asarray(xf.GetLocalToWorldTransform(mesh)).T @ np.c_[xyz, np.ones(len(xyz))].T).T)
            assert vertices, 'Missing finger mesh: ' + path
            vertices = np.concatenate(vertices)
            from audit_task2_workspace import corners
            points[path] = corners(vertices[:, :3].min(axis=0), vertices[:, :3].max(axis=0))
            finger_names.append(path)
        assert len(finger_names) >= 8, 'Incomplete measured finger geometry'
    targets = [dict(layout=2, origin=[4.619789389644037, .029999825915178503, .92]),
               dict(layout=1, origin=[4.619789369670302, -1.3008002087454045, .92])]
    results = []
    for target in targets:
        layout = next(x for x in layouts if x['id'] == target['layout'])
        assert sha(layout['scene']) == layout['scene_sha256']
        # Asset +Y up, +Z front; the front faces the west/operator edge.
        target_m = np.array([[0, 0, -1, target['origin'][0]], [-1, 0, 0, target['origin'][1]],
                             [0, 1, 0, target['origin'][2]], [0, 0, 0, 1]], float)
        obstacles = [b for b in layout['blockers'] if b['max'][2] > 1.2 and b['min'][2] < 1.8]
        runs = []
        for selected in policy['selected']:
            record = next(r for r in evidence['records'] if r['api'] == selected['api'] and r['episode'] == selected['episode'] and r['verdict'] == 'success')
            run = Path(record['run'])
            tp = run / 'run/episode/trace.json'
            bp = TASK / 'resources/source_episodes' / selected['api'] / selected['episode'] / 'public_binding.json'
            for f in [tp, bp, run / 'run/report.json']:
                assert sha(f) == record['hashes'][str(f)]
            trace = json.loads(tp.read_text())
            binding = json.loads(bp.read_text())
            obs = [trace['initial_observation']] + [s['observation'] for s in trace['steps']]
            obs = sorted({o['step_index']: o for o in obs}.values(), key=lambda o: o['step_index'])
            assert len(obs) == record['frames'] and all(o['source'] == 'isaac_physics' for o in obs)
            assert np.allclose(np.diff([o['time_s'] for o in obs]), .05, atol=1e-5)
            qs = [{**o['joint_positions_rad'], **o['joint_positions_m']} for o in obs]
            n = len(qs)
            inv = np.linalg.inv(matrix(binding['world']['machine']))
            remap = target_m @ inv
            frames = {ROOT + '/base_link': np.broadcast_to(base, (n, 4, 4))}

            def fk(body):
                if body in frames:
                    return frames[body]
                a, name, f0, f1, axis, angular = parents[body]
                m = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
                if axis is not None:
                    values = np.array([q[name] for q in qs])
                    assert np.isfinite(values).all()
                    if angular:
                        m[:, :3, :3] = Rotation.from_rotvec(values[:, None] * np.eye(3)[axis]).as_matrix()
                    else:
                        m[:, axis, 3] = values
                frames[body] = fk(a) @ f0 @ m @ f1
                return frames[body]

            checks = []
            union_min, union_max = [], []
            for body, pts in points.items():
                transform = remap @ fk(body)
                lo, hi = pts[:, :3].min(axis=0), pts[:, :3].max(axis=0)
                center = np.einsum('nij,j->ni', transform, np.r_[(lo + hi) / 2, 1])[:, :3]
                half = (hi - lo) / 2
                axes = transform[:, :3, :3]
                assert np.allclose(np.linalg.det(axes), 1, atol=1e-5)
                extent = np.einsum('nij,j->ni', np.abs(axes), half)
                union_min.append((center - extent).min(axis=0)); union_max.append((center + extent).max(axis=0))
                for obstacle in obstacles:
                    blo, bhi = np.array(obstacle['min']), np.array(obstacle['max'])
                    margins = obb_aabb_separation(center, axes, half, blo, bhi)
                    checks.append((body, obstacle['path'], margins, 'per_link_oriented_visual_box'))
            for side in (() if measured_fingers else ('l', 'r')):
                center = (remap @ fk(ROOT + '/gripper_' + side + '_base_link'))[:, :3, 3]
                radius = radii[side]
                union_min.append((center - radius).min(axis=0)); union_max.append((center + radius).max(axis=0))
                for obstacle in obstacles:
                    lo, hi = np.array(obstacle['min']), np.array(obstacle['max'])
                    nearest = np.maximum(lo, np.minimum(center, hi))
                    margins = np.linalg.norm(center - nearest, axis=1) - radius
                    checks.append(('gripper_' + side + '_sphere', obstacle['path'], margins, 'conservative_finger_chain_sphere'))
            any_suspect = np.zeros(n, bool)
            suspects = []
            for body, obstacle, margins, kind in checks:
                idx = np.where(margins <= 1e-7)[0]
                if not len(idx):
                    continue
                any_suspect[idx] = True
                worst = int(np.argmin(margins))
                suspects.append(dict(body=body, obstacle=obstacle, geometry=kind, suspect_frames=len(idx),
                                     first_step=obs[int(idx[0])]['step_index'], first_time_s=obs[int(idx[0])]['time_s'],
                                     worst_step=obs[worst]['step_index'], worst_time_s=obs[worst]['time_s'],
                                     separating_axis_margin_m=float(margins[worst])))
            closest = min(checks, key=lambda row: float(np.min(row[2])))
            runs.append(dict(api=selected['api'], episode=selected['episode'], frames=n,
                             source_hashes={str(p): sha(p) for p in (tp, bp)},
                             projected_base=(remap @ base).tolist(),
                             global_arm_bounds=dict(min=np.min(union_min, axis=0).tolist(), max=np.max(union_max, axis=0).tolist()),
                             suspect_frame_count=int(any_suspect.sum()), suspects=sorted(suspects, key=lambda s: s['separating_axis_margin_m']),
                             minimum_separating_axis_margin_m=float(np.min(closest[2])),
                             closest_body=closest[0], closest_obstacle=closest[1]))
            print('CHECK', target['layout'], selected['api'], n, int(any_suspect.sum()), flush=True)
        results.append(dict(**target, scene_sha256=layout['scene_sha256'], target_machine_matrix=target_m.tolist(),
                            obstacle_count=len(obstacles), records=runs))
    report = dict(schema='arena.coffee.counter_frame_clearance.v2' if measured_fingers else 'arena.coffee.counter_frame_clearance.v1', results=results, frames_per_target=sum(r['frames'] for r in results[0]['records']),
                  measured_fingers=measured_fingers, finger_links=finger_names,
                  diagnostic_script_sha256=sha(__file__),
                  initial_fk_error=fk_error, source_geometry_hashes_match=True,
                  policy_sha256=sha(WEB / 'coffee-standard-zone-v2.json'), physical_validation=False,
                  limitations=['Rigid remapping of four independent measured 20 Hz segments; no target IK, navigation or physics replay.',
                               'Current public robot geometry, not recovered historical meshes; geometry layer hashes match the existing standard-frame audit.',
                               ('Oriented per-link arm and finger visual boxes reconstructed using recorded joint angles, with no replacement finger spheres.' if measured_fingers else 'Oriented per-link visual boxes and conservative finger spheres against obstacle visual AABBs.') + ' Positive separation excludes overlap; a suspect is NOT proven mesh contact.',
                               'Only elevated obstacles intersecting the upper workspace height were checked; support, lower-body, carried objects, all room boundaries and frame-intermediate motion are not certified.',
                               'No existing placement rule or standard box was changed.'])
    dest = WEB / ('coffee-counter-clearance-v2.json' if measured_fingers else 'coffee-counter-clearance-v1.json')
    assert not dest.exists(), 'Preserve prior diagnostic output'
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
