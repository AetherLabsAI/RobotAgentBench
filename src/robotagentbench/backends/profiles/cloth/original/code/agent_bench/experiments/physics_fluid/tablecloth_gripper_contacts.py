"""Build detailed, closed CAD convex parts for cloth-contacting fingers.

CoACD is an offline optional dependency. The saved numeric geometry records
its source arrays and approximation settings; runtime does not run meshing.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def union_contact_parts(parts):
    """Remove internal decomposition surfaces with a watertight solid union."""
    import manifold3d as m
    solids = []
    for vertices, faces in parts:
        solid = m.Manifold(m.Mesh(np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.uint32)))
        if solid.status() != m.Error.NoError:
            raise ValueError(f'Invalid convex contact part: {solid.status()}')
        solids.append(solid)
    solid = m.Manifold.batch_boolean(solids, m.OpType.Add)
    if solid.status() != m.Error.NoError or solid.is_empty():
        raise ValueError(f'Contact union failed: {solid.status()}')
    mesh = solid.to_mesh()
    return np.asarray(mesh.vert_properties[:, :3], dtype=np.float32), np.asarray(mesh.tri_verts, dtype=np.int32)


def contact_parts(calibration, link, detailed=None):
    from tablecloth_robot_collisions import convex_contact_mesh
    if detailed is None or f'part_count_{link}' not in detailed:
        return [convex_contact_mesh(calibration[f'points_{link}'])]
    if not np.array_equal(detailed[f'source_points_{link}'], calibration[f'points_{link}']):
        raise ValueError(f'Detailed contact geometry belongs to different CAD: {link}')
    if not np.array_equal(detailed[f'source_triangles_{link}'], calibration[f'triangles_{link}']):
        raise ValueError(f'Detailed contact topology belongs to different CAD: {link}')
    count = int(detailed[f'part_count_{link}'])
    if count < 1:
        raise ValueError(f'Detailed contact geometry must not remove a collider: {link}')
    return [(detailed[f'vertices_{link}_{k}'], detailed[f'faces_{link}_{k}']) for k in range(count)]


def main():
    import coacd
    import trimesh
    from tablecloth_robot_collisions import convex_contact_mesh
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tolerance-m', type=float, default=.0005)
    parser.add_argument('--all-links', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = dict(np.load(args.calibration))
    output, records, cache = {}, [], []
    for link in source['links'].tolist():
        if not args.all_links and not link.endswith('_link4'):
            continue
        v, f = source[f'points_{link}'], source[f'triangles_{link}']
        parts = next((p for cv, cf, p in cache if np.array_equal(cv, v) and np.array_equal(cf, f)), None)
        if parts is None:
            mesh = trimesh.Trimesh(v, f, process=True)
            print(f'[CAD] decomposing {link}: {len(mesh.vertices)} vertices', flush=True)
            raw = coacd.run_coacd(coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces)),
                threshold=args.tolerance_m, real_metric=True, preprocess_resolution=100,
                resolution=2000, mcts_iterations=40, mcts_nodes=15, merge=True,
                decimate=False, seed=8)
            parts = [union_contact_parts([convex_contact_mesh(vertices) for vertices, _ in raw])]
            cache.append((v, f, parts))
        output[f'source_points_{link}'], output[f'source_triangles_{link}'] = v, f
        output[f'part_count_{link}'] = len(parts)
        output[f'concave_{link}'] = True
        for k, (vertices, faces) in enumerate(parts):
            output[f'vertices_{link}_{k}'], output[f'faces_{link}_{k}'] = vertices, faces
        records.append(dict(link=link, parts=len(parts), vertices=sum(len(p[0]) for p in parts),
                            triangles=sum(len(p[1]) for p in parts)))
        print(records[-1], flush=True)
    metadata = dict(schema='real2sim.detailed_gripper_contacts.v1', source=str(args.calibration.resolve()),
                    algorithm='Watertight solid union of CoACD CAD parts', tolerance_m=args.tolerance_m,
                    tolerance_role='mesher concavity target, not measured maximum surface error',
                    records=records)
    output['metadata_json'] = json.dumps(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
