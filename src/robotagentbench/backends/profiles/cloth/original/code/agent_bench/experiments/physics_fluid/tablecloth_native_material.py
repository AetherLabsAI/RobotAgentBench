"""Explicit PhysX spring material in SI units, independent of render triangles.

Newton's triangle/dihedral coefficients are not PhysX distance-spring constants.
This rectangular-grid approximation is diagnostic until its drape is calibrated.
All springs have flat UV rest lengths and act only through the native solver.
"""
from __future__ import annotations
import numpy as np


def woven_springs(uv, stretch=5000., shear=5000., bend=100., damping=1.):
    uv = np.asarray(uv, dtype=float)
    values = np.asarray([stretch, shear, bend, damping], dtype=float)
    if uv.ndim != 2 or uv.shape[1] != 2 or not np.isfinite(uv).all():
        raise ValueError('Expected finite flat material coordinates')
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Invalid SI spring parameters')
    u, iu = np.unique(uv[:, 0], return_inverse=True)
    v, iv = np.unique(uv[:, 1], return_inverse=True)
    if min(len(u), len(v)) < 3 or len(u)*len(v) != len(uv):
        raise ValueError('Woven spring topology requires a complete rectangular UV grid')
    index = iv*len(u)+iu
    if len(np.unique(index)) != len(uv):
        raise ValueError('Duplicate UV vertex')
    for axis in (u, v):
        if not np.allclose(np.diff(axis), np.diff(axis).mean(), rtol=1e-4, atol=1e-8):
            raise ValueError('Expected uniformly spaced material grid')
    grid = np.empty((len(v), len(u)), dtype=np.int32)
    grid[iv, iu] = np.arange(len(uv))
    # Both shear diagonals are equally stiff. Two-step warp/weft springs resist
    # bending; a triangle's opposite diagonal is shear, not a bending hinge.
    categories = [('warp', 0, 1, stretch), ('weft', 1, 0, stretch),
                  ('shear_positive', 1, 1, shear), ('shear_negative', 1, -1, shear),
                  ('bend_warp', 0, 2, bend), ('bend_weft', 2, 0, bend)]
    pairs, coefficients, labels = [], [], []
    for name, dy, dx, stiffness in categories:
        ys, xs = np.mgrid[0:len(v)-dy, max(0, -dx):min(len(u), len(u)-dx)]
        a, b = grid[ys, xs].ravel(), grid[ys+dy, xs+dx].ravel()
        pairs.append(np.sort(np.column_stack([a, b]), axis=1))
        coefficients.extend([stiffness]*len(a))
        labels.extend([name]*len(a))
    pairs = np.concatenate(pairs)
    if len(np.unique(pairs, axis=0)) != len(pairs):
        raise ValueError('Duplicate material spring')
    rest = np.linalg.norm(uv[pairs[:, 0]]-uv[pairs[:, 1]], axis=1)
    if (rest <= 0).any():
        raise ValueError('Degenerate material spring')
    metadata = dict(model='symmetric_woven_distance_springs',
                    stiffness_units='N/m', damping_units='kg/s',
                    stretch=stretch, shear=shear, bend=bend, damping=damping,
                    grid=[len(u)-1, len(v)-1],
                    counts={name:labels.count(name) for name, *_ in categories},
                    rest_configuration='flat UV; no fold-shaped rest constraints',
                    calibrated=False,
                    limitation='Distance bending is a nonlinear approximation, not a dihedral/FEM material')
    return pairs, np.asarray(coefficients), np.full(len(pairs), damping), rest, metadata


def curvature_metrics(q, uv, faces, cloth_id):
    """Observations only: adjacent-face folds, grid-scale corrugation, spread.

A folded initial sheet legitimately has high curvature. Compare the same initial
state/action across materials; these measurements are not task success gates.
"""
    results = {}
    q = np.asarray(q)
    for cid in np.unique(cloth_id):
        selected = cloth_id == cid
        f = faces[np.all(selected[faces], axis=1)]
        adjacency = {}
        for face_id, face in enumerate(f):
            for a, b in ((face[0],face[1]),(face[1],face[2]),(face[2],face[0])):
                adjacency.setdefault(tuple(sorted((int(a),int(b)))), []).append(face_id)
        pairs = np.asarray([ids for ids in adjacency.values() if len(ids)==2])
        cross = np.cross(q[:,f[:,1]]-q[:,f[:,0]], q[:,f[:,2]]-q[:,f[:,0]])
        normals = cross / np.maximum(np.linalg.norm(cross,axis=-1,keepdims=True),1e-12)
        dot = np.sum(normals[:,pairs[:,0]]*normals[:,pairs[:,1]],axis=-1)
        angles = np.degrees(np.arccos(np.clip(dot,-1,1)))
        local_uv = uv[selected]
        ids = np.flatnonzero(selected)
        nu, nv = len(np.unique(local_uv[:,0])), len(np.unique(local_uv[:,1]))
        grid_ids = ids[np.lexsort((local_uv[:,0],local_uv[:,1]))].reshape(nv,nu)
        grid = q[:,grid_ids]
        residual = grid[:,1:-1,1:-1]-(grid[:,:-2,1:-1]+grid[:,2:,1:-1]+grid[:,1:-1,:-2]+grid[:,1:-1,2:])/4
        residual = np.linalg.norm(residual,axis=-1).reshape(len(q),-1)
        results[str(int(cid))] = dict(
            adjacent_fold_angle_p95_deg=np.quantile(angles,.95,axis=1).tolist(),
            adjacent_fold_angle_max_deg=angles.max(axis=1).tolist(),
            sharp_fold_fraction=(angles>45).mean(axis=1).tolist(),
            grid_corrugation_p95_m=np.quantile(residual,.95,axis=1).tolist(),
            world_height_range_m=np.ptp(q[:,selected,2],axis=1).tolist())
    return dict(per_cloth=results, observation_only=True, calibrated_material=False,
                note='Compare identical initial state/action; curvature is not a task checker.')
