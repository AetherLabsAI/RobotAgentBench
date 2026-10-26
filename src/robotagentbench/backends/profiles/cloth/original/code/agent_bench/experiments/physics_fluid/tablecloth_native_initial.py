"""Rest-length-respecting folded cloth initialization, before physics starts only."""
from __future__ import annotations
import numpy as np


def refine_initial_mesh(points, uv, faces, cloth_ids):
    """Owner midpoint refinement, before physics; preserve the entire surface.

    Adapted from the delivered standalone runner's mesh refinement. Shared
    edges have one midpoint and cloths remain grouped for native tensor views.
    This changes resolution only, never an episode's simulated state.
    """
    points, uv = np.asarray(points), np.asarray(uv)
    faces, cloth_ids = np.asarray(faces), np.asarray(cloth_ids)
    if (points.ndim != 2 or points.shape[1] != 3 or uv.shape != (len(points), 2)
            or cloth_ids.shape != (len(points),) or faces.ndim != 2
            or faces.shape[1] != 3 or not len(faces) or faces.dtype.kind not in 'iu'
            or faces.min() < 0 or faces.max() >= len(points)
            or not np.isfinite(points).all() or not np.isfinite(uv).all()):
        raise ValueError('Invalid initial material mesh')
    edges, edge_index = np.unique(np.sort(np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1),
        axis=0, return_inverse=True)
    if np.any(cloth_ids[edges[:, 0]] != cloth_ids[edges[:, 1]]):
        raise ValueError('Material edge crosses cloth identities')
    ab, bc, ca = edge_index.reshape(3, len(faces)) + len(points)
    refined_faces = np.concatenate([
        np.stack([faces[:, 0], ab, ca], axis=1),
        np.stack([ab, faces[:, 1], bc], axis=1),
        np.stack([ca, bc, faces[:, 2]], axis=1),
        np.stack([ab, bc, ca], axis=1)])
    q = np.concatenate([points, points[edges].mean(axis=1)])
    material = np.concatenate([uv, uv[edges].mean(axis=1)])
    identities = np.concatenate([cloth_ids, cloth_ids[edges[:, 0]]])
    order = np.argsort(identities, kind='stable')
    inverse = np.empty(len(order), dtype=int)
    inverse[order] = np.arange(len(order))
    return q[order], material[order], inverse[refined_faces], identities[order]


def rounded_basket_fold(uv, basket_bounds, radius=.012, clearance=.004, isometric=False):
    uv=np.asarray(uv,dtype=float);bounds=np.asarray(basket_bounds,dtype=float)
    if uv.ndim!=2 or uv.shape[1]!=2 or bounds.shape!=(2,3) or not np.isfinite(uv).all() or not np.isfinite(bounds).all():
        raise ValueError('Invalid material coordinates or basket bounds')
    u,v=(uv-uv.min(0)).T
    width=float(np.ptp(v));folds=4
    if radius<=0 or clearance<=0 or np.max(np.diff(np.unique(v)))>radius:
        raise ValueError('Fold radius must resolve the material grid; refine the mesh')
    straight=(width-(folds-1)*np.pi*radius)/folds
    if straight<=0:raise ValueError('Cloth too short for the rounded folds')
    y=np.zeros(len(v));z=np.zeros(len(v));start=0.
    for k in range(folds):
        direction=1 if k%2==0 else -1
        y+=direction*np.clip(v-start,0,straight)
        start+=straight
        if k<folds-1:
            theta=np.clip((v-start)/radius,0,np.pi)
            y+=direction*radius*np.sin(theta)
            z+=radius*(1-np.cos(theta))
            start+=np.pi*radius
    if isometric:
        # A sampled circular arc has chords shorter than its material lengths.
        # Preserve each row interval exactly while retaining the rounded turn
        # directions. Extruding this polygon along u is an isometry of every
        # material triangle, including its diagonal, before simulation starts.
        levels, first, inverse = np.unique(v, return_index=True, return_inverse=True)
        curve = np.column_stack((y[first], z[first]))
        segments = np.diff(curve, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        if np.any(lengths <= 0):
            raise ValueError('Degenerate fold segment')
        segments *= (np.diff(levels)/lengths)[:, None]
        curve = np.vstack((np.zeros(2), np.cumsum(segments, axis=0)))
        y, z = curve[inverse].T
    q=np.column_stack((u,-y,z))
    q-=.5*(q.min(0)+q.max(0))
    angle=np.deg2rad(105.)
    rotation=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    q=q@rotation.T
    # Same owner basket: its interior floor is 25 mm above assembly lower bound.
    q[:,2]+=bounds[0,2]+.025+clearance-q[:,2].min()
    below_rim=q[:,2]<=bounds[1,2]+clearance
    if not below_rim.any():raise ValueError('Fold does not reach basket')
    # Owner's old 12 mm rim inset intersected the actual rim. Keep a physical
    # clearance to the inside of the approximately 25 mm basket wall.
    q[:,0]+=bounds[1,0]-.025-clearance-q[below_rim,0].max()
    q[:,1]+=bounds[1,1]-.025-clearance-q[:,1].max()
    return q


def initial_edge_ratios(points,uv,faces):
    edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
    return np.linalg.norm(points[edges[:,1]]-points[edges[:,0]],axis=1)/np.linalg.norm(uv[edges[:,1]]-uv[edges[:,0]],axis=1)
