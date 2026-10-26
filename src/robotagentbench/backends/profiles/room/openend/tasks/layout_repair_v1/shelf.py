"""Approved canonical-size replacement for the three scaled source shelves."""
import numpy as np
def canonical_delta(original,target,path):
    from pxr import Gf
    from openend.tasks.scene_requirements.descendant_transforms import geometry
    aa,bb=geometry(original,path),geometry(target,path)
    if not aa or aa.keys()!=bb.keys() or any(aa[k].shape!=bb[k].shape for k in aa):
        raise ValueError('Cannot infer shelf orientation from different geometry')
    a,b=[np.concatenate([v[k] for k in sorted(v)]) for v in (aa,bb)]
    # Fit yaw only. Nonuniform source scale is deliberately NOT transferred.
    ac,bc=a[:,:2].mean(0),b[:,:2].mean(0)
    u,_,vt=np.linalg.svd((a[:,:2]-ac).T@(b[:,:2]-bc));rot2=u@vt
    if np.linalg.det(rot2)<0:raise ValueError('Reflected shelf cannot be canonicalized')
    rot=np.eye(3);rot[:2,:2]=rot2
    ar=a@rot;acenter=(ar.min(0)+ar.max(0))/2;bcenter=(b.min(0)+b.max(0))/2
    translation=bcenter-acenter
    matrix=np.eye(4);matrix[:3,:3]=rot;matrix[3,:3]=translation
    return Gf.Matrix4d(*matrix.reshape(-1).tolist()),dict(method='approved_original_size_shelf_replacement',
        matrix=matrix.tolist(),source_scaled_bounds=[b.min(0).tolist(),b.max(0).tolist()],
        canonical_bounds=[(ar.min(0)+translation).tolist(),(ar.max(0)+translation).tolist()],
        shared_asset_modified=False,scale_transferred=False,collision_check_required=True)
