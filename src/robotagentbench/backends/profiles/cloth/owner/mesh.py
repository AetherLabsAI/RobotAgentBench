"""Material topology and area-weighted native mesh measurements, no Isaac writes."""
from dataclasses import dataclass
import hashlib
import numpy as np


@dataclass(frozen=True)
class MaterialMesh:
    uv: np.ndarray
    faces: np.ndarray

    def __post_init__(self):
        uv=np.array(self.uv,dtype=float,copy=True);f=np.array(self.faces,copy=True)
        if uv.ndim!=2 or uv.shape[1]!=2 or not np.isfinite(uv).all():raise ValueError('Finite UV material coordinates required')
        if f.dtype.kind not in 'iu' or f.ndim!=2 or f.shape[1]!=3 or not len(f) or f.min()<0 or f.max()>=len(uv):raise ValueError('Invalid material triangle indices')
        f=f.astype(np.int64)
        a,b=uv[f[:,1]]-uv[f[:,0]],uv[f[:,2]]-uv[f[:,0]]
        cross=a[:,0]*b[:,1]-a[:,1]*b[:,0]
        if np.any(cross==0) or not(np.all(cross>0)or np.all(cross<0)):raise ValueError('Degenerate/inconsistent material winding')
        if len(np.unique(np.sort(f,axis=1),axis=0))!=len(f) or len(np.unique(f))!=len(uv):raise ValueError('Duplicate or unbound material faces')
        uv.flags.writeable=f.flags.writeable=False
        object.__setattr__(self,'uv',uv);object.__setattr__(self,'faces',f)

    @property
    def areas(self):
        uv,f=self.uv,self.faces
        a,b=uv[f[:,1]]-uv[f[:,0]],uv[f[:,2]]-uv[f[:,0]]
        return abs(a[:,0]*b[:,1]-a[:,1]*b[:,0])/2

    @property
    def area(self):return float(self.areas.sum())

    @property
    def node_weights(self):
        weights=np.zeros(len(self.uv));np.add.at(weights,self.faces.ravel(),np.repeat(self.areas/3,3));return weights

    @property
    def identity(self):
        h=hashlib.sha256();h.update(self.uv.astype('<f8').tobytes());h.update(self.faces.astype('<i8').tobytes());return h.hexdigest()

    def positions(self,q):
        q=np.asarray(q,float)
        if q.shape!=(len(self.uv),3)or not np.isfinite(q).all():raise ValueError('Exact finite native topology required')
        return q

    def vertex_normals(self,q):
        q=self.positions(q);f=self.faces
        cross=np.cross(q[f[:,1]]-q[f[:,0]],q[f[:,2]]-q[f[:,0]])
        normals=np.zeros_like(q);np.add.at(normals,f.ravel(),np.repeat(cross,3,axis=0))
        length=np.linalg.norm(normals,axis=1)
        return np.divide(normals,length[:,None],out=np.zeros_like(normals),where=length[:,None]>0)


def split_materials(initial):
    """Separate exact cloth instances without mixing faces, preserving node order."""
    q,uv,faces,ids=(np.asarray(initial[k])for k in ('initial_cloth_q','uv','faces','cloth_id'))
    if ids.shape!=(len(q),)or not np.all(ids[faces]==ids[faces][:,:1]):raise ValueError('Cloth identity crosses topology')
    result={}
    for cid in np.unique(ids):
        nodes=np.flatnonzero(ids==cid);reverse=np.full(len(q),-1,dtype=int);reverse[nodes]=np.arange(len(nodes))
        f=reverse[faces[np.all(ids[faces]==cid,axis=1)]]
        result[int(cid)]=(MaterialMesh(uv[nodes],f),np.array(q[nodes],copy=True))
    return result
