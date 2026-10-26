"""Read-only triangle-surface distance queries against native CAD, no physics writes."""
import numpy as np
import warp as wp

@wp.kernel
def _distance(mesh:wp.uint64,points:wp.array(dtype=wp.vec3),result:wp.array(dtype=float)):
    i=wp.tid();p=points[i];query=wp.mesh_query_point_no_sign(mesh,p,10.0)
    result[i]=10.0
    if query.result:
        closest=wp.mesh_eval_position(mesh,query.face,query.u,query.v)
        result[i]=wp.length(p-closest)

class CadSurfaceDistance:
    def __init__(self,points,triangles):
        self.mesh=wp.Mesh(points=wp.array(np.asarray(points,dtype=np.float32),dtype=wp.vec3,device='cpu'),indices=wp.array(np.asarray(triangles,dtype=np.int32).ravel(),dtype=int,device='cpu'))
    def query(self,points):
        q=wp.array(np.asarray(points,dtype=np.float32),dtype=wp.vec3,device='cpu');out=wp.empty(len(points),dtype=float,device='cpu')
        wp.launch(_distance,dim=len(points),inputs=[self.mesh.id,q,out],device='cpu')
        return out.numpy(),None
