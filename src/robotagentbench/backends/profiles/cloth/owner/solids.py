"""Conservative exact convex-volume clearance and physical surface proximity."""
import numpy as np
from contact_geometry import convex_surface,surface_distance


class StaticSolid:
    def __init__(self,*,path,pieces_world_m):
        if not path.startswith('/World/')or not pieces_world_m:raise ValueError('Bound physical solid required')
        self.path=path;self.pieces=[]
        for vertices in pieces_world_m:
            v=np.asarray(vertices,float);planes,triangles=convex_surface(v)
            self.pieces.append((v.min(0),v.max(0),planes,triangles))

    def node_proximity(self,points,*,shell_m):
        if shell_m<=0:raise ValueError('Existing physical contact shell required')
        q=np.asarray(points,float);near=np.zeros(len(q),bool)
        for lo,hi,planes,triangles in self.pieces:
            ids=np.flatnonzero(np.all(q>=lo-shell_m,axis=1)&np.all(q<=hi+shell_m,axis=1))
            if len(ids):near[ids]|=surface_distance(q[ids],triangles)<=shell_m
        return near

    def clear_of_entire_surface(self,mesh,points,*,shell_m):
        q=mesh.positions(points);tri=q[mesh.faces];tlo=tri.min(1);thi=tri.max(1)
        for lo,hi,planes,_ in self.pieces:
            ids=np.flatnonzero(np.all(thi>=lo-shell_m,axis=1)&np.all(tlo<=hi+shell_m,axis=1))
            for triangle in tri[ids]:
                polygon=list(triangle)
                # Clip the whole triangle, not just its material vertices. A
                # convex collider can pierce a face with all vertices outside.
                for plane in planes:
                    if not polygon:break
                    output=[];previous=polygon[-1];dp=previous@plane[:3]+plane[3]-shell_m
                    for current in polygon:
                        dc=current@plane[:3]+plane[3]-shell_m
                        if (dc<=0)!=(dp<=0):output.append(previous+(current-previous)*(dp/(dp-dc)))
                        if dc<=0:output.append(current)
                        previous,dp=current,dc
                    polygon=output
                if polygon:return False
        return True


def points_in_polygon_xy(points,polygon):
    q=np.asarray(points,float);p=np.asarray(polygon,float)
    if q.ndim!=2 or q.shape[1]<2 or p.ndim!=2 or p.shape[1]!=2 or len(p)<3:raise ValueError('Measured XY polygon required')
    inside=np.zeros(len(q),bool)
    for a,b in zip(p,np.roll(p,-1,axis=0)):
        if b[1]==a[1]:continue
        crossed=((a[1]>q[:,1])!=(b[1]>q[:,1]))&(q[:,0]<(b[0]-a[0])*(q[:,1]-a[1])/(b[1]-a[1])+a[0])
        inside^=crossed
    return inside
