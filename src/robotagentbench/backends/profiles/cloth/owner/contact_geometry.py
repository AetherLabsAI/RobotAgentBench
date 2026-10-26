"""Exact convex-pad surface distances; diagnostic until contact policy approval.

This is geometric evidence from actual native body poses, not contact force.
No command, jaw target, bounding-box hit or old grasp is a contact measurement.
"""
from itertools import combinations
import numpy as np


def quaternion_matrix_xyzw(q):
    q=np.asarray(q,float)
    if q.shape!=(4,)or not np.isfinite(q).all()or abs(np.linalg.norm(q)-1)>1e-4:raise ValueError('Native unit quaternion required')
    x,y,z,w=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def convex_surface(vertices):
    """Supporting planes of the actual cooked convex; no SciPy dependency."""
    v=np.asarray(vertices,float)
    if v.ndim!=2 or v.shape[1]!=3 or len(v)<4 or not np.isfinite(v).all():raise ValueError('Finite cooked vertices required')
    eps=256*np.finfo(float).eps*max(1.,float(abs(v).max()))
    unique={}
    for ids in combinations(range(len(v)),3):
        a,b,c=v[list(ids)];n=np.cross(b-a,c-a);length=np.linalg.norm(n)
        if length<=eps:continue
        n/=length;d=-n@a;signed=v@n+d
        if signed.max()<=eps:pass
        elif signed.min()>=-eps:n=-n;d=-d
        else:continue
        key=tuple(np.round(np.r_[n,d],11));unique[key]=np.r_[n,d]
    if len(unique)<4:raise ValueError('Cooked hull has no enclosed volume')
    planes=np.array(list(unique.values()));triangles=[]
    for plane in planes:
        face=v[abs(v@plane[:3]+plane[3])<=eps*4]
        center=face.mean(0);u=face[0]-center;u/=np.linalg.norm(u);w=np.cross(plane[:3],u)
        angles=np.arctan2((face-center)@w,(face-center)@u);face=face[np.argsort(angles)]
        for k in range(1,len(face)-1):
            triangle=np.array([face[0],face[k],face[k+1]])
            if np.linalg.norm(np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0]))>eps:triangles.append(triangle)
    return planes,np.array(triangles)


def surface_distance(points,triangles):
    """Point-to-triangle distance including edges/corners, bounded memory."""
    p=np.asarray(points,float);tri=np.asarray(triangles,float)
    if p.ndim!=2 or p.shape[1]!=3 or tri.ndim!=3 or tri.shape[1:]!=(3,3):raise ValueError('Exact triangle geometry required')
    distances=np.full(len(p),np.inf)
    for start in range(0,len(p),256):
        q=p[start:start+256,None,:];a,b,c=(tri[None,:,i,:]for i in (0,1,2))
        ab,ac=b-a,c-a;normal=np.cross(ab,ac);normal2=(normal*normal).sum(-1)
        if np.any(normal2<=0):raise ValueError('Degenerate collision triangle')
        signed=((q-a)*normal).sum(-1);projected=q-normal*(signed/normal2)[:,:,None]
        v=projected-a;aa=(ab*ab).sum(-1);bb=(ab*ac).sum(-1);cc=(ac*ac).sum(-1)
        va=(v*ab).sum(-1);vc=(v*ac).sum(-1);den=aa*cc-bb*bb
        s=(va*cc-vc*bb)/den;t=(vc*aa-va*bb)/den
        within=(s>=0)&(t>=0)&(s+t<=1)
        best=np.where(within,signed*signed/normal2,np.inf)
        for x,y in ((a,b),(b,c),(c,a)):
            delta=y-x;u=np.clip(((q-x)*delta).sum(-1)/(delta*delta).sum(-1),0,1)
            offset=q-(x+u[:,:,None]*delta);best=np.minimum(best,(offset*offset).sum(-1))
        distances[start:start+len(q)]=np.sqrt(best.min(1))
    return distances


def surface_proximity(points,triangles,*,shell_m):
    """Same exact distance threshold, skipping only provably distant points."""
    q=np.asarray(points,float);tri=np.asarray(triangles,float)
    if (shell_m<=0 or not np.isfinite(shell_m) or q.ndim!=2 or q.shape[1]!=3
            or tri.ndim!=3 or tri.shape[1:]!=(3,3)or not len(tri)
            or not np.isfinite(q).all()or not np.isfinite(tri).all()):
        raise ValueError('Finite native triangles/points and positive shell required')
    n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    if np.any((n*n).sum(1)<=0):raise ValueError('Degenerate collision triangle')
    vertices=tri.reshape(-1,3)
    # Outside an axis-expanded enclosing box, Euclidean surface distance must
    # exceed shell_m. Inside it, retain the original point/triangle algorithm.
    guard=256*np.finfo(float).eps*max(1.,float(abs(vertices).max()),float(abs(q).max())if len(q)else 0.)
    # Round-off only widens candidate selection. The final distance comparison
    # below still uses exactly shell_m, with no added scoring tolerance.
    candidate_shell=shell_m+guard
    ids=np.flatnonzero(np.all(q>=vertices.min(0)-candidate_shell,axis=1)&np.all(q<=vertices.max(0)+candidate_shell,axis=1))
    near=np.zeros(len(q),bool)
    if len(ids):near[ids]=surface_distance(q[ids],tri)<=shell_m
    return near


class NativePadGeometry:
    def __init__(self,cooked_records):
        self.parts=[]
        for row in cooked_records:
            for c in row['convexes']:self.parts.append(convex_surface(c['vertices_body_m']))
        if not self.parts:raise ValueError('No actual cooked pad geometry')

    def measure(self,points,body_transform_xyzw,*,surface_shell_m):
        if surface_shell_m<=0:raise ValueError('Explicit source contact shell required')
        pose=np.asarray(body_transform_xyzw,float)
        if pose.shape!=(7,):raise ValueError('Measured native body pose required')
        local=(np.asarray(points,float)-pose[:3])@quaternion_matrix_xyzw(pose[3:])
        distance=np.full(len(local),np.inf);inside=np.zeros(len(local),bool);computed=np.zeros(len(local),bool)
        for planes,triangles in self.parts:
            bounds=triangles.reshape(-1,3)
            ids=np.flatnonzero(np.all(local>=bounds.min(0)-surface_shell_m,axis=1)&np.all(local<=bounds.max(0)+surface_shell_m,axis=1))
            if not len(ids):continue
            inside[ids]|=np.all(local[ids]@planes[:,:3].T+planes[:,3]<=0,axis=1)
            distance[ids]=np.minimum(distance[ids],surface_distance(local[ids],triangles));computed[ids]=True
        # Interior penetration is diagnosed, never made into a valid touch.
        return dict(distances_m=distance,distance_computed=computed,interior=inside,near_surface=(distance<=surface_shell_m)&~inside,
                    source='native_cooked_pad_surface_and_measured_body_pose',force_measured=False)

    def ray_sides(self,points,body_transform_xyzw,axis_world):
        pose=np.asarray(body_transform_xyzw,float);r=quaternion_matrix_xyzw(pose[3:])
        local=(np.asarray(points,float)-pose[:3])@r;axis=np.asarray(axis_world,float)@r
        if abs(np.linalg.norm(axis)-1)>1e-4:raise ValueError('Native unit closing axis required')
        negative=np.zeros(len(local),bool);positive=np.zeros(len(local),bool)
        for planes,_ in self.parts:
            direction=planes[:,:3]@axis;distance=local@planes[:,:3].T+planes[:,3]
            pos=direction>1e-12;neg=direction<-1e-12;parallel=~(pos|neg)
            lower=np.max(-distance[:,neg]/direction[neg],axis=1)if neg.any()else np.full(len(local),-np.inf)
            upper=np.min(-distance[:,pos]/direction[pos],axis=1)if pos.any()else np.full(len(local),np.inf)
            hit=(lower<=upper)&np.all(distance[:,parallel]<=0,axis=1)
            negative|=hit&(upper<=0);positive|=hit&(lower>=0)
        return negative,positive


def material_between_jaws(points,inner,outer,inner_pose,outer_pose,hand_pose):
    axis=quaternion_matrix_xyzw(np.asarray(hand_pose)[3:])[:,1]
    ineg,ipos=inner.ray_sides(points,inner_pose,axis);oneg,opos=outer.ray_sides(points,outer_pose,axis)
    return (ineg&opos)|(oneg&ipos)
