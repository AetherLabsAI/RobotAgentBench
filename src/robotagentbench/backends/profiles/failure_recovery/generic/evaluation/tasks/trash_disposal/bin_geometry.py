"""Closed detection cavity from the actual open bag, not its outer AABB.

No asset writes. The lowest escape saddle of the folded bag defines its rim;
the floor-connected surface below that rim is capped for containment queries.
The cap is a scoring boundary only, never a physical collider.
"""
from collections import Counter, defaultdict, deque
import hashlib
import heapq
import numpy as np
from scipy.spatial import ConvexHull

EPS = 1e-7  # Numerical metres only, not a collision/fit tolerance.


def edges_of(faces):
    return Counter(tuple(sorted((int(a),int(b)))) for f in faces for a,b in zip(f,np.roll(f,1)))


def escape_height(vertices, faces):
    edges=edges_of(faces)
    if any(n>2 for n in edges.values()):raise ValueError('Non-manifold bag surface')
    boundary={i for edge,n in edges.items() if n==1 for i in edge}
    if not boundary:raise ValueError('Bag has no free hem')
    adj=[set() for _ in vertices]
    for a,b in edges:adj[a].add(b);adj[b].add(a)
    start=int(vertices[:,2].argmin());cost=np.full(len(vertices),np.inf)
    cost[start]=vertices[start,2];queue=[(cost[start],start)]
    while queue:
        h,i=heapq.heappop(queue)
        if h!=cost[i]:continue
        if i in boundary:return float(h)
        for j in adj[i]:
            nh=max(h,vertices[j,2])
            if nh<cost[j]:cost[j]=nh;heapq.heappush(queue,(nh,j))
    raise ValueError('Bag floor disconnected from rim')


def clip_polygon(poly, normal, offset):
    """Sutherland-Hodgman clipping against n.x+offset <= 0."""
    if not len(poly):return []
    out=[]
    for a,b in zip(poly,[*poly[1:],poly[0]]):
        da,db=np.dot(a,normal)+offset,np.dot(b,normal)+offset
        if da<=0:out.append(a)
        if (da<0<db) or (db<0<da):out.append(a+(b-a)*(da/(da-db)))
    return out


def orient_closed(vertices, faces):
    faces=np.array(faces,dtype=int,copy=True)
    users=defaultdict(list)
    for i,f in enumerate(faces):
        for a,b in zip(f,np.roll(f,-1)):users[tuple(sorted((a,b)))].append((i,a<b))
    if any(len(v)!=2 for v in users.values()):raise ValueError('Detection cavity is not closed manifold')
    neighbours=defaultdict(list)
    for pair in users.values():
        (a,sa),(b,sb)=pair
        neighbours[a].append((b,sa==sb));neighbours[b].append((a,sa==sb))
    flip={0:False};queue=deque([0])
    while queue:
        i=queue.popleft()
        for j,different in neighbours[i]:
            value=flip[i]^different
            if j in flip:
                if flip[j]!=value:raise ValueError('Non-orientable cavity')
            else:flip[j]=value;queue.append(j)
    if len(flip)!=len(faces):raise ValueError('Disconnected cavity boundary')
    for i,value in flip.items():
        if value:faces[i]=faces[i,::-1]
    tri=vertices[faces]
    volume=np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6
    if abs(volume)<1e-12:raise ValueError('Empty cavity')
    if volume<0:faces=faces[:,::-1]
    return faces,abs(float(volume))


def cap_loop(vertices, loop):
    """Ear-clip one simple horizontal boundary; supports a nonconvex rim."""
    ids=list(loop)
    xy=vertices[:,:2]
    def cross(a,b,c):
        u,v=xy[b]-xy[a],xy[c]-xy[a]
        return u[0]*v[1]-u[1]*v[0]
    area=sum(xy[a,0]*xy[b,1]-xy[b,0]*xy[a,1] for a,b in zip(ids,ids[1:]+ids[:1]))
    if area<0:ids.reverse()
    triangles=[]
    while len(ids)>3:
        for i,b in enumerate(ids):
            a,c=ids[i-1],ids[(i+1)%len(ids)]
            if cross(a,b,c)<=1e-16:continue
            inside=False
            for p in ids:
                if p in (a,b,c):continue
                if min(cross(a,b,p),cross(b,c,p),cross(c,a,p))>=-1e-16:
                    inside=True;break
            if not inside:
                triangles.append([a,b,c]);ids.pop(i);break
        else:raise ValueError('Cannot triangulate bag rim (non-simple or degenerate)')
    triangles.append(ids)
    return triangles


class Cavity:
    def __init__(self, vertices, faces, *, rim_z):
        self.vertices=np.asarray(vertices,dtype=float)
        self.faces,self.volume_m3=orient_closed(self.vertices,faces)
        self.triangles=self.vertices[self.faces]
        self.low=self.vertices.min(axis=0);self.high=self.vertices.max(axis=0)
        self.tri_low=self.triangles.min(axis=1);self.tri_high=self.triangles.max(axis=1)
        self.rim_z=float(rim_z)

    @classmethod
    def from_bag(cls, vertices, faces):
        vertices=np.asarray(vertices,dtype=float);faces=np.asarray(faces,dtype=int)
        if vertices.ndim!=2 or vertices.shape[1]!=3 or not np.isfinite(vertices).all():
            raise ValueError('Missing finite bag vertices')
        rim=escape_height(vertices,faces)
        # At the exact saddle the inner and outer surfaces touch at a vertex.
        # Sub-micrometre clipping separates them without changing fit policy.
        level=rim-EPS
        points=[];index={};clipped=[]
        def add(p):
            key=tuple(np.round(p,10))
            if key not in index:index[key]=len(points);points.append(p)
            return index[key]
        for tri in vertices[faces]:
            poly=clip_polygon(list(tri),np.array([0.,0.,1.]),-level)
            ids=[add(p) for p in poly]
            for i in range(1,len(ids)-1):
                f=[ids[0],ids[i],ids[i+1]]
                if len(set(f))==3:clipped.append(f)
        v=np.asarray(points);f=np.asarray(clipped)
        incident=defaultdict(list)
        for i,face in enumerate(f):
            for p in face:incident[p].append(i)
        start=int(v[:,2].argmin());seen=set();queue=list(incident[start])
        while queue:
            i=queue.pop()
            if i in seen:continue
            seen.add(i)
            for p in f[i]:queue.extend(j for j in incident[p] if j not in seen)
        f=f[sorted(seen)]
        edges=edges_of(f);boundary=[e for e,n in edges.items() if n==1]
        adj=defaultdict(list)
        for a,b in boundary:adj[a].append(b);adj[b].append(a)
        if not boundary or any(len(vv)!=2 for vv in adj.values()):raise ValueError('Ambiguous cavity opening')
        if not np.allclose(v[list(adj),2],level,rtol=0,atol=EPS/10):raise ValueError('Cavity reached free hem, not rim')
        first=next(iter(adj));loop=[first];previous=None;current=first
        while True:
            nxt=next(n for n in adj[current] if n!=previous)
            if nxt==first:break
            if nxt in loop:raise ValueError('Invalid cavity opening loop')
            loop.append(nxt);previous,current=current,nxt
        if len(loop)!=len(adj):raise ValueError('Multiple cavity openings need explicit geometry audit')
        f=np.vstack([f,cap_loop(v,loop)])
        used,inverse=np.unique(f,return_inverse=True)
        return cls(v[used],inverse.reshape(-1,3),rim_z=rim)

    def points_inside(self, points):
        points=np.asarray(points)
        # Solid-angle winding number handles nonconvex walls; no AABB success.
        a=self.triangles[None,:,0]-points[:,None,:]
        b=self.triangles[None,:,1]-points[:,None,:]
        c=self.triangles[None,:,2]-points[:,None,:]
        la=np.linalg.norm(a,axis=-1);lb=np.linalg.norm(b,axis=-1);lc=np.linalg.norm(c,axis=-1)
        numerator=np.einsum('...i,...i->...',a,np.cross(b,c))
        denominator=la*lb*lc+np.einsum('...i,...i->...',a,b)*lc+np.einsum('...i,...i->...',b,c)*la+np.einsum('...i,...i->...',c,a)*lb
        winding=np.abs((2*np.arctan2(numerator,denominator)).sum(axis=1))/(4*np.pi)
        return winding>.5

    def contains_hull(self, vertices, hull_faces):
        """All hull vertices inside AND no concave wall cuts its interior."""
        vertices=np.asarray(vertices,dtype=float)
        low,high=vertices.min(axis=0),vertices.max(axis=0)
        if np.any(low<self.low-EPS) or np.any(high>self.high+EPS):return False
        if not self.points_inside(vertices).all():return False
        # A cavity triangle entering the hull interior would invalidate
        # containment even if every hull vertex happens to lie inside.
        tris=vertices[np.asarray(hull_faces)]
        normals=np.cross(tris[:,1]-tris[:,0],tris[:,2]-tris[:,0])
        normals/=np.linalg.norm(normals,axis=1)[:,None]
        offsets=-np.einsum('ij,ij->i',normals,tris[:,0])
        centre=vertices.mean(axis=0)
        sign=np.where(normals@centre+offsets>0,-1.,1.)
        normals*=sign[:,None];offsets*=sign
        candidates=np.all(self.tri_high>low+EPS,axis=1)&np.all(self.tri_low<high-EPS,axis=1)
        for tri in self.triangles[candidates]:
            d=tri@normals.T+offsets
            if np.any(np.all(d>=-EPS,axis=0)):continue
            poly=list(tri)
            for n,off in zip(normals,offsets):
                poly=clip_polygon(poly,n,off+EPS)
                if len(poly)<3:break
            if len(poly)>=3:return False
        return True


class SceneGeometry:
    def __init__(self, stage, definition):
        from pxr import Gf,Usd,UsdGeom,UsdPhysics
        if UsdGeom.GetStageUpAxis(stage)!='Z' or UsdGeom.GetStageMetersPerUnit(stage)!=1:
            raise ValueError('Expected metre Z-up scene')
        self.stage=stage;self.definition=definition
        def world(p):return UsdGeom.Xformable(p).ComputeLocalToWorldTransform(0)
        root=stage.GetPrimAtPath(definition.destination)
        if not root or any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in Usd.PrimRange(root)):
            raise ValueError('Expected existing static bin')
        self.bin_world=world(root)
        self.bin_rigid=np.array(self.bin_world.RemoveScaleShear()).T
        def points(p,frame):
            tf=world(p)*frame.GetInverse()
            return np.array([list(tf.Transform(Gf.Vec3d(*v))) for v in UsdGeom.Mesh(p).GetPointsAttr().Get()])
        bag=stage.GetPrimAtPath(definition.destination+'/geo/Trash_Bag/Trash_Bag_002')
        if not bag or not bag.HasAPI(UsdPhysics.CollisionAPI) or bag.GetAttribute('physics:collisionEnabled').Get() is False or bag.GetAttribute('physics:approximation').Get()!='none':
            raise ValueError('Expected active original triangle bag collider')
        v=points(bag,self.bin_world.RemoveScaleShear());mesh=UsdGeom.Mesh(bag)
        faces=[];offset=0;indices=mesh.GetFaceVertexIndicesAttr().Get()
        for n in mesh.GetFaceVertexCountsAttr().Get():
            ids=indices[offset:offset+n];offset+=n
            faces.extend([[ids[0],ids[i],ids[i+1]] for i in range(1,n-1)])
        self.cavity=Cavity.from_bag(v,faces)
        bag_hash=hashlib.sha256(v.astype('<f8').tobytes()+np.asarray(faces,dtype='<i8').tobytes()).hexdigest()
        target=stage.GetPrimAtPath(definition.target);frame=world(target).RemoveScaleShear();verts=[]
        for p in Usd.PrimRange(target):
            if not p.IsA(UsdGeom.Boundable):continue
            imageable=UsdGeom.Imageable(p)
            collision=p.HasAPI(UsdPhysics.CollisionAPI) and p.GetAttribute('physics:collisionEnabled').Get() is not False
            visible=imageable.ComputeVisibility()!='invisible' and imageable.ComputePurpose() in ('default','render')
            if not (collision or visible):continue
            if p.IsA(UsdGeom.Mesh):verts.extend(points(p,frame))
            elif p.IsA(UsdGeom.Cube):
                half=float(UsdGeom.Cube(p).GetSizeAttr().Get())/2
                tf=world(p)*frame.GetInverse()
                verts.extend([list(tf.Transform(Gf.Vec3d(x,y,z)))
                              for x in (-half,half) for y in (-half,half) for z in (-half,half)])
            elif p.IsA(UsdGeom.Cylinder):
                # 2026-10-02: source-aligned can collider (Task7 replay). Envelope = the two
                # end circles sampled at 48 points; geometry only, no physics opinion.
                cyl=UsdGeom.Cylinder(p);r=float(cyl.GetRadiusAttr().Get());h=float(cyl.GetHeightAttr().Get())
                axis=str(cyl.GetAxisAttr().Get() or 'Z');tf=world(p)*frame.GetInverse()
                for k in range(48):
                    a=2*np.pi*k/48;u,v=r*np.cos(a),r*np.sin(a)
                    for s_ in (-h/2,h/2):
                        local={'X':(s_,u,v),'Y':(u,s_,v),'Z':(u,v,s_)}[axis]
                        verts.append(list(tf.Transform(Gf.Vec3d(*local))))
            else:raise ValueError('Unsupported active target geometry: '+str(p.GetPath()))
        hull=ConvexHull(np.asarray(verts));ids=hull.vertices
        self.target_vertices=np.asarray(verts)[ids]
        self.target_faces=ConvexHull(self.target_vertices).simplices
        self.receipt=dict(policy='trash_disposal.bag_cavity.v1',bag_geometry_sha256=bag_hash,
            target_geometry_sha256=hashlib.sha256(np.asarray(verts,dtype='<f8').tobytes()).hexdigest(),
            cavity_volume_m3=self.cavity.volume_m3,rim_z_bin_local_m=self.cavity.rim_z,
            cavity_triangle_count=len(self.cavity.faces),target_hull_vertices=len(self.target_vertices),
            fit_representation='convex envelope of visible mesh vertices and active mesh/cube collision vertices',
            numerical_epsilon_m=EPS,physical_collider_added=False)

    def measure(self, state):
        from scipy.spatial.transform import Rotation
        from pxr import UsdGeom
        if state['target']!=self.definition.target:raise ValueError('Wrong live target')
        current=UsdGeom.Xformable(self.stage.GetPrimAtPath(self.definition.destination)).ComputeLocalToWorldTransform(0)
        if not np.allclose(np.array(current),np.array(self.bin_world),rtol=0,atol=1e-9):raise RuntimeError('Static bin moved after geometry binding')
        q=np.asarray(state['target_wxyz']);world=Rotation.from_quat(q[[1,2,3,0]]).apply(self.target_vertices)+state['target_xyz_m']
        local=(world-self.bin_rigid[:3,3])@self.bin_rigid[:3,:3]
        return self.cavity.contains_hull(local,self.target_faces)
