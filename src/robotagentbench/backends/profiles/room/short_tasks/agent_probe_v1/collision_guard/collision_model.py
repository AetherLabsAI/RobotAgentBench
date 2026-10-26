"""FCL collision model derived from USD, independent of live physics.

No collision pair is newly suppressed. USD filteredPairs and joints whose
collisionEnabled is false are mirrored. Mesh convex decomposition is represented
by its enclosing convex hull for conservative *diagnostic* queries, explicitly
reported; this is not claimed identical to PhysX's cooked decomposition.
"""
import itertools
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor'))
import fcl

def pose(position, quaternion_wxyz):
    t=np.eye(4)
    q=np.asarray(quaternion_wxyz,float)
    t[:3,:3]=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
    t[:3,3]=position
    return t

def native_pose(row):
    t=np.eye(4)
    t[:3,:3]=Rotation.from_quat(row[3:7]).as_matrix()
    t[:3,3]=row[:3]
    return t

def convex(points):
    points=np.asarray(points,dtype=float)
    hull=ConvexHull(points)
    tris=hull.simplices.copy()
    for i,face in enumerate(tris):
        a,b,c=points[face]
        if np.dot(np.cross(b-a,c-a),hull.equations[i,:3])<0:
            tris[i]=face[::-1]
    # Identical convex hull, omit interior vertices from support searches.
    vertices=np.unique(tris)
    remap=np.full(len(points),-1,dtype=np.int32)
    remap[vertices]=np.arange(len(vertices))
    return fcl.Convex(points[vertices],len(tris),
        np.column_stack((np.full(len(tris),3),remap[tris])).astype(np.int32).ravel())

class CollisionModel:
    def __init__(self,model):
        self.data=model
        self.shapes=[]
        self.joints=[]
        self.exclusions={tuple(sorted(pair)) for pair in model['filters']}
        self.approximations=[]
        for j in model['joints']:
            if j['attrs'].get('physics:jointEnabled',True) is False:
                continue
            if len(j['body0'])!=1 or len(j['body1'])!=1:
                continue
            a,b=j['body0'][0],j['body1'][0]
            if not (a.startswith('/World/Robot/') and b.startswith('/World/Robot/')):
                continue
            if j['attrs'].get('physics:collisionEnabled',False) is False:
                self.exclusions.add(tuple(sorted((a,b))))
            if j['attrs'].get('physics:excludeFromArticulation',False):
                continue
            if j['type'] not in ('PhysicsRevoluteJoint','PhysicsPrismaticJoint','PhysicsFixedJoint'):
                raise ValueError('Unsupported tree joint '+j['name'])
            attrs=j['attrs']
            self.joints.append(dict(name=j['name'],type=j['type'],parent=a,child=b,
                t0=pose(attrs.get('physics:localPos0',[0,0,0]),attrs.get('physics:localRot0',[1,0,0,0])),
                it1=np.linalg.inv(pose(attrs.get('physics:localPos1',[0,0,0]),attrs.get('physics:localRot1',[1,0,0,0]))),
                axis=np.eye(3)['XYZ'.index(attrs.get('physics:axis','X'))]))
        for s in model['shapes']:
            local=np.array(s['local'],float)
            scale=np.linalg.norm(local[:3,:3],axis=0)
            rot=local[:3,:3]/scale
            if not np.allclose(rot.T@rot,np.eye(3),atol=1e-6) or np.linalg.det(rot)<0:
                raise ValueError('Unsupported sheared/reflected collision '+s['path'])
            rigid=local.copy();rigid[:3,:3]=rot
            typ=s['type']
            if typ=='Cube':
                size=np.array([s['size']]*3)*scale
                geometry=fcl.Box(*size)
                radius=float(np.linalg.norm(size)/2)
            elif typ=='Cylinder':
                axis='XYZ'.index(s.get('axis','Z'))
                radial=[i for i in range(3) if i!=axis]
                if not np.isclose(scale[radial[0]],scale[radial[1]],rtol=1e-5):
                    raise ValueError('Elliptical cylinder '+s['path'])
                radius0=s['radius']*scale[radial[0]]
                height=s['height']*scale[axis]
                geometry=fcl.Cylinder(radius0,height)
                axisrot=np.eye(4)
                if axis==0:axisrot[:3,:3]=Rotation.from_rotvec([0,np.pi/2,0]).as_matrix()
                if axis==1:axisrot[:3,:3]=Rotation.from_rotvec([-np.pi/2,0,0]).as_matrix()
                rigid=rigid@axisrot
                radius=float(np.hypot(radius0,height/2))
            elif typ=='Sphere':
                if not np.allclose(scale,scale[0],rtol=1e-5):
                    geometry=fcl.Ellipsoid(*(s['radius']*scale))
                else:geometry=fcl.Sphere(s['radius']*scale[0])
                radius=float(s['radius']*max(scale))
            elif typ=='Mesh':
                approx=s.get('physics:approximation')
                if approx not in ('convexHull','convexDecomposition'):
                    raise ValueError('Unsupported mesh approximation '+str(approx))
                points=np.array(s['points'],float)*scale
                geometry=convex(points)
                radius=float(np.max(np.linalg.norm(points,axis=1)))
                if approx=='convexDecomposition':
                    self.approximations.append(dict(path=s['path'],method='conservative_enclosing_convex_hull'))
            else:
                raise ValueError('Unsupported shape '+typ)
            self.shapes.append(dict(path=s['path'],body=s['body'],local=rigid,
                radius=radius,obj=fcl.CollisionObject(geometry)))
        self.pairs=[(i,j) for i,j in itertools.combinations(range(len(self.shapes)),2)
                    if self.shapes[i]['body']!=self.shapes[j]['body'] and
                    tuple(sorted((self.shapes[i]['body'],self.shapes[j]['body']))) not in self.exclusions]
        self._ordered=[]
        known={'/World/Robot/base_link'}
        pending=self.joints[:]
        while pending:
            selected=[j for j in pending if j['parent'] in known]
            if not selected:raise ValueError('Disconnected/cyclic articulation tree')
            for j in selected:
                if j['child'] in known:raise ValueError('Multiple tree parents')
                self._ordered.append(j);known.add(j['child']);pending.remove(j)

    def fk(self,q,root):
        links={'/World/Robot/base_link':root}
        for j in self._ordered:
            motion=np.eye(4)
            if j['type']=='PhysicsRevoluteJoint':
                motion[:3,:3]=Rotation.from_rotvec(j['axis']*q[j['name']]).as_matrix()
            elif j['type']=='PhysicsPrismaticJoint':
                motion[:3,3]=j['axis']*q[j['name']]
            links[j['child']]=links[j['parent']]@j['t0']@motion@j['it1']
        return links
