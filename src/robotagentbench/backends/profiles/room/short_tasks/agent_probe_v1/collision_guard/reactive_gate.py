"""Current-overlap joint veto, not a trajectory planner or safety certificate.

One current-geometry scan per physics substep. No distance queries, predictive
path subdivision, clearance margin, or velocity reserve. Accepts discrete-step
interpenetration. Exact received targets pass except closing joints, which hold
the measured position at the start of that veto. Contact normals and analytical point Jacobians decide
closing direction; there is no synthesized escape command.
"""
import math
import time
import numpy as np
from .collision_model import CollisionModel, fcl


class ReactiveGate:
    def __init__(self, data):
        self.model=CollisionModel(data)
        self.tree={j['child']:j for j in self.model.joints}
        self.joints={j['name']:j for j in self.model.joints
                     if j['type']!='PhysicsFixedJoint'}
        def ancestors(body):
            result=set()
            while body in self.tree:
                j=self.tree[body]
                if j['type']!='PhysicsFixedJoint': result.add(j['name'])
                body=j['parent']
            return result
        self.ancestors=[ancestors(s['body']) for s in self.model.shapes]
        def rigid_root(body):
            while body in self.tree and self.tree[body]['type']=='PhysicsFixedJoint':
                body=self.tree[body]['parent']
            return body
        def gripper(body):
            name=body.split('/')[-1]
            return name[:9] if name.startswith(('gripper_l','gripper_r')) else None
        retained=[]; self.scope_exemptions=[]
        for i,j in self.model.pairs:
            a,b=(self.model.shapes[k]['body'] for k in (i,j))
            reason=None
            if gripper(a) is not None and gripper(a)==gripper(b):
                reason='same_gripper_original_physics_only'
            elif rigid_root(a)==rigid_root(b): reason='rigid_invariant_clearance'
            if reason: self.scope_exemptions.append(dict(bodies=[a,b],reason=reason))
            else: retained.append((i,j))
        self.model.pairs=retained
        self.pair_indices=np.asarray(retained,dtype=int)
        self.radii=np.asarray([s['radius'] for s in self.model.shapes])
        self.pair_radii=self.radii[self.pair_indices].sum(axis=1)
        self.request=fcl.CollisionRequest(num_max_contacts=1,enable_contact=True)
        self.hold_positions={}

    def scan(self, links):
        """One boolean narrowphase call per broadphase candidate, one contact.

        One convex contact witness is deliberately a lightweight local test,
        not a complete contact manifold or finite-motion guarantee.
        """
        start=time.perf_counter()
        centers=[]
        for s in self.model.shapes:
            t=links[s['body']]@s['local']
            s['obj'].setTransform(fcl.Transform(t[:3,:3],t[:3,3]))
            centers.append(t[:3,3])
        centers=np.asarray(centers)
        difference=centers[self.pair_indices[:,0]]-centers[self.pair_indices[:,1]]
        candidates=np.flatnonzero(np.einsum('ij,ij->i',difference,difference)<=self.pair_radii**2)
        contacts=[]
        for index in candidates:
            i,j=map(int,self.pair_indices[index]); a,b=(self.model.shapes[k] for k in (i,j))
            result=fcl.CollisionResult()
            if not fcl.collide(a['obj'],b['obj'],self.request,result): continue
            if not result.contacts: raise RuntimeError('Collision without directional evidence')
            c=result.contacts[0]
            normal=np.asarray(c.normal,dtype=float); position=np.asarray(c.pos,dtype=float)
            depth=float(c.penetration_depth)
            if not np.all(np.isfinite([*normal,*position,depth])) or np.linalg.norm(normal)<.5:
                raise RuntimeError('Invalid collision contact')
            # FCL normal points from first shape toward second shape.
            contacts.append(dict(indices=[i,j],bodies=[a['body'],b['body']],
                shapes=[a['path'],b['path']],normal=normal.tolist(),position=position.tolist(),
                penetration_depth_m=depth))
        return dict(overlap=bool(contacts),contacts=contacts,narrowphase_calls=len(candidates),
                    scan_wall_s=time.perf_counter()-start)

    def filter_command(self,q,desired,links):
        start=time.perf_counter()
        if set(desired)-set(q): raise ValueError('Unknown commanded joint')
        if not all(math.isfinite(float(v)) for v in [*q.values(),*desired.values()]):
            raise ValueError('Nonfinite state or command')
        state=self.scan(links)
        applied=dict(desired); reasons=[]; held=set(); derivatives={}
        for contact in state['contacts']:
            i,j=contact['indices']; a,b=self.ancestors[i],self.ancestors[j]
            point=np.asarray(contact['position']); normal=np.asarray(contact['normal'])
            closing=[]
            # Shared ancestors move both bodies rigidly; never hold those.
            for name in sorted((a^b)&desired.keys()):
                joint=self.joints[name]; delta=desired[name]-q[name]
                t=links[joint['parent']]@joint['t0']; axis=t[:3,:3]@joint['axis']
                velocity=axis if joint['type']=='PhysicsPrismaticJoint' else np.cross(axis,point-t[:3,3])
                separation_derivative=float(normal@velocity)*(1. if name in b else -1.)
                derivatives.setdefault(name,[]).append(separation_derivative)
                if separation_derivative*delta < -1e-12:
                    held.add(name); closing.append(name)
            reasons.append(dict(bodies=contact['bodies'],depth_m=contact['penetration_depth_m'],
                                held_joints=closing))
        # Keep a fixed target while vetoed; following the measured position on
        # every substep would ratchet the stop along with inertial drift.
        # Release immediately when no contact opposes the received direction.
        anchors={}
        for name in held:
            anchor=self.hold_positions.get(name,q[name])
            # Another contact may make a previous anchor itself a closing
            # direction. Do not automatically drive toward that old anchor.
            if any(d*(anchor-q[name]) < -1e-12 for d in derivatives.get(name,[])):
                anchor=q[name]
            anchors[name]=anchor; applied[name]=anchor
        self.hold_positions=anchors
        return applied,dict(raw=dict(desired),measured={n:q[n] for n in desired},applied=applied,
            hold_positions=dict(anchors),
            held_joints=sorted(held),held_reasons=reasons,state=state,
            wall_s=time.perf_counter()-start,mode='current_overlap_reactive',
            path_queries=0,zero_penetration_guaranteed=False)
