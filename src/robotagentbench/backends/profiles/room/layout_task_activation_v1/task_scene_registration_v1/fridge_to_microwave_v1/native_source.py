"""Read actual room bodies and contacts; never step/reset/write physical state."""
import math
from dataclasses import asdict
import numpy as np
from scipy.spatial.transform import Rotation
from evaluation.tasks.microwave_put.physical_geometry import rotation, transform, inverse, under, PLATE_BODY
from evaluation.tasks.microwave_put.physical_contacts import PutContactStream
from evaluation.tasks.microwave_put.process_rules import PutFacts
from evaluation.tasks.microwave_put.standards import door_state
from evaluation.tasks.microwave_put.isaac_evidence import final_facts
from evaluation.tasks.microwave_put.openended_fridge_to_microwave_v1.fridge_rules import Frame, CLOSED_DEG
from evaluation.tasks.refrigerator_take.isaac_evidence import outside_bounds, opposed_branches
from .scene import DOORS, ROBOT
from .composite import Sample


class BoundPutContacts(PutContactStream):
    def __init__(self,scene,*args):
        super().__init__(scene.geometry,*args,payload_variant=scene.variant)
        # Explicit existing object binding, not a new body or scene alias.
        self.payload=scene.target


def joint_angle(joint,poses):
    bodies,frames,axis=joint
    a,b=[rotation(poses[p]['quaternion_wxyz']).as_matrix()@f for p,f in zip(bodies,frames)]
    # Joint frames define zero independent of the episode's initial door pose.
    rv=Rotation.from_matrix(a.T@b).as_rotvec()
    return float(np.degrees(rv['XYZ'.index(axis)]))


class NativeSource:
    def __init__(self,scene,clock,view):
        from pxr import PhysicsSchemaTools,UsdPhysics
        from omni.physx import get_physx_simulation_interface
        from omni.physx.bindings._physx import ContactEventType
        from evaluation.tasks.coffee_machine.openended_composite_v1.native_source import _BoundBodies
        self.scene,self.clock=scene,clock
        paths={scene.target,PLATE_BODY,*DOORS}
        for bodies,_,_ in scene.joints.values():paths.update(bodies)
        self.views={p:_BoundBodies(view,[p]) for p in paths}
        com=np.asarray(self.views[scene.target].view.get_coms(),float)
        if com.shape!=(1,7) or not np.isfinite(com).all():raise ValueError('Missing native COM')
        self.com=com[0,:3].copy()
        self.interface=get_physx_simulation_interface();self.decode=PhysicsSchemaTools.intToSdfPath
        def sleeping(sid,actor):
            p=scene.stage.GetPrimAtPath(self.decode(actor))
            if not p:raise RuntimeError('Unresolved actor')
            return bool(not p.HasAPI(UsdPhysics.RigidBodyAPI) or p.GetAttribute('physics:kinematicEnabled').Get() is True
                        or self.interface.is_sleeping(sid,actor))
        self.contacts=BoundPutContacts(scene,self.decode,sleeping,
            (ContactEventType.CONTACT_FOUND,ContactEventType.CONTACT_PERSIST,ContactEventType.CONTACT_LOST))
        self.branches={f'{ROBOT}/gripper_{hand}_{side}_link{link}':(h,s)
                       for h,hand in enumerate(('l','r')) for s,side in enumerate(('inner','outer')) for link in (2,4)}
        self.last_time=float(clock.current_time);self.latest=None;self.closed=False;self.paths={}

    def allowed(self,a,b):
        s=self.scene
        if under(a,ROBOT) and under(b,ROBOT):return True
        for hand,other in ((a,b),(b,a)):
            if under(hand,ROBOT) and '/gripper_' in hand and (under(other,s.target) or any(under(other,d) for d in DOORS)):return True
            if under(hand,s.target) and other in s.source_paths:return True
        return False

    def fridge_contacts(self,report,dt,out):
        # Original refrigerator reader: current positive impulses, no cached
        # awake load or command-based grasp. Source paths are room-bound.
        headers,data=report[:2];forces=np.zeros((2,2,3));collision=external=False
        for h in headers:
            pair=[]
            for token in (h.collider0,h.collider1):
                key=int(token)
                if key not in self.paths:self.paths[key]=str(self.decode(token))
                pair.append(self.paths[key])
            if not any(under(p,ROBOT) or under(p,self.scene.target) for p in pair):continue
            start,count=int(h.contact_data_offset),int(h.num_contact_data)
            if count==0:continue
            if start<0 or count<0 or start+count>len(data):raise ValueError('Incomplete contact range')
            impulses=[np.asarray(tuple(data[i].impulse),float) for i in range(start,start+count)]
            if not all(np.isfinite(v).all() for v in impulses):raise ValueError('Nonfinite impulse')
            if not any(np.linalg.norm(v)>1e-9 for v in impulses):continue
            collision |= not self.allowed(*pair)
            indices=[i for i,p in enumerate(pair) if under(p,self.scene.target)]
            if not indices:continue
            idx=indices[0];other=pair[1-idx]
            load=sum(impulses,np.zeros(3))*(1 if idx==0 else -1)
            external |= not under(other,ROBOT) and load[2]>1e-9
            for path,(hand,side) in self.branches.items():
                if not under(other,path):continue
                for i,impulse in zip(range(start,start+count),impulses):
                    n=np.asarray(tuple(data[i].normal),float);length=np.linalg.norm(n)
                    if not np.isfinite(n).all() or length<1e-9:raise ValueError('Invalid normal')
                    n/=length;forces[hand,side]+=(1 if idx==1 else -1)*n*abs(np.dot(impulse,n))/dt
                break
        right=np.zeros_like(forces);right[1]=forces[1]
        return dict(held=bool(opposed_branches(forces) and not(out and external)),
                    right=bool(opposed_branches(right) and not(out and external)),
                    collision=bool(collision),external_support=bool(external),branch_forces_N=forces.tolist())

    def after_physics(self,helper_report,phase):
        if self.closed:raise RuntimeError('Closed source')
        if helper_report['events'] or helper_report['docked_object_count'] or helper_report['tray_dock_enabled']:
            raise RuntimeError('Unapproved assisted payload evidence')
        now=float(self.clock.current_time);dt=float(self.clock.get_physics_dt())
        if not any(abs(dt-v)<1e-10 for v in (1/60,1/240)) or abs(now-self.last_time-dt)>1e-6:raise RuntimeError('Incomplete native clock')
        poses={}
        for p,v in self.views.items():
            pos,q=v.get_world_pose();rotation(q)
            poses[p]=dict(position_m=pos.tolist(),quaternion_wxyz=q.tolist())
        payload=poses[self.scene.target];plate=poses[PLATE_BODY]
        points=inverse(transform(self.scene.geometry.extrema['payload'],payload),poses[DOORS[1]])
        out=outside_bounds(points,self.scene.lower,self.scene.upper)
        report=self.interface.get_contact_report()
        fc=self.fridge_contacts(report,dt,out)
        pc=self.contacts.read(report,rotation(plate['quaternion_wxyz']).apply([0,0,1]))
        angles={k:joint_angle(j,poses) for k,j in self.scene.joints.items()}
        opened,closed=door_state(angles['microwave']*self.scene.geometry.opening_sign)
        pp=dict(payload=payload,plate=plate);contents=self.scene.geometry.contents(pp)
        held,right=(fc['held'],fc['right']) if phase=='refrigerator' else (pc.held_by_robot,pc.held_by_right)
        facts=PutFacts(self.scene.variant,self.scene.target,opened,closed,
            contents['payload']['inside'],bool(pc.plate_load and self.scene.geometry.com_on_plate(pp,self.com)),
            held,right,pc.released)
        doors=tuple(abs(angles[k]) for k in ('left','right'))
        self.latest=dict(time_s=now,dt=dt,facts=facts,angles=doors,out=out,held=held,
            collision=fc['collision'],wall=pc.wall_impact,
            take_hold=bool(held and out and all(a<=CLOSED_DEG for a in doors)),put_hold=final_facts(facts),
            details=dict(poses=poses,angles_deg=angles,fridge_contacts=fc,put_contacts=asdict(pc),contents=contents))
        self.last_time=now

    def frame(self,episode,index,rows):
        r=self.latest
        if r is None:raise RuntimeError('No native facts')
        ff=Frame(episode,self.scene.target,index,r['time_s'],r['angles'][1],r['angles'],r['held'],r['out'],
                 any(x['collision'] for x in rows),all(x['take_hold'] for x in rows),True)
        return Sample(episode,self.scene.target,index,r['time_s'],ff,r['facts'],
                      any(x['wall'] for x in rows),any(not x['put_hold'] for x in rows))

    def close(self):
        self.contacts.close();self.views.clear();self.closed=True
