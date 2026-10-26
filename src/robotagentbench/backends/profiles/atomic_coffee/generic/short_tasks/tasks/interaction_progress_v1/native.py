"""Read-only PhysX evidence; distance at 20 Hz, contact edges every substep."""
import json
import numpy as np
from scipy.spatial.transform import Rotation
from .bindings import declarations
from .contact import ContactEdges
from .geometry import collision_surfaces,scoring_distance,DISTANCE_CAP_M
from .policy import active_roles
from .scoring import Progress,Sample,Hand


class NativeProgress:
    def __init__(self,stage,plan,manifest,ports,episode_id,output,fingers):
        from pxr import PhysicsSchemaTools
        from omni.physx.bindings._physx import ContactEventType
        self.plan=plan;self.ports=ports;self.output=output
        self.bindings=declarations(plan,manifest,ports,stage)
        self.surfaces={key:collision_surfaces(stage,row['roots']) for key,row in self.bindings.items()}
        self.progress=Progress(episode_id,self.bindings)
        self.pads={h:tuple('/World/Robot/gripper_'+h+'_'+s+'_link4' for s in ('inner','outer')) for h in ('l','r')}
        groups={h:tuple(p for p in fingers if '/gripper_'+h+'_' in p) for h in ('l','r')}
        if not all(groups.values()):raise ValueError('Both existing finger contact groups required')
        self.contacts=ContactEdges(self.bindings,groups,PhysicsSchemaTools.intToSdfPath,
            (ContactEventType.CONTACT_FOUND,ContactEventType.CONTACT_PERSIST,ContactEventType.CONTACT_LOST))
        self.events=set();self.previous={};self.views={};self.index=-1;self.rebase_needed=False
        self.last_written=-1

    def initialize(self,pool):
        paths=set(p for ss in self.surfaces.values() for p in ss)
        paths.update(p for pp in self.pads.values() for p in pp)
        for path in sorted(paths):
            view=pool.create_rigid_body_view([path])
            if view.count!=1 or tuple(view.prim_paths)!=(path,):raise ValueError('Nonunique native interaction body '+path)
            self.views[path]=view

    def eligible(self):
        out={}
        for key,row in self.bindings.items():
            roles=active_roles(self.plan['task_name'],self.ports[row['owner']])
            out[key]=roles is None or row['role'] in roles
        return out

    def after_physics(self,report,*,admitted):
        entered=self.contacts.update(report)
        if admitted:
            active=self.eligible()
            self.events.update((key,h) for key,h in entered if active[key])
        else:self.events.clear()

    def begin_control(self):
        # A new operation becomes eligible before this interval's robot action,
        # not retroactively for the interval that completed its prerequisite.
        self.progress.set_active_flags(self.eligible())

    def sample(self,time_s,*,baseline=False):
        transforms={p:np.asarray(v.get_transforms(),dtype=float).copy()[0] for p,v in self.views.items()}
        if any(t.shape!=(7,) or not np.isfinite(t).all() or abs(np.linalg.norm(t[3:])-1)>1e-3 for t in transforms.values()):
            raise ValueError('Invalid native interaction pose')
        active=self.eligible();samples={}
        for key,surfaces in self.surfaces.items():
            hands=[]
            previous=self.previous.get(key,transforms)
            for h in ('l','r'):
                points=np.asarray([transforms[p][:3] for p in self.pads[h]])
                def distance(poses):
                    return min(scoring_distance(
                        Rotation.from_quat(poses[body][3:]).inv().apply(points-poses[body][:3]),triangles)
                        for body,triangles in surfaces.items())
                hands.append(Hand(distance(transforms),distance(previous),(key,h) in self.events))
            samples[key]=Sample(bool(active[key]),*hands)
            self.previous[key]={p:transforms[p].copy() for p in surfaces}
        self.index+=1
        self.progress.observe(self.progress.episode_id,self.index,float(time_s),samples,
                              baseline=baseline or self.rebase_needed)
        self.events.clear();self.rebase_needed=False
        if len(self.progress.events)!=self.last_written:
            self.output.write_text(json.dumps(self.summary(),indent=2)+'\n')
            self.last_written=len(self.progress.events)

    def summary(self):
        return dict(self.progress.summary(),bindings=self.bindings,
            distance_reference='nearest inner/outer fingertip link4 origin to collision surface',
            distance_values_capped_at_m=DISTANCE_CAP_M,
            distance_hz=20,contact_sampling='every_physics_substep',evidence_source='native_physx',
            success_unchanged=True)

    def pause(self):
        self.events.clear();self.rebase_needed=True
