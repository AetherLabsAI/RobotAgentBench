"""Measured pose/contact adapter for the standalone rotary startup task.

Never writes knob poses or joint drives. Simulator integration must supply actual
PhysX poses and per-step native contact records, not command targets. This adapter
alone is not a robot manipulation rollout or full collision qualification.
"""
import math
import numpy as np
from .rotary_start_cycle import RotaryStartCycle


def rotation(q):
    q=np.asarray(q,dtype=float)
    if q.shape!=(4,) or not np.isfinite(q).all() or np.linalg.norm(q)<1e-12:
        raise ValueError('Expected finite nonzero actual wxyz quaternion')
    w,x,y,z=q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def measured_y_rotation(body_q,knob_q,rest_relative):
    relative=np.asarray(rest_relative).T@rotation(body_q).T@rotation(knob_q)
    angle=math.atan2(relative[0,2]-relative[2,0],relative[0,0]+relative[2,2])
    c,s=math.cos(angle),math.sin(angle)
    expected=np.array([[c,0,s],[0,1,0],[-s,0,c]])
    swing=math.degrees(math.acos(float(np.clip((np.trace(expected.T@relative)-1)/2,-1,1))))
    return math.degrees(angle),swing


def finger_knob_contact(records,knob_path,finger_paths):
    """Exact actor/collider pairing, positive native impulse; proximity is not force."""
    def under(path,root):return path==root or path.startswith(root+'/')
    force=False;deep=False
    for row in records:
        paths=row['paths']
        if len(paths)!=4:raise ValueError('Expected native actor/collider pairs')
        sides=[paths[:2],paths[2:]]
        relevant=any(any(under(p,knob_path) for p in sides[i])
            and any(under(p,f) for p in sides[1-i] for f in finger_paths) for i in [0,1])
        if not relevant:continue
        impulse=float(row['peak_point_impulse_ns']);separation=float(row['minimum_separation_m'])
        if not math.isfinite(impulse) or impulse<0 or not math.isfinite(separation):
            raise ValueError('Invalid native contact measurement')
        deep|=separation<-.001
        force|=impulse>1e-6
    return force,deep


class RotaryStartRuntime:
    def __init__(self,authored_rest_relative,knob_path,finger_paths,lamp_intensity_attr=None,
                 on_intensity=600.,duration_seconds=3.):
        if not math.isfinite(on_intensity) or on_intensity<=0:raise ValueError('Invalid lamp intensity')
        if not knob_path.startswith('/') or not finger_paths:raise ValueError('Explicit physical paths required')
        rest=np.asarray(authored_rest_relative,dtype=float)
        if (rest.shape!=(3,3) or not np.isfinite(rest).all()
                or not np.allclose(rest.T@rest,np.eye(3),atol=1e-8)
                or abs(np.linalg.det(rest)-1)>1e-8):
            raise ValueError('Authored zero-angle joint calibration required, never reset from live pose')
        self.rest_relative=rest.copy()
        self.knob_path=knob_path;self.finger_paths=tuple(finger_paths)
        self.lamp=lamp_intensity_attr;self.on_intensity=on_intensity
        self.cycle=RotaryStartCycle(duration_seconds=duration_seconds)
        if self.lamp is not None:self.lamp.Set(0.)

    @classmethod
    def from_passive_joint(cls,stage,joint_path,finger_paths,lamp_intensity_attr=None,**options):
        from pxr import UsdPhysics
        joint=UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(joint_path))
        if not joint or joint.GetAxisAttr().Get()!='Y':raise ValueError('Expected passive Y revolute joint')
        if joint.GetLowerLimitAttr().Get()!=0 or joint.GetUpperLimitAttr().Get()!=70:
            raise ValueError('Unexpected rotary limits')
        if any('DriveAPI' in str(x) for x in joint.GetPrim().GetAppliedSchemas()):
            raise ValueError('Motor mechanism previews cannot validate robot startup')
        bodies=joint.GetBody1Rel().GetTargets()
        if len(bodies)!=1:raise ValueError('One independent knob body required')
        knob=stage.GetPrimAtPath(bodies[0])
        if (not knob.HasAPI(UsdPhysics.RigidBodyAPI)
                or UsdPhysics.RigidBodyAPI(knob).GetKinematicEnabledAttr().Get()):
            raise ValueError('Dynamic physical knob required')
        def quat(q):return [q.GetReal(),*q.GetImaginary()]
        rest=rotation(quat(joint.GetLocalRot0Attr().Get()))@rotation(quat(joint.GetLocalRot1Attr().Get())).T
        return cls(rest,str(bodies[0]),finger_paths,lamp_intensity_attr,**options)

    def step(self,now,body_q,knob_q,door_angle_deg,native_contacts):
        angle,swing=measured_y_rotation(body_q,knob_q,self.rest_relative)
        contact,deep=finger_knob_contact(native_contacts,self.knob_path,self.finger_paths)
        if swing>2 or angle<-.5 or angle>70.5 or deep:
            if self.cycle.state!='fault':
                self.cycle.events.append(dict(event='invalid_knob_physics',seconds=now,
                    measured_angle_deg=angle,swing_deg=swing,deep_finger_contact=deep))
            self.cycle.state='fault'
        result=self.cycle.step(now,angle,door_angle_deg,contact)
        if self.lamp is not None:self.lamp.Set(self.on_intensity if result['lamp_on'] else 0.)
        return dict(**result,measured_knob_angle_deg=angle,swing_deg=swing,
                    native_finger_force_contact=contact,deep_finger_contact=deep)
