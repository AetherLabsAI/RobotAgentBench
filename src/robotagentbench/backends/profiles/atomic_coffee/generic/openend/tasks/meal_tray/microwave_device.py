"""Native readout for the approved existing knob/light mechanism.

Prepare before PhysX initialization, initialize tensor views after it, then call
after_physics on EVERY actual substep (also when the task scorer is inactive).
Only the original mechanism return drive and light are written, in the episode
session. No door/robot/payload poses, control rate or solver parameters are set.
This component alone does not register a meal task with the room factory.
"""
import math
import numpy as np
from .rotary_component.rotary_start_runtime import rotation
from .rotary_component.rotary_return_runtime import RotaryReturnRuntime
from .microwave_illumination import IlluminationGate

ROOT='/World/Assets/DiningTableMicrowave/Asset'
ROBOT='/World/Robot'


def contact_records(report,decode):
    """Current native events only; retain zero-impulse proximity events too."""
    headers,data=report;rows=[]
    for h in headers:
        count=int(h.num_contact_data);start=int(h.contact_data_offset)
        if count<0 or start<0 or start+count>len(data):
            raise ValueError('Malformed native contact span')
        if not count:continue
        points=data[start:start+count]
        forces=[float(np.linalg.norm(p.impulse)) for p in points]
        gaps=[float(p.separation) for p in points]
        if not all(math.isfinite(v) for v in forces+gaps):
            raise ValueError('Nonfinite native contact evidence')
        rows.append(dict(paths=[str(decode(v)) for v in (h.actor0,h.collider0,h.actor1,h.collider1)],
            peak_point_impulse_ns=max(forces),minimum_separation_m=min(gaps)))
    return rows


class MicrowaveDevice:
    def __init__(self,stage):
        from pxr import Usd,UsdPhysics,PhysxSchema
        from .rotary_component.geometry import collect_body_points
        if not stage.GetSessionLayer().anonymous:
            raise ValueError('Anonymous episode session required')
        self.stage=stage;self.body=ROOT+'/body';self.door=ROOT+'/door'
        self.knob=ROOT+'/StartKnob';self.joint=ROOT+'/StartKnobJoint'
        self.lamp=ROOT+'/OpenendCavityLamp';self.gate=None;self.views=None
        required=(self.body,self.door,self.knob,self.joint,self.lamp,ROBOT)
        if any(not stage.GetPrimAtPath(p) for p in required):
            raise ValueError('Room must contain the approved rotary mechanism and robot')
        self.fingers=tuple(str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(ROBOT))
            if p.HasAPI(UsdPhysics.RigidBodyAPI) and p.GetName().startswith(
                ('gripper_r_inner_link','gripper_r_outer_link')))
        if len(self.fingers)<6:raise ValueError('Complete native right-finger bodies required')
        self.bounds={}
        for path in (*self.fingers,self.knob):
            points,_=collect_body_points(stage,path)
            centre=(points.min(0)+points.max(0))/2
            self.bounds[path]=(centre,float(np.linalg.norm(points-centre,axis=1).max()))
        hinge=UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(ROOT+'/door_hinge'))
        if not hinge or hinge.GetAxisAttr().Get()!='Z':raise ValueError('Existing Z door hinge required')
        if list(map(str,hinge.GetBody0Rel().GetTargets()))!=[self.body] or list(map(str,hinge.GetBody1Rel().GetTargets()))!=[self.door]:
            raise ValueError('Unexpected door joint bodies')
        def quat(q):return (float(q.GetReal()),*map(float,q.GetImaginary()))
        self.door_rest=rotation(quat(hinge.GetLocalRot0Attr().Get()))@rotation(quat(hinge.GetLocalRot1Attr().Get())).T
        self.door_sign=-1. if hinge.GetLowerLimitAttr().Get()<0 and hinge.GetUpperLimitAttr().Get()<=0 else 1.
        intensity=stage.GetPrimAtPath(self.lamp).GetAttribute('inputs:intensity')
        if not intensity or intensity.Get()!=0.:raise ValueError('Initial cavity lamp must be off')
        self.paths=(*self.fingers,self.knob,self.body,self.door)
        with Usd.EditContext(stage,stage.GetSessionLayer()):
            # Exactly the source runtime defaults, including resistance/return.
            self.runtime=RotaryReturnRuntime.from_passive_joint(stage,self.joint,self.fingers,
                lamp_intensity_attr=intensity,on_intensity=600.)
            for path in self.paths:
                p=stage.GetPrimAtPath(path);a=p.GetAttribute('physxContactReport:threshold')
                if a and a.HasAuthoredValueOpinion() and a.Get()!=0.:
                    raise ValueError('Conflicting native contact reporter')
                PhysxSchema.PhysxContactReportAPI.Apply(p).CreateThresholdAttr(0.)

    def initialize(self,clock,view):
        if self.views is not None:raise RuntimeError('Device initialized twice')
        self.clock=clock;self.views={}
        for path in self.paths:
            v=view.create_rigid_body_view(path)
            if v.count!=1 or list(v.prim_paths)!=[path]:raise ValueError('Wrong native body '+path)
            self.views[path]=v
        self.robot=view.create_articulation_view(ROBOT)
        if self.robot.count!=1:raise ValueError('One native robot articulation required')
        names=list(self.robot.shared_metatype.dof_names)
        self.jaws=tuple(names.index(n) for n in ('idx81_gripper_r_outer_joint1','idx71_gripper_r_inner_joint1'))

    def accept_initial(self,binding,*,gate_factory=IlluminationGate):
        if self.views is None or self.gate is not None:raise RuntimeError('Wrong device lifecycle')
        from .rotary_component.rotary_start_runtime import measured_y_rotation
        facts=self._measure()
        angle,swing=measured_y_rotation(facts['body_q'],facts['knob_q'],self.runtime.rest_relative)
        from openend.tasks.layout_repair_v1.initial_knob import validate
        initial_measurement=validate(angle,swing,facts['door_angle_deg'])
        # Settle is not scored and cannot operate the knob on behalf of a policy.
        self.gate=gate_factory(binding,self.runtime,initial_time_s=self.clock.current_time)
        return dict(self.gate.summary(),initial_measurement=initial_measurement)

    def _measure(self):
        poses={}
        for path,v in self.views.items():
            t=np.asarray(v.get_transforms()).copy()[0]
            if t.shape!=(7,) or not np.isfinite(t).all():raise ValueError('Invalid native body pose')
            poses[path]=(t[:3],t[[6,3,4,5]])
        centres={p:rotation(poses[p][1])@c+poses[p][0] for p,(c,r) in self.bounds.items()}
        gap=min(np.linalg.norm(centres[p]-centres[self.knob])-self.bounds[p][1]
                -self.bounds[self.knob][1]-.020 for p in self.fingers)
        q=np.asarray(self.robot.get_dof_positions()).copy()[0]
        if not np.isfinite(q).all():raise ValueError('Invalid measured robot joint position')
        opened=bool(q[self.jaws[0]]>=.55 and q[self.jaws[1]]<=-.55)
        relative=self.door_rest.T@rotation(poses[self.body][1]).T@rotation(poses[self.door][1])
        angle=self.door_sign*math.degrees(math.atan2(relative[1,0],relative[0,0]))
        return dict(body_q=poses[self.body][1],knob_q=poses[self.knob][1],door_angle_deg=angle,
            measured_gripper_open=opened,minimum_finger_clearance_m=float(gap))

    def after_physics(self,report):
        if self.gate is None:raise RuntimeError('Device initial state not accepted')
        from pxr import Usd,PhysicsSchemaTools
        facts=self._measure()
        with Usd.EditContext(self.stage,self.stage.GetSessionLayer()):
            return self.gate.advance(self.clock.current_time,1/self.clock.hz,
                native_contacts=contact_records(report,PhysicsSchemaTools.intToSdfPath),**facts)

    def close(self):
        from pxr import Usd
        with Usd.EditContext(self.stage,self.stage.GetSessionLayer()):
            self.runtime.return_drive.disable()
            self.runtime.lamp.Set(0.)
