"""Opt-in timer return for the independent rotary task, never a pose reset.

The caller must supply MEASURED gripper opening and minimum finger/knob surface
clearance, as well as actual poses and native contacts. Loss of impulse alone is
not proof of release. A small force-limited session drive is disabled throughout
manual startup and while fingers are nearby. The current resistance variant uses
zero-stiffness, zero-velocity limited damping when return is disabled, not a
position detent. Faults remove all drive. No static-friction claim is made.
This implementation is not proof that a robot has operated the knob.
"""
import math
from .rotary_start_runtime import RotaryStartRuntime


class ReturnController:
    def __init__(self,rate_deg_s=3.,release_dwell_s=.15,home_dwell_s=.2,timeout_s=40.):
        values=[rate_deg_s,release_dwell_s,home_dwell_s,timeout_s]
        if not all(math.isfinite(x) and x>0 for x in values):raise ValueError('Invalid return parameters')
        self.rate=rate_deg_s;self.release_dwell=release_dwell_s
        self.home_dwell=home_dwell_s;self.timeout=timeout_s
        self.previous_time=None;self.previous_angle=None;self.clear_since=None
        self.home_since=None;self.running_since=None;self.target=None
        self.was_enabled=False

    def step(self,now,angle,running,hand_clear,*,home_latched=False):
        if not all(math.isfinite(x) for x in [now,angle]):raise ValueError('Invalid measurement')
        if self.previous_time is not None and now<self.previous_time:raise ValueError('Time reversed')
        dt=0. if self.previous_time is None else now-self.previous_time
        speed=0. if dt<=0 else (angle-self.previous_angle)/dt
        self.previous_time=now;self.previous_angle=angle
        if running and self.running_since is None:self.running_since=now
        timed_out=running and now-self.running_since>=self.timeout
        if not (running or home_latched) or not hand_clear or timed_out:
            self.clear_since=None;self.home_since=None;self.target=None;self.was_enabled=False
            return dict(enabled=False,target_deg=0.,home_confirmed=False,timeout=bool(timed_out))
        if self.clear_since is None:self.clear_since=now
        enabled=now-self.clear_since>=self.release_dwell-1e-9
        if enabled:
            # Reacquire from actual angle after every interruption: no accumulated
            # target jump while a hand is near the knob.
            if not self.was_enabled:self.target=max(0.,angle)
            else:self.target=max(0.,self.target-self.rate*dt)
        self.was_enabled=enabled
        settled=enabled and self.target<=1e-9 and abs(angle)<=.75 and abs(speed)<=2.
        if not settled:self.home_since=None
        elif self.home_since is None:self.home_since=now
        home=settled and now-self.home_since>=self.home_dwell-1e-9
        return dict(enabled=enabled,target_deg=self.target if enabled else 0.,
                    home_confirmed=bool(home),timeout=False)


class SessionReturnDrive:
    """USD angular force drive: maxForce is torque in Nm, targetPosition degrees."""
    def __init__(self,stage,joint_path,max_torque_nm=.002):
        from pxr import UsdPhysics
        if stage.GetEditTarget().GetLayer()!=stage.GetSessionLayer():
            raise ValueError('Return drive must be authored only in the session layer')
        if not math.isfinite(max_torque_nm) or not 0<max_torque_nm<=.002:
            raise ValueError('Return torque must be in (0, .002] Nm')
        self.limit=max_torque_nm
        self.drive=UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath(joint_path),'angular')
        self.drive.CreateTypeAttr('force');self.drive.CreateTargetVelocityAttr(0.)
        self.drive.CreateTargetPositionAttr(0.)
        self.drive.CreateStiffnessAttr(0.);self.drive.CreateDampingAttr(0.)
        self.drive.CreateMaxForceAttr(0.)

    def apply(self,enabled,target_deg):
        if not math.isfinite(target_deg) or not 0<=target_deg<=70.5:
            self.disable();raise ValueError('Invalid return target')
        self.drive.GetTargetPositionAttr().Set(target_deg)
        self.drive.GetStiffnessAttr().Set(.04 if enabled else 0.)
        self.drive.GetDampingAttr().Set(.008 if enabled else 0.)
        self.drive.GetMaxForceAttr().Set(self.limit if enabled else 0.)

    def disable(self):
        self.drive.GetMaxForceAttr().Set(0.)
        self.drive.GetStiffnessAttr().Set(0.);self.drive.GetDampingAttr().Set(0.)


class SessionResistanceDrive(SessionReturnDrive):
    """Dynamic friction approximation supported by standalone revolute joints.

PhysxJointAPI jointFriction is articulation-only. The documented alternative is
zero stiffness / zero velocity target / damping. Resistance cannot attract a
particular angle. The small torque cap allows intentional manual turning.
"""
    def __init__(self,stage,joint_path,max_torque_nm=.002,
                 resistance_torque_nm=.0003,resistance_damping=2.):
        if (not math.isfinite(resistance_torque_nm) or not 0<resistance_torque_nm<=.0003
                or not math.isfinite(resistance_damping) or not 0<resistance_damping<=2.):
            raise ValueError('Invalid bounded resistance')
        self.resistance_limit=resistance_torque_nm;self.resistance_damping=resistance_damping
        super().__init__(stage,joint_path,max_torque_nm)
        self.apply(False,0.)

    def apply(self,enabled,target_deg):
        super().apply(enabled,target_deg)
        if not enabled:
            self.drive.GetStiffnessAttr().Set(0.)
            self.drive.GetTargetVelocityAttr().Set(0.)
            self.drive.GetDampingAttr().Set(self.resistance_damping)
            self.drive.GetMaxForceAttr().Set(self.resistance_limit)


class RotaryReturnRuntime(RotaryStartRuntime):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.cycle.require_return_home=True
        self.return_controller=ReturnController()
        self.return_drive=None

    @classmethod
    def from_passive_joint(cls,stage,joint_path,finger_paths,lamp_intensity_attr=None,**options):
        # Verify original passive joint BEFORE adding our disabled return drive.
        if stage.GetEditTarget().GetLayer()!=stage.GetSessionLayer():
            raise ValueError('Use a session layer, never alter the source asset')
        runtime=super().from_passive_joint(stage,joint_path,finger_paths,lamp_intensity_attr,**options)
        runtime.return_drive=SessionResistanceDrive(stage,joint_path)
        return runtime

    def step(self,now,body_q,knob_q,door_angle_deg,native_contacts,
             *,measured_gripper_open=False,minimum_finger_clearance_m=None):
        try:
            result=super().step(now,body_q,knob_q,door_angle_deg,native_contacts)
            # A measured gap AND open hand are mandatory, even during zero-force
            # contact frames. Missing clearance evidence fails closed.
            gap=minimum_finger_clearance_m
            if gap is not None and not math.isfinite(gap):raise ValueError('Invalid measured clearance')
            def under(p,root):return p==root or p.startswith(root+'/')
            near_event=any(
                any(any(under(p,self.knob_path) for p in row['paths'][i:i+2])
                    and any(under(p,f) for p in row['paths'][2-i:4-i] for f in self.finger_paths)
                    for i in [0,2]) for row in native_contacts)
            clear=bool(measured_gripper_open and gap is not None and gap>=.005 and not near_event)
            control=self.return_controller.step(now,result['measured_knob_angle_deg'],
                self.cycle.state=='running',clear)
            if control['timeout']:
                self.cycle.state='fault'
                self.cycle.events.append(dict(event='return_timeout',seconds=now))
            if control['home_confirmed']:
                self.cycle.step(now,result['measured_knob_angle_deg'],door_angle_deg,False,
                                return_home_confirmed=True)
            enabled=control['enabled'] and self.cycle.state=='running'
            if self.return_drive is not None:
                if self.cycle.state=='fault':self.return_drive.disable()
                else:self.return_drive.apply(enabled,control['target_deg'])
            if self.lamp is not None:self.lamp.Set(self.on_intensity if self.cycle.lamp_on else 0.)
            return dict(result,state=self.cycle.state,lamp_on=self.cycle.lamp_on,
                may_open=self.cycle.may_open,return_drive_enabled=enabled,
                idle_resistance_enabled=self.cycle.state!='fault' and not enabled,
                return_target_deg=control['target_deg'],return_home_confirmed=control['home_confirmed'],
                release_clearance_confirmed=clear)
        except Exception:
            if self.return_drive is not None:self.return_drive.disable()
            self.cycle.state='fault'
            if self.lamp is not None:self.lamp.Set(0.)
            raise
