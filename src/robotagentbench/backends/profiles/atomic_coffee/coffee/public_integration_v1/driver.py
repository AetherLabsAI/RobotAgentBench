"""Source joint20 direct drives plus the existing mobile base controller.

The older source asset has passive fingers, not the new Mimic contract. Do not
add drives. Public target names/units stay robot22 and all native34 states are
measured. Internal fingers remain the original physical mechanism.
"""
from .robot_api import robot_profiles as public_rp
from .contract import public_base


class SourceDriver:
    def __init__(self, session, robot, stage, mobile, capture):
        import numpy as np
        from pxr import UsdPhysics
        from arena_api import robot_profiles as rp, policy_execution as pe
        self.session, self.robot, self.mobile, self.capture_png = session, robot, mobile, capture
        self.pe, self.profile = pe, rp.JOINT20
        if tuple(public_rp.get(public_rp.MOBILE22).order) != tuple(rp.get(rp.JOINT20).order):
            raise ValueError('Source/public native command order mismatch')
        self.names = tuple(rp.get(rp.JOINT20).order)
        driven, limits = set(), {}
        if set(robot.dof_names) != set(public_rp.NATIVE_JOINTS):
            raise ValueError('Complete measured native34 source robot required')
        for name in robot.dof_names:
            prim = stage.GetPrimAtPath('/World/Robot/joints/'+name)
            angular = prim.IsA(UsdPhysics.RevoluteJoint)
            if not angular and not prim.IsA(UsdPhysics.PrismaticJoint):
                raise ValueError('Unknown native joint type')
            if any(str(s).startswith('PhysicsDriveAPI:') for s in prim.GetAppliedSchemas()):
                driven.add(name)
            lo, hi = (prim.GetAttribute('physics:'+k).Get() for k in ('lowerLimit','upperLimit'))
            if lo is not None and hi is not None:
                limits[name] = [float(np.radians(v)) if angular else float(v) for v in (lo,hi)]
        if driven != set(self.names):
            raise ValueError('Exactly original twenty driven joints required; no finger drive retrofit')
        self.previous, self.limits = pe.build_limits(self.measured_targets(), limits, rp.JOINT20)
        self.indices = np.array([robot.dof_names.index(n) for n in self.names])

    def measured(self):
        return {n: float(v) for n,v in zip(self.robot.dof_names, self.robot.get_joint_positions())}

    def measured_targets(self):
        values = self.measured()
        return {n: values[n] for n in self.names}

    def validate_targets(self, targets):
        self.pe.validate(targets, self.limits, self.profile)

    def base_state(self):
        return public_base(self.mobile.observe())

    def observe(self):
        now, step = self.session.clock.current_time, self.session.control_index
        joints, base = self.measured(), self.base_state()
        images = self.capture_png()
        if self.session.clock.current_time != now or self.session.control_index != step:
            raise RuntimeError('Camera capture advanced physics')
        return dict(episode_id=self.session.episode_id, step=step, time_s=now,
            instruction=self.session.plan['instruction'], joints=joints, base_state=base,
            image_step=step, images=images)

    def apply_action(self, targets, velocity):
        self.interval = self.pe.plan_interval(targets, self.measured_targets(), self.previous,
                                             self.limits, self.profile)
        self.targets = dict(targets)
        self.substep = 0
        self.mobile.apply(velocity)

    def advance(self, dt):
        import numpy as np
        from isaacsim.core.utils.types import ArticulationAction
        if abs(dt-1/240) > 1e-12 or self.substep >= 12:
            raise ValueError('Exactly original 240/20 Hz direct-target intervals required')
        planned = self.interval['interpolated_substep_targets_rad'][self.substep]
        command, changes = self.pe.direct_gripper_substep(self.targets, planned,
            self.measured_targets(), self.limits, self.profile)
        if changes or command != self.targets:
            raise RuntimeError('Direct native commands must not be rewritten')
        self.robot.apply_action(ArticulationAction(joint_positions=np.array([command[n] for n in self.names]),
                                                  joint_indices=self.indices))
        self.session.before_physics(dt)
        self.session.world.step(render=False)
        self.fridge.after_physics_step()
        # after_physics owns the sole reader update and clock acceptance.
        self.substep += 1
        self.previous = dict(command)
        return self.fridge.report()
