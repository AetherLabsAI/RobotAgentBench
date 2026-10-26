"""Session-only adapter of openend/probe_native.py's L1 planar base executor.

No wheel model, root teleport, auto-navigation, contact change or task imports.
The original fixed anchor holds initialization. Drives use the existing L1
force mode, zero stiffness, damping/max force 100000, X/Z axes and limits.
"""
import math

from . import mobile_base as contract


class Controller:
    def __init__(self, stage):
        from pxr import Gf, UsdGeom, UsdPhysics
        self.stage, self.Gf = stage, Gf
        self.previous = dict(contract.ZERO)
        self.mode = 'anchor'
        self.packer = contract.StatePacker()
        self.view = self.simulation_view = None
        base_path = '/World/Robot/base_link'
        base = stage.GetPrimAtPath(base_path)
        self.anchor = UsdPhysics.FixedJoint(stage.GetPrimAtPath(base_path + '/FixedJoint'))
        if not base or not self.anchor or self.anchor.GetJointEnabledAttr().Get() is False:
            raise ValueError('22D requires the existing enabled L1 fixed-base initialization anchor')
        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(base)
        position, rotation = matrix.ExtractTranslation(), matrix.ExtractRotationQuat()
        self.joints, self.drives = {}, {}
        # All edits are ephemeral, never saved into the supplied scene/asset.
        if stage.GetEditTarget().GetLayer() != stage.GetSessionLayer():
            raise ValueError('Mobile base constraints must be authored in the episode session layer')
        for kind, cls, axis, drive_type in (
                ('drive', UsdPhysics.PrismaticJoint, 'X', 'linear'),
                ('turn', UsdPhysics.RevoluteJoint, 'Z', 'angular')):
            path = '/World/ArenaMobile22_' + kind
            if stage.GetPrimAtPath(path):
                raise ValueError('Mobile constraint path already exists: ' + path)
            joint = cls.Define(stage, path)
            joint.CreateBody1Rel().SetTargets([base_path])
            joint.CreateExcludeFromArticulationAttr(True)
            joint.CreateJointEnabledAttr(False)
            joint.CreateLocalPos0Attr(Gf.Vec3f(position))
            joint.CreateLocalRot0Attr(Gf.Quatf(rotation))
            joint.CreateLocalPos1Attr(Gf.Vec3f(0))
            joint.CreateLocalRot1Attr(Gf.Quatf(1))
            joint.CreateAxisAttr(axis)
            joint.CreateLowerLimitAttr(-170 if kind == 'turn' else -10)
            joint.CreateUpperLimitAttr(170 if kind == 'turn' else 10)
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), drive_type)
            drive.CreateTypeAttr('force')
            drive.CreateStiffnessAttr(0)
            drive.CreateDampingAttr(100000)
            drive.CreateMaxForceAttr(100000)
            drive.CreateTargetVelocityAttr(0)
            self.joints[kind], self.drives[kind] = joint, drive

    def bind(self):
        import omni.physics.tensors as tensors
        import omni.physx
        self.simulation_view = tensors.create_simulation_view('numpy')
        self.simulation_view.set_subspace_roots('/')
        self.view = self.simulation_view.create_articulation_view('/World/Robot')
        if self.view.count != 1:
            raise RuntimeError('Expected exactly one measured robot articulation')
        self.flush = omni.physx.get_physx_simulation_interface().flush_changes
        self.observe()  # Reject non-finite/missing measured root state before ready.

    def pose(self):
        return self.view.get_root_transforms()[0].copy()

    def observe(self):
        return self.packer.pack(self.pose(), self.view.get_root_velocities()[0])

    def apply(self, velocity):
        velocity = contract.validate(velocity, self.previous)
        v, w = (velocity[n] for n in contract.NAMES)
        desired = 'drive' if abs(v) > 1e-8 else 'turn' if abs(w) > 1e-8 else self.mode
        switched = desired != self.mode
        if switched:
            before = self.pose()
            self.anchor.GetJointEnabledAttr().Set(False)
            for joint in self.joints.values():
                joint.GetJointEnabledAttr().Set(False)
            joint = self.joints[desired]
            joint.GetLocalPos0Attr().Set(self.Gf.Vec3f(*map(float, before[:3])))
            joint.GetLocalRot0Attr().Set(self.Gf.Quatf(
                float(before[6]), self.Gf.Vec3f(*map(float, before[3:6]))))
            joint.GetJointEnabledAttr().Set(True)
            self.flush()
            if list(before) != list(self.pose()):
                raise RuntimeError('Base constraint switch unexpectedly changed measured pose')
            self.mode = desired
        for kind, drive in self.drives.items():
            drive.GetTargetVelocityAttr().Set(v if kind == 'drive' else math.degrees(w))
        self.previous = velocity
        self.verify()
        return dict(command=velocity, constraint_mode=self.mode, switched=switched,
                    mechanism='L1_planar_constraints_not_wheels',
                    root_pose_setter_used=False)

    def verify(self):
        expected_anchor = self.mode == 'anchor'
        if bool(self.anchor.GetJointEnabledAttr().Get()) != expected_anchor:
            raise RuntimeError('Unexpected base anchor state')
        for kind, joint in self.joints.items():
            drive = self.drives[kind]
            limit = 170 if kind == 'turn' else 10
            if (bool(joint.GetJointEnabledAttr().Get()) != (kind == self.mode)
                    or [str(p) for p in joint.GetBody1Rel().GetTargets()] != ['/World/Robot/base_link']
                    or joint.GetBody0Rel().GetTargets()
                    or joint.GetExcludeFromArticulationAttr().Get() is not True
                    or joint.GetAxisAttr().Get() != ('Z' if kind == 'turn' else 'X')
                    or joint.GetLowerLimitAttr().Get() != -limit
                    or joint.GetUpperLimitAttr().Get() != limit
                    or drive.GetTypeAttr().Get() != 'force'
                    or drive.GetStiffnessAttr().Get() != 0
                    or drive.GetDampingAttr().Get() != 100000
                    or drive.GetMaxForceAttr().Get() != 100000):
                raise RuntimeError('Existing L1 mobile constraint configuration changed')
            expected = (self.previous[contract.NAMES[0]] if kind == 'drive'
                        else math.degrees(self.previous[contract.NAMES[1]]))
            if not math.isclose(drive.GetTargetVelocityAttr().Get(), expected, abs_tol=1e-6):
                raise RuntimeError('Base target velocity changed outside action execution')
