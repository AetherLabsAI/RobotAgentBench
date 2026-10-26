"""Task-owned human visual mesh and CPU kinematic rigid collision links.

This module authors ONLY the new actor. It never writes robot, target,
materials/physics on shared assets, or simulation solver parameters.
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation, Slerp

HERE = Path(__file__).resolve().parent


def pose_matrix(position, quaternion_xyzw):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_quat(quaternion_xyzw).as_matrix()
    value[:3, 3] = position
    return value


def transform_points(matrix, points):
    points = np.asarray(points, dtype=float)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


class HumanAsset:
    def __init__(self, path=None):
        self.path = Path(path or HERE / 'assets/derived/right_arm.json').resolve()
        self.data = json.loads(self.path.read_text())
        if self.data['schema'] != 'arena.human.cpu_bake.v1' or self.data['unit_scale_m'] != 0.1:
            raise ValueError('Unverified human bake or scale')
        self.frames = self.data['frames']
        self.bones = tuple(self.frames[0]['bones'])

    def sample(self, closure):
        if not np.isfinite(closure) or not 0 <= closure <= 1:
            raise ValueError('Closure must be in [0,1]')
        v = closure * (len(self.frames) - 1)
        i = min(int(v), len(self.frames) - 2)
        f = v - i
        a, b = self.frames[i:i+2]
        points = (1-f) * np.asarray(a['points']) + f * np.asarray(b['points'])
        transforms = {}
        for name in self.bones:
            m, n = np.array(a['bones'][name]), np.array(b['bones'][name])
            out = np.eye(4)
            out[:3, 3] = (1-f)*m[:3, 3] + f*n[:3, 3]
            out[:3, :3] = Slerp([0,1], Rotation.from_matrix([m[:3,:3],n[:3,:3]]))([f]).as_matrix()[0]
            transforms[name] = out
        return points, transforms

    def collision_groups(self):
        groups = {}
        for bone, indices in self.data['collider_vertices'].items():
            if bone.startswith('palm_'):
                bone = 'hand.R'
            elif bone in ('clavicle.R', 'deltoid.R'):
                bone = 'upper_arm.R'
            groups.setdefault(bone, []).extend(indices)
        return groups


class NativeHuman:
    """Geometry + kinematic target actuator, bound to the owner's CPU view.

    The visual mesh is deformed on CPU. GPU is not used for motion or physics.
    No calls to set_transforms for the target or robot exist here.
    """
    physics_backend = 'cpu_physx'
    motion_backend = 'cpu'

    def __init__(self, stage, root, initial_matrix, asset=None):
        from pxr import UsdGeom, UsdPhysics, PhysxSchema, Sdf, Gf, UsdShade
        from .cpu_contract import validate_cpu_scene
        validate_cpu_scene(stage)
        if stage.GetPrimAtPath(root):
            raise ValueError('Human root already exists; refusing overwrite')
        self.stage, self.root = stage, root
        self.asset = asset or HumanAsset()
        self.links = {}
        self.initial_matrix = np.asarray(initial_matrix, dtype=float)
        self.last_matrix = self.initial_matrix.copy()
        UsdGeom.Xform.Define(stage, root)
        points, bones = self.asset.sample(0)
        mesh = UsdGeom.Mesh.Define(stage, root+'/Visual')
        mesh.CreatePointsAttr(transform_points(self.initial_matrix, points).tolist())
        mesh.CreateFaceVertexCountsAttr([len(f) for f in self.asset.data['faces']])
        mesh.CreateFaceVertexIndicesAttr([v for f in self.asset.data['faces'] for v in f])
        mesh.CreateSubdivisionSchemeAttr('none')
        mesh.CreateDoubleSidedAttr(True)
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.faceVarying).Set(self.asset.data['uv'])
        material = UsdShade.Material.Define(stage,root+'/Skin')
        surface = UsdShade.Shader.Define(stage,root+'/Skin/Surface')
        surface.CreateIdAttr('UsdPreviewSurface')
        surface.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(0.65)
        texture = UsdShade.Shader.Define(stage,root+'/Skin/Texture')
        texture.CreateIdAttr('UsdUVTexture')
        texture.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(str((self.asset.path.parent/self.asset.data['texture']).resolve()))
        uv = UsdShade.Shader.Define(stage,root+'/Skin/UV')
        uv.CreateIdAttr('UsdPrimvarReader_float2')
        uv.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
        texture.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(),'result')
        surface.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(),'rgb')
        material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(),'surface')
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        self.visual = mesh
        for bone, indices in self.asset.collision_groups().items():
            p = transform_points(np.linalg.inv(bones[bone]), points[indices])
            if len(p)<4 or np.linalg.matrix_rank(p-p.mean(axis=0))<3:
                raise ValueError('Degenerate anatomical collider '+bone)
            hull = ConvexHull(p)
            name = bone.replace('.', '_')
            path = root+'/Bodies/'+name
            body = UsdGeom.Xform.Define(stage,path)
            m = self.initial_matrix @ bones[bone]
            body.AddTransformOp().Set(Gf.Matrix4d(m.T.tolist()))
            api = UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            api.CreateRigidBodyEnabledAttr(True)
            api.CreateKinematicEnabledAttr(True)
            PhysxSchema.PhysxContactReportAPI.Apply(body.GetPrim()).CreateThresholdAttr(0.0)
            shape = UsdGeom.Mesh.Define(stage,path+'/Collision')
            shape.CreatePointsAttr(p.tolist())
            shape.CreateFaceVertexCountsAttr([3]*len(hull.simplices))
            shape.CreateFaceVertexIndicesAttr(hull.simplices.flatten().tolist())
            shape.CreateVisibilityAttr('invisible')
            UsdPhysics.CollisionAPI.Apply(shape.GetPrim()).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(shape.GetPrim()).CreateApproximationAttr('convexHull')
            self.links[bone] = path
        # Adjacent anatomical links intersect at skin seams. Filter only this
        # actor against itself, never human vs robot/target/environment.
        group = UsdPhysics.CollisionGroup.Define(stage,root+'/SelfGroup')
        group.GetCollidersCollectionAPI().CreateIncludesRel().SetTargets([root+'/Bodies'])
        group.CreateFilteredGroupsRel().SetTargets([root+'/SelfGroup'])
        self.views = {}
        self.appearance_state = 'active'

    def _set_present(self, present):
        from pxr import UsdGeom, UsdPhysics
        for path in self.links.values():
            UsdPhysics.CollisionAPI(self.stage.GetPrimAtPath(path+'/Collision')).GetCollisionEnabledAttr().Set(present)
        imageable = UsdGeom.Imageable(self.stage.GetPrimAtPath(self.root))
        if present:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()
        return dict(visible=present, collision_enabled=present, rigid_bodies_enabled=True,
                    kinematic_records_retained_for_tensor_stability=True)

    def wait_for_trigger(self):
        """Absent while waiting: neither visible skin nor invisible obstacles."""
        if self.appearance_state != 'active' or self.views:
            raise RuntimeError('Arm initial absence before native binding only')
        self.appearance_state = 'waiting'
        return self._set_present(False)

    def activate_on_trigger(self):
        """Restore original shapes at the admitted entry pose, exactly once."""
        if self.appearance_state != 'waiting':
            raise RuntimeError('Human may appear only from waiting state')
        self.appearance_state = 'active'
        return self._set_present(True)

    def bind(self, simulation_view):
        for bone,path in self.links.items():
            view = simulation_view.create_rigid_body_view(path)
            if view.count != 1:
                raise RuntimeError('Human body not resolved '+path)
            self.views[bone] = view

    def command(self, matrix, closure):
        from pxr import Gf
        if self.appearance_state != 'active':
            raise RuntimeError('Cannot move an absent human actor')
        if len(self.views) != len(self.links):
            raise RuntimeError('Native human not bound')
        points, bones = self.asset.sample(closure)
        matrix = np.asarray(matrix)
        self.visual.GetPointsAttr().Set(transform_points(matrix,points).tolist())
        for bone,view in self.views.items():
            m = matrix @ bones[bone]
            pose = np.concatenate((m[:3,3],Rotation.from_matrix(m[:3,:3]).as_quat())).astype(np.float32)[None,:]
            view.set_kinematic_targets(pose,np.array([0],dtype=np.uint32))
        self.last_matrix = matrix.copy()

    def retire_after_exit(self):
        """Explicit end-of-intervention despawn: visual AND collision stop.

        Caller must finish the visible retreat first and release the target.
        Does not affect robot, target, scene, or any scoring history.
        """
        if self.appearance_state != 'active':
            raise RuntimeError('Only an active intervention can retire')
        self.appearance_state = 'retired'
        # Keep the kinematic actor records allocated: disabling rigid bodies
        # invalidates the world's tensor views, including unrelated targets.
        # With every shape disabled these records exert no contact forces.
        return dict(self._set_present(False), retired_body_count=len(self.links))


class TemporaryHumanGrasp:
    """User-approved external intervention constraint, never a robot grasp.

    Joint anchors are computed from measured native poses at acquisition.
    The caller must supply actual hand-target contact and closed-hand evidence.
    """
    def __init__(self, stage, path, hand_path, target_path):
        from pxr import UsdPhysics
        if stage.GetPrimAtPath(path):
            raise ValueError('Constraint already exists')
        self.joint = UsdPhysics.FixedJoint.Define(stage,path)
        self.joint.CreateBody0Rel().SetTargets([hand_path])
        self.joint.CreateBody1Rel().SetTargets([target_path])
        self.joint.CreateJointEnabledAttr(False)
        # Preserve contacts while constrained: human/target is not penalty.
        self.joint.CreateCollisionEnabledAttr(True)
        self.active = False
        self.ever_acquired = False

    def acquire(self, hand_pose, target_pose, *, hand_target_contact, hand_closed):
        from pxr import Gf
        if self.active or self.ever_acquired:
            raise RuntimeError('One external acquisition only')
        if not hand_target_contact or not hand_closed:
            raise RuntimeError('Cannot attach without measured human contact and closure')
        h, t = pose_matrix(hand_pose[:3],hand_pose[3:]), pose_matrix(target_pose[:3],target_pose[3:])
        relative = np.linalg.inv(h) @ t
        q = Rotation.from_matrix(relative[:3,:3]).as_quat()
        self.joint.CreateLocalPos0Attr(Gf.Vec3f(*relative[:3,3]))
        self.joint.CreateLocalRot0Attr(Gf.Quatf(float(q[3]),Gf.Vec3f(*q[:3])))
        self.joint.CreateLocalPos1Attr(Gf.Vec3f(0))
        self.joint.CreateLocalRot1Attr(Gf.Quatf(1))
        self.joint.GetJointEnabledAttr().Set(True)
        self.active = self.ever_acquired = True

    def release(self):
        if not self.active:
            raise RuntimeError('No active external grasp')
        self.joint.GetJointEnabledAttr().Set(False)
        self.active = False

    def close(self):
        self.joint.GetJointEnabledAttr().Set(False)
        self.active = False
