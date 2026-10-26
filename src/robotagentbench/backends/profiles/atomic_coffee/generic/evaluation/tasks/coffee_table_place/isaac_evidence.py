"""Bound table/tray and plate/payload evidence. No controller/physics writes."""
from collections import deque
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from .contact_evidence import PlacementContacts
from .process_rules import HeldFacts, PayloadFacts, PlacementProcess
from arena_api.physics_defaults import PHYSICS_DT


class TableGeometry:
    def __init__(self, stage, binding, tray_floor_z=None):
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
        from scipy.spatial import ConvexHull
        if UsdGeom.GetStageUpAxis(stage) != 'Z' or UsdGeom.GetStageMetersPerUnit(stage) != 1:
            raise ValueError('Expected metres/Z-up')
        self.stage, self.binding = stage, binding
        if binding.destination_kind == 'tray':
            if tray_floor_z is None or not np.isfinite(tray_floor_z):
                raise ValueError('Verified source tray floor required')
            meshes = [p for p in Usd.PrimRange(stage.GetPrimAtPath(binding.destination))
                      if p.IsA(UsdGeom.Mesh) and p.HasAPI(UsdPhysics.CollisionAPI)
                      and p.GetAttribute('physics:collisionEnabled').Get() is not False]
            if len(meshes) != 1:
                raise ValueError('Expected audited single-mesh tray')
            self.path = str(meshes[0].GetPath())
        else:
            self.path = binding.destination+'/Mesh'
        table = stage.GetPrimAtPath(self.path)
        if not table or not table.IsA(UsdGeom.Mesh) or not table.HasAPI(UsdPhysics.CollisionAPI):
            raise ValueError('Missing actual coffee-table collider')
        if table.GetAttribute('physics:collisionEnabled').Get() is False:
            raise ValueError('Disabled destination collision')
        self.table_tf = UsdGeom.Xformable(table).ComputeLocalToWorldTransform(0)
        vertices = np.asarray([self.table_tf.Transform(Gf.Vec3d(*p))
                               for p in table.GetAttribute('points').Get()], dtype=float)
        if binding.destination_kind == 'tray':
            # Source floor z is rounded in its manifest. Select the matching
            # geometric plane, not a permissive legacy rectangular tray zone.
            floor = vertices[np.abs(vertices[:, 2]-tray_floor_z) <= 1e-6]
            if len(floor) < 3 or np.ptp(floor[:, 2]) > 1e-7:
                raise ValueError('Source tray floor does not match actual geometry')
            self.top = float(floor[:, 2].mean())
            surface = floor[:, :2]
        else:
            self.top = float(vertices[:, 2].max())
            surface = vertices[np.abs(vertices[:, 2]-self.top) <= 1e-7, :2]
        self.region = ConvexHull(surface).equations
        self.vertices = self.body_vertices(stage, binding.held_body)
        # 目的地坐标系：静止窗口按托盘系判断，托盘若移动也不会被误判为“没动”。
        self.destination_inverse = self.table_tf.GetInverse()
        # 落地与出可操作区域沿用 4-x 的做法（用户 2026-09-23）：任务目录内复用
        # shelf_pick 的几何判据，与 tidy_sofa 复用同一模块的先例一致。
        from evaluation.tasks.shelf_pick.geometry import Geometry as ShelfGeometry
        from evaluation.tasks.shelf_pick.targets import spec as shelf_spec
        # 手中物品可能来自任一层库存（5-9…5-14 是 Tier2Items），两个库存根都要量，
        # 否则取不到该物体的半径，工作区判据会直接报 Missing target radius。
        self.workspace = ShelfGeometry(stage, shelf_spec('4-1'),
                                       stock_roots=('/World/Assets/ShelfItems',
                                                    '/World/Assets/Tier2Items'),
                                       targets=(binding.held_body,))
        if binding.payload:
            self.payload_vertices = self.body_vertices(stage, binding.payload)
            self.held_footprint = ConvexHull(self.vertices[:, :2]).equations

    @staticmethod
    def body_vertices(stage, path):
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
        from scipy.spatial import ConvexHull
        import itertools
        body = stage.GetPrimAtPath(path)
        if not body or not body.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError('Missing held rigid body')
        api = UsdPhysics.RigidBodyAPI(body)
        if api.GetKinematicEnabledAttr().Get() or api.GetRigidBodyEnabledAttr().Get() is False:
            raise ValueError('Expected an enabled dynamic object')
        inverse = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(0).RemoveScaleShear().GetInverse()
        local = []
        for prim in Usd.PrimRange(body):
            if not (prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Cube)):
                continue
            visible = UsdGeom.Imageable(prim).ComputeVisibility() != 'invisible'
            collider = prim.HasAPI(UsdPhysics.CollisionAPI) and prim.GetAttribute('physics:collisionEnabled').Get() is not False
            if visible or collider:
                tf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)*inverse
                points = (prim.GetAttribute('points').Get() if prim.IsA(UsdGeom.Mesh) else
                          itertools.product([-float(UsdGeom.Cube(prim).GetSizeAttr().Get())/2,
                                              float(UsdGeom.Cube(prim).GetSizeAttr().Get())/2], repeat=3))
                local.extend(tf.Transform(Gf.Vec3d(*v)) for v in points)
        local = np.asarray(local, dtype=float)
        if local.ndim != 2 or local.shape[1] != 3 or not np.isfinite(local).all():
            raise ValueError('Missing held-object geometry')
        return local[ConvexHull(local).vertices]

    def pose_in_destination(self, position, quaternion):
        """Held-object pose expressed in the destination (tray) frame."""
        from pxr import Gf
        from evaluation.tasks.dining_place.isaac_detection import rotation
        local = self.destination_inverse.Transform(Gf.Vec3d(*[float(v) for v in position]))
        basis = Gf.Transform(self.destination_inverse).GetRotation().GetQuat()
        relative = (rotation([basis.GetReal(), *basis.GetImaginary()]) * rotation(quaternion)).as_quat()
        return ([float(local[0]), float(local[1]), float(local[2])],
                [float(relative[3]), float(relative[0]), float(relative[1]), float(relative[2])])

    def in_region(self, position, quaternion):
        from evaluation.tasks.dining_place.isaac_detection import rotation
        points = rotation(quaternion).apply(self.vertices) + position
        return bool(np.all(points[:, :2]@self.region[:, :2].T+self.region[:, 2] <= 1e-7))

    def payload_geometry(self, held_pos, held_q, payload_pos, payload_q, com):
        from evaluation.tasks.dining_place.isaac_detection import rotation
        if self.binding.payload is None:
            raise ValueError('No payload role')
        # Same COM-in-footprint/above-plate semantics as existing dining_place;
        # actual loaded contact must independently establish support.
        world_com = rotation(payload_q).apply(com)+payload_pos
        local = rotation(held_q).inv().apply(world_com-held_pos)
        return dict(inside=bool(np.all(self.held_footprint[:, :2]@local[:2]+
                                      self.held_footprint[:, 2] <= 1e-7)),
                    above=bool(local[2] > 0), height_m=float(local[2]))

    def check_destination_unchanged(self):
        from pxr import UsdGeom
        tf = UsdGeom.Xformable(self.stage.GetPrimAtPath(self.path)).ComputeLocalToWorldTransform(0)
        if tf != self.table_tf:
            raise RuntimeError('Destination moved; region needs a new reset')


class PlacementReader:
    def __init__(self, stage, binding, episode_id, history_steps, tray_floor_z=None, output=None):
        from pxr import Usd, UsdPhysics, PhysxSchema, PhysicsSchemaTools
        from omni.physx import get_physx_simulation_interface
        from omni.physx.bindings._physx import ContactEventType
        self.binding, self.episode_id = binding, episode_id
        self.geometry = TableGeometry(stage, binding, tray_floor_z)
        self.interface = get_physx_simulation_interface()
        self.step = -1
        self.history = deque(maxlen=history_steps)
        self.latest = None
        self.process = None
        # 逐物理步落盘，和 4-x 的 shelf_detection/physical_facts.jsonl 对等。
        # 没有它，CPU 复算只能拿公共 worker 每控制步一帧的 1/12 降采样，
        # 1 秒静止窗与掉落瞬间都可能落在采样缝里。
        self.trace = None
        self._pending = None
        if output is not None:
            output = Path(output)
            output.mkdir(parents=True, exist_ok=True)
            self.trace = (output/'placement_facts.jsonl').open('x')
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(binding.held_body)).CreateThresholdAttr(0.)
            if binding.payload:
                PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(binding.payload)).CreateThresholdAttr(0.)

        def sleeping(stage_id, actor):
            prim = stage.GetPrimAtPath(PhysicsSchemaTools.intToSdfPath(actor))
            if not prim:
                raise RuntimeError('Missing contact actor')
            return bool(not prim.HasAPI(UsdPhysics.RigidBodyAPI) or self.interface.is_sleeping(stage_id, actor))

        self.contacts = PlacementContacts(binding, PhysicsSchemaTools.intToSdfPath, sleeping,
            (ContactEventType.CONTACT_FOUND, ContactEventType.CONTACT_PERSIST, ContactEventType.CONTACT_LOST),
            self.geometry.top, floor_paths=tuple(self.geometry.workspace.floor_paths))

    def initialize(self, world):
        from evaluation.physics_readers import ReadOnlyRigidBody
        self.world = world
        self.last_time = float(world.current_time)
        self.view = ReadOnlyRigidBody(prim_path=self.binding.held_body)
        self.view.initialize()
        # 机器人底座位姿用于“出可操作区域”的参照（4-x 的同一判据）。
        from evaluation.tasks.shelf_pick.geometry import BASE_LINK
        try:
            self.robot_base = ReadOnlyRigidBody(prim_path=BASE_LINK)
            self.robot_base.initialize()
        except Exception as error:
            raise RuntimeError('Operable-region checks need the robot base link: ' + str(error))
        self.payload_view = None
        if self.binding.payload:
            self.payload_view = ReadOnlyRigidBody(prim_path=self.binding.payload)
            self.payload_view.initialize()
            self.payload_com = np.asarray(self.payload_view.get_com()[0]).reshape(3)

    def after_step(self):
        from evaluation.tasks.dining_place.isaac_detection import rotation
        now = float(self.world.current_time)
        if abs(now-self.last_time-PHYSICS_DT) > 1e-6:
            raise RuntimeError('Missing/repeated physics evidence')
        pos, q = self.view.get_world_pose()
        if not np.isfinite(pos).all():
            raise ValueError('Nonfinite body position')
        up = rotation(q).apply([0,0,1])
        contact = self.contacts.read(self.interface.get_contact_report(), up)
        self.step += 1
        local_pos, local_q = self.geometry.pose_in_destination(pos, q)
        base_pos, _ = self.robot_base.get_world_pose() if self.robot_base is not None else (None, None)
        outside = bool(base_pos is not None and self.geometry.workspace.outside_workspace(
            pos, base_pos, self.binding.held_body))
        facts = HeldFacts(self.step, now, float(pos[2]), contact['held_grasped'],
                          contact['held_supported_by_destination'], self.geometry.in_region(pos, q),
                          tuple(float(x) for x in q),
                          position_in_destination_m=tuple(local_pos),
                          quaternion_in_destination_wxyz=tuple(local_q),
                          on_floor=bool(contact.get('held_on_floor', False)),
                          outside_workspace=outside)
        self.latest = dict(facts=asdict(facts), contacts=contact,
            pose=dict(position_m=pos.tolist(), quaternion_wxyz=q.tolist()))
        payload = None
        if self.payload_view is not None:
            box_pos, box_q = self.payload_view.get_world_pose()
            if not np.isfinite(box_pos).all() or not np.isfinite(self.payload_com).all():
                raise ValueError('Nonfinite payload pose/COM')
            geo = self.geometry.payload_geometry(pos, q, box_pos, box_q, self.payload_com)
            payload = PayloadFacts(float(box_pos[2]), geo['height_m'],
                bool(contact['payload_supported_by_held_body'] and geo['inside'] and geo['above']),
                geo['inside'], geo['above'], contact['payload_grasped'], contact['payload_on_other_support'])
            self.latest.update(payload_facts=asdict(payload),
                payload_pose=dict(position_m=box_pos.tolist(), quaternion_wxyz=box_q.tolist()))
        self.history.append(self.latest)
        if self.process is not None:
            self.latest['events'] = self.process.observe(HeldFacts(**dict(
                self.latest['facts'], step=self.step-self.start_step)), payload)
        self._trace()
        self.last_time = now

    def _trace(self):
        """每个物理步恰好一行。

        落盘滞后一步：begin_episode 会在同一步的 after_step 之后再给 latest 补上
        事件，所以只有下一步到来（或收尾）时这一行的内容才算定稿。
        """
        # 有的测试/探针绕过 __init__ 直接构造 reader，没有 trace 就不落盘。
        if getattr(self, 'trace', None) is None:
            return
        if self._pending is not None:
            self.trace.write(json.dumps(self._pending, allow_nan=False, default=float)+'\n')
            self.trace.flush()
        self._pending = self.latest

    def close(self):
        if getattr(self, 'trace', None) is not None:
            if self._pending is not None:
                self.trace.write(json.dumps(self._pending, allow_nan=False, default=float)+'\n')
            self.trace.close()
            self.trace = None
            self._pending = None

    def begin_episode(self):
        if self.process is not None or self.latest is None:
            raise RuntimeError('Invalid episode start')
        self.start_step = self.step
        self.process = PlacementProcess(self.episode_id, task_id=self.binding.task_id,
                                        carries_payload=self.binding.payload is not None)
        payload = PayloadFacts(**self.latest['payload_facts']) if self.binding.payload else None
        self.latest['events'] = self.process.observe(HeldFacts(**dict(self.latest['facts'], step=0)), payload)
        if getattr(self, 'trace', None) is not None:
            self._pending = self.latest   # 同一步的行尚未落盘，补上的事件会一起写出
