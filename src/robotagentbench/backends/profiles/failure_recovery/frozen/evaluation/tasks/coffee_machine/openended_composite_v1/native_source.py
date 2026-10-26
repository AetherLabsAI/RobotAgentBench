"""Native, read-only PhysX source. Missing instrumentation is a hard blocker.

The scene owner installs its usual contact sensors BEFORE physics starts.
This adapter neither authors contact thresholds nor modifies any rigid body.
"""
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics
from evaluation.tasks.coffee_machine import isaac_evidence as old
from evaluation.tasks.coffee_machine.camera_contacts import CAMERA_COLLIDERS, CAMERA_ROOTS, camera_contacts
from evaluation.physics_readers import ReadOnlyRigidBody
from .scene_service import Substep


class _BoundBodies:
    """Read-only explicit native view for the existing direct PhysX room runner."""
    def __init__(self, simulation_view, paths):
        self.view = simulation_view.create_rigid_body_view(list(paths))
        reported = list(map(str, self.view.prim_paths))
        if self.view.count != len(paths) or sorted(reported) != sorted(paths):
            raise RuntimeError('Native view bound missing or other rigid bodies')
        self.order = np.asarray([reported.index(p) for p in paths])

    def poses(self):
        data = self.view.get_transforms()
        if hasattr(data, 'detach'): data = data.detach().cpu().numpy()
        elif hasattr(data, 'numpy'): data = data.numpy()
        data = np.asarray(data)
        if data.shape != (len(self.order), 7) or not np.isfinite(data).all():
            raise RuntimeError('Invalid measured native poses')
        return data[self.order].copy()

    def positions(self): return self.poses()[:, :3]

    def get_world_pose(self):
        values = self.poses()
        if len(values) != 1: raise RuntimeError('Expected one rigid body')
        return values[0, :3], values[0, [6, 3, 4, 5]]


def preflight(service, floor):
    stage, roles = service.stage, service.roles
    issues = []
    for path in (floor, *getattr(service,'counter_paths',(roles['counter'],)), roles['cup_support'], roles['machine'] + '/Body/Collision/Tray', *CAMERA_COLLIDERS):
        p = stage.GetPrimAtPath(path)
        if not p or not p.IsActive() or not p.HasAPI(UsdPhysics.CollisionAPI) or p.GetAttribute('physics:collisionEnabled').Get() is False:
            issues.append('Missing enabled collider: ' + path)
    robot = stage.GetPrimAtPath(old.ROBOT)
    bodies = [roles['cup'], roles['bag'], roles['button'], roles['machine'] + '/Body']
    if robot:
        bodies += [str(p.GetPath()) for p in Usd.PrimRange(robot) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    else:
        issues.append('Missing robot')
    missing_reports = []
    for path in bodies:
        p = stage.GetPrimAtPath(path)
        if not p or not p.HasAPI(UsdPhysics.RigidBodyAPI):
            issues.append('Missing rigid body: ' + path)
        elif ('PhysxContactReportAPI' not in p.GetAppliedSchemas()
              or p.GetAttribute('physxContactReport:threshold').Get() != 0):
            missing_reports.append(path)
    for path in (*old.HANDS.values(), *(p for pads in old.PADS.values() for p in pads)):
        if not stage.GetPrimAtPath(path): issues.append('Missing hand/pad: ' + path)
    machine_body = stage.GetPrimAtPath(roles['machine'] + '/Body')
    if machine_body and machine_body.GetAttribute('physics:kinematicEnabled').Get() is not True:
        issues.append('Expected existing kinematic machine body')
    if missing_reports:
        issues.append('Scene-owned contact report setup missing; adapter does not author sensors')
    return dict(ready=not issues, issues=issues, missing_contact_report_bodies=missing_reports,
                physics_modified=False, gpu_fluid_required=False)


class NativeCoffeeSource:
    provenance = 'native_physx'

    def __init__(self, service, world, *, floor, allow_idle_60hz=False, simulation_view=None):
        check = preflight(service, floor)
        if not check['ready']:
            raise RuntimeError(str(check))
        self.allowed_dts = (old.PHYSICS_DT, 1/60) if allow_idle_60hz else (old.PHYSICS_DT,)
        if not any(abs(float(world.get_physics_dt()) - dt) <= 1e-10 for dt in self.allowed_dts):
            raise RuntimeError('Current physics clock does not match existing coffee evidence contract')
        from pxr import PhysicsSchemaTools
        from omni.physx import get_physx_simulation_interface
        from omni.physx.bindings._physx import ContactEventType
        self.service, self.world, self.floor = service, world, floor
        self.interface = get_physx_simulation_interface()
        self.last_time = float(world.current_time)
        self.views = {}
        self.machine_body = service.roles['machine'] + '/Body'
        cache = UsdGeom.XformCache()
        self.expected_machine_body = cache.GetLocalToWorldTransform(service.stage.GetPrimAtPath(self.machine_body))
        for path in (*service.objects, service.roles['button'], self.machine_body, *old.HANDS.values()):
            if simulation_view is None:
                view = ReadOnlyRigidBody(path)
                view.initialize()
            else:
                view = _BoundBodies(simulation_view, [path])
            self.views[path] = view
        self.beans = (old.Bodies(service.binding.resolved['bean_bodies']) if simulation_view is None else
                      _BoundBodies(simulation_view, service.binding.resolved['bean_bodies']))

        def sleeping(stage_id, actor):
            p = service.stage.GetPrimAtPath(PhysicsSchemaTools.intToSdfPath(actor))
            if not p: raise RuntimeError('Unknown contact actor')
            if not p.HasAPI(UsdPhysics.RigidBodyAPI) or p.GetAttribute('physics:kinematicEnabled').Get() is True:
                return True
            return bool(self.interface.is_sleeping(stage_id, actor))

        self.contacts = old.Contacts(PhysicsSchemaTools.intToSdfPath, sleeping,
            (ContactEventType.CONTACT_FOUND, ContactEventType.CONTACT_PERSIST, ContactEventType.CONTACT_LOST),
            roots=(*service.objects, service.roles['machine'], *CAMERA_ROOTS), ignored=(service.roles['beans'],))
        self.last_camera_contacts = []

    def after_physics(self, helper_report):
        # Require the existing runner's explicit no-assistance receipt; missing != zero.
        if helper_report['events'] or helper_report['docked_object_count'] or helper_report['tray_dock_enabled']:
            raise RuntimeError('Manipulation assistance invalidates coffee evidence')
        now = float(self.world.current_time)
        observed_dt = float(self.world.get_physics_dt())
        matches = [dt for dt in self.allowed_dts if abs(observed_dt-dt) <= 1e-10]
        if not matches:
            raise RuntimeError('Unexpected native clock')
        dt = matches[0]
        if abs(now - self.last_time - dt) > 1e-6:
            raise RuntimeError('Missing native substep')
        poses = {}
        for path, view in self.views.items():
            p, q = view.get_world_pose()
            poses[path] = dict(position_m=list(map(float, p)), quaternion_wxyz=list(map(float, q)))
        # Machine anchors are fixed; refuse a moved body instead of retaining stale anchors.
        actual = poses[self.machine_body]
        expected = np.asarray(self.expected_machine_body)
        if (not np.allclose(actual['position_m'], expected[3, :3], rtol=0, atol=1e-6)
                or not np.allclose(old.rotation(actual['quaternion_wxyz']).as_matrix().T, expected[:3, :3], rtol=0, atol=1e-6)):
            raise RuntimeError('Native machine pose changed; bound anchors are stale')
        current = UsdGeom.XformCache().GetLocalToWorldTransform(self.service.stage.GetPrimAtPath(self.service.roles['machine']))
        if current != self.service.machine:
            raise RuntimeError('Authored machine frame changed')
        self.contacts.ingest(self.interface.get_contact_report())
        pairs = tuple(self.contacts.pairs())
        self.last_camera_contacts = camera_contacts(pairs)
        self.service.advance(Substep(now, poses, self.beans.positions(), pairs), floor=self.floor, physics_dt=dt)
        self.last_time = now
