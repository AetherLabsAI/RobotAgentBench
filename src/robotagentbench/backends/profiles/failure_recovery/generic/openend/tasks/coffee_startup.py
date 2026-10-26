"""Unified-scene coffee startup, explicitly imported by the scene owner.

Call prepare before attaching PhysX. Only session instrumentation and the
approved L1 wrist-camera restoration are authored. No assets, clock, drives,
poses, task registration or scoring parameters are changed.
"""
import hashlib
import json

CAMERA_SOURCE = {
    'host': '64.62.194.196',
    'path': '/mnt/agent-bench-data/lingjun/runs/microwave-diverse-20260911/install_head_camera_colliders.py',
    'sha256': '2d81b351e8a1953d001f43184a7ba8e3817dc940738cb1c4a62a42ed971ec950',
    'revision': 'head_camera_v2',
    'source_mesh_sha256': 'f58b881307b67f42ea849c93f4e66736c9eacd3c832ff9839f7a1d394327b6d0',
    'source_robot_sha256': 'c38464eedf13b5b678f7e027c7569b6be095c1a69a90f712502e420bb89743a3',
}
MESH_FIELDS = ('points', 'faceVertexCounts', 'faceVertexIndices', 'subdivisionScheme')
MESH_SIGNATURES = {
    'NONE': '2c74b758c66946afd88b42a306181e03ea4e32354ba4b4984388f7c086d0d41b',
    'cam': '2988a59887b08fa714f157bd517f6f0736890d9f41191422f140e949a1e0bc7f',
}
# Exact values from CAMERA_SOURCE, not new tuning.
CAMERA_VALUES = {'physics:collisionEnabled': True, 'physics:approximation': 'convexHull',
                 'physxCollision:contactOffset': .001, 'physxCollision:restOffset': 0.}


def mesh_signature(prim):
    return hashlib.sha256(json.dumps({n: str(prim.GetAttribute(n).Get()) for n in MESH_FIELDS},
                                    sort_keys=True).encode()).hexdigest()


def equal_value(actual, expected):
    # USD float32 representation of the original .001, not a configurable margin.
    import struct
    if type(expected) is float:
        return actual == struct.unpack('f', struct.pack('f', expected))[0]
    return actual == expected


def install_instrumentation(stage, binding):
    """Preflight every edit, then atomically author into the anonymous session."""
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
    from evaluation.tasks.coffee_machine.camera_contacts import CAMERA_COLLIDERS
    robot = stage.GetPrimAtPath('/World/Robot')
    if not robot or not stage.GetSessionLayer().anonymous:
        raise ValueError('Existing robot and anonymous episode session required')
    roles = binding.roles
    bodies = [roles['cup'], roles['bag'], roles['button'], roles['machine'] + '/Body']
    bodies += [str(p.GetPath()) for p in Usd.PrimRange(robot) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    for path in bodies:
        p = stage.GetPrimAtPath(path)
        if not p or not p.IsActive() or not p.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError('Missing report body: ' + path)
        a = p.GetAttribute('physxContactReport:threshold')
        if a and a.HasAuthoredValueOpinion() and a.Get() != 0:
            raise ValueError('Conflicting existing contact threshold: ' + path)
    cache = UsdGeom.XformCache()
    for path in CAMERA_COLLIDERS:
        p = stage.GetPrimAtPath(path)
        if not p or not p.IsActive() or p.GetTypeName() != 'Mesh' or p.IsInstanceProxy():
            raise ValueError('Missing editable camera mesh: ' + path)
        if mesh_signature(p) != MESH_SIGNATURES[p.GetName()]:
            raise ValueError('Camera mesh differs from approved L1 source: ' + path)
        # L1 robot.usda already has cam scale=(1,1,1.02), NONE has unit scale.
        # Preserve that authored configuration; do not normalize the shell.
        m = cache.GetLocalToWorldTransform(p).ExtractRotationMatrix()
        scales = (1., 1., 1.02 if p.GetName() == 'cam' else 1.)
        if any(abs(sum(m[i][k]*m[j][k] for k in range(3)) - (scales[i]**2 if i == j else 0.)) > 1e-6
               for i in range(3) for j in range(3)):
            raise ValueError('Scaled camera mesh: ' + path)
        for name, value in CAMERA_VALUES.items():
            a = p.GetAttribute(name)
            if a and a.HasAuthoredValueOpinion() and not equal_value(a.Get(), value):
                raise ValueError('Conflicting camera physics; will not overwrite: ' + path + '.' + name)
    session = stage.GetSessionLayer()
    before = session.ExportToString()
    try:
        with Usd.EditContext(stage, session):
            for path in CAMERA_COLLIDERS:
                p = stage.GetPrimAtPath(path)
                UsdPhysics.CollisionAPI.Apply(p).CreateCollisionEnabledAttr(True)
                UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr('convexHull')
                api = PhysxSchema.PhysxCollisionAPI.Apply(p)
                api.CreateContactOffsetAttr(.001)
                api.CreateRestOffsetAttr(0.)
            for path in bodies:
                PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(path)).CreateThresholdAttr(0.)
    except Exception:
        session.ImportFromString(before)
        raise
    return dict(camera_source=CAMERA_SOURCE, camera_paths=list(CAMERA_COLLIDERS),
                camera_values=CAMERA_VALUES, contact_report_bodies=bodies,
                contact_threshold=0., authoring_layer='anonymous_session_only',
                source_assets_modified=False, clock_modified=False)


class CoffeeSceneStartup:
    """Scene-owner lifecycle, not a replacement global runner or task registry.

prepare -> host initializes native physics -> on_physics_ready ->
after_physics per substep. Host must separately accept measured initial state
before attach_score; preparation is never a physical acceptance certificate.
"""
    def __init__(self):
        self.service = self.source = self.bridge = None
        self.hotwake = None
        self.closed = False

    def prepare(self, stage, manifest, *, episode_id, instance_id):
        if self.service is not None or self.closed:
            raise RuntimeError('One preparation per scene episode; wake/re-entry cannot reset coffee')
        from evaluation.tasks.coffee_machine.openended_composite_v1.scene_binding import ReadOnlySceneBinding
        from openend.tasks.holding_follow_v1.coffee import FollowCoffeeService as SceneCoffeeService
        from evaluation.tasks.coffee_machine.openended_composite_v1.native_source import preflight
        binding = ReadOnlySceneBinding(stage, manifest)
        binding.task_binding(episode_id, instance_id)
        session = stage.GetSessionLayer()
        before = session.ExportToString()
        try:
            receipt = install_instrumentation(stage, binding)
            service = SceneCoffeeService(stage, manifest, episode_id, instance_id)
            from openend.tasks.scene_requirements.source_geometry import floor_paths
            floors = floor_paths(stage)
            if len(floors) != 1:
                raise ValueError('Unique physical floor required for coffee')
            floor = floors[0]
            check = preflight(service, floor)
            if not check['ready']:
                raise RuntimeError(str(check))
        except Exception:
            session.ImportFromString(before)
            raise
        self.service, self.floor = service, floor
        self.receipt = dict(receipt, instrumentation_preflight=check,
                            status='prepared_not_physically_validated',
                            native_physics_verified=False, runtime_registered=False)
        return self.receipt

    def clock_check(self, physics_dt):
        from arena_api.policy_execution import PHYSICS_DT, SUBSTEPS
        import math
        compatible = math.isclose(float(physics_dt), PHYSICS_DT, rel_tol=0, abs_tol=1e-10)
        return dict(compatible=compatible, observed_physics_dt_s=float(physics_dt),
                    required_physics_dt_s=PHYSICS_DT, control_hz=20, substeps=SUBSTEPS,
                    changes_clock=False)

    def enable_hotwake(self, clock):
        if self.source is not None or self.hotwake is not None:
            raise RuntimeError('Select hotwake before native binding, once per scene')
        from ..clock import CoffeeHotwakeCycle
        self.hotwake = CoffeeHotwakeCycle(self, clock)
        return self.hotwake

    def on_physics_ready(self, world, *, simulation_view=None):
        if self.closed or self.service is None or self.source is not None:
            raise RuntimeError('Prepare once, then bind native source once')
        compatible = self.clock_check(world.get_physics_dt())['compatible']
        idle = self.hotwake is not None and abs(world.get_physics_dt()-1/60) <= 1e-10
        if not (compatible or idle):
            raise RuntimeError('Coffee requires existing 240 Hz physics / 20 Hz control; '
                               '60 Hz navigation cannot be silently reused or reconfigured')
        from evaluation.tasks.coffee_machine.openended_composite_v1.native_source import NativeCoffeeSource
        self.source = NativeCoffeeSource(self.service, world, floor=self.floor,
                                        allow_idle_60hz=self.hotwake is not None, simulation_view=simulation_view)

    def after_physics(self, helper_report):
        if self.closed or self.source is None:
            raise RuntimeError('No live native source')
        self.source.after_physics(helper_report)

    def attach_score(self, *, accepted_binding):
        if self.closed or self.source is None or self.bridge is not None:
            raise RuntimeError('Native evidence and a single scorer required')
        from evaluation.tasks.coffee_machine.openended_composite_v1.scene_service import CoffeeEvidenceBridge
        self.bridge = CoffeeEvidenceBridge(self.service, accepted_binding=accepted_binding,
                                           evidence_source=self.source.provenance)
        return self.bridge

    def close(self):
        # Host destroys/detaches its PhysX scene and anonymous session. No asset cleanup writes.
        if self.service is not None:
            self.service.closed = True
        if self.source is not None:
            self.source.contacts.active.clear()
        self.source = self.bridge = None
        self.closed = True
