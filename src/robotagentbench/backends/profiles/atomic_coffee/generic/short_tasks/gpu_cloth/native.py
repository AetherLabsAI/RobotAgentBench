"""Scene-owned native PhysX cloth, authored before attach and read after reset.

Only the new cloth subtree is authored. This module never steps/resets a World,
changes solver settings, animates vertices, or substitutes USD for live facts.
Its measurements are diagnostics; the cloth scoring rules are not frozen.
"""
from copy import deepcopy
import json
import math

import numpy as np

from arena_api.scene_transitions import PhysicsBoundary
from evaluation.tasks.tabletop_pick.soft_cloth.clearance_transition import ClothSurface
from evaluation.tasks.tabletop_pick.soft_cloth.live_surface import LiveSurfaceReader
from evaluation.tasks.tabletop_pick.soft_cloth import native as source_native
from short_tasks.common.contracts import TaskError


BODY_ROOT = source_native.ROOT
MESH_PATH = BODY_ROOT + "/Surface"
PHYSICS_HZ = 240
PHYSICS_DT_S = 1 / PHYSICS_HZ
SAMPLE_SCHEMA = "openend.short_task.gpu_cloth_sample.v1"


def host_array(value):
    """Copy a native tensor for CPU diagnostics, preserving shape and dtype.

    CUDA-to-host transfer completes before the existing reader checks the
    physical clock again. This helper never reshapes or trims native buffers.
    NumPy values are supported for CPU contract tests and authored references;
    admission to the runtime reader separately requires a GPU simulation view.
    """
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


class TorchClothBodyViewHostReader:
    """Read-only host copies over an actual GPU surface-deformable body view."""
    def __init__(self, native_view):
        self.native_view = native_view

    def __getattr__(self, name):
        # Native identity/count metadata is preserved without reinterpretation.
        if name.startswith("set_"):
            raise AttributeError("GPU cloth evidence adapter has no write methods")
        return getattr(self.native_view, name)

    def get_simulation_element_indices(self):
        return host_array(self.native_view.get_simulation_element_indices())

    def get_simulation_nodal_positions(self):
        return host_array(self.native_view.get_simulation_nodal_positions())


class DiagnosticLiveSurfaceReader(LiveSurfaceReader):
    """Keep the approved strict read contract; expose actual native mismatch."""
    @classmethod
    def bind(cls, *, stage, world, episode_id, body_root, expected_surface):
        """Bind the GPU tensor pipeline; do not retry a CPU or USD reader.

        Only the frontend/buffer transport differs from the frozen reader.
        Its strict identity, node counts, exact topology, same-step reuse and
        before/after clock checks continue to run without relaxed conditions.
        """
        from pxr import UsdGeom
        import omni.physics.tensors as tensors

        if (UsdGeom.GetStageUpAxis(stage) != "Z"
                or UsdGeom.GetStageMetersPerUnit(stage) != 1.):
            raise RuntimeError("Require the existing metre-scale Z-up stage")
        body, mesh = stage.GetPrimAtPath(body_root), stage.GetPrimAtPath(expected_surface.mesh)
        if (not body or not body.HasAPI("OmniPhysicsDeformableBodyAPI")
                or not mesh or not mesh.HasAPI("OmniPhysicsSurfaceDeformableSimAPI")):
            raise RuntimeError("Native surface schemas absent; reader will not create them")
        simulation_view = tensors.create_simulation_view("torch")
        ordinal = simulation_view.device_ordinal
        if type(ordinal) is not int or ordinal < 0:
            raise TaskError("Native surface deformable reads require the GPU tensor pipeline",
                kind="load", reason="gpu_cloth_tensor_pipeline_unavailable")
        simulation_view.set_subspace_roots("/")
        native_view = simulation_view.create_surface_deformable_body_view(body_root)
        reader = cls(episode_id=episode_id, body_root=body_root,
            expected_surface=expected_surface, simulation_view=simulation_view,
            body_view=TorchClothBodyViewHostReader(native_view),
            read_clock=lambda: (int(world.current_time_step_index), float(world.current_time)))
        reader.tensor_frontend = "torch"
        reader.tensor_device_ordinal = ordinal
        return reader

    def _validate_view(self):
        try:
            return super()._validate_view()
        except Exception as exc:
            view = self.body_view
            detail = dict(expected_nodes=self.node_count, expected_triangles=len(self.triangles))
            for name in ("count", "prim_paths", "simulation_mesh_prim_paths", "collision_mesh_prim_paths",
                    "num_nodes_per_element", "max_simulation_nodes_per_body", "max_simulation_elements_per_body",
                    "max_collision_nodes_per_body", "max_collision_elements_per_body"):
                try:
                    value = getattr(view, name)
                    detail[name] = list(value) if isinstance(value,(tuple,list)) else int(value)
                except Exception as metadata_error:
                    detail[name] = str(metadata_error)
            for getter in ("get_simulation_element_indices", "get_simulation_nodal_positions"):
                try:
                    values = host_array(getattr(view,getter)())
                    detail[getter] = dict(shape=list(values.shape), dtype=str(values.dtype))
                    if getter == "get_simulation_element_indices" and values.size:
                        detail[getter].update(minimum=int(values.min()), maximum=int(values.max()))
                except Exception as metadata_error:
                    detail[getter] = dict(error=str(metadata_error))
            raise RuntimeError(str(exc)+"; actual_native_view="+json.dumps(detail,sort_keys=True)) from exc


def _runtime_settings(settings):
    handle = settings.get("runtime_settings")
    if handle is None:
        import carb
        handle = carb.settings.get_settings()
    return handle


def _require_settings(settings, *, authoring=False):
    if not isinstance(settings, dict):
        raise TaskError("Trusted native GPU cloth settings are required", kind="load", reason="gpu_cloth_settings_required")
    if (settings.get("physics_backend") != "physx"
            or settings.get("tensor_pipeline") != "gpu"
            or settings.get("tensor_frontend") != "torch"
            or settings.get("deformable_beta_enabled") is not True
            or type(settings.get("physics_dt_s")) not in (int, float)
            or not math.isclose(settings["physics_dt_s"], PHYSICS_DT_S, rel_tol=0, abs_tol=1e-12)):
        raise TaskError("Native PhysX GPU cloth requires beta support and fixed 240 Hz", kind="binding", reason="gpu_cloth_backend_mismatch")
    if authoring and settings.get("physics_attached") is not False:
        raise TaskError("Author cloth only before the caller attaches physics", kind="execution", reason="cloth_authoring_after_attach")
    if settings.get("cloth_contact_preset", "source") not in ("source", "thin_contact_v1"):
        raise TaskError("Unknown scene-owned cloth contact preset", kind="binding", reason="cloth_contact_preset_unknown")


def _descriptor(manifest):
    cloth = manifest.get("gpu_cloth") if isinstance(manifest, dict) else None
    if (not isinstance(cloth, dict)
            or cloth.get("schema") != "openend.short_task.gpu_cloth_scene.v1"
            or cloth.get("body_root") != BODY_ROOT or cloth.get("mesh") != MESH_PATH
            or cloth.get("meters_per_unit") != 1 or cloth.get("up_axis") != "Z"):
        raise TaskError("Exact scene-owned native cloth manifest is required", kind="binding", reason="gpu_cloth_manifest_mismatch")
    center, yaw = cloth.get("initial_center_world_m"), cloth.get("initial_yaw_rad")
    if (not isinstance(center, (list, tuple)) or len(center) != 3
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in center)
            or type(yaw) not in (int, float) or not math.isfinite(yaw)):
        raise TaskError("Finite scene-owned initial cloth transform is required", kind="binding")
    return cloth


def verify_backend(stage, settings):
    """Read existing scene configuration. Never enable or switch a solver."""
    from pxr import UsdPhysics
    _require_settings(settings)
    runtime_settings = _runtime_settings(settings)
    if runtime_settings.get("/persistent/physics/enableDeformableBeta") is not True:
        raise TaskError("Actual native deformable beta support is disabled", kind="load", reason="gpu_cloth_beta_disabled")
    try:
        source_native.preflight_runtime(stage, settings["physics_dt_s"],
            deformable_beta_enabled=settings["deformable_beta_enabled"])
    except RuntimeError as exc:
        raise TaskError(str(exc), kind="binding", reason="gpu_cloth_scene_profile_mismatch") from exc
    scenes = [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if scenes != ["/World/PhysicsScene"]:
        raise TaskError("One exact caller-owned PhysicsScene is required", kind="binding", reason="gpu_cloth_scene_profile_mismatch")
    scene = stage.GetPrimAtPath("/World/PhysicsScene")
    if (scene.GetAttribute("physxScene:timeStepsPerSecond").Get() != PHYSICS_HZ
            or scene.GetAttribute("physxScene:solverType").Get() != "TGS"):
        raise TaskError("The existing GPU scene must use 240 Hz and TGS", kind="binding", reason="gpu_cloth_scene_profile_mismatch")
    direction = scene.GetAttribute("physics:gravityDirection").Get()
    magnitude = scene.GetAttribute("physics:gravityMagnitude").Get()
    if (direction is None or tuple(direction) != (0., 0., -1.)
            or type(magnitude) not in (int, float)
            or not math.isclose(magnitude, 9.81, rel_tol=0, abs_tol=1e-5)):
        raise TaskError("The existing scene must use the declared Earth gravity", kind="binding", reason="gpu_cloth_scene_profile_mismatch")
    return dict(verified=True, physics_backend="physx", physics_hz=PHYSICS_HZ,
                gpu_dynamics=True, broadphase="GPU", solver="TGS", settings_changed=False,
                tensor_pipeline="gpu", tensor_frontend="torch")


def _edge_drape_geometry(manifest, cloth, points, bounds):
    """Permit partial support only for this exact scene-owned diagnostic."""
    diagnostic = manifest.get("geometry_diagnostic")
    if diagnostic is None:
        return None
    from .fixture import edge_drape_definition
    expected = edge_drape_definition()
    if (not isinstance(diagnostic, dict) or set(diagnostic) != set(expected)
            or diagnostic != expected or diagnostic.get("benchmark") is not False
            or manifest.get("full_room_layout") is not False
            or manifest.get("robot_loaded") is not False
            or manifest.get("initialization_owner") != "gpu_cloth_scene_owner.v1"
            or cloth.get("source_role") != "coffee_table"
            or cloth.get("destination_role") != "cloth_basket"
            or cloth["table_region"].get("prim_path") != "/World/Assets/CoffeeTable"
            or not np.allclose(cloth["initial_center_world_m"], [0., 0., .775], rtol=0, atol=1e-9)
            or not math.isclose(cloth["initial_yaw_rad"], 0., rel_tol=0, abs_tol=1e-9)):
        raise TaskError("Only the exact isolated collect edge-drape diagnostic may overhang",
            kind="binding", reason="cloth_edge_drape_diagnostic_mismatch")
    table_low, table_high = bounds["table_region"]
    if (not np.allclose((table_low+table_high)/2, [0., 0., .725], rtol=0, atol=1e-6)
            or not np.allclose(table_high-table_low, expected["table_size_m"], rtol=0, atol=1e-6)
            or not np.allclose(np.ptp(points[:, :2], axis=0), expected["cloth_size_m"], rtol=0, atol=1e-9)
            or (points[:, 1] < table_low[1]-1e-6).any()
            or (points[:, 1] > table_high[1]+1e-6).any()
            or points[:, 2].min() < table_high[2]-1e-6):
        raise TaskError("Edge-drape cloth and actual table geometry differ from the declared diagnostic",
            kind="binding", reason="cloth_edge_drape_geometry_mismatch")
    left = table_low[0]-points[:, 0].min()
    right = points[:, 0].max()-table_high[0]
    supported = np.all((points[:, :2] >= table_low[:2]-1e-6)
                       & (points[:, :2] <= table_high[:2]+1e-6), axis=1)
    unsupported = np.flatnonzero(~supported)
    area_fraction = float((table_high[0]-table_low[0])/np.ptp(points[:, 0]))
    if (not np.allclose([left, right], expected["overhang_x_each_side_m"], rtol=0, atol=1e-6)
            or not math.isclose(area_fraction, expected["initial_support_area_fraction"], rel_tol=0, abs_tol=1e-6)
            or not supported.any() or not len(unsupported)
            or not (points[unsupported, 0] < table_low[0]).any()
            or not (points[unsupported, 0] > table_high[0]).any()):
        raise TaskError("Edge-drape requires verified partial support and off-table nodes on both sides",
            kind="binding", reason="cloth_edge_drape_support_mismatch")
    return dict(**expected, geometry_verified=True,
        node_count=len(points), supported_node_count=int(supported.sum()),
        supported_node_fraction=float(supported.mean()),
        unsupported_node_indices=unsupported.tolist(),
        initial_minimum_table_top_clearance_m=float(points[:, 2].min()-table_high[2]),
        native_support_contact_verified=False, task_initial_state_verified=False)


def verify_fixture_geometry(stage, manifest, *, existing_cloth=False):
    """Read fixture bounds and reject occupied/duplicate initial cloth geometry.

    This verifies the isolated scene's authored initial arrangement, not grasp,
    contact support, settling, coverage or any benchmark success predicate.
    """
    from pxr import Usd, UsdGeom, UsdPhysics
    cloth = _descriptor(manifest)
    if manifest.get("layout_scope") != "isolated_component_preview":
        raise TaskError("Shared-world cloth admission is not validated", kind="load", reason="shared_world_cloth_preview_not_validated")
    if stage.GetPrimAtPath(BODY_ROOT) and not existing_cloth:
        raise TaskError("A cloth body already exists; no duplicate cloth is authored", kind="binding", reason="duplicate_cloth_geometry")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy, UsdGeom.Tokens.render])
    roots, bounds = [], {}
    for name in ("table_region", "storage_region", "basket_region"):
        region = cloth.get(name)
        path = None if not isinstance(region, dict) else region.get("prim_path")
        if not isinstance(path, str) or not path.startswith("/World/Assets/") or path == BODY_ROOT:
            raise TaskError("An exact native fixture prim_path is required for " + name, kind="binding", reason="fixture_geometry_unverified")
        prim = stage.GetPrimAtPath(path)
        if not prim:
            raise TaskError("Declared cloth fixture is missing: " + path, kind="load", reason="fixture_geometry_missing")
        envelope = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if envelope.IsEmpty():
            raise TaskError("Cloth fixture has no measured world bounds", kind="binding", reason="fixture_geometry_unverified")
        low, high = np.asarray(envelope.GetMin()), np.asarray(envelope.GetMax())
        center, size = np.asarray(region.get("center_world_m")), np.asarray(region.get("size_m"))
        if (center.shape != (3,) or size.shape != (3,) or not np.isfinite(center).all()
                or not np.isfinite(size).all() or not (size > 0).all()
                or not np.allclose((low+high)/2, center, rtol=0, atol=1e-6)
                or not np.allclose(high-low, size, rtol=0, atol=1e-6)):
            raise TaskError("Declared region differs from actual fixture geometry", kind="binding", reason="fixture_geometry_mismatch")
        colliders = [p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.CollisionAPI)
                     and p.GetAttribute("physics:collisionEnabled").Get() is True]
        if not colliders or any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in Usd.PrimRange(prim)):
            raise TaskError("The isolated cloth fixture must be a static enabled support", kind="binding", reason="fixture_geometry_unverified")
        roots.append(path)
        bounds[name] = (low, high)
    if len(set(roots)) != 3:
        raise TaskError("Cloth fixture region identities must be distinct", kind="binding")
    local_points = np.asarray(source_native.grid()[0], dtype=float)
    angle, center = cloth["initial_yaw_rad"], np.asarray(cloth["initial_center_world_m"])
    rotation = np.asarray([[math.cos(angle), -math.sin(angle), 0.],
                           [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]])
    points = local_points @ rotation.T + center
    low, high = points.min(axis=0), points.max(axis=0)
    source_region = "storage_region" if cloth.get("source_role") == "cloth_storage" else "table_region"
    source_low, source_high = bounds[source_region]
    geometry_diagnostic = _edge_drape_geometry(manifest, cloth, points, bounds)
    if geometry_diagnostic is None and ((low[:2] < source_low[:2]-1e-6).any() or (high[:2] > source_high[:2]+1e-6).any()
            or low[2] < source_high[2]-1e-6):
        raise TaskError("Initial cloth must lie over the declared source support", kind="binding", reason="cloth_source_geometry_mismatch")
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path == BODY_ROOT or path.startswith(BODY_ROOT+"/"):
            continue
        if any(path == p or path.startswith(p+"/") for p in roots):
            continue
        if not (prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.RigidBodyAPI)
                or prim.IsA(UsdGeom.Mesh)):
            continue
        envelope = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if envelope.IsEmpty():
            continue
        other_low, other_high = np.asarray(envelope.GetMin()), np.asarray(envelope.GetMax())
        if (np.all(other_high[:2] >= low[:2]) and np.all(other_low[:2] <= high[:2])
                and other_high[2] > source_high[2]+1e-6):
            raise TaskError("Foreign initial geometry occupies the cloth source footprint: " + path,
                kind="binding", reason="cloth_initial_occupancy_conflict")
    receipt = dict(fixture_geometry_verified=True, task_initial_state_verified=False,
        initial_occupancy_verification="usd_geometry_checked_before_attach",
        native_support_contact_verified=False)
    if geometry_diagnostic is not None:
        receipt.update(initial_arrangement_diagnostic=True, geometry_diagnostic=geometry_diagnostic,
            initial_occupancy_verification="usd_geometry_diagnostic_checked_before_attach")
    return receipt


def author_component(stage, manifest, settings):
    """Create a native dynamic surface in the anonymous episode edit layer."""
    _require_settings(settings, authoring=True)
    cloth = _descriptor(manifest)
    from short_tasks.scene import verify_manifest
    verify_manifest(manifest)
    backend = verify_backend(stage, settings)
    fixture = verify_fixture_geometry(stage, manifest)
    layer = stage.GetEditTarget().GetLayer()
    if not layer.anonymous:
        raise TaskError("Native cloth authoring requires an anonymous episode layer", kind="binding")
    before = layer.ExportToString()
    try:
        description = source_native.author(stage,
            center_world_m=cloth["initial_center_world_m"], yaw_rad=cloth["initial_yaw_rad"])
        from .contact_preview import apply as select_contact
        contact = select_contact(stage, manifest,
            preset=settings.get("cloth_contact_preset", "source"), physics_attached=False)
    except Exception:
        layer.ImportFromString(before)
        raise
    return dict(schema="openend.short_task.gpu_cloth_authoring.v1", **description,
                body_root=BODY_ROOT, mesh=MESH_PATH, backend=backend,
                fixture_validation=fixture,
                contact_selection=contact,
                initialization_owner="gpu_cloth_scene_owner.v1", authoring_only=True,
                native_physics_verified=False, runtime_physics_validated=False)


def expected_surface(stage, manifest):
    """Read authored topology once to bind exact native node identities."""
    from pxr import Gf, UsdGeom, UsdPhysics
    _descriptor(manifest)
    if (UsdGeom.GetStageUpAxis(stage) != "Z"
            or not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1., abs_tol=1e-12)):
        raise TaskError("Existing cloth stage must use metres and Z-up", kind="binding")
    body, mesh_prim = stage.GetPrimAtPath(BODY_ROOT), stage.GetPrimAtPath(MESH_PATH)
    if (not body or not mesh_prim
            or not body.HasAPI("OmniPhysicsDeformableBodyAPI")
            or not body.HasAPI("PhysxSurfaceDeformableBodyAPI")
            or not mesh_prim.HasAPI("OmniPhysicsSurfaceDeformableSimAPI")
            or body.GetAttribute("omniphysics:kinematicEnabled").Get() is not False
            or body.HasAPI(UsdPhysics.RigidBodyAPI)):
        raise TaskError("The scene does not contain the declared dynamic native cloth", kind="binding", reason="native_cloth_absent")
    mesh = UsdGeom.Mesh(mesh_prim)
    points = mesh.GetPointsAttr().Get()
    counts, indices = mesh.GetFaceVertexCountsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get()
    if not points or not counts or any(n != 3 for n in counts) or len(indices) != 3*len(counts):
        raise TaskError("Complete triangle cloth topology is required", kind="binding")
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(mesh_prim)
    # Authored coordinates are only an identity/topology reference. Runtime
    # observations below always come from the native tensor surface view.
    return ClothSurface(MESH_PATH,
        tuple(tuple(float(v) for v in transform.Transform(Gf.Vec3d(*p))) for p in points),
        tuple(tuple(int(v) for v in indices[i:i+3]) for i in range(0, len(indices), 3)))


class NativeClothComponent:
    """One exact body sampled at the caller's completed native boundaries."""
    def __init__(self, stage, manifest, settings, *, episode_id, task_name):
        if not isinstance(episode_id, str) or not episode_id or task_name not in ("spread_tablecloth", "collect_tablecloth"):
            raise TaskError("Explicit cloth task and episode identity are required", kind="binding")
        _require_settings(settings)
        _descriptor(manifest)
        # Keep the caller's read-only native settings handle; it is not JSON or
        # copied simulator state. The config values themselves remain fixed.
        self.stage, self.manifest, self.settings = stage, deepcopy(manifest), dict(settings)
        self.episode_id, self.task_name = episode_id, task_name
        self.clock = self.world = self.reader = self.reference = self.edges = None
        self.actual_settings = None
        self.last = self.last_boundary = self.initial_boundary = None
        self.fixture_receipt = None
        self.closed = self.accepted = False

    def boundary(self):
        if self.world is None:
            raise TaskError("The caller must initialize the existing World first", kind="execution")
        step, time = self.world.current_time_step_index, self.world.current_time
        if (type(step) is not int or step < 0 or type(time) not in (int, float)
                or not math.isfinite(time) or time < 0):
            raise TaskError("Actual native physics step and time are required", kind="execution")
        return PhysicsBoundary(self.episode_id, step, float(time))

    def initialize(self, clock, view=None):
        if self.reader is not None or self.clock is not None or self.closed:
            raise TaskError("Native cloth binds once after World reset", kind="execution")
        verify_backend(self.stage, self.settings)
        self.fixture_receipt = verify_fixture_geometry(self.stage, self.manifest, existing_cloth=True)
        world = getattr(clock, "world", clock)
        dt = world.get_physics_dt()
        if (not math.isclose(dt, PHYSICS_DT_S, rel_tol=0, abs_tol=1e-12)
                or world.get_physics_context().is_gpu_dynamics_enabled() is not True):
            raise TaskError("The caller World must already use fixed 240 Hz", kind="binding")
        self.clock, self.world = clock, world
        self.actual_settings = _runtime_settings(self.settings)
        surface = expected_surface(self.stage, self.manifest)
        self.reader = DiagnosticLiveSurfaceReader.bind(stage=self.stage, world=world,
            episode_id=self.episode_id, body_root=BODY_ROOT, expected_surface=surface)
        pairs = sorted({tuple(sorted((a,b))) for tri in surface.triangles
                        for a,b in ((tri[0],tri[1]), (tri[1],tri[2]), (tri[2],tri[0]))})
        self.edges = np.asarray(pairs, dtype=int)
        self._read(self.boundary())

    def _read(self, boundary):
        sample = self.reader.read(boundary)
        if self.initial_boundary is None:
            self.initial_boundary = boundary
        points = np.asarray(sample.surface.points, dtype=float)
        if self.reference is None:
            self.reference = points.copy()
        displacement = np.linalg.norm(points-self.reference, axis=1)
        lengths = np.linalg.norm(points[self.edges[:,0]]-points[self.edges[:,1]], axis=1)
        initial_lengths = np.linalg.norm(self.reference[self.edges[:,0]]-self.reference[self.edges[:,1]], axis=1)
        measured = dict(vertex_count=len(points), triangle_count=len(sample.surface.triangles),
            centroid_world_m=points.mean(axis=0).tolist(),
            aabb_world_m=dict(low=points.min(axis=0).tolist(), high=points.max(axis=0).tolist()),
            finite=True, displacement_rms_m=float(np.sqrt(np.mean(displacement**2))),
            displacement_max_m=float(displacement.max()),
            edge_length_change_rms_m=float(np.sqrt(np.mean((lengths-initial_lengths)**2))))
        self.last = dict(schema=SAMPLE_SCHEMA, task_name=self.task_name, instance_id="gpu_cloth",
            body_root=BODY_ROOT, mesh=MESH_PATH, state_source=sample.source,
            diagnostics_only=True, scoring_available=False, live_physics_verified=True,
            tensor_frontend="torch", tensor_pipeline="gpu",
            actor_bound=True, physics_advance_verified=boundary.physics_step > self.initial_boundary.physics_step,
            evidence_physics_step=boundary.physics_step, evidence_time_s=boundary.time_s,
            native_state=dict(points_world_m=points.tolist(),
                triangles=[list(t) for t in sample.surface.triangles],
                physics_step=boundary.physics_step, time_s=boundary.time_s),
            measured=measured, scores={k:None for k in ("process","current_final","deductions","total","maximum")},
            success=None)
        self.last_boundary = boundary
        return self.summary()

    def after_physics(self, boundary=None):
        if self.reader is None or self.closed:
            raise TaskError("No live native cloth reader", kind="execution")
        if (not math.isclose(self.world.get_physics_dt(), PHYSICS_DT_S, rel_tol=0, abs_tol=1e-12)
                or self.world.get_physics_context().is_gpu_dynamics_enabled() is not True
                or self.actual_settings.get("/persistent/physics/enableDeformableBeta") is not True):
            raise TaskError("Episode-wide native cloth capability changed", kind="binding", reason="gpu_cloth_backend_mismatch")
        boundary = self.boundary() if boundary is None else boundary
        previous = self.last_boundary
        if (not isinstance(boundary, PhysicsBoundary)
                or boundary.episode_id != self.episode_id
                or boundary.physics_step != previous.physics_step+1
                or not math.isclose(boundary.time_s-previous.time_s, PHYSICS_DT_S, rel_tol=0, abs_tol=1e-9)):
            raise TaskError("Each completed 240 Hz native boundary must be sampled once", kind="execution", reason="physics_evidence_discontinuous")
        return self._read(boundary)

    def accept_initial(self):
        if self.accepted or self.closed or self.last is None or self.boundary() != self.last_boundary:
            raise TaskError("Measured initial cloth admission occurs once at the current boundary", kind="binding", reason="initial_state_unverified")
        self.accepted = True
        return dict(accepted=True, preview_admission_only=True, initialization_owner="gpu_cloth_scene_owner.v1",
            body_root=BODY_ROOT, mesh=MESH_PATH, actor_bound=True,
            physics_advance_verified=self.last["physics_advance_verified"],
            evidence_physics_step=self.last_boundary.physics_step, evidence_time_s=self.last_boundary.time_s,
            scoring_available=False, runtime_action_validated=False,
            initial_conditions_verified=False, **self.fixture_receipt)

    def summary(self):
        if self.last is None:
            raise TaskError("Native cloth state has not been sampled", kind="execution")
        return deepcopy(self.last)

    @property
    def latest_sample(self):
        return None if self.reader is None else self.reader.last

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.reader is not None:
            self.reader.close()
        self.reader = None
        self.reference = self.edges = None
        # Retain the last immutable receipt; never reset the shared tensor service.
