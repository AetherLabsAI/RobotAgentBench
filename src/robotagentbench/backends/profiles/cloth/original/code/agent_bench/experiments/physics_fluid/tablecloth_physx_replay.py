"""Diagnostic command replay with native robot dynamics and PhysX cloth.

No attachments, kinematic robot, state replay or post-start pose writes. Material
conversion is experimental and must be calibrated before any production use.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def material_springs(uv, faces, stretch, bend, damping):
    """Rest lengths from flat material coordinates, never from the folded pose."""
    edges = set()
    opposite = {}
    for face in faces:
        for a, b, c in ((face[0], face[1], face[2]), (face[1], face[2], face[0]),
                        (face[2], face[0], face[1])):
            key = tuple(sorted((int(a), int(b))))
            edges.add(key)
            opposite.setdefault(key, []).append(int(c))
    pairs = sorted(edges)
    stiffness = [stretch] * len(pairs)
    for vertices in opposite.values():
        if len(vertices) == 2:
            pair = tuple(sorted(vertices))
            if pair not in edges:
                pairs.append(pair)
                stiffness.append(bend)
                edges.add(pair)
    pairs = np.asarray(pairs, dtype=np.int32)
    rest = np.linalg.norm(uv[pairs[:, 0]] - uv[pairs[:, 1]], axis=1)
    if not np.isfinite(rest).all() or (rest <= 0).any():
        raise ValueError('Invalid material rest lengths')
    return pairs, np.asarray(stiffness), np.full(len(pairs), damping), rest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--robot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--diagnostic-canonical', action='store_true', required=True,
                        help='Explicitly exclude this material/action diagnostic from randomized production')
    parser.add_argument('--diagnostic-unbaked-table', action='store_true',
                        help='A/B diagnostic only: retain original table mesh transform to reproduce contact failure')
    parser.add_argument('--diagnostic-bake-basket', action='store_true',
                        help='A/B diagnostic only: also prepare the unchanged static basket mesh transform')
    parser.add_argument('--diagnostic-basket-sdf-resolution', type=int,
                        help='Delivered owner SDF collider diagnostic; original visible mesh remains unchanged')
    parser.add_argument('--diagnostic-preload-collisions', action='store_true',
                        help='Before reset, load physics and verify the native SDF shape; no simulation steps')
    parser.add_argument('--diagnostic-sdf-witnesses', type=Path,
                        help='Query captured geometric witness points against native SDF; never restore these as state')
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--substeps', type=int,
                        help='Diagnostic timestep override; default preserves the source House physics frequency')
    parser.add_argument('--stretch', type=float, default=5000.)
    parser.add_argument('--bend', type=float, default=100., help='Distance-spring stiffness in N/m, not Newton dihedral stiffness')
    parser.add_argument('--damping', type=float, default=1.)
    parser.add_argument('--velocity-damping', type=float, default=.8,
                        help='Native PBD material global damping; separate from spring damping')
    parser.add_argument('--material-model', choices=('legacy_triangle', 'woven', 'fem_surface'), default='woven')
    parser.add_argument('--shear', type=float, default=5000.)
    parser.add_argument('--youngs-modulus', type=float, default=1.3333333e7)
    parser.add_argument('--surface-thickness', type=float, default=.001)
    parser.add_argument('--surface-bend-modulus', type=float, default=1.22e6)
    parser.add_argument('--elasticity-damping', type=float, default=.01)
    parser.add_argument('--bend-damping', type=float, default=.01)
    parser.add_argument('--surface-contact-offset', type=float,
                        help='FEM contact generation distance; physical rest thickness stays unchanged')
    parser.add_argument('--surface-rest-offset', type=float,
                        help='Explicit FEM collision rest offset; default retains reference particle radius')
    parser.add_argument('--surface-pair-updates', type=int, default=1,
                        help='FEM collision pair update frequency, fixed before episode')
    parser.add_argument('--surface-collision-multiplier', type=int, default=1,
                        help='FEM collision iteration multiplier, fixed before episode')
    parser.add_argument('--surface-max-linear-velocity', type=float,
                        help='Explicit native FEM velocity bound; -1 selects the solver adaptive mode')
    parser.add_argument('--surface-contact-capacity', type=int,
                        help='FEM contact buffer capacity, resolved once before reset; no rigid solver or clock change')
    parser.add_argument('--observe-substeps', type=int, default=0,
                        help='Read-only samples of the last frame for diagnosing high-frequency vibration')
    parser.add_argument('--observe-all-substeps', action='store_true',
                        help='Save every physics step after reset for independent contact audits')
    parser.add_argument('--diagnostic-refine-mesh', action='store_true',
                        help='Before physics, split each material triangle into four without changing its surface')
    parser.add_argument('--clean-initial-shape', choices=('reference', 'rounded', 'isometric'), default='reference',
                        help='Before-episode basket fold geometry; rounded preserves material arc length')
    parser.add_argument('--clean-initial-offset', type=float, nargs=3, default=(0., 0., 0.),
                        help='Explicit fixed setup repair in world metres, before episode only; never a runtime pose write')
    parser.add_argument('--dirty-initial-offset', type=float, nargs=3, default=(0., 0., 0.),
                        help='Explicit pre-episode setup diagnostic, recorded against original geometry')
    parser.add_argument('--iterations', type=int, default=64)
    parser.add_argument('--commands', type=Path,
                        help='Replacement joint commands only; never a replacement physical state')
    args = parser.parse_args()
    coefficients = [args.stretch, args.shear, args.bend, args.damping, args.velocity_damping,
                    args.elasticity_damping, args.bend_damping]
    if (args.substeps is not None and args.substeps < 1) or not 1 <= args.iterations <= 255 or not np.isfinite(coefficients).all() or min(coefficients) < 0:
        parser.error('Invalid solver configuration')
    if min(args.surface_pair_updates, args.surface_collision_multiplier) < 1:
        parser.error('FEM collision update counts must be positive')
    if (args.surface_pair_updates > args.iterations
            or args.surface_collision_multiplier > max(1, args.iterations//2)):
        parser.error('FEM collision update counts exceed the native iteration bounds')
    if args.surface_max_linear_velocity is not None and (
            not np.isfinite(args.surface_max_linear_velocity)
            or (args.surface_max_linear_velocity <= 0 and args.surface_max_linear_velocity != -1)):
        parser.error('FEM velocity bound must be positive or -1 for adaptive mode')
    if args.surface_contact_capacity is not None and args.surface_contact_capacity < 1:
        parser.error('FEM contact buffer capacity must be positive')
    if args.diagnostic_basket_sdf_resolution is not None and args.diagnostic_basket_sdf_resolution < 1:
        parser.error('SDF resolution must be positive')
    if args.diagnostic_sdf_witnesses and args.diagnostic_basket_sdf_resolution is None:
        parser.error('SDF witnesses require the explicit basket SDF diagnostic')
    if args.observe_substeps < 0 or (args.substeps is not None and args.observe_substeps > args.substeps):
        parser.error('Substep observation count must be between zero and substeps')
    args.output.mkdir(parents=True, exist_ok=False)
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True})
    try:
        run(args, app)
    except BaseException:
        import traceback
        failure = traceback.format_exc()
        print(failure, file=sys.stderr, flush=True)
        (args.output/'startup_or_runtime_failure.json').write_text(
            json.dumps({'failure': failure, 'diagnostic_only': True, 'eligible_for_production': False}, indent=2))
        # Kit shutdown can terminate Python before the exception propagates.
        # Preserve a failing process status as well as the failure artifact.
        import omni.kit.app
        omni.kit.app.get_app().post_quit(1)
        raise
    finally:
        app.close()


def run(args, app):
    import torch
    import carb.settings
    import omni.usd
    import omni.physics.tensors
    from pxr import Usd, UsdUtils, UsdGeom, UsdPhysics, UsdShade, PhysxSchema, Gf, Sdf, Vt
    from omni.physx.scripts import particleUtils
    from isaacsim.core.api import SimulationContext
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.types import ArticulationAction
    from scipy.spatial.transform import Rotation
    from tablecloth_scene_binding import station_xyyaw, source_record
    from tablecloth_physics_scene import enable_required_contacts, audit_required_contacts
    from tablecloth_native_material import woven_springs, curvature_metrics
    from git_provenance import source_revision
    # Capture the actual entrypoint/dependencies, including dirty source. A
    # commit alone hid the difference between experiments and this entrypoint.
    source_paths = [Path(__file__).with_name(name) for name in (
        'tablecloth_physx_replay.py', 'tablecloth_native_geometry.py',
        'tablecloth_native_surface.py', 'tablecloth_native_material.py',
        'tablecloth_native_initial.py', 'tablecloth_physics_scene.py',
        'tablecloth_scene_binding.py')]
    source_paths.append(REPO/'real2sim/teleop/tablecloth_basket_contract.py')
    source_paths.append(REPO/'arena_scene/v1_2/demo_sim/scene_room_baseline.usda')
    robot_tracked = (args.robot.resolve().is_relative_to(REPO) and subprocess.run(
        ['git', '-C', str(REPO), 'ls-files', '--error-unmatch', '--', str(args.robot.resolve().relative_to(REPO))],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0)
    if robot_tracked:
        source_paths.append(args.robot)
    git_source = source_revision(source_paths, repo=REPO)
    start_source = source_record()
    start_source['commit'] = git_source['git_commit']
    robot_asset = dict(path=str(args.robot.resolve()), size_bytes=args.robot.stat().st_size,
                       git_tracked=robot_tracked)
    (args.output/'git_source.json').write_text(json.dumps(git_source, indent=2)+'\n')
    surface = args.material_model == 'fem_surface'
    if surface:
        from omni.physx.bindings._physx import SETTING_ENABLE_DEFORMABLE_BETA
        carb.settings.get_settings().set_bool(SETTING_ENABLE_DEFORMABLE_BETA, True)
        from tablecloth_native_surface import create_surface, SurfaceReadback
        if min(args.youngs_modulus, args.surface_thickness, args.surface_bend_modulus) <= 0:
            raise ValueError('Surface material parameters must be positive')
        args.surface_friction = None  # resolved from the same owner cloth config below

    meta = json.loads((args.reference/'trial.json').read_text())
    with np.load(args.reference/'action.npz') as z:
        action = {k: z[k] for k in ('joint_target', 'joint_names', 'time_s')}
    with np.load(args.reference/'full_state.npz') as z:
        initial = z['initial_cloth_q'].copy()
        faces, uv = z['faces'], z['uv']
        cloth_ids = z['cloth_id'] if 'cloth_id' in z.files else np.zeros(len(initial), dtype=int)
        phases = z['phase']
    reference_initial = initial.copy()
    dirty_initial_offset = np.asarray(args.dirty_initial_offset, dtype=float)
    if not np.isfinite(dirty_initial_offset).all():
        raise ValueError('Invalid dirty-cloth initial offset')
    initial[cloth_ids == 0] += dirty_initial_offset
    initial_shape_record = dict(model=args.clean_initial_shape)
    if args.clean_initial_shape in ('rounded', 'isometric'):
        from tablecloth_native_initial import rounded_basket_fold, initial_edge_ratios
        clean = np.flatnonzero(cloth_ids == 1)
        if not len(clean):
            raise ValueError('No clean cloth exists for the requested fold repair')
        local_index = np.full(len(initial), -1, dtype=int)
        local_index[clean] = np.arange(len(clean))
        local_faces = local_index[faces[np.all(cloth_ids[faces] == 1, axis=1)]]
        initial[clean] = rounded_basket_fold(uv[clean], meta['basket_bounds_m'],
                                            isometric=args.clean_initial_shape == 'isometric')
        ratios = initial_edge_ratios(initial[clean], uv[clean], local_faces)
        minimum = .99999 if args.clean_initial_shape == 'isometric' else .97
        if ratios.min() < minimum or ratios.max() > 1.00001:
            raise ValueError('Rounded initialization does not preserve material lengths')
        initial_shape_record.update(radius_m=.012, clearance_m=.004,
            edge_length_over_rest_min=float(ratios.min()), edge_length_over_rest_max=float(ratios.max()))
    initial_offset = np.asarray(args.clean_initial_offset, dtype=float)
    if not np.isfinite(initial_offset).all():
        raise ValueError('Invalid initial clearance offset')
    if np.any(initial_offset):
        if not np.any(cloth_ids == 1):
            raise ValueError('No clean cloth exists for the requested setup correction')
        initial[cloth_ids == 1] += initial_offset
    mesh_refinement = dict(applied=False, nodes_before=len(initial), faces_before=len(faces))
    if args.diagnostic_refine_mesh:
        from tablecloth_native_initial import refine_initial_mesh
        reference_initial = refine_initial_mesh(reference_initial, uv, faces, cloth_ids)[0]
        initial, uv, faces, cloth_ids = refine_initial_mesh(initial, uv, faces, cloth_ids)
        mesh_refinement.update(applied=True, applied_before_episode=True,
            nodes_after=len(initial), faces_after=len(faces),
            method='shared-edge midpoints; piecewise linear surface and material area unchanged')
    if args.commands:
        with np.load(args.commands, allow_pickle=False) as z:
            action = {k: z[k] for k in ('joint_target', 'joint_names', 'time_s')}
            phases = z['phase']
    fps = int(meta['fps'])
    count = len(action['joint_target'])
    if (action['joint_target'].shape != (count, len(action['joint_names']))
            or not np.isfinite(action['joint_target']).all()
            or len(phases) != count or len(action['time_s']) != count
            or not np.allclose(action['time_s'], np.arange(count)/fps, atol=1e-8, rtol=0)):
        raise ValueError('Reference command clock or joint array invalid')
    if not np.isfinite(initial).all():
        raise ValueError('Invalid initial cloth')
    # One anonymous working layer: never save or mutate the source House asset.
    source = Usd.Stage.Open(str(args.scene))
    stage = Usd.Stage.Open(source.Flatten())
    del source
    geometry_preparation = []
    if surface and not args.diagnostic_unbaked_table:
        from tablecloth_native_geometry import prepare_static_contact_mesh
        geometry_preparation.append(prepare_static_contact_mesh(stage))
    if args.diagnostic_bake_basket:
        if not surface:
            raise ValueError('Basket transform diagnostic requires FEM surface')
        from tablecloth_native_geometry import prepare_static_contact_mesh
        from tablecloth_basket_contract import BASKET_COLLISION_PATH
        geometry_preparation.append(prepare_static_contact_mesh(stage, BASKET_COLLISION_PATH))
    sdf_candidate = None
    if args.diagnostic_basket_sdf_resolution is not None:
        if not surface:
            raise ValueError('Basket SDF diagnostic requires FEM surface')
        from tablecloth_basket_contract import BASKET_COLLISION_PATH
        basket = stage.GetPrimAtPath(BASKET_COLLISION_PATH)
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(basket)
        previous = mesh_collision.GetApproximationAttr().Get()
        mesh_collision.CreateApproximationAttr('sdf')
        sdf = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(basket)
        sdf.CreateSdfResolutionAttr(args.diagnostic_basket_sdf_resolution)
        sdf.CreateSdfSubgridResolutionAttr(6)
        contact = PhysxSchema.PhysxCollisionAPI.Apply(basket)
        previous_contact = contact.GetContactOffsetAttr().Get()
        rest = contact.GetRestOffsetAttr().Get()
        contact.CreateContactOffsetAttr(.03)
        sdf_candidate = dict(path=BASKET_COLLISION_PATH, previous_approximation=previous,
            approximation='sdf', resolution=sdf.GetSdfResolutionAttr().Get(),
            subgrid_resolution=sdf.GetSdfSubgridResolutionAttr().Get(),
            previous_contact_offset_m=(float(previous_contact) if np.isfinite(previous_contact) else None),
            previous_contact_offset_automatic=not np.isfinite(previous_contact), contact_offset_m=.03,
            rest_offset_unchanged=contact.GetRestOffsetAttr().Get() == rest,
            source='delivered basket_sdf_contact60_probe.py',
            original_mesh_preserved=True, shared_asset_writes=False,
            diagnostic_only=True, common_scene_regression_pending=True)
    (args.output/'contact_geometry_preparation.json').write_text(json.dumps(
        dict(rows=geometry_preparation, diagnostic_unbaked_table=args.diagnostic_unbaked_table,
             shared_scene_modified=False), indent=2)+'\n')
    stage.RemovePrim('/World/Robot')
    robot = stage.DefinePrim('/World/Robot', 'Xform')
    robot.GetReferences().AddReference(str(args.robot))
    x, y, yaw = station_xyyaw(meta['benchmark_task_id'])
    root = np.eye(4)
    root[:3, :3] = Rotation.from_euler('z', yaw, degrees=True).as_matrix()
    root[:2, 3] = [x, y]
    UsdGeom.Xformable(robot).AddTransformOp().Set(Gf.Matrix4d(root.T.tolist()))
    physics = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if len(physics) != 1:
        raise ValueError('Expected one authoritative House physics scene')
    physics_path = physics[0].GetPath()
    px = PhysxSchema.PhysxSceneAPI.Apply(physics[0])
    source_surface_capacity = px.GetGpuMaxDeformableSurfaceContactsAttr().Get()
    if args.surface_contact_capacity is not None:
        if not surface:
            raise ValueError('Surface contact capacity requires native FEM surface')
        px.CreateGpuMaxDeformableSurfaceContactsAttr(args.surface_contact_capacity)
    surface_capacity_record = dict(source=source_surface_capacity,
        resolved=px.GetGpuMaxDeformableSurfaceContactsAttr().Get(),
        requested=args.surface_contact_capacity,
        source_of_readback='USD scene attribute before reset, not GPU buffer occupancy',
        fixed_for_entire_episode=True)
    source_frequency = int(px.GetTimeStepsPerSecondAttr().Get())
    timestep_override_requested = args.substeps is not None
    if args.substeps is None:
        if source_frequency <= 0 or source_frequency % fps:
            raise ValueError('House physics frequency must be an integer multiple of the action rate')
        args.substeps = source_frequency // fps
    if args.observe_substeps > args.substeps:
        raise ValueError('Substep observations exceed resolved House steps per action')
    px.CreateEnableGPUDynamicsAttr(True)
    px.CreateBroadphaseTypeAttr('GPU')
    # Native robot gains, limits, mass, friction and cameras are never overridden.
    native = {str(a.GetPath()): str(a.Get()) for p in Usd.PrimRange(robot)
              for a in p.GetAttributes() if a.GetName().startswith(('physics:', 'physx', 'drive:'))
              and not any(k in a.GetName() for k in ('targetPosition', 'targetVelocity', 'velocity', 'angularVelocity'))}
    articulation_roots = [p for p in Usd.PrimRange(robot) if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if len(articulation_roots) != 1:
        raise ValueError('Expected one native robot articulation')
    self_collision = PhysxSchema.PhysxArticulationAPI(articulation_roots[0]).GetEnabledSelfCollisionsAttr().Get()
    if self_collision is not True:
        raise ValueError('Native robot self collision must be enabled')
    # Reporting is observational; leave collision geometry, filters and response intact.
    for prim in Usd.PrimRange(robot):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
    from omni.physx import get_physx_simulation_interface
    from pxr import PhysicsSchemaTools
    contact_rows = []
    current_frame = [-1]
    def on_contact(headers, data):
        for header in headers:
            if not header.num_contact_data:
                continue
            points = data[header.contact_data_offset:header.contact_data_offset + header.num_contact_data]
            contact_rows.append(dict(frame=current_frame[0],
                collider0=str(PhysicsSchemaTools.intToSdfPath(header.collider0)),
                collider1=str(PhysicsSchemaTools.intToSdfPath(header.collider1)),
                impulse_norm_sum=float(sum(np.linalg.norm(p.impulse) for p in points)),
                min_separation=float(min(p.separation for p in points))))
    contact_subscription = get_physx_simulation_interface().subscribe_contact_report_events(on_contact)
    collision_repairs = enable_required_contacts(stage)
    collision_audit = audit_required_contacts(stage)
    (args.output/'collision_inventory.json').write_text(json.dumps(collision_audit,indent=2))
    existing_cloth = [p for p in stage.Traverse() if p.HasAPI(PhysxSchema.PhysxParticleClothAPI)]
    if existing_cloth:
        raise ValueError('House already contains cloth; refuse duplicate or silently replaced state')
    ps_path = Sdf.Path('/World/TableclothPhysics/Particles')
    radius = float(meta['config']['radius'])
    particleUtils.add_physx_particle_system(stage, ps_path, simulation_owner=physics_path,
        contact_offset=radius*1.5, rest_offset=radius, particle_contact_offset=radius*1.5,
        solid_rest_offset=radius, solver_position_iterations=args.iterations,
        global_self_collision_enabled=True, non_particle_collision_enabled=True)
    PhysxSchema.PhysxParticleSystem(stage.GetPrimAtPath(ps_path)).CreateEnableCCDAttr(True)
    material_path = '/World/TableclothPhysics/Material'
    particleUtils.add_pbd_particle_material(stage, material_path,
        friction=float(meta['config']['friction']), damping=args.velocity_damping)
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(ps_path)).Bind(
        UsdShade.Material(stage.GetPrimAtPath(material_path)), materialPurpose='physics')
    meshes, particle_counts, masses, materials, nodal_mass_estimates = [], [], [], [], []
    for cid in np.unique(cloth_ids):
        selected = np.flatnonzero(cloth_ids == cid)
        remap = np.full(len(initial), -1, dtype=int)
        remap[selected] = np.arange(len(selected))
        local_faces = remap[faces[np.all(cloth_ids[faces] == cid, axis=1)]]
        local_uv = uv[selected]
        path = f'/World/TableclothPhysics/Cloth_{cid}'
        mass = float(meta['config']['density'] * np.prod(np.ptp(local_uv, axis=0)))
        if surface:
            args.surface_friction = float(meta['config']['friction'])
            node_mass, material = create_surface(stage, path, initial[selected], local_uv,
                local_faces, mass, radius, args)
            nodal_mass_estimates.append(node_mass)
            materials.append(material)
            meshes.append(path); particle_counts.append(len(selected)); masses.append(mass)
            continue
        if args.material_model == 'woven':
            springs, stiffness, damping, rest, material = woven_springs(local_uv,
                args.stretch, args.shear, args.bend, args.damping)
        else:
            springs, stiffness, damping, rest = material_springs(local_uv, local_faces,
                args.stretch, args.bend, args.damping)
            material = dict(model='legacy_triangle', calibrated=False,
                known_defect='Triangulation-dependent shear anisotropy; Newton bend copied into distance springs')
        materials.append(material)
        path = f'/World/TableclothPhysics/Cloth_{cid}'
        flat = np.column_stack((local_uv, np.zeros(len(selected))))
        particleUtils.add_physx_particle_cloth_with_constraints(stage, path, ps_path,
            Vt.Vec3fArray.FromNumpy(initial[selected].astype(np.float32)), [],
            Vt.Vec3fArray.FromNumpy(flat.astype(np.float32)), Vt.Vec3fArray.FromNumpy(np.zeros_like(flat, dtype=np.float32)),
            Vt.IntArray.FromNumpy(local_faces.reshape(-1).astype(np.int32)), Vt.Vec2iArray.FromNumpy(springs),
            stiffness.tolist(), damping.tolist(), rest.tolist(),
            self_collision=True, self_collision_filter=True, particle_group=int(cid))
        mass = float(meta['config']['density'] * np.prod(np.ptp(local_uv, axis=0)))
        if mass <= 0:
            raise ValueError('Cloth mass must be positive')
        UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(path)).CreateMassAttr(mass)
        meshes.append(path); particle_counts.append(len(selected)); masses.append(mass)
    # Only initialize joint state before the episode starts.
    command_names = action['joint_names'].tolist()
    driven = []
    for p in Usd.PrimRange(robot):
        if not p.IsA(UsdPhysics.Joint) or p.GetName() not in command_names:
            continue
        kind = 'linear' if p.IsA(UsdPhysics.PrismaticJoint) else 'angular'
        value = float(action['joint_target'][0, command_names.index(p.GetName())])
        value *= 1 if kind == 'linear' else 180/np.pi
        p.AddAppliedSchema('PhysicsJointStateAPI:'+kind)
        p.CreateAttribute(f'state:{kind}:physics:position', Sdf.ValueTypeNames.Float).Set(value)
        p.CreateAttribute(f'state:{kind}:physics:velocity', Sdf.ValueTypeNames.Float).Set(0.)
        target = p.GetAttribute(f'drive:{kind}:physics:targetPosition')
        if target and target.Get() is not None:
            target.Set(value)
            driven.append(p.GetName())
    if not driven:
        raise ValueError('Native robot has no commandable drives')
    context = omni.usd.get_context()
    if not context.attach_stage_with_callback(UsdUtils.StageCache.Get().Insert(stage).ToLongInt()):
        raise RuntimeError('Could not attach the anonymous House stage')
    while context.get_stage_loading_status()[2]:
        app.update()
    sim = SimulationContext(physics_dt=1/(fps*args.substeps), rendering_dt=1/fps,
                            physics_prim_path=str(physics_path), device='cuda:0', backend='torch')
    sim.get_physics_context().enable_gpu_dynamics(True)
    cooking_preload = dict(requested=args.diagnostic_preload_collisions)
    if args.diagnostic_preload_collisions:
        from omni.physx import get_physx_interface
        before = float(sim.current_time)
        get_physx_interface().force_load_physics_from_usd()
        # This installed PhysX version removed the legacy cooking wait/count
        # methods. Verify a real native shape after the blocking physics load.
        if sdf_candidate is not None:
            early_sim = omni.physics.tensors.create_simulation_view('torch')
            early_sim.set_subspace_roots('/')
            early_sdf = early_sim.create_sdf_shape_view(sdf_candidate['path'], 1)
            cooking_preload['native_sdf_shape_count_before_reset'] = early_sdf.count
            if early_sdf.count != 1:
                raise RuntimeError('Basket SDF unavailable before reset')
            del early_sdf, early_sim
        cooking_preload.update(simulation_time_before_s=before,
                               simulation_time_after_s=float(sim.current_time))
        if float(sim.current_time) != before:
            raise RuntimeError('Collision preload unexpectedly advanced physics')
        (args.output/'collision_preload.json').write_text(json.dumps(cooking_preload, indent=2)+'\n')
    sim.reset()
    tensor_sim = omni.physics.tensors.create_simulation_view('torch')
    tensor_sim.set_subspace_roots('/')
    if sdf_candidate is not None:
        sdf_view = tensor_sim.create_sdf_shape_view(sdf_candidate['path'], 1)
        sdf_candidate['native_sdf_shape_count'] = sdf_view.count
        (args.output/'basket_sdf_candidate.json').write_text(json.dumps(sdf_candidate, indent=2)+'\n')
        if sdf_view.count != 1:
            raise RuntimeError('Requested basket SDF was not present in the native physics backend')
        if args.diagnostic_sdf_witnesses:
            with np.load(args.diagnostic_sdf_witnesses, allow_pickle=False) as witness:
                world = witness['world_points']
                labels = witness['labels']
            if (world.ndim != 2 or world.shape[1] != 3 or not len(world)
                    or not np.isfinite(world).all() or labels.shape != (len(world),)):
                raise ValueError('Invalid read-only SDF witness geometry')
            transform = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(
                stage.GetPrimAtPath(sdf_candidate['path'])))
            local = (np.c_[world, np.ones(len(world))] @ np.linalg.inv(transform))[:, :3]
            witness_view = tensor_sim.create_sdf_shape_view(sdf_candidate['path'], len(world))
            values = {}
            for label, points in [('world', world), ('mesh_local', local)]:
                query = torch.as_tensor(points[None], dtype=torch.float32, device='cuda:0')
                values[label+'_raw_sdf_and_gradients'] = witness_view.get_sdf_and_gradients(
                    query).detach().cpu().numpy().copy()
            np.savez_compressed(args.output/'basket_sdf_witness.npz', world_points=world,
                mesh_local_points=local, mesh_world_transform=transform, labels=labels,
                query_only_not_simulation_state=True, **values)
    if not px.GetEnableGPUDynamicsAttr().Get():
        raise RuntimeError('Particle cloth requires GPU dynamics on its actual simulation owner')
    art = SingleArticulation('/World/Robot')
    art.initialize(physics_sim_view=tensor_sim)
    names = list(art.dof_names)
    if any(n not in names for n in driven):
        raise ValueError('Native drive missing from articulation')
    indices = np.array([names.index(n) for n in driven])
    columns = [command_names.index(n) for n in driven]
    # reset already advances physics. Preserve that physical state, including
    # passive motion; never restore targets by writing articulation state.
    initial_joint_position = art.get_joint_positions().detach().cpu().numpy().copy()
    initial_joint_velocity = art.get_joint_velocities().detach().cpu().numpy().copy()
    if not np.isfinite(initial_joint_position).all() or not np.isfinite(initial_joint_velocity).all():
        raise RuntimeError('Nonfinite native initialization readback')
    initial_readback = dict(joint_names=names,
        actual_joint_position=initial_joint_position.tolist(),
        actual_joint_velocity=initial_joint_velocity.tolist(),
        command_error_max_native_units=float(np.abs(initial_joint_position[indices]
            - action['joint_target'][0, columns]).max()),
        initialization_elapsed_s=float(sim.current_time), post_reset_joint_state_writes=0,
        note='Measured after reset warmup; natural dynamics are not overwritten to match commands')
    (args.output/'native_initial_joint_readback.json').write_text(json.dumps(initial_readback, indent=2)+'\n')
    # GPU particle state is not necessarily mirrored into USD. Read the solver
    # tensors directly rather than mistaking unchanged authored mesh for state.
    contact_sensors = [str(p.GetPath()) for p in Usd.PrimRange(robot)
                       if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    rigid_contact_view = tensor_sim.create_rigid_contact_view(contact_sensors)
    contact_forces = []
    cloth_views = ([SurfaceReadback(tensor_sim.create_surface_deformable_body_view(path), n)
                    for path, n in zip(meshes, particle_counts)] if surface else
                   [tensor_sim.create_particle_cloth_view(path) for path in meshes])
    particle_masses = []
    for cloth_index, (view, expected) in enumerate(zip(cloth_views, particle_counts)):
        if view.count != 1 or view.max_particles_per_cloth != expected:
            raise RuntimeError('Native solver cloth handle/count invalid')
        mass = (nodal_mass_estimates[cloth_index] if surface else
                view.get_masses().detach().cpu().numpy().reshape(-1).copy())
        if len(mass) != expected or not np.isfinite(mass).all() or np.any(mass <= 0):
            raise RuntimeError('Native cloth has invalid or fixed particle masses')
        particle_masses.append(mass)
    solver_initial_cloth = np.concatenate([view.get_positions().detach().cpu().numpy().reshape(-1,3).copy()
                                           for view in cloth_views])
    solver_initial_velocity = np.concatenate([view.get_velocities().detach().cpu().numpy().reshape(-1,3).copy()
                                              for view in cloth_views])
    solver_warmup_time = float(sim.current_time)
    articulation_view = art._articulation_view
    body_names = list(articulation_view.body_names)
    body_records = []
    root_records = []
    record, failure = [], None
    substep_positions, substep_velocities, substep_times = [], [], []
    try:
        limit = min(count, args.max_frames) if args.max_frames else count
        for frame in range(limit):
            current_frame[0] = frame
            targets = action['joint_target'][frame, columns]
            frame_force = None
            for substep in range(args.substeps):
                art.apply_action(ArticulationAction(
                    joint_positions=torch.as_tensor(targets, dtype=torch.float32, device='cuda:0'),
                    joint_indices=torch.as_tensor(indices, dtype=torch.int64, device='cuda:0')))
                sim.step(render=False)
                if args.observe_all_substeps or (frame == limit-1 and substep >= args.substeps-args.observe_substeps):
                    substep_positions.append(np.concatenate([
                        view.get_positions().detach().cpu().numpy().reshape(-1,3).copy() for view in cloth_views]))
                    substep_velocities.append(np.concatenate([
                        view.get_velocities().detach().cpu().numpy().reshape(-1,3).copy() for view in cloth_views]))
                    substep_times.append(frame/fps+(substep+1)/(fps*args.substeps))
                force = rigid_contact_view.get_net_contact_forces(1/(fps*args.substeps)).detach().cpu().numpy().copy()
                if frame_force is None:
                    frame_force = force
                else:
                    replace = np.linalg.norm(force, axis=-1) > np.linalg.norm(frame_force, axis=-1)
                    frame_force[replace] = force[replace]
            chunks = [view.get_positions().detach().cpu().numpy().reshape(-1,3).copy()
                      for view in cloth_views]
            if [len(q) for q in chunks] != particle_counts:
                raise RuntimeError('Particle readback count changed')
            cloth = np.concatenate(chunks)
            velocities = np.concatenate([view.get_velocities().detach().cpu().numpy().reshape(-1,3).copy()
                                         for view in cloth_views])
            for view, initial_mass in zip(cloth_views, particle_masses):
                if not surface and not np.array_equal(view.get_masses().detach().cpu().numpy().reshape(-1), initial_mass):
                    raise RuntimeError('Native particle masses changed during episode')
            joints = art.get_joint_positions().detach().cpu().numpy().copy()
            if not np.isfinite(cloth).all() or not np.isfinite(joints).all():
                raise RuntimeError('Nonfinite physical state')
            body_records.append(articulation_view._physics_view.get_link_transforms().detach().cpu().numpy()[0].copy())
            root_records.append(articulation_view._physics_view.get_root_transforms().detach().cpu().numpy()[0].copy())
            contact_forces.append(frame_force)
            record.append((cloth.copy(), joints, art.get_joint_velocities().detach().cpu().numpy().copy(), velocities.copy()))
            if frame % 20 == 0:
                print(json.dumps({'frame': frame, 'cloth_min_z': float(cloth[:,2].min()),
                    'joint_tracking_error_max': float(np.abs(joints[indices]-targets).max())}), flush=True)
        if record and np.max(np.linalg.norm(record[-1][0] - initial, axis=1)) < 1e-6:
            raise RuntimeError('Cloth did not respond physically; particle simulation/readback may be inactive')
        if limit < count:
            failure = 'Explicit diagnostic frame limit; incomplete episode'
    except BaseException as exc:
        failure = repr(exc)
        raise
    finally:
        if substep_positions:
            np.savez_compressed(args.output/'substep_observations.npz',
                cloth_q=np.stack(substep_positions), cloth_qd=np.stack(substep_velocities),
                time_s=np.asarray(substep_times), cloth_id=cloth_ids,
                physical_time_s=np.asarray(substep_times)+solver_warmup_time,
                all_post_reset_steps_recorded=args.observe_all_substeps)
        (args.output/'rigid_contacts.json').write_text(json.dumps(contact_rows, indent=2)+'\n')
        if record:
            (args.output/'material_shape.json').write_text(json.dumps(curvature_metrics(
                np.stack([r[0] for r in record]), uv, faces, cloth_ids), indent=2)+'\n')
            np.savez_compressed(args.output/'full_state.npz', cloth_q=np.stack([r[0] for r in record]),
                cloth_qd=np.stack([r[3] for r in record]),
                joint_position=np.stack([r[1] for r in record]), joint_velocity=np.stack([r[2] for r in record]),
                joint_names=names, faces=faces, uv=uv, cloth_id=cloth_ids, initial_cloth_q=initial,
                reference_initial_cloth_q=reference_initial,
                solver_initial_cloth_q=solver_initial_cloth, solver_initial_cloth_qd=solver_initial_velocity,
                solver_initial_joint_position=initial_joint_position,
                solver_initial_joint_velocity=initial_joint_velocity,
                particle_mass=np.concatenate(particle_masses),
                body_q=np.stack(body_records), body_names=body_names, robot_root_pose_xyzw=np.stack(root_records),
                rigid_contact_force_peak=np.stack(contact_forces), rigid_contact_sensor_paths=contact_sensors,
                benchmark_task_id=meta["benchmark_task_id"],
                time_s=(np.arange(len(record))+1)/fps, phase=phases[:len(record)])
            np.savez_compressed(args.output/'action.npz',
                joint_target=action['joint_target'][:len(record)], joint_names=action['joint_names'],
                time_s=np.arange(len(record))/fps, fps=fps,
                control_mode='native_joint_drive_targets', command_joint_names=driven)
        now = {str(a.GetPath()): str(a.Get()) for p in Usd.PrimRange(robot)
               for a in p.GetAttributes() if str(a.GetPath()) in native}
        report = dict(source=start_source, reference=str(args.reference), frames=len(record),
            git_source=git_source,
            robot_asset=robot_asset,
            benchmark_task_id=meta['benchmark_task_id'], fps=fps, scenario_hidden_prims=[],
            collision_inventory='collision_inventory.json', robot_self_collision_enabled=self_collision,
            rigid_contacts='rigid_contacts.json',
            rigid_force_signal_observed=bool(contact_forces and np.any(np.asarray(contact_forces) != 0)),
            rigid_force_signal_is_grasp_validation=False,
            initial_collision_filter_repairs=collision_repairs,
            contact_geometry_preparation=geometry_preparation,
            basket_sdf_candidate=sdf_candidate,
            collision_preload=cooking_preload,
            mesh_refinement=mesh_refinement,
            surface_contact_capacity=surface_capacity_record,
            initial_joint_readback='native_initial_joint_readback.json',
            diagnostic_only=True, eligible_for_production=False,
            solver='PhysX FEM surface and native articulation' if surface else 'PhysX particle cloth and native articulation',
            cloth_solver_exception='GPU FEM surface with flat rest shape; calibration pending' if surface else
                'GPU particle cloth with explicit flat-rest springs; material calibration pending',
            particle_mass_source='authored_total_and_rest_area_estimates' if surface else 'native_GPU_tensors',
            material_models=materials,
            initial_setup_correction=dict(cloth_id=1, translation_m=initial_offset.tolist(),
                shape=initial_shape_record, applied_before_episode=True, randomized=False),
            dirty_initial_setup_correction=dict(cloth_id=0, translation_m=dirty_initial_offset.tolist(),
                applied_before_episode=True, randomized=False),
            robot_parameters_unchanged=now == native, attachment_count=0,
            physics_clock=dict(source_hz=source_frequency, actual_hz=fps*args.substeps,
                matches_source=fps*args.substeps == source_frequency,
                diagnostic_override_requested=timestep_override_requested,
                initialization_elapsed_s=solver_warmup_time),
            post_start_pose_or_velocity_writes=False, cloth_masses_kg=masses,
            particle_mass_min=float(np.concatenate(particle_masses).min()), particle_state_readback="native GPU tensors",
            driven_joint_names=driven, failure=failure, owner_approved=False,
            command_interval='reference frame k command applied over [k/fps, (k+1)/fps]',
            config={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()})
        (args.output/'trial.json').write_text(json.dumps(report,indent=2)+'\n')
        sim.stop()


if __name__ == '__main__':
    main()
