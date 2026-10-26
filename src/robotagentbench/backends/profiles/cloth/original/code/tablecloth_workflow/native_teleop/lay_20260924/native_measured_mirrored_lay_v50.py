"""Diagnostic command replay with shared fixed-base robot and PhysX cloth.

No attachments, kinematic robot, state replay or post-start pose writes. Material
conversion is experimental and must be calibrated before any production use.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path('/home/aether/workspace/agent_bench')
NATIVE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NATIVE_DIR))
RUN_ROOT = NATIVE_DIR.parent


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
    parser.add_argument('--headless-smoke', action='store_true', help='Hold-only solver smoke test; no keyboard or scripted task motion')
    parser.add_argument('--headless-command-replay', action='store_true',
                        help='Diagnostic native-drive replay with shared fixed-base dynamic articulation')
    parser.add_argument('--probe-grasp-after-stretch', action='store_true',
                        help='Continue a rejected replay to observe gripper contact; never eligible as task data')
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--robot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--diagnostic-canonical', action='store_true', required=True,
                        help='Explicitly exclude this material/action diagnostic from randomized production')
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--stretch', type=float, default=5000.)
    parser.add_argument('--bend', type=float, default=100., help='Distance-spring stiffness in N/m, not Newton dihedral stiffness')
    parser.add_argument('--damping', type=float, default=1.)
    parser.add_argument('--velocity-damping', type=float, default=.8,
                        help='Native PBD material global damping; separate from spring damping')
    parser.add_argument('--cloth-friction', type=float,
                        help='Diagnostic PhysX cloth dynamic friction override; recorded, not a Newton coefficient conversion')
    parser.add_argument('--material-model', choices=('legacy_triangle', 'woven', 'fem_surface'), default='woven')
    parser.add_argument('--shear', type=float, default=5000.)
    parser.add_argument('--youngs-modulus', type=float, default=1.3333333e7)
    parser.add_argument('--surface-thickness', type=float, default=.001)
    parser.add_argument('--surface-bend-modulus', type=float, default=1.22e6)
    parser.add_argument('--clean-surface-bend-modulus', type=float, default=None, help='Explicit per-asset diagnostic candidate, fixed before episode')
    parser.add_argument('--elasticity-damping', type=float, default=.01)
    parser.add_argument('--bend-damping', type=float, default=.01)
    parser.add_argument('--surface-contact-offset', type=float,
                        help='FEM contact generation distance; physical rest thickness stays unchanged')
    parser.add_argument('--diagnostic-convex-table-collider-candidate', action='store_true',
                        help='Pre-episode shared House collision candidate; never production or task-only override')
    parser.add_argument('--diagnostic-decomposed-basket-collider-candidate', action='store_true',
                        help='Pre-episode shared basket collision candidate; preserve open container geometry')
    parser.add_argument('--diagnostic-basket-box-collider-candidate', action='store_true',
                        help='Add measured basket floor and four wall colliders before reset')
    parser.add_argument('--diagnostic-basket-box-replace-source', action='store_true',
                        help='Disable only the superseded basket mesh after a passing canonical collision audit; verify all five replacement colliders')
    parser.add_argument('--observe-substeps', type=int, default=0,
                        help='Read-only samples of the last frame for diagnosing high-frequency vibration')
    parser.add_argument('--clean-rest-fold-deg', type=float, default=0.)
    parser.add_argument('--clean-initial-shape', choices=('reference', 'rimfit_isometric', 'rounded', 'isometric', 'photo_lean', 'photo_wall', 'right_rim_drape', 'right_rim_drape_high', 'right_rim_drape_front', 'front_staggered_corner_drape'), default='reference',
                        help='Before-episode basket fold geometry; rounded preserves material arc length')
    parser.add_argument('--clean-initial-offset', type=float, nargs=3, default=(0., 0., 0.),
                        help='Explicit fixed setup repair in world metres, before episode only; never a runtime pose write')
    parser.add_argument('--clean-diagonal-rotate-deg', type=float, default=0.,
                        help='Diagnostic rigid pre-episode rotation of the folded clean sheet about basket vertical')
    parser.add_argument('--clean-diagonal-center-xy', type=float, nargs=2,
                        help='Diagnostic pre-episode center of the rotated clean sheet, in world metres')
    parser.add_argument('--clean-tilt-x-rad', type=float, default=0.,
                        help='Diagnostic pre-episode rigid tilt about basket horizontal X')
    parser.add_argument('--clean-tilt-y-rad', type=float, default=0.,
                        help='Diagnostic pre-episode rigid tilt about basket horizontal Y')
    parser.add_argument('--adaptive-clean-corner-uv', type=float, nargs=2,
                        help='Observe the settled material corner and shift only subsequent right-arm joint commands')
    parser.add_argument('--adaptive-clean-corner-nominal-xyz', type=float, nargs=3,
                        help='World corner position used by the precomputed action; no cloth state write')
    parser.add_argument('--adaptive-clean-corner-settle-frame', type=int, default=400)
    parser.add_argument('--adaptive-clean-corner-arm-start-frame', type=int, default=580)
    parser.add_argument('--clean-rim-overhang-extra', type=float, default=0.,
                        help='Diagnostic enlargement of physical rim oval, before episode only; requires isometric rim fold')
    parser.add_argument('--dirty-initial-offset', type=float, nargs=3, default=(0., 0., 0.),
                        help='Explicit pre-episode setup diagnostic, recorded against original geometry')
    parser.add_argument('--raw-state-storage',type=Path,default=None,help='New native output only, never a checkpoint input')
    parser.add_argument('--surface-self-filter-distance',type=float,default=None,help='Explicit cloth-only self-contact diagnostic; fixed before episode')
    parser.add_argument('--surface-max-linear-velocity',type=float,default=None,help='Native cloth velocity solver bound, negative for supported automatic mode')
    parser.add_argument('--iterations', type=int, default=64)
    parser.add_argument('--commands', type=Path,
                        help='Replacement joint commands only; never a replacement physical state')
    parser.add_argument('--adaptive-repeat-brush', action='store_true',
                        help='Diagnostic: repeat open-finger brush until live corner overhang is sufficient, then retarget the physical pinch')
    parser.add_argument('--adaptive-follow-brush-corner', action='store_true', help='Retarget repeat brush along table edge using live corner displacement')
    parser.add_argument('--brush-target-overhang-m', type=float, default=.04,
                        help='Measured table-local corner overhang required before pinch in adaptive diagnostic')
    parser.add_argument('--adaptive-corner-free-inplane-roll', action='store_true')
    parser.add_argument('--adaptive-clean-corner-nominal-normal', type=float, nargs=3)
    parser.add_argument('--natural-arm-bend-limit-deg', type=float, default=None)
    parser.add_argument('--diagnostic-grasp-uv', type=float, nargs=2, default=None)
    parser.add_argument('--surface-rest-offset', type=float, default=None)
    parser.add_argument('--surface-collision-multiplier', type=int, default=1)
    parser.add_argument('--surface-pair-updates', type=int, default=1)
    parser.add_argument('--force-api-probe',action='store_true')
    args = parser.parse_args()
    if not 1 <= args.iterations <= 255:
        parser.error('Native FEM surface iteration range is 1..255; do not author unsupported values')
    if args.headless_smoke and args.headless_command_replay:
        parser.error('Choose hold-only or command replay')
    if args.headless_command_replay != bool(args.commands):
        parser.error('--headless-command-replay requires --commands and vice versa')
    if args.probe_grasp_after_stretch and not args.headless_command_replay:
        parser.error('Stretch-failure grasp probe requires command replay')
    if args.adaptive_repeat_brush and not args.headless_command_replay:
        parser.error('Adaptive brush requires native command replay')
    if args.diagnostic_convex_table_collider_candidate and args.material_model != 'fem_surface':
        parser.error('Convex table collision diagnostic is only meaningful for native FEM surface')
    if args.diagnostic_decomposed_basket_collider_candidate and args.material_model != 'fem_surface':
        parser.error('Basket collision diagnostic is only meaningful for native FEM surface')
    if args.diagnostic_basket_box_collider_candidate and args.material_model != 'fem_surface':
        parser.error('Basket box collision diagnostic is only meaningful for native FEM surface')
    if args.diagnostic_basket_box_collider_candidate and args.diagnostic_decomposed_basket_collider_candidate:
        parser.error('Choose only one basket collider candidate')
    if args.diagnostic_basket_box_replace_source and not args.diagnostic_basket_box_collider_candidate:
        parser.error('Basket source replacement requires the five physical box colliders')
    if args.clean_rim_overhang_extra and args.clean_initial_shape != 'rimfit_isometric':
        parser.error('Rim oval adjustment requires --clean-initial-shape rimfit_isometric')
    if not 0 <= args.clean_rim_overhang_extra <= .015:
        parser.error('Rim oval adjustment must be in [0, 0.015] metres')
    coefficients = [args.stretch, args.shear, args.bend, args.damping, args.velocity_damping,
                    args.elasticity_damping, args.bend_damping]
    if args.iterations < 1 or not np.isfinite(coefficients).all() or min(coefficients) < 0 or (args.cloth_friction is not None and (not np.isfinite(args.cloth_friction) or args.cloth_friction < 0)):
        parser.error('Invalid solver configuration')
    if args.observe_substeps < 0:
        parser.error('Substep observation count must be nonnegative')
    args.output.mkdir(parents=True, exist_ok=False)
    if args.raw_state_storage is not None:
        assert args.raw_state_storage.is_absolute() and not args.raw_state_storage.exists() and args.raw_state_storage.parent.is_dir()
        (args.output/'full_state.npz').symlink_to(args.raw_state_storage)
        (args.output/'raw_state_storage.json').write_text(json.dumps(dict(kind='newly generated native output, not a checkpoint input',path=str(args.raw_state_storage),episode_state_restores=0),indent=2))
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': args.headless_smoke or args.headless_command_replay, 'width':1280, 'height':800})
    try:
        run(args, app)
    except BaseException:
        import traceback
        failure = traceback.format_exc()
        print(failure, file=sys.stderr, flush=True)
        (args.output/'startup_or_runtime_failure.json').write_text(
            json.dumps({'failure': failure, 'diagnostic_only': True, 'eligible_for_production': False}, indent=2))
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
    surface = args.material_model == 'fem_surface'
    if surface:
        from omni.physx.bindings._physx import SETTING_ENABLE_DEFORMABLE_BETA
        carb.settings.get_settings().set_bool(SETTING_ENABLE_DEFORMABLE_BETA, True)
        from native_surface_contact_candidate_v1 import create_surface, SurfaceReadback
        if min(args.youngs_modulus, args.surface_thickness, args.surface_bend_modulus) <= 0:
            raise ValueError('Surface material parameters must be positive')
        args.surface_friction = None  # resolved from the same owner cloth config below

    meta = json.loads((args.reference/'trial.json').read_text())
    cloth_friction = (float(meta['config']['friction']) if args.cloth_friction is None
                      else args.cloth_friction)
    with np.load(args.reference/'action.npz') as z:
        action = {k: z[k] for k in ('joint_target', 'joint_names', 'time_s')}
    with np.load(args.reference/'full_state.npz') as z:
        initial = z['initial_cloth_q'].copy()
        faces, uv = z['faces'], z['uv']
        cloth_ids = z['cloth_id'] if 'cloth_id' in z.files else np.zeros(len(initial), dtype=int)
        phases = z['phase']
        if 'grasp_reference_corner' in z: action['grasp_reference_corner'] = z['grasp_reference_corner']
        if 'right_grasp_reference_corner' in z: action['right_grasp_reference_corner'] = z['right_grasp_reference_corner']
    # Refine the cloth simulation mesh before episode initialization. Mass, UV
    # area, material, robot clock and collision gates remain unchanged.
    mesh_edges, edge_index = np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0,return_inverse=True)
    assert np.all(cloth_ids[mesh_edges[:,0]]==cloth_ids[mesh_edges[:,1]])
    old_node_count=len(initial); face_count=len(faces)
    ab,bc,ca=(edge_index.reshape(3,face_count)+old_node_count)
    initial=np.concatenate([initial,initial[mesh_edges].mean(axis=1)])
    uv=np.concatenate([uv,uv[mesh_edges].mean(axis=1)])
    cloth_ids=np.concatenate([cloth_ids,cloth_ids[mesh_edges[:,0]]])
    faces=np.concatenate([np.stack([faces[:,0],ab,ca],axis=1),np.stack([ab,faces[:,1],bc],axis=1),np.stack([ca,bc,faces[:,2]],axis=1),np.stack([ab,bc,ca],axis=1)])
    print(json.dumps(dict(mesh_refinement=dict(before=old_node_count,after=len(initial),faces=len(faces),applied_before_episode=True))),flush=True)
    # Group refined nodes exactly as the native cloth tensor views are concatenated.
    order=np.argsort(cloth_ids,kind='stable');inverse=np.empty(len(order),dtype=int);inverse[order]=np.arange(len(order))
    initial=initial[order];uv=uv[order];cloth_ids=cloth_ids[order];faces=inverse[faces]
    assert np.all(np.diff(cloth_ids)>=0)
    reference_initial = initial.copy()
    import importlib.util
    spec=importlib.util.spec_from_file_location('approved_fold', RUN_ROOT/'protocol_spread_rimfit/fold_initial.py')
    fold_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(fold_module)
    from tablecloth_scene_binding import load_scene_collisions
    geometry=np.load(meta['config']['geometry'])
    basket=next(v for e,v,f in load_scene_collisions(geometry) if e['group']=='basket' and 'riser' not in e['prim_path'])
    basket_bounds=np.array([basket.min(0),basket.max(0)])
    clean=cloth_ids==1
    if np.any(clean):
        if args.clean_rim_overhang_extra:
            from rimfit_initial import long_midpoint_fold
            initial[clean], fold_record = long_midpoint_fold(uv[clean],np.ptp(uv[clean],axis=0),
                basket_bounds,float(meta['config']['radius']),args.clean_rim_overhang_extra)
        else:
            initial[clean], fold_record=fold_module.long_midpoint_fold(uv[clean],np.ptp(uv[clean],axis=0),basket_bounds,float(meta['config']['radius']))
    rimfit_record = None
    if args.clean_initial_shape in ('photo_lean', 'photo_wall'):
        from rimfit_initial import photo_basket_fold, photo_wall_fold, isometric_rows
        initial[clean], photo_record = (photo_wall_fold if args.clean_initial_shape == 'photo_wall' else photo_basket_fold)(uv[clean], basket_bounds,
                                                         float(meta['config']['radius']))
        initial[clean], correction = isometric_rows(initial[clean], uv[clean],
                                                    float(np.ptp(uv[clean], axis=0)[1])/2)
        rimfit_record = dict(photo_record, **correction)
    if args.clean_initial_shape in ('right_rim_drape', 'right_rim_drape_high', 'right_rim_drape_front'):
        from right_rim_drape import right_rim_drape
        initial[clean], rimfit_record = right_rim_drape(uv[clean], high=args.clean_initial_shape == 'right_rim_drape_high', front_fold=args.clean_initial_shape == 'right_rim_drape_front')
    if args.clean_initial_shape == 'front_staggered_corner_drape':
        from front_folded_bridge_v2 import front_staggered_corner_drape
        initial[clean], rimfit_record = front_staggered_corner_drape(uv[clean])
    if args.clean_initial_shape == 'rimfit_isometric':
        from rimfit_initial import isometric_rows
        initial[clean], rimfit_record = isometric_rows(initial[clean], uv[clean],
                                                        float(np.ptp(uv[clean], axis=0)[1])/2)
    fixed=json.loads((RUN_ROOT/'protocol_v14/config.json').read_text())['joint_targets']
    first=action['joint_target'][0].copy()
    for j,n in enumerate(action['joint_names']):
        if n in fixed:first[j]=fixed[n]
        if str(n).endswith('gripper_l_outer_joint1') or str(n).endswith('gripper_r_outer_joint1'):first[j]=.75
    action['joint_target']=np.tile(first,(args.max_frames or 6000,1))
    action['time_s']=np.arange(len(action['joint_target']))/int(meta['fps'])
    phases=np.full(len(action['joint_target']),'manual_native_teleop')
    dirty_initial_offset = np.asarray(args.dirty_initial_offset, dtype=float)
    if not np.isfinite(dirty_initial_offset).all():
        raise ValueError('Invalid dirty-cloth initial offset')
    initial[cloth_ids == 0] += dirty_initial_offset
    initial_shape_record = dict(model=args.clean_initial_shape)
    if rimfit_record is not None:
        from tablecloth_native_initial import initial_edge_ratios
        clean_indices = np.flatnonzero(clean)
        remap = np.full(len(initial), -1, dtype=int)
        remap[clean_indices] = np.arange(len(clean_indices))
        local_faces = remap[faces[np.all(cloth_ids[faces] == 1, axis=1)]]
        ratios = initial_edge_ratios(initial[clean], uv[clean], local_faces)
        if ratios.min() < (.94 if args.clean_initial_shape == 'right_rim_drape_high' else .96 if args.clean_initial_shape in ('right_rim_drape','right_rim_drape_front','front_staggered_corner_drape') else .99999) or ratios.max() > (1.05 if args.clean_initial_shape == 'right_rim_drape_high' else 1.04 if args.clean_initial_shape in ('right_rim_drape','right_rim_drape_front','front_staggered_corner_drape') else 1.00001):
            raise ValueError('Isometric rim initialization changed material edge lengths')
        initial_shape_record.update(rimfit_record, rim_overhang_extra_m=args.clean_rim_overhang_extra,
                                    edge_length_over_rest_min=float(ratios.min()),
                                    edge_length_over_rest_max=float(ratios.max()))
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
    if args.clean_diagonal_rotate_deg or args.clean_diagonal_center_xy is not None:
        if args.clean_initial_shape != 'photo_wall' or args.clean_diagonal_center_xy is None:
            raise ValueError('Diagonal fold diagnostic requires photo_wall and an explicit XY center')
        if not -45. <= args.clean_diagonal_rotate_deg <= 45.:
            raise ValueError('Diagonal fold rotation outside tested range')
        folded = initial[clean].copy()
        center_before = folded[:, :2].mean(axis=0)
        radians = np.deg2rad(args.clean_diagonal_rotate_deg)
        rotation = np.array([[np.cos(radians), -np.sin(radians)],
                             [np.sin(radians), np.cos(radians)]])
        folded[:, :2] = ((folded[:, :2] - center_before) @ rotation.T
                          + np.asarray(args.clean_diagonal_center_xy, dtype=float))
        if args.clean_tilt_x_rad or args.clean_tilt_y_rad:
            if max(abs(args.clean_tilt_x_rad), abs(args.clean_tilt_y_rad)) > .3:
                raise ValueError('Pre-episode fold tilt outside diagnostic search range')
            pivot = np.array([*args.clean_diagonal_center_xy, folded[:, 2].min()])
            tilt = Rotation.from_rotvec([args.clean_tilt_x_rad,
                                         args.clean_tilt_y_rad, 0.]).as_matrix()
            folded = (folded - pivot) @ tilt.T + pivot
        below_rim = folded[:, 2] < .45
        cavity_min = np.array([-2.419, -1.013])
        cavity_max = np.array([-2.061, -.737])
        if np.any((folded[below_rim, :2] < cavity_min).any(axis=1) |
                  (folded[below_rim, :2] > cavity_max).any(axis=1)):
            raise ValueError('Fold starts inside physical basket wall below the rim')
        initial[clean] = folded
        initial_shape_record['diagonal_basket_orientation'] = dict(
            rotation_deg=args.clean_diagonal_rotate_deg,
            center_xy_m=list(map(float, args.clean_diagonal_center_xy)),
            tilt_rotvec_rad=[args.clean_tilt_x_rad,args.clean_tilt_y_rad,0.],
            right_rim_overhang_vertices_initial=int(((folded[:, 0] > -2.061) &
                                                       (folded[:, 2] > .45)).sum()),
            rigid_material_transform_before_episode=True)
    if args.commands:
        with np.load(args.commands, allow_pickle=False) as z:
            action = {k: z[k] for k in ('joint_target', 'joint_names', 'time_s')}
            phases = z['phase']
            if 'grasp_reference_corner' in z: action['grasp_reference_corner'] = z['grasp_reference_corner']
            if 'right_grasp_reference_corner' in z: action['right_grasp_reference_corner'] = z['right_grasp_reference_corner']
            if 'right_grasp_reference_uv' in z: action['right_grasp_reference_uv'] = z['right_grasp_reference_uv']
            if 'right_grasp_layer_uv' in z: action['right_grasp_layer_uv'] = z['right_grasp_layer_uv']
    fps = int(meta['fps'])
    count = len(action['joint_target'])
    if (action['joint_target'].shape != (count, len(action['joint_names']))
            or not np.isfinite(action['joint_target']).all()
            or len(phases) != count or len(action['time_s']) != count
            or not np.allclose(action['time_s'], np.arange(count)/fps, atol=1e-8, rtol=0)):
        raise ValueError('Reference command clock or joint array invalid')
    if not np.isfinite(initial).all():
        raise ValueError('Invalid initial cloth')
    material_edges = np.unique(np.sort(np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    material_rest = np.linalg.norm(uv[material_edges[:, 0]]-uv[material_edges[:, 1]], axis=1)
    if np.any(material_rest <= 0):
        raise ValueError('Invalid cloth rest edge')
    from cloth_self_intersection_audit_v1 import audit_self_intersections
    initial_self = audit_self_intersections(initial, faces)
    (args.output/'authored_initial_self_intersection.json').write_text(json.dumps(initial_self,indent=2))
    if initial_self['pairs_over_50micron']:
        raise RuntimeError('Reject self-intersecting cloth initialization before physics')
    # One anonymous working layer: never save or mutate the source House asset.
    source = Usd.Stage.Open(str(args.scene))
    stage = Usd.Stage.Open(source.Flatten())
    del source
    stage.RemovePrim('/World/Robot')
    robot = stage.DefinePrim('/World/Robot', 'Xform')
    robot.GetReferences().AddReference(str(args.robot))
    x, y, yaw = station_xyyaw(meta['benchmark_task_id'])
    root = np.eye(4)
    root[:3, :3] = Rotation.from_euler('z', yaw, degrees=True).as_matrix()
    root[:2, 3] = [x, y]
    UsdGeom.Xformable(robot).AddTransformOp().Set(Gf.Matrix4d(root.T.tolist()))
    # All House tasks use the same robot.usda world anchor. Keep the base
    # fixed while every articulated rigid body and native drive remains dynamic.
    anchor = Sdf.Path('/World/Robot/base_link/FixedJoint')
    if not stage.GetPrimAtPath(anchor) or not stage.GetPrimAtPath(anchor).IsActive():
        raise ValueError('Shared robot.usda fixed base anchor is required')
    physics = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if len(physics) != 1:
        raise ValueError('Expected one authoritative House physics scene')
    physics_path = physics[0].GetPath()
    px = PhysxSchema.PhysxSceneAPI.Apply(physics[0])
    source_frequency = int(px.GetTimeStepsPerSecondAttr().Get())
    if source_frequency != 60 or fps != 20:
        raise ValueError('Shared House physics/action clock must be 60/20 Hz')
    args.substeps = source_frequency // fps
    if args.observe_substeps > args.substeps:
        raise ValueError('Substep observations exceed resolved House steps per action')
    px.CreateEnableGPUDynamicsAttr(True)
    px.CreateBroadphaseTypeAttr('GPU')
    # Native robot gains, limits, mass, friction and cameras are never overridden.
    native = {str(a.GetPath()): str(a.Get()) for p in Usd.PrimRange(robot)
              for a in p.GetAttributes() if a.GetName().startswith(('physics:', 'physx', 'drive:'))
              and not any(k in a.GetName() for k in ('targetPosition', 'targetVelocity', 'velocity', 'angularVelocity'))}
    # Explicit20260927 user-requested friction diagnostic; anonymous stage only.
    grip_prim=stage.GetPrimAtPath('/World/Robot/PhysicsMaterials/GripMat')
    if not grip_prim:raise RuntimeError('Native shared GripMat missing')
    grip_friction_override={}
    for key,value in [('physics:staticFriction',12.),('physics:dynamicFriction',10.)]:
        attr=grip_prim.GetAttribute(key);before=attr.Get();attr.Set(value)
        grip_friction_override[key]=dict(before=before,after=attr.Get())
    (args.output/'grip_friction_override.json').write_text(json.dumps(dict(user_requested=True,applied_before_episode=True,scope='Four native gripper pads in anonymous stage; shared robot.usda unchanged',values=grip_friction_override),indent=2))
    articulation_roots = [p for p in Usd.PrimRange(robot) if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if len(articulation_roots) != 1:
        raise ValueError('Expected one native robot articulation')
    self_collision = PhysxSchema.PhysxArticulationAPI(articulation_roots[0]).GetEnabledSelfCollisionsAttr().Get()
    if self_collision is not True:
        raise ValueError('Native robot self collision must be enabled')
    robot_bodies = [p for p in Usd.PrimRange(robot) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    kinematic_bodies = [str(p.GetPath()) for p in robot_bodies
                        if UsdPhysics.RigidBodyAPI(p).GetKinematicEnabledAttr().Get() is True]
    if not robot_bodies or kinematic_bodies:
        raise ValueError(f'Robot requires fully dynamic articulation bodies; kinematic={kinematic_bodies}')
    world_fixed_joints = []
    for prim in Usd.PrimRange(robot):
        if prim.IsA(UsdPhysics.FixedJoint):
            joint = UsdPhysics.Joint(prim)
            if not joint.GetBody0Rel().GetTargets() or not joint.GetBody1Rel().GetTargets():
                world_fixed_joints.append(str(prim.GetPath()))
    if world_fixed_joints != [str(anchor)]:
        raise ValueError(f'Unexpected world anchors for shared robot: {world_fixed_joints}')
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
    table_collider_candidate = None
    if args.diagnostic_convex_table_collider_candidate:
        table_prim = stage.GetPrimAtPath('/World/Assets/CoffeeTable/Mesh')
        if not table_prim or not table_prim.IsA(UsdGeom.Mesh) or not table_prim.HasAPI(UsdPhysics.CollisionAPI):
            raise RuntimeError('Canonical coffee-table physical mesh collider not found')
        table_collision = UsdPhysics.MeshCollisionAPI.Apply(table_prim)
        before = table_collision.GetApproximationAttr().Get()
        table_collision.CreateApproximationAttr('convexHull')
        table_collider_candidate = dict(path=str(table_prim.GetPath()), previous_approximation=str(before),
                                        candidate_approximation='convexHull', applied_before_episode=True,
                                        shared_scene_review_required=True)
    basket_collider_candidate = None
    replaced_basket_mesh_path = None
    if args.diagnostic_decomposed_basket_collider_candidate:
        basket_meshes = [p for p in Usd.PrimRange(stage.GetPrimAtPath('/World/Assets/TableclothBasket/model'))
                         if p.IsA(UsdGeom.Mesh) and p.HasAPI(UsdPhysics.CollisionAPI)]
        if len(basket_meshes) != 1:
            raise RuntimeError('Expected exactly one canonical basket physical mesh')
        basket_prim = basket_meshes[0]
        basket_collision = UsdPhysics.MeshCollisionAPI.Apply(basket_prim)
        before = basket_collision.GetApproximationAttr().Get()
        basket_collision.CreateApproximationAttr('convexDecomposition')
        basket_collider_candidate = dict(path=str(basket_prim.GetPath()), previous_approximation=str(before),
                                         candidate_approximation='convexDecomposition', applied_before_episode=True,
                                         shared_scene_review_required=True)
    if args.diagnostic_basket_box_collider_candidate:
        basket_meshes = [p for p in Usd.PrimRange(stage.GetPrimAtPath('/World/Assets/TableclothBasket/model'))
                         if p.IsA(UsdGeom.Mesh) and p.HasAPI(UsdPhysics.CollisionAPI)]
        if len(basket_meshes) != 1:
            raise RuntimeError('Expected exactly one canonical basket physical mesh')
        basket_prim = basket_meshes[0]
        # The original mesh remains for the canonical audit. An explicit
        # replacement diagnostic can retire it afterward, with an exact
        # before/after collision inventory check below.
        # These bounds come from the existing visible basket triangles, not
        # from the cloth pose. They replace its unsupported GPU triangle-mesh
        # collider with floor and wall surfaces at the same location.
        basket_boxes = {
            'floor': ((-2.419, -2.061), (-1.013, -.737), (.260, .285063)),
            'wall_x_min': ((-2.439, -2.419), (-1.034, -.716), (.260, .450)),
            'wall_x_max': ((-2.061, -2.041), (-1.034, -.716), (.260, .450)),
            'wall_y_min': ((-2.419, -2.061), (-1.034, -1.013), (.260, .450)),
            'wall_y_max': ((-2.419, -2.061), (-.737, -.716), (.260, .450)),
        }
        candidate_root = '/World/Assets/TableclothBasket/PhysicalColliderCandidate'
        UsdGeom.Xform.Define(stage, candidate_root)
        basket_parent = stage.GetPrimAtPath('/World/Assets/TableclothBasket')
        parent_translation = UsdGeom.XformCommonAPI(basket_parent).GetXformVectors(Usd.TimeCode.Default())[0]
        if any(abs(float(parent_translation[k]) - expected) > 1e-5 for k, expected in enumerate((0., .28091587972998144, 0.))):
            raise RuntimeError('Basket box candidate requires canonical local basket transform')
        profile=json.loads((Path(__file__).parent/'basket_profile_split_colliders.json').read_text())
        records=profile['colliders']
        for record in records:
            name=record['name'];points=np.array(record['world_vertices_m'])-np.asarray(parent_translation,dtype=float)
            mesh=UsdGeom.Mesh.Define(stage,f'{candidate_root}/{name}')
            mesh_center=points.mean(axis=0)
            mesh.CreatePointsAttr((points-mesh_center).tolist())
            UsdGeom.XformCommonAPI(mesh).SetTranslate(tuple(map(float,mesh_center)))
            triangles=np.asarray(record['triangles'],dtype=int)
            mesh.CreateFaceVertexCountsAttr([3]*len(triangles));mesh.CreateFaceVertexIndicesAttr(triangles.ravel().tolist())
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
        (args.output/'basket_profile_colliders.json').write_text(json.dumps(profile,indent=2)+'\n')
        basket_collider_candidate = dict(path=candidate_root, source_mesh_still_enabled=str(basket_prim.GetPath()),
                                         previous_approximation='none', candidate_approximation='visible_cross_section_convex_wall_pieces',
                                         box_world_bounds=records, applied_before_episode=True,
                                         diagnostic_overlay_only=True, shared_scene_review_required=True)
    from basket_profile_guard_all_cloths_v1 import BasketWallGuard
    basket_guard=BasketWallGuard(basket_collider_candidate['box_world_bounds'],cloth_ids,faces)
    initial_basket_check=basket_guard.check(initial,tolerance=.00001)
    basket_rows=[dict(stage='authored_initial',**initial_basket_check)]
    (args.output/'basket_initial_clearance.json').write_text(json.dumps(initial_basket_check,indent=2)+'\n')
    if not initial_basket_check['valid']:raise RuntimeError('Authored cloth crosses basket before episode: '+str(initial_basket_check))
    collision_repairs = enable_required_contacts(stage)
    collision_audit = audit_required_contacts(stage)
    if args.diagnostic_basket_box_replace_source:
        source_path = basket_collider_candidate['source_mesh_still_enabled']
        source_prim = stage.GetPrimAtPath(source_path)
        source_collision = UsdPhysics.CollisionAPI(source_prim)
        if source_collision.GetCollisionEnabledAttr().Get() is False:
            raise RuntimeError('Basket source mesh was already disabled before replacement audit')
        source_collision.CreateCollisionEnabledAttr(False)
        before = {row['path']: row['enabled'] for row in collision_audit['colliders']}
        after = {str(p.GetPath()): p.GetAttribute('physics:collisionEnabled').Get() is not False
                 for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)}
        changed = {path for path in before if before[path] != after.get(path)}
        required_boxes = {f"{basket_collider_candidate['path']}/{name}"
                          for name in [r['name'] for r in records]}
        if (set(after) != set(before) or changed != {source_path} or not before[source_path]
                or after[source_path] or not all(after.get(path) for path in required_boxes)):
            raise RuntimeError('Basket replacement collision inventory differs from exactly one retired mesh and all enabled profiled convex pieces')
        replaced_basket_mesh_path = source_path
        collision_audit['colliders'] = [dict(path=path, enabled=enabled) for path, enabled in after.items()]
        collision_audit['basket_replacement'] = dict(retired_source_mesh=source_path,
            enabled_replacements=sorted(required_boxes), canonical_audit_passed_before_replacement=True,
            other_collision_enabled_states_unchanged=True, filters_unchanged=True,
            diagnostic_only=True, shared_scene_review_required=True)
        basket_collider_candidate['source_mesh_still_enabled'] = None
        basket_collider_candidate['source_mesh_retired_before_episode'] = replaced_basket_mesh_path
        basket_collider_candidate['diagnostic_overlay_only'] = False
    sys.path.insert(0, str(REPO / 'real2sim/teleop'))
    from robot_collision_profile import audit_full_robot_collision_profile
    shared_robot_profile = audit_full_robot_collision_profile(stage)
    (args.output/'collision_inventory.json').write_text(json.dumps(collision_audit,indent=2))
    # Compare the cached planning pad geometry against this episode's actual native USD geometry.
    from scipy.spatial import cKDTree
    cached_pads=np.load('/home/aether/workspace/agent_bench/.codex-tmp/tablecloth-action-loop-20260905/robot_native_collision.npz')
    pad_geometry_rows=[]
    xcache=UsdGeom.XformCache()
    for side in ('inner','outer'):
        link='gripper_r_'+side+'_link4'
        body_prim=stage.GetPrimAtPath('/World/Robot/'+link)
        body_world=np.array(xcache.GetLocalToWorldTransform(body_prim)).T
        samples=[];colliders=[]
        for child in Usd.PrimRange(body_prim):
            if child.IsA(UsdGeom.Mesh) and child.HasAPI(UsdPhysics.CollisionAPI):
                mesh=UsdGeom.Mesh(child);points=np.asarray(mesh.GetPointsAttr().Get(),dtype=float)
                relative=np.linalg.inv(body_world)@np.array(xcache.GetLocalToWorldTransform(child)).T
                samples.append(points@relative[:3,:3].T+relative[:3,3])
                colliders.append(dict(path=str(child.GetPath()),approximation=str(child.GetAttribute('physics:approximation').Get()),rest_offset=str(child.GetAttribute('physxCollision:restOffset').Get()),contact_offset=str(child.GetAttribute('physxCollision:contactOffset').Get())))
        if not samples:raise RuntimeError('No actual native pad collision mesh')
        actual_points=np.concatenate(samples);cached_points=cached_pads['points_'+link]
        delta=max(float(cKDTree(cached_points).query(actual_points)[0].max()),float(cKDTree(actual_points).query(cached_points)[0].max()))
        pad_geometry_rows.append(dict(link=link,actual_mesh_vertices=len(actual_points),cached_samples=len(cached_points),maximum_bidirectional_vertex_distance_m=delta,colliders=colliders))
        (args.output/'native_pad_geometry_audit.json').write_text(json.dumps(pad_geometry_rows,indent=2)+'\n')
        if delta>.0001:raise RuntimeError('Cached planning pad geometry differs from actual robot by '+str(delta))
    (args.output/'native_pad_geometry_audit.json').write_text(json.dumps(pad_geometry_rows,indent=2)+'\n')
    existing_cloth = [p for p in stage.Traverse() if p.HasAPI(PhysxSchema.PhysxParticleClothAPI)]
    if existing_cloth:
        raise ValueError('House already contains cloth; refuse duplicate or silently replaced state')
    if not surface:
        from physx_cloth_backend import PhysXClothBackend, ClothConfig
        backend=PhysXClothBackend(stage,initial,uv,faces,cloth_ids,ClothConfig(radius=float(meta['config']['radius']),density=float(meta['config']['density']),friction=cloth_friction,stretch=args.stretch,shear=args.shear,bend=args.bend,spring_damping=args.damping,velocity_damping=args.velocity_damping,iterations=args.iterations))
        backend.author();meshes=backend.paths;particle_counts=backend.counts;materials=backend.materials
        masses=[float(meta['config']['density']*np.prod(np.ptp(uv[cloth_ids==cid],axis=0))) for cid in np.unique(cloth_ids)];nodal_mass_estimates=[]
    else:
        radius = float(meta['config']['radius'])
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
                args.surface_friction = cloth_friction
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
    sim.reset()
    tensor_sim = omni.physics.tensors.create_simulation_view('torch')
    tensor_sim.set_subspace_roots('/')
    if surface:
        material_backend_readback=[]
        for material_body_path in meshes:
            material_path=material_body_path+'_Material'
            material_view=tensor_sim.create_deformable_material_view(material_path)
            if material_view.count!=1:raise RuntimeError('Expected exactly one native deformable material')
            measured_E=float(material_view.get_youngs_modulus().detach().cpu().numpy().reshape(-1)[0])
            measured_mu=float(material_view.get_dynamic_friction().detach().cpu().numpy().reshape(-1)[0])
            if not np.isclose(measured_E,args.youngs_modulus,rtol=1e-6):raise RuntimeError(f'Native Young modulus mismatch {measured_E}')
            mp=stage.GetPrimAtPath(material_path)
            authored={str(attr.GetName()):str(attr.Get()) for attr in mp.GetAttributes() if str(attr.GetName()).startswith(('omniphysics:','physxDeformableMaterial:'))}
            material_backend_readback.append(dict(path=material_path,backend_youngs_modulus_Pa=measured_E,backend_dynamic_friction=measured_mu,usd_material_attributes=authored,read_only=True))
        (args.output/'material_backend_readback.json').write_text(json.dumps(material_backend_readback,indent=2)+'\n')
        print(json.dumps({'material_backend_readback':material_backend_readback}),flush=True)

    pad_material_readback={}
    try:
        pad_view=tensor_sim.create_rigid_body_view('/World/Robot/gripper_*_link4')
        props=pad_view.get_material_properties().detach().cpu().numpy()
        pad_material_readback=dict(method='rigid body tensor get_material_properties',count=pad_view.count,values=props.tolist(),columns=['static_friction','dynamic_friction','restitution'])
    except Exception as e:pad_material_readback=dict(error=repr(e),authored_values_only=True)
    (args.output/'grip_material_backend_readback.json').write_text(json.dumps(pad_material_readback,indent=2))
    print(json.dumps({'pad_material_backend_readback':pad_material_readback}),flush=True)
    if surface:
        material_backend_readback=[]
        for material_body_path in meshes:
            material_path=material_body_path+'_Material'
            material_view=tensor_sim.create_deformable_material_view(material_path)
            if material_view.count!=1:raise RuntimeError('Expected exactly one native deformable material')
            measured_E=float(material_view.get_youngs_modulus().detach().cpu().numpy().reshape(-1)[0])
            measured_mu=float(material_view.get_dynamic_friction().detach().cpu().numpy().reshape(-1)[0])
            if not np.isclose(measured_mu,cloth_friction,rtol=1e-6):raise RuntimeError(f'Native friction mismatch {measured_mu}')
            if not np.isclose(measured_E,args.youngs_modulus,rtol=1e-6):raise RuntimeError(f'Native Young modulus mismatch {measured_E}')
            mp=stage.GetPrimAtPath(material_path)
            authored={str(attr.GetName()):str(attr.Get()) for attr in mp.GetAttributes() if str(attr.GetName()).startswith(('omniphysics:','physxDeformableMaterial:'))}
            material_backend_readback.append(dict(path=material_path,backend_youngs_modulus_Pa=measured_E,backend_dynamic_friction=measured_mu,usd_material_attributes=authored,read_only=True))
        (args.output/'material_backend_readback.json').write_text(json.dumps(material_backend_readback,indent=2)+'\n')
        print(json.dumps({'material_backend_readback':material_backend_readback}),flush=True)

    if not px.GetEnableGPUDynamicsAttr().Get():
        raise RuntimeError('Particle cloth requires GPU dynamics on its actual simulation owner')
    art = SingleArticulation('/World/Robot')
    art.initialize(physics_sim_view=tensor_sim)
    names = list(art.dof_names)
    if any(n not in names for n in driven):
        raise ValueError('Native drive missing from articulation')
    indices = np.array([names.index(n) for n in driven])
    columns = [command_names.index(n) for n in driven]
    # Joint states and drive targets were authored before reset. Do not write
    # articulation state after the cloth has taken its first solver step.
    q0 = art.get_joint_positions().detach().cpu().numpy().copy()
    initial_joint_error = float(np.abs(q0[indices] - action['joint_target'][0, columns]).max())
    if initial_joint_error > .02:
        raise RuntimeError(f'Native joint initialization differs from authored state: {initial_joint_error:.6f} rad; errors={dict(zip(driven, (q0[indices] - action["joint_target"][0, columns]).tolist()))}')
    # GPU particle state is not necessarily mirrored into USD. Read the solver
    # tensors directly rather than mistaking unchanged authored mesh for state.
    contact_sensors = [str(p.GetPath()) for p in Usd.PrimRange(robot)
                       if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    rigid_contact_view = tensor_sim.create_rigid_contact_view(contact_sensors)
    pad_pair_view=tensor_sim.create_rigid_contact_view('/World/Robot/gripper_r_inner_link4',['/World/Robot/gripper_r_outer_link4'])
    left_pad_pair_view=tensor_sim.create_rigid_contact_view('/World/Robot/gripper_l_inner_link4',['/World/Robot/gripper_l_outer_link4'])
    pad_sensor_indices=[contact_sensors.index('/World/Robot/gripper_r_'+side+'_link4') for side in ('inner','outer')]
    external_contact_forces=[]
    pad_pair_contact_forces=[]
    contact_forces = []
    cloth_views = ([SurfaceReadback(tensor_sim.create_surface_deformable_body_view(path), n)
                    for path, n in zip(meshes, particle_counts)] if surface else
                   [tensor_sim.create_particle_cloth_view(path) for path in meshes])
    identity_rows=[]
    for cid, mesh_path, cloth_view in zip(np.unique(cloth_ids), meshes, cloth_views):
        body_prim=stage.GetPrimAtPath(mesh_path)
        bound,_=UsdShade.MaterialBindingAPI(body_prim).ComputeBoundMaterial(materialPurpose='physics')
        if str(bound.GetPath()) != mesh_path+'_Material':
            raise RuntimeError('Cloth resolved material binding mismatch')
        sim_points=cloth_view.get_positions().detach().cpu().numpy().reshape(-1,3)
        identity_rows.append(dict(cloth_id=int(cid), body_path=mesh_path,
            resolved_physics_material=str(bound.GetPath()), target_is_clean_basket_cloth=bool(cid==1),
            node_count=len(sim_points), simulation_bounds_m=[sim_points.min(0).tolist(),sim_points.max(0).tolist()],
            body_parameters={str(a.GetName()):str(a.Get()) for a in body_prim.GetAttributes()
                if str(a.GetName()).startswith(('physxDeformableBody:','physxCollision:'))},
            iteration_count_is_usd_authored_not_backend_readback=True,
            available_body_readback_methods=[n for n in dir(cloth_view.view) if 'iteration' in n or 'material' in n]))
    (args.output/'cloth_identity_audit.json').write_text(json.dumps(identity_rows,indent=2)+'\n')
    print(json.dumps({'cloth_identity_audit':identity_rows}),flush=True)
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
    solver_initial_basket_check=basket_guard.check(solver_initial_cloth)
    (args.output/'solver_initial_basket_clearance.json').write_text(json.dumps(solver_initial_basket_check,indent=2)+'\n')
    adaptive_brush = None
    if args.adaptive_repeat_brush:
        from removal167_adaptive_right_v1 import AdaptiveRightBasket as AdaptiveBrush
        adaptive_brush = AdaptiveBrush(dict(action, phase=phases), uv, cloth_ids, stage,
                                       root, args.brush_target_overhang_m, args.adaptive_follow_brush_corner)
    if (args.adaptive_clean_corner_uv is None) != (args.adaptive_clean_corner_nominal_xyz is None):
        raise ValueError('Adaptive clean corner requires both material UV and nominal XYZ')
    corner_feedback = None
    corner_kinematics = None
    corner_arm_names = None
    if args.adaptive_clean_corner_uv is not None:
        if not 1 <= args.adaptive_clean_corner_settle_frame < args.adaptive_clean_corner_arm_start_frame:
            raise ValueError('Corner feedback requires a positive settle frame before arm motion')
        from tablecloth_robot_kinematics import Robot as KinematicRobot
        cad_record = json.loads(Path('/home/aether/workspace/agent_bench/.codex-tmp/tablecloth-action-loop-20260905/robot_native_collision.json').read_text())
        corner_kinematics = KinematicRobot(cad_record, station_xyyaw(meta['benchmark_task_id']))
        corner_arm_names = KinematicRobot.arm_names('R')
        if args.natural_arm_bend_limit_deg is not None:
            for n in (corner_arm_names[3], corner_arm_names[5]):
                corner_kinematics.by_name[n]['limits'] = np.deg2rad([-args.natural_arm_bend_limit_deg+1, args.natural_arm_bend_limit_deg-1]).tolist()
        corner_indices = np.flatnonzero(cloth_ids == 1)
        corner_index = int(corner_indices[np.linalg.norm(
            uv[corner_indices] - np.asarray(args.adaptive_clean_corner_uv), axis=1).argmin()])
        corner_feedback = dict(material_uv_m=list(args.adaptive_clean_corner_uv),
                               nominal_xyz_m=list(args.adaptive_clean_corner_nominal_xyz),
                               material_vertex_index=corner_index, applied_to_joint_commands_only=True,
                               settle_frame=args.adaptive_clean_corner_settle_frame,
                               arm_start_frame=args.adaptive_clean_corner_arm_start_frame,
                               observed_settled_xyz_m=None, correction_xyz_m=None)
    solver_warmup_time = float(sim.current_time)
    articulation_view = art._articulation_view
    body_names = list(articulation_view.body_names)
    from force_readback_probe import ForceReadback
    force_probe=ForceReadback(args.output,articulation_view._physics_view,stage,robot,names,body_names,cloth_views)
    if args.force_api_probe:
        force_probe.sample(0,0,0,'read_only_api_probe',np.zeros((len(contact_sensors),3)),np.zeros(3));force_probe.save();return
    from read_dual_pad_convexes import read_pad_convexes
    from cooked_closure_guard import CookedClosureGuard
    read_pad_convexes(stage,args.output)
    read_pad_convexes(stage,args.output,"l")
    cooked_guard=CookedClosureGuard(args.output/'native_pad_cooked_convexes.json')
    body_records = []
    root_records = []
    from keyboard_control import KeyboardControl
    limits={}
    for prim in Usd.PrimRange(robot):
        if prim.IsA(UsdPhysics.RevoluteJoint):
            j=UsdPhysics.RevoluteJoint(prim);limits[prim.GetName()]=np.deg2rad([j.GetLowerLimitAttr().Get(),j.GetUpperLimitAttr().Get()]).tolist()
    control=KeyboardControl(driven,action['joint_target'][0,columns],limits,
                            args.headless_smoke or args.headless_command_replay)
    commanded=[]
    continuation_start_frame=None
    gripper_command_audit=[]
    from profile_horizontal_clearance_exact import make_clearance
    horizontal_clearance = make_clearance(corner_kinematics.metadata) if corner_kinematics is not None else None
    executed_phases=[]
    command_source_frames=[]
    import time
    wall_start=time.monotonic()
    if not (args.headless_smoke or args.headless_command_replay):
        from isaacsim.core.utils.viewports import set_camera_view
        set_camera_view(eye=np.array([-3.6,-2.2,1.8]),target=np.array([-2.05,-.45,.5]))
    diagnostic_events=[]
    def diagnostic_warning(message):
        diagnostic_events.append(dict(frame=int(frame),source_frame=int(source_frame),message=str(message)))
        if len(diagnostic_events)<=5 or frame%20==0:
            print(json.dumps({'diagnostic_continue_warning':diagnostic_events[-1]}),flush=True)
    from bimanual_lay_extension_v38 import LayExtension
    bimanual=LayExtension(args.output,cloth_ids)
    bimanual.force_readback=force_probe
    force_probe.meta['meaning']['left_pad_pair_force']='Left rigid inner-pad versus outer-pad force, not cloth pressure.'
    record, failure = [], None
    self_intersection_rows=[]
    max_edge_ratio_seen = 0.
    first_edge_limit_failure = None
    substep_positions, substep_velocities, substep_times = [], [], []
    numerical_state_history = []
    try:
        limit = min(count, args.max_frames) if args.max_frames else count
        source_frame = 0
        closure_wait_frames = 0
        closure_stable_frames = 0
        strain_hold_active=False
        strain_hold_frames=0
        strain_hold_rows=[]
        live_grip_rows=[]
        lost_grip_frames=0
        force_sweep_frames=0
        tighter_target=.009
        tighter_last_adjust=160
        tighter_geometry_rows=[]
        small_lift_hold_frames=0
        small_lift_start_z=None
        small_lift_stop_source=None
        for frame in range(18000):
            if source_frame>=count:break
            if source_frame >= count:
                break
            current_frame[0] = frame
            while not control.running and not control.quit and app.is_running():
                # Render/UI only while paused; do not advance the physical episode.
                sim.render();time.sleep(.01)
            if control.quit or not app.is_running():break
            template = action['joint_target'][source_frame].copy()
            if corner_feedback is not None and source_frame == args.adaptive_clean_corner_settle_frame and not corner_feedback.get('online_replanned',False):
                observed = record[-1][0][corner_index].astype(float)
                correction = observed - np.asarray(args.adaptive_clean_corner_nominal_xyz)
                # This diagnostic replans the full episode from measured cloth, rather than warping
                # an old trajectory by an offset. Validate the canonical basket work region instead.
                workspace_lower=np.array([-2.55,-1.20,.26]);workspace_upper=np.array([-2.30,-.95,.65])
                if not np.isfinite(observed).all() or np.any(observed<workspace_lower) or np.any(observed>workspace_upper):
                    raise RuntimeError(f'Settled corner left canonical basket work region: {observed}')
                corner_feedback['observation_validation']=dict(mode='fresh_full_path_replan',nominal_offset_m=correction.tolist(),workspace_lower_m=workspace_lower.tolist(),workspace_upper_m=workspace_upper.tolist(),diagnostic_validation_thresholds_unchanged_execution_continues=True)
                if args.adaptive_clean_corner_nominal_normal is not None:
                    from scipy.spatial.transform import Rotation
                    cu, cv = args.adaptive_clean_corner_uv
                    tangent_ids = [int(corner_indices[np.linalg.norm(uv[corner_indices]-point,axis=1).argmin()])
                                   for point in ([cu-.019,cv],[cu,cv+.01888889])]
                    measured_normal = np.cross(record[-1][0][tangent_ids[0]]-observed,
                                               record[-1][0][tangent_ids[1]]-observed)
                    measured_normal /= np.linalg.norm(measured_normal)
                    nominal_normal = np.asarray(args.adaptive_clean_corner_nominal_normal)
                    nominal_normal /= np.linalg.norm(nominal_normal)
                    if np.dot(measured_normal, nominal_normal) < 0: measured_normal = -measured_normal
                    rotation = Rotation.align_vectors([measured_normal],[nominal_normal])[0]
                    if np.rad2deg(rotation.magnitude()) > 60:
                        raise RuntimeError('Settled corner plane drift exceeds 60 degree feedback bound')
                    corner_feedback['observed_settled_normal'] = measured_normal.tolist()
                    corner_feedback['normal_rotation_vector'] = rotation.as_rotvec().tolist()
                corner_feedback['observed_settled_xyz_m'] = observed.tolist()
                corner_feedback['correction_xyz_m'] = correction.tolist()
                print(json.dumps({'adaptive_clean_corner': corner_feedback}), flush=True)
                # Plan joint targets from this episode's observed, naturally settled cloth.
                # Physics pauses during computation; no object state or robot state is written.
                import runpy
                from contextlib import redirect_stdout
                online_dir = args.output / 'online_protocol'
                online_dir.mkdir(exist_ok=False)
                snapshot = online_dir / 'settled_observation.npz'
                np.savez_compressed(snapshot, cloth_id=cloth_ids, uv=uv, cloth_q=np.asarray([record[-1][0]]))
                exposed=corner_indices[(uv[corner_indices,1]<.075)&(record[-1][0][corner_indices,1]<-1.034)]
                outer_material=float(.342-uv[exposed,0].min()) if len(exposed) else 0.
                horizontal_outside=float(max(0.,-1.034-record[-1][0][corner_indices[uv[corner_indices,1]<.075],1].min()))
                exposure_report=dict(material_outside_m=outer_material,horizontal_outside_m=horizontal_outside,required_approx50mm=True)
                (online_dir/'exposure_acceptance.json').write_text(json.dumps(exposure_report,indent=2))
                if not (.025<=outer_material<=.075 and horizontal_outside<=.065):
                    diagnostic_warning('Settled exposed strip fails approximately50mm requirement: '+str(exposure_report))
                candidates_file=online_dir/'candidates.json'
                saved_argv=sys.argv[:]
                with (online_dir/'planning.log').open('w') as planning_log, redirect_stdout(planning_log):
                    try:
                        sys.argv=['search_pad_face_safe_v3.py',str(snapshot),str(candidates_file)]
                        runpy.run_path(str(Path(__file__).parent/'search_pad_face_safe_v3.py'),run_name='__main__')
                        candidates=json.loads(candidates_file.read_text())
                        candidates=[c for c in candidates if c['position_error']<.002 and abs(c['yaw_deg'])<.1 and abs(c['pitch_deg'])<55 and c['hand_basket_clearance_m']>.008 and c['camera_mount_outward_m']<-.015]
                        if args.diagnostic_grasp_uv is not None:
                            candidates=[c for c in candidates if np.linalg.norm(np.asarray(c['uv'])-args.diagnostic_grasp_uv)<1e-6]
                        candidates.sort(key=lambda c:(float(np.linalg.norm(np.asarray(c['uv'])-[.31825,.0283333333])),abs(c['pitch_deg'])))
                        planned=None; attempts=[]
                        for candidate_number,candidate in enumerate(candidates[:6]):
                            settled_index=int(corner_indices[np.linalg.norm(uv[corner_indices]-np.asarray(candidate['uv']),axis=1).argmin()])
                            recent=np.asarray([r[0][settled_index] for r in record[-100:]])
                            motion=float(np.max(np.linalg.norm(recent-recent[-1],axis=1)))
                            if False: # User requests full diagnostic motion even if initial cloth still drifts.
                                attempts.append(dict(candidate=candidate_number,status='not_settled',maximum_corner_motion_m=motion));continue
                            candidate['maximum_corner_motion_last_5s_m']=motion
                            chosen=online_dir/f'candidate_{candidate_number}.json';chosen.write_text(json.dumps([candidate],indent=2))
                            planned_path=online_dir/f'candidate_{candidate_number}.action.npz'
                            try:
                                sys.argv=['plan_pad_face_interior.py',str(chosen),str(planned_path)]
                                runpy.run_path(str(Path(__file__).parent/'plan_pad_face_interior.py'),run_name='__main__')
                                with np.load(planned_path) as planned_file:
                                    candidate_action={k:planned_file[k].copy() for k in planned_file.files}
                                if candidate_action['joint_names'].tolist()!=action['joint_names'].tolist(): raise RuntimeError('Online joint name mismatch')
                                if not np.array_equal(candidate_action['joint_target'][:400],action['joint_target'][:400]): raise RuntimeError('Online initial targets changed')
                                fixed_columns=[i for i,n in enumerate(candidate_action['joint_names'].tolist()) if '_arm_r_' not in n and '_gripper_r_' not in n]
                                if np.any(np.ptp(candidate_action['joint_target'][:,fixed_columns],axis=0)!=0):raise RuntimeError('Online non-right commands moved')
                                for check_frame in range(0,len(candidate_action['joint_target']),10):
                                    check_values=dict(zip(candidate_action['joint_names'].tolist(),candidate_action['joint_target'][check_frame]))
                                    _,check_axes=corner_kinematics.fk(check_values,True)
                                    if check_frame>=700 and check_axes[corner_arm_names[0]][0][1]-check_axes[corner_arm_names[2]][0][1]<0:diagnostic_warning(f'Upper arm is inward at planned frame {check_frame}')
                                    if np.max(horizontal_clearance(corner_kinematics.fk(check_values)))>.012: diagnostic_warning(f'Online clearance check failed {check_frame}')
                                rate=float(np.max(np.abs(np.diff(candidate_action['joint_target'][:,[candidate_action['joint_names'].tolist().index(n) for n in corner_arm_names]],axis=0)))*fps)
                                if rate>1.5: diagnostic_warning(f'Online rate check failed {rate}')
                                planned=candidate_action;attempts.append(dict(candidate=candidate_number,status='planned',max_joint_rate=rate));break
                            except Exception as plan_error:
                                attempts.append(dict(candidate=candidate_number,status='rejected',reason=repr(plan_error)))
                        (online_dir/'attempts.json').write_text(json.dumps(attempts,indent=2))
                        if planned is None: raise RuntimeError('No settled-corner candidate admits complete fixed-body horizontal protocol; see online_protocol evidence')
                    finally:
                        sys.argv=saved_argv
                if len(planned['joint_target'])!=count: raise RuntimeError('Online protocol frame count mismatch')
                action['joint_target']=planned['joint_target'];phases=planned['phase'];template=action['joint_target'][source_frame].copy()
                corner_feedback['online_replanned']=True
                corner_feedback['selected_material_uv_m']=candidate['uv']
                (args.output/'selected_grasp_material.json').write_text(json.dumps(dict(selected_material_uv_m=candidate['uv'],selected_settled_xyz_m=candidate['edge'],read_only_material_identity=True),indent=2))
                corner_feedback['selected_settled_xyz_m']=candidate['edge']
                corner_feedback['gripper_basket_planned_clearance_m']=candidate['hand_basket_clearance_m']
                corner_feedback['opposite_panel_settled_clearance_m']=candidate['opposite_panel_clearance_m']
                corner_feedback['single_corner_grasp_required']=True
                corner_feedback['pinch_point_link4_local_m']=[.03285,0,0,1.]
                corner_feedback['pinch_geometry_correction']=dict(previous_local_x_m=.02645,corrected_local_x_m=.03285,reason='Move target5mm proximally on pad and target an interior cloth patch to fill the pad face; preserve native pad geometry.')
                corner_feedback['requested_material_inboard_offset_m']=.03
                corner_feedback['closure_strategy']=dict(initial_command_rad=.012,requested_tighter_command_rad=.009,maximum_backoff_command_rad=.01175,description='Physically tighten then allow small command backoff if actual cooked containment is invalid; native drives unchanged')
                corner_feedback['requested_additional_mouth_axis_insertion_m']=.010
                corner_feedback['closure_arm_targets_fixed']=True
                corner_feedback['pad_local_grasp_depth_increase_m']=.005
                corner_feedback['approach_opening_command_rad']=.35
                corner_feedback['user_requested_pad_face_engagement']=True
                corner_feedback['depth_reference_run']='single_corner_stiff100_zero_0085'
                corner_feedback['actual_corner_observations']=[]
                corner_feedback['locked_mouth_rotation']=json.loads(planned_path.with_suffix('.json').read_text())['locked_rotation']
                corner_feedback['online_action_path']=str(planned_path)
                corner_feedback['selected_corner_motion_last_5s_m']=candidate['maximum_corner_motion_last_5s_m']
                corner_feedback['physics_state_writes_during_planning']=0
                print(json.dumps({'online_horizontal_protocol':corner_feedback}),flush=True)

            if corner_feedback is not None and not corner_feedback.get('online_replanned',False) and source_frame >= args.adaptive_clean_corner_arm_start_frame:
                if corner_feedback['correction_xyz_m'] is None:
                    raise RuntimeError('Missing physical settled-corner observation')
                correction = np.asarray(corner_feedback['correction_xyz_m']) * min(1., (source_frame - args.adaptive_clean_corner_arm_start_frame) / 100.)
                if np.any(correction) or corner_feedback.get('normal_rotation_vector') is not None:
                    names_action = action['joint_names'].tolist()
                    command_values = dict(zip(names_action, map(float, template)))
                    posed = corner_kinematics.fk(command_values)
                    hand = posed['gripper_r_base_link']
                    pad_local = np.array([.04785 - .025, 0., 0., 1.])
                    middle = np.mean([(posed[f'gripper_r_{side}_link4'] @ pad_local)[:3]
                                      for side in ('inner', 'outer')], axis=0)
                    local = hand[:3, :3].T @ (middle - hand[:3, 3])
                    from scipy.optimize import least_squares
                    from scipy.spatial.transform import Rotation
                    rotation_correction = Rotation.from_rotvec(np.asarray(corner_feedback.get('normal_rotation_vector',[0.,0.,0.])) * min(1., (source_frame - args.adaptive_clean_corner_arm_start_frame) / 100.)).as_matrix()
                    corrected_orientation = rotation_correction @ hand[:3,:3]
                    target_middle = middle + correction
                    corner_feedback['approach_rotated_with_settled_plane'] = False
                    ik_bounds = np.array([corner_kinematics.by_name[n]['limits'] for n in corner_arm_names])
                    pad_gap = posed['gripper_r_outer_link4'][:3,3] - posed['gripper_r_inner_link4'][:3,3]
                    # Use the same pad-surface centers as the position objective.
                    pad_gap = (posed['gripper_r_outer_link4'] @ pad_local)[:3] - (posed['gripper_r_inner_link4'] @ pad_local)[:3]
                    local_gap = hand[:3,:3].T @ (pad_gap / np.linalg.norm(pad_gap))
                    goal_gap = corrected_orientation @ local_gap
                    seed = np.array([command_values[n] for n in corner_arm_names])
                    if commanded:
                        seed = np.array([commanded[-1][command_names.index(n)] for n in corner_arm_names])
                    continuity_bounds = ik_bounds.copy()
                    if commanded:
                        continuity_bounds[:,0] = np.maximum(continuity_bounds[:,0], seed-1.485/fps)
                        continuity_bounds[:,1] = np.minimum(continuity_bounds[:,1], seed+1.485/fps)
                    def corner_residual(q):
                        vals = command_values | dict(zip(corner_arm_names, q))
                        all_poses, all_axes = corner_kinematics.fk(vals, True)
                        h = all_poses['gripper_r_base_link']
                        pos_error = h[:3,:3] @ local + h[:3,3] - target_middle
                        rot_error = Rotation.from_matrix(corrected_orientation @ h[:3,:3].T).as_rotvec()
                        if args.adaptive_corner_free_inplane_roll:
                            fore = all_axes[corner_arm_names[6]][0]-all_axes[corner_arm_names[4]][0]
                            fore /= np.linalg.norm(fore)
                            horizontal = 1. if str(phases[source_frame]) in ('R_horizontal_corner_approach','R_corner_physical_close','R_corner_physical_hold') else max(0., min(1.,(source_frame-600)/100.)) if str(phases[source_frame]) == 'R_retract_camera_outward' else 0.
                            camera_axis = Rotation.from_quat([-.6989317357080854,-.6868383037850344,-.14219926877000738,-.1397388322373386]).apply([0.,0.,-1.])
                            camera_y = (h[:3,:3]@camera_axis)[1]
                            return np.r_[pos_error, .12*(h[:3,:3]@local_gap-goal_gap), .001*rot_error, .0003*(q-seed), .05*max(0.,abs(fore[2])-np.sin(np.deg2rad(8)))*horizontal, .013*max(0.,camera_y+.3)*horizontal, horizontal_clearance(all_poses)/3.]
                        return np.r_[pos_error, .12*rot_error]
                    opt = least_squares(corner_residual, np.clip(seed, continuity_bounds[:,0]+1e-7, continuity_bounds[:,1]-1e-7),
                                        bounds=(continuity_bounds[:,0], continuity_bounds[:,1]), max_nfev=100,
                                        ftol=1e-9, xtol=1e-9, gtol=1e-9)
                    corner_feedback['continuous_previous_joint_seed'] = True
                    corrected = command_values | dict(zip(corner_arm_names, opt.x))
                    residual = corner_residual(opt.x)
                    error = float(np.linalg.norm(residual[:3]))
                    gap_or_rotation_error_deg = float(np.rad2deg(np.linalg.norm(residual[3:6])/.12))
                    corner_feedback['maximum_alignment_error_deg'] = max(corner_feedback.get('maximum_alignment_error_deg', 0.), gap_or_rotation_error_deg)
                    if gap_or_rotation_error_deg > 2:
                        raise RuntimeError(f'Corner feedback pad alignment error at {frame}: {gap_or_rotation_error_deg}')
                    if error > .002:
                        raise RuntimeError(f'Right corner feedback IK error at {frame}: {error}')
                    for name in corner_arm_names:
                        template[names_action.index(name)] = corrected[name]
            if adaptive_brush is not None:
                template = adaptive_brush.command(source_frame, frame, template)
            requested_right_opening=float(template[command_names.index("idx81_gripper_r_outer_joint1")])
            in_continuation=continuation_start_frame is not None and source_frame>=continuation_start_frame
            if in_continuation:
                opening=None  # Continuation owns its native jaw targets, including reclosure and release.
            elif corner_feedback is not None and source_frame==829:
                force_sweep_frames+=1
                if force_sweep_frames<=100: opening=.012
                elif force_sweep_frames<=160: opening=.012+(.009-.012)*(force_sweep_frames-100)/60
                else: opening=tighter_target
            elif corner_feedback is not None and source_frame>=830 and not (bimanual.released or str(phases[source_frame])=='LR_release_on_table'): opening=tighter_target
            else: opening=None
            if opening is not None:
                gv=corner_kinematics.calibrate_loops(dict(zip(command_names,map(float,template)))|{'idx71_gripper_r_inner_joint1':-opening,'idx81_gripper_r_outer_joint1':opening})
                for k,n in enumerate(command_names):
                    if '_gripper_r_' in n:template[k]=gv[n]
            targets=(template[columns].copy() if args.headless_command_replay else
                     control.step(1/fps,art.get_joint_positions().detach().cpu().numpy()[indices]))
            full_target=template if args.headless_command_replay else action['joint_target'][0].copy()
            full_target[columns]=targets
            if args.adaptive_corner_free_inplane_roll and commanded:
                arm_columns = [command_names.index(n) for n in corner_arm_names]
                rate = float(np.max(np.abs(full_target[arm_columns]-commanded[-1][arm_columns]))*fps)
                corner_feedback['maximum_joint_command_rate_rad_s'] = max(corner_feedback.get('maximum_joint_command_rate_rad_s',0.),rate)
                if rate > 1.5: raise RuntimeError(f'Right arm command speed exceeds natural motion bound at {frame}: {rate}')
            if frame%20==0 or (in_continuation and source_frame==continuation_start_frame):
                gripper_command_audit.append(dict(frame=frame,phase=str(phases[source_frame]),continuation=bool(in_continuation),requested_right_rad=requested_right_opening,applied_right_rad=float(full_target[command_names.index("idx81_gripper_r_outer_joint1")]),legacy_override_rad=opening))
                (args.output/"gripper_command_audit.json").write_text(json.dumps(gripper_command_audit,indent=2))
            commanded.append(full_target)
            executed_phases.append(str(phases[source_frame]))
            command_source_frames.append(source_frame)
            frame_wall=time.monotonic()
            frame_force = None
            frame_external_force=None
            frame_pad_pair_force=np.zeros(3)
            for substep in range(args.substeps):
                art.apply_action(ArticulationAction(
                    joint_positions=torch.as_tensor(targets, dtype=torch.float32, device='cuda:0'),
                    joint_indices=torch.as_tensor(indices, dtype=torch.int64, device='cuda:0')))
                sim.step(render=False)
                # Read-only numerical diagnostics, never used to restore/splice physical state.
                numerical_q = np.concatenate([v.get_positions().detach().cpu().numpy().reshape(-1,3).copy() for v in cloth_views])
                numerical_qd = np.concatenate([v.get_velocities().detach().cpu().numpy().reshape(-1,3).copy() for v in cloth_views])
                numerical_j = art.get_joint_positions().detach().cpu().numpy().copy()
                numerical_jd = art.get_joint_velocities().detach().cpu().numpy().copy()
                numerical_state_history.append((numerical_q,numerical_qd,numerical_j,numerical_jd,frame/fps+(substep+1)/(fps*args.substeps)))
                numerical_state_history = numerical_state_history[-7:]
                bad = {key:int((~np.isfinite(value)).sum()) for key,value in [('cloth_position',numerical_q),('cloth_velocity',numerical_qd),('joint_position',numerical_j),('joint_velocity',numerical_jd)]}
                if any(bad.values()):
                    np.savez_compressed(args.output/'nonfinite_substep_state.npz',cloth_q=np.stack([x[0] for x in numerical_state_history]),cloth_qd=np.stack([x[1] for x in numerical_state_history]),joint_position=np.stack([x[2] for x in numerical_state_history]),joint_velocity=np.stack([x[3] for x in numerical_state_history]),time_s=np.array([x[4] for x in numerical_state_history]),joint_target=full_target,uv=uv,cloth_id=cloth_ids)
                    (args.output/'nonfinite_substep_state.json').write_text(json.dumps(dict(frame=frame,substep=substep,phase=str(phases[source_frame]),nonfinite_counts=bad,read_only_diagnostic=True,physical_state_restores=0),indent=2))
                    raise RuntimeError('Nonfinite physical substep: '+str(bad))
                if frame == limit-1 and substep >= args.substeps-args.observe_substeps:
                    substep_positions.append(np.concatenate([
                        view.get_positions().detach().cpu().numpy().reshape(-1,3).copy() for view in cloth_views]))
                    substep_velocities.append(np.concatenate([
                        view.get_velocities().detach().cpu().numpy().reshape(-1,3).copy() for view in cloth_views]))
                    substep_times.append(frame/fps+(substep+1)/(fps*args.substeps))
                force = rigid_contact_view.get_net_contact_forces(1/(fps*args.substeps)).detach().cpu().numpy().copy()
                pair_force=pad_pair_view.get_contact_force_matrix(1/(fps*args.substeps)).detach().cpu().numpy()[0,0].copy()
                force_probe.sample(frame,substep,source_frame,str(phases[source_frame]),force,pair_force)
                force_probe.rows[-1]['left_pad_pair_force']=left_pad_pair_view.get_contact_force_matrix(1/(fps*args.substeps)).detach().cpu().numpy()[0,0].copy()
                external_force=force.copy()
                external_force[pad_sensor_indices[0]]-=pair_force
                external_force[pad_sensor_indices[1]]+=pair_force
                if frame_external_force is None:frame_external_force=external_force
                else:
                    replace_external=np.linalg.norm(external_force,axis=-1)>np.linalg.norm(frame_external_force,axis=-1)
                    frame_external_force[replace_external]=external_force[replace_external]
                if np.linalg.norm(pair_force)>np.linalg.norm(frame_pad_pair_force):frame_pad_pair_force=pair_force

                if frame_force is None:
                    frame_force = force
                else:
                    replace = np.linalg.norm(force, axis=-1) > np.linalg.norm(frame_force, axis=-1)
                    frame_force[replace] = force[replace]
            if not (args.headless_smoke or args.headless_command_replay):
                sim.render()
                time.sleep(max(0.,1/fps-(time.monotonic()-frame_wall)))
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
            external_contact_forces.append(frame_external_force)
            pad_pair_contact_forces.append(frame_pad_pair_force)
            record.append((cloth.copy(), joints, art.get_joint_velocities().detach().cpu().numpy().copy(), velocities.copy()))
            if frame==0 and not solver_initial_basket_check['valid']:diagnostic_warning('Solver startup cloth-basket penetration; see solver_initial_basket_clearance.json')
            if frame%20==0:
                self_row=audit_self_intersections(cloth,faces);self_row.update(frame=int(frame),source_frame=int(source_frame),phase=str(phases[source_frame]));self_intersection_rows.append(self_row)
                (args.output/'live_self_intersections.json').write_text(json.dumps(self_intersection_rows,indent=2))
                if self_row['pairs_over_50micron']:diagnostic_warning('Cloth self-intersection detected; diagnostic remains invalid: '+str(self_row['pairs_over_50micron']))
            basket_check=basket_guard.check(cloth)
            if frame%20==0 or not basket_check['valid']:basket_rows.append(dict(frame=frame,source_frame=source_frame,**basket_check))
            if not basket_check['valid']:diagnostic_warning('Physical cloth-basket penetration: '+str(basket_check))
            if args.natural_arm_bend_limit_deg is not None and source_frame>=700:
                bend_names = KinematicRobot.arm_names('R')
                bends = np.rad2deg(np.abs(joints[[names.index(bend_names[k]) for k in (3, 5)]]))
                if np.max(bends) > args.natural_arm_bend_limit_deg:
                    diagnostic_warning(f'Actual elbow/wrist bend exceeded limit at frame {frame}: {bends.tolist()}')
            if corner_feedback is not None and corner_feedback.get('online_replanned',False) and str(phases[source_frame]) in ('R_straight_mouth_insertion','R_immediate_physical_close'):
                actual_poses,actual_axes=corner_kinematics.fk(dict(zip(names,map(float,joints))),True)
                hand=actual_poses['gripper_r_base_link']; locked=np.asarray(corner_feedback['locked_mouth_rotation'])
                rotation_error=float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(locked.T@hand[:3,:3]).as_rotvec())))
                camera_outward=float(-(hand[:3,:3]@np.array([.07593,-.009,.021]))[1])
                if abs(float(np.rad2deg(np.arctan2(hand[1,2],hand[0,2]))))>1:diagnostic_warning(f'Actual forearm yaw not parallel to basket long edge at {frame}')
                if rotation_error>2:diagnostic_warning(f'Actual straight-insertion orientation drift {rotation_error} deg at {frame}')
                if camera_outward>0:diagnostic_warning(f'Unexpected outward wrist flip at {frame}')
                if source_frame in (400,600,700,750,799):
                    observed_index=int(corner_indices[np.linalg.norm(uv[corner_indices]-np.asarray(corner_feedback['selected_material_uv_m']),axis=1).argmin()])
                    corner_feedback['actual_corner_observations'].append(dict(frame=frame,xyz_m=cloth[observed_index].tolist()))
                if str(phases[source_frame])=='R_straight_mouth_insertion':
                    selected=int(corner_indices[np.linalg.norm(uv[corner_indices]-np.asarray(corner_feedback['selected_material_uv_m']),axis=1).argmin()])
                    local=(np.linalg.inv(hand)@np.r_[cloth[selected],1.])[:3]
                    pads=[(np.linalg.inv(hand)@actual_poses['gripper_r_'+side+'_link4']@np.array([.03285,0,0,1.]))[:3] for side in ('inner','outer')]
                    center=(pads[0]+pads[1])/2
                    contains=bool(abs(local[0]-center[0])<.012 and min(t[1] for t in pads)+.003<local[1]<max(t[1] for t in pads)-.003 and abs(local[2]-center[2])<.006)
                    corner_feedback['preclose_actual_corner_in_open_fingers']=contains
                    corner_feedback['preclose_corner_local']=local.tolist()
                    cloth_local=(cloth[corner_indices]-hand[:3,3])@hand[:3,:3]
                    # Deep insertion: inspect the entire actual pad span, not a narrow box at the target corner.
                    native_pad=np.load('/home/aether/workspace/agent_bench/.codex-tmp/tablecloth-action-loop-20260905/robot_native_collision.npz')
                    pad_bounds=[]
                    for side in ('inner','outer'):
                        points=native_pad['points_gripper_r_'+side+'_link4']
                        transform=np.linalg.inv(hand)@actual_poses['gripper_r_'+side+'_link4']
                        points=points@transform[:3,:3].T+transform[:3,3]
                        pad_bounds.append((points.min(0),points.max(0)))
                    native_pad.close()
                    xmin=max(b[0][0] for b in pad_bounds);xmax=min(b[1][0] for b in pad_bounds)
                    zmin=max(b[0][2] for b in pad_bounds);zmax=min(b[1][2] for b in pad_bounds)
                    ymin=pad_bounds[0][1][1];ymax=pad_bounds[1][0][1]
                    in_corridor=(cloth_local[:,0]>xmin)&(cloth_local[:,0]<xmax)&(cloth_local[:,2]>zmin)&(cloth_local[:,2]<zmax)&(cloth_local[:,1]>ymin)&(cloth_local[:,1]<ymax)
                    corner_feedback['preclose_depth_target_reached']=contains
                    corner_feedback['preclose_actual_corner_in_open_fingers']=bool(xmin<local[0]<xmax and ymin<local[1]<ymax and zmin<local[2]<zmax)
                    corner_feedback['preclose_actual_pad_span_hand_z_m']=[float(zmin),float(zmax)]
                    corner_feedback['preclose_actual_pad_corridor_vertex_count']=int(in_corridor.sum())
                    opposite=in_corridor&(uv[corner_indices,1]>.303)
                    corner_feedback['preclose_opposite_layer_vertices_in_corridor']=int(opposite.sum())
                    corner_feedback['preclose_captured_material_uv']=uv[corner_indices[in_corridor]].tolist()
                    if opposite.any():diagnostic_warning('Opposite folded layer entered shallow pinch corridor')

                    corner_patch=in_corridor&(uv[corner_indices,0]>.31)&(uv[corner_indices,1]<.06)
                    corner_feedback['preclose_single_corner_patch_vertices']=int(corner_patch.sum())
                    full_stroke_contact=bool(source_frame==799 and corner_patch.sum()>=6 and not opposite.any())
                    corner_feedback['closure_on_full_stroke_actual_pad_containment']=full_stroke_contact
                    # The additional commanded insertion may push a free corner. Use actual mesh containment
                    # at the completed stroke, never pretend the moving corner reached the nominal depth.
                    if full_stroke_contact: contains=True
                    if False: # Full-motion diagnostic uses the complete preplanned tail, without state grafts.
                        import contextlib
                        names_action=action['joint_names'].tolist()
                        live_values=dict(zip(names_action,map(float,action['joint_target'][source_frame])))
                        # Freeze arm command at insertion end; closing fingers alone changes pad geometry.
                        predicted_values=corner_kinematics.calibrate_loops(live_values|{'idx71_gripper_r_inner_joint1':-.012,'idx81_gripper_r_outer_joint1':.012})
                        predicted_poses=corner_kinematics.fk(predicted_values)
                        predicted_mid=sum((predicted_poses['gripper_r_'+side+'_link4']@np.array([.03285,0,0,1.]))[:3] for side in ('inner','outer'))/2
                        eligible=corner_indices[corner_patch]
                        deeper_selected=int(eligible[np.linalg.norm(cloth[eligible]-predicted_mid,axis=1).argmin()])
                        corner_feedback['tracking_point_source']='Chosen once before closure from actual single-layer open-pad corridor, nearest predicted closed-pad center; fixed throughout subsequent closure and lift'
                        corner_feedback['approach_reference_material_uv_m']=corner_feedback['selected_material_uv_m']
                        corner_feedback['selected_material_uv_m']=uv[deeper_selected].tolist()
                        corner_feedback['predicted_closed_pad_mid_m']=predicted_mid.tolist()
                        corner_feedback['actual_deeper_patch_point_m']=cloth[deeper_selected].tolist()
                        corner_feedback['deeper_patch_distance_to_predicted_center_m']=float(np.linalg.norm(cloth[deeper_selected]-predicted_mid))
                        selected=deeper_selected
                        live_mid=(hand@np.r_[center,1.])[:3]
                        live_candidate=dict(values=live_values,edge=live_mid.tolist(),mouth_rotation=hand[:3,:3].tolist(),uv=corner_feedback['selected_material_uv_m'])
                        online_dir=args.output/'online_protocol'; live_input=online_dir/'live_close.json';live_path=online_dir/'live_close.action.npz'
                        live_input.write_text(json.dumps([live_candidate],indent=2))
                        saved_argv=sys.argv[:]
                        try:
                            with open(online_dir/'live_close_planning.log','w') as live_log,contextlib.redirect_stdout(live_log),contextlib.redirect_stderr(live_log):
                                sys.argv=['plan_short50_aligned_tail.py',str(live_input),str(live_path)]
                                runpy.run_path(str(Path(__file__).parent/'plan_short50_aligned_tail.py'),run_name='__main__')
                        finally:sys.argv=saved_argv
                        with np.load(live_path) as live:
                            if live['joint_names'].tolist()!=action['joint_names'].tolist():diagnostic_warning('Live closure joint order mismatch')
                            fixed=[i for i,n in enumerate(names_action) if '_arm_r_' not in n and '_gripper_r_' not in n]
                            if not np.array_equal(live['joint_target'][800:,fixed],action['joint_target'][800:,fixed]):diagnostic_warning('Live close moved non-right commands')
                            armcols=[names_action.index(n) for n in corner_arm_names]
                            jump=float(np.max(np.abs(live['joint_target'][800,armcols]-action['joint_target'][source_frame,armcols]))*fps)
                            if jump>1.5:diagnostic_warning(f'Live close transition rate {jump}')
                            action['joint_target'][800:]=live['joint_target'][800:];phases[800:]=live['phase'][800:]
                        corner_feedback.update(live_close_triggered=True,live_close_frame=frame,live_close_source_frame=source_frame,live_close_actual_local_error_m=(local-center).tolist(),live_close_corner_xyz_m=cloth[selected].tolist(),live_close_transition_rate_rad_s=jump,next_source_override=800,locked_mouth_rotation=hand[:3,:3].tolist(),physics_state_writes_during_live_close_planning=0)
                        print(json.dumps({'live_single_corner_close':corner_feedback}),flush=True)
                    if not contains and source_frame==799:
                        corner_feedback['physical_pick_failed']=True
                        corner_feedback['empty_hand_pose_review_only']=False
                        diagnostic_warning(f'Actual measured corner missed open fingers before closure: {local.tolist()}')
            if (adaptive_brush is not None or args.natural_arm_bend_limit_deg is not None) and str(phases[source_frame]).startswith('R_'):
                arm_contact=[(name,float(np.linalg.norm(frame_force[i]))) for i,name in enumerate(contact_sensors) if '/arm_' in name and np.linalg.norm(frame_force[i])>1.]
                gripper_contact=[(name,float(np.linalg.norm(frame_external_force[i]))) for i,name in enumerate(contact_sensors) if '/gripper_r_' in name and np.linalg.norm(frame_external_force[i])>1.]
                if gripper_contact:diagnostic_warning(f'Unexpected external rigid gripper contact at {frame}: {gripper_contact}')
                if arm_contact:diagnostic_warning(f'Unexpected rigid arm contact during right action at {frame}: {arm_contact}')
                if adaptive_brush is not None:
                    adaptive_brush.on_physics(source_frame,frame,cloth,body_records[-1],body_names)
            edge_ratio = np.linalg.norm(cloth[material_edges[:, 0]]-cloth[material_edges[:, 1]], axis=1)/material_rest
            max_edge_ratio_seen = max(max_edge_ratio_seen, float(edge_ratio.max()))
            if max_edge_ratio_seen > 1.15 and first_edge_limit_failure is None:
                first_edge_limit_failure = dict(frame=frame, ratio=max_edge_ratio_seen)
                if not args.probe_grasp_after_stretch:
                    diagnostic_warning(f'Physical cloth edge stretch exceeded prior 1.15 limit at frame {frame}: {max_edge_ratio_seen:.6f}')
            if max_edge_ratio_seen > 2.0:
                diagnostic_warning(f'Cloth diverged during rejected grasp probe at frame {frame}: {max_edge_ratio_seen:.6f}')
            if frame % 20 == 0:
                print(json.dumps({'frame': frame, 'cloth_min_z': float(cloth[:,2].min()),
                    'joint_tracking_error_max': float(np.abs(joints[indices]-targets).max())}), flush=True)
            next_source = (adaptive_brush.on_frame(source_frame, frame, cloth)
                           if adaptive_brush is not None else corner_feedback.pop('next_source_override',source_frame+1) if corner_feedback is not None else source_frame+1)
            if next_source is None:
                failure = adaptive_brush.failure
                break
            if corner_feedback is not None and source_frame==829:
                closure_wait_frames += 1
                grip_ids=[names.index(n) for n in ('idx71_gripper_r_inner_joint1','idx81_gripper_r_outer_joint1')]
                grip_angle=np.abs(joints[grip_ids])
                grip_speed=np.abs(record[-1][2][grip_ids])
                # Native contacts cause small substep velocity chatter. Verify the recorded angle
                # remains stable over half a second, with the same net angular drift limit.
                closure_window=np.asarray([r[1][grip_ids] for r in record[-11:]])
                angle_span=np.ptp(closure_window,axis=0)
                drift=np.abs(closure_window[-1]-closure_window[0])/(max(1,len(closure_window)-1)/fps)
                settled_closed=bool(len(closure_window)==11 and np.max(grip_angle)<=.0175 and np.max(angle_span)<=.001 and np.max(drift)<=.002)
                closure_stable_frames=closure_stable_frames+1 if settled_closed else 0
                geometry=cooked_guard.check(cloth[selected],body_records[-1],body_names)
                if force_sweep_frames>=180 and not geometry['geometry_allows_lift'] and force_sweep_frames-tighter_last_adjust>=20:
                    if tighter_target<.01175:
                        tighter_target=min(.01175,tighter_target+.00025);tighter_last_adjust=force_sweep_frames
                        print(json.dumps({'tighter_closure_backoff':dict(frame=frame,target_rad=float(full_target[command_names.index("idx81_gripper_r_outer_joint1")]),geometry=geometry)}),flush=True)
                if frame%20==0:tighter_geometry_rows.append(dict(frame=frame,phase='closure_hold',target_rad=float(opening),geometry=geometry))
                ready=force_sweep_frames>=260 and closure_stable_frames>=10 and force_sweep_frames-tighter_last_adjust>=60 and geometry['geometry_allows_lift']
                corner_feedback['physical_closure_gate']=dict(frame=frame,wait_frames=closure_wait_frames,actual_abs_angle_rad=grip_angle.tolist(),actual_abs_speed_rad_s=grip_speed.tolist(),closed_command_rad=float(opening),angle_limit_rad=.0175,speed_limit_rad_s=.002,stable_frames=closure_stable_frames,angle_span_last_half_second_rad=angle_span.tolist(),net_drift_rad_s=drift.tolist(),ready=ready)
                if closure_wait_frames>=260:
                    if not ready:diagnostic_warning('Closure verification failed after13s; continuing full-motion review by user request')
                    corner_feedback['diagnostic_continued_without_verified_grip']=not ready
                    geometry=cooked_guard.check(cloth[selected],body_records[-1],body_names)
                    corner_feedback['cooked_closure_guard']=geometry
                    if not geometry['geometry_allows_lift']:
                        corner_feedback['physical_closure_gate']['ready']=False
                        diagnostic_warning('Cooked pinch gap or selected-corner containment invalid; lift withheld: '+str(geometry))
                    print(json.dumps({'physical_closure_before_lift':corner_feedback['physical_closure_gate']}),flush=True)
                elif closure_wait_frames>=460:
                    diagnostic_warning('Gripper did not reach stable physical closure within force-sweep diagnostic time budget; lift withheld')
                else:
                    next_source=829
            if corner_feedback is not None and source_frame==399:
                stable_ids=corner_indices[(uv[corner_indices,0]>=.28)&(uv[corner_indices,1]<=.06)]
                recent=np.asarray([r[0][stable_ids] for r in record[-100:]])
                motion=float(np.linalg.norm(recent-recent[-1],axis=-1).max())
                if frame%100==0:
                    print(json.dumps({'settling_patch_motion':dict(frame=frame,last5s_max_m=motion)}),flush=True)
                    np.savez_compressed(args.output/'settle_monitor.npz',cloth_id=cloth_ids,uv=uv,cloth_q=np.asarray([cloth]),frame=frame)
                if frame<799:
                    if frame>=1599:diagnostic_warning(f'Cloth corner still moving after 80 seconds: {motion}')
                    next_source=399
                else:
                    corner_feedback['actual_settle_duration_s']=(frame+1)/fps
                    corner_feedback['all_near_corner_last5s_motion_m']=motion
            if str(phases[source_frame]) in ('R_lift_outside_clear','R_extend_to_eye_height','R_eye_height_display'):
                pad_distances=[]
                for side in ('inner','outer'):
                    bname='gripper_r_'+side+'_link4'
                    body=body_records[-1][body_names.index(bname)]
                    local=(cloth-body[:3])@Rotation.from_quat(body[3:]).as_matrix()
                    pad_distances.append(cKDTree(cached_pads['points_'+bname]).query(local)[0])
                both=(pad_distances[0]<.005)&(pad_distances[1]<.005)
                patch=both&(uv[:,0]>.28)&(uv[:,1]<.06)
                lost_grip_frames=lost_grip_frames+1 if not patch.any() else 0
                grip_row=dict(frame=frame,phase=str(phases[source_frame]),same_corner_bilateral_count=int(patch.sum()),opposite_panel_bilateral_count=int((both&(uv[:,1]>.303)).sum()),max_edge_ratio=float(edge_ratio.max()))
                live_grip_rows.append(grip_row)
                if frame%20==0:print(json.dumps({'live_grip':grip_row}),flush=True)
                if lost_grip_frames>=10:diagnostic_warning('Same-corner bilateral proximity lost for 10 lift frames; stop empty-hand continuation')
            if corner_feedback is not None and source_frame>=830 and not bimanual.released:
                geometry=cooked_guard.check(cloth[selected],body_records[-1],body_names)
                if frame%20==0:tighter_geometry_rows.append(dict(frame=frame,phase=str(phases[source_frame]),target_rad=float(full_target[command_names.index("idx81_gripper_r_outer_joint1")]),geometry=geometry))
                if not geometry['geometry_allows_lift']:
                    tighter_geometry_rows.append(dict(frame=frame,phase=str(phases[source_frame]),target_rad=float(full_target[command_names.index("idx81_gripper_r_outer_joint1")]),geometry=geometry,stop=True))
                    diagnostic_warning('Selected cloth corner left valid cooked pinch gap during lift: '+str(geometry))
            bimanual.observe(frame,str(phases[source_frame]),dict(zip(names,map(float,joints))),cloth,uv)
            if next_source>=count:
                current_command=dict(zip(command_names,map(float,commanded[-1])))
                observed_values=dict(zip(names,map(float,joints)))
                continuation=bimanual.append_at_end(current_command,observed_values,cloth,uv,record)
                if continuation:
                    if continuation_start_frame is None:continuation_start_frame=count
                    new_rows=[]
                    for values,phase_name in continuation:
                        new_rows.append([values[n] for n in command_names])
                    action['joint_target']=np.vstack([action['joint_target'],np.asarray(new_rows)])
                    phases=np.r_[phases,np.asarray([x[1] for x in continuation],dtype='U80')]
                    count=len(phases);action['time_s']=np.arange(count)/fps
                    print(json.dumps({'bimanual_extension':bimanual.stage,'count':count,'frame':frame}),flush=True)
                elif bimanual.failure:
                    diagnostic_warning(bimanual.failure)
            source_frame = next_source
        if record and np.max(np.linalg.norm(record[-1][0] - initial, axis=1)) < 1e-6:
            raise RuntimeError('Cloth did not respond physically; particle simulation/readback may be inactive')
        if failure is None and source_frame < count:
            failure = 'Explicit diagnostic frame limit; incomplete episode'
    except BaseException as exc:
        failure = repr(exc)
        raise
    finally:
        bimanual.save()
        (args.output/'diagnostic_continuation_events.json').write_text(json.dumps(diagnostic_events,indent=2)+'\n')
        (args.output/'basket_wall_readback.json').write_text(json.dumps(basket_rows,indent=2)+'\n')
        force_probe.save()
        (args.output/'tighter_closure_geometry.json').write_text(json.dumps(tighter_geometry_rows,indent=2)+'\n')
        (args.output/'live_grip_audit.json').write_text(json.dumps(live_grip_rows,indent=2)+'\n')
        (args.output/'strain_relaxation_hold.json').write_text(json.dumps(strain_hold_rows,indent=2)+'\n')
        if substep_positions:
            np.savez_compressed(args.output/'substep_observations.npz',
                cloth_q=np.stack(substep_positions), cloth_qd=np.stack(substep_velocities),
                time_s=np.asarray(substep_times), cloth_id=cloth_ids)
        (args.output/'teleop_events.json').write_text(json.dumps(control.events,indent=2))
        input_mode=('native_feedback_brush_joint_drive' if adaptive_brush is not None else
                    'native_drive_command_replay' if args.headless_command_replay else
                    'hold_only_smoke' if args.headless_smoke else 'keyboard_joint_jog')
        (args.output/'teleop_timing.json').write_text(json.dumps(dict(wall_seconds=time.monotonic()-wall_start,simulation_seconds=len(record)/fps,frames=len(record),input=input_mode,native_joint_drives=True,material_calibrated=False),indent=2))
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
                particle_mass=np.concatenate(particle_masses),
                body_q=np.stack(body_records), body_names=body_names, robot_root_pose_xyzw=np.stack(root_records),
                rigid_contact_force_peak=np.stack(contact_forces), rigid_contact_sensor_paths=contact_sensors,
                rigid_external_contact_force_peak=np.stack(external_contact_forces), right_pad_pair_contact_force_peak=np.stack(pad_pair_contact_forces),
                benchmark_task_id="8.10",reference_benchmark_task_id=meta["benchmark_task_id"],
                time_s=(np.arange(len(record))+1)/fps,
                phase=np.asarray(executed_phases[:len(record)]),
                command_source_frame=np.asarray(command_source_frames[:len(record)]))
            np.savez_compressed(args.output/'action.npz',
                joint_target=np.asarray(commanded[:len(record)]), joint_names=action['joint_names'],
                time_s=np.arange(len(record))/fps, fps=fps,
                control_mode='native_joint_drive_targets', command_joint_names=driven,
                command_source_frame=np.asarray(command_source_frames[:len(record)]))
        if failure is None and diagnostic_events:failure='Full motion completed with recorded failed physical metrics; see diagnostic_continuation_events.json'
        now = {str(a.GetPath()): str(a.Get()) for p in Usd.PrimRange(robot)
               for a in p.GetAttributes() if str(a.GetPath()) in native}
        report = dict(diagnostic_full_motion_user_authorized=True, diagnostic_metric_events=len(diagnostic_events), production_count=0, motion_completed=bool(source_frame>=count), physical_success=False, source=source_record(), reference=str(args.reference), frames=len(record),
            benchmark_task_id='8.10',reference_benchmark_task_id=meta['benchmark_task_id'], fps=fps, scenario_hidden_prims=[],
            collision_inventory='collision_inventory.json', robot_self_collision_enabled=self_collision,
            rigid_contacts='rigid_contacts.json',
            rigid_force_signal_observed=bool(contact_forces and np.any(np.asarray(contact_forces) != 0)),
            rigid_force_signal_is_grasp_validation=False,
            initial_collision_filter_repairs=collision_repairs,
            diagnostic_table_collider_candidate=table_collider_candidate,
            diagnostic_basket_collider_candidate=basket_collider_candidate,
            diagnostic_only=True, eligible_for_production=False,
            solver='PhysX FEM surface and native articulation' if surface else 'PhysX particle cloth and native articulation',
            cloth_solver_exception='GPU FEM surface with flat rest shape; calibration pending' if surface else
                'GPU particle cloth with explicit flat-rest springs; material calibration pending',
            particle_mass_source='authored_total_and_rest_area_estimates' if surface else 'native_GPU_tensors',
            material_models=materials,
            initial_setup_correction=dict(cloth_id=1, translation_m=initial_offset.tolist(),
                shape=initial_shape_record, applied_before_episode=True, randomized=False),
            initial_joint_error_max_rad=initial_joint_error,
            initial_joint_error_within_native_gate=initial_joint_error <= .02,
            cloth_dynamic_friction=args.surface_friction if surface else cloth_friction,
            cloth_dynamic_friction_requested=cloth_friction,
            dirty_initial_setup_correction=dict(cloth_id=0, translation_m=dirty_initial_offset.tolist(),
                applied_before_episode=True, randomized=False),
            robot_parameters_unchanged=now == native,user_requested_friction_override=grip_friction_override,native_robot_changes={k:dict(before=native.get(k),after=v)for k,v in now.items()if v!=native.get(k)},
            robot_source_world_anchor_removed_for_diagnostic=None,
            attachment_count=0,
            contact_classification=dict(method="native pair-wise contact force matrix each physics substep", allowed_pair=["gripper_r_inner_link4","gripper_r_outer_link4"], external_gripper_force_limit_N=1.0, raw_forces_preserved=True),
            robot_dynamic_body_count=len(robot_bodies), robot_kinematic_body_paths=kinematic_bodies,
            robot_world_fixed_joint_paths=world_fixed_joints,
            shared_robot_collision_profile=shared_robot_profile,
            physics_clock=dict(source_hz=source_frequency, actual_hz=fps*args.substeps,
                matches_source=fps*args.substeps == source_frequency,
                diagnostic_override_requested=False,
                initialization_elapsed_s=solver_warmup_time),
            post_start_pose_or_velocity_writes=False, cloth_masses_kg=masses,
            particle_mass_min=float(np.concatenate(particle_masses).min()), particle_state_readback="native GPU tensors",
            max_edge_ratio_seen=max_edge_ratio_seen, edge_ratio_diagnostic_limit=1.15,
            first_edge_limit_failure=first_edge_limit_failure,
            rejected_stretch_continuation_for_grasp_probe=args.probe_grasp_after_stretch,
            adaptive_brush=None if adaptive_brush is None else adaptive_brush.report(),
            adaptive_clean_corner=corner_feedback,bimanual_stage=bimanual.stage,bimanual_motion_completed=bimanual.complete,
            driven_joint_names=driven, failure=failure, owner_approved=False,
            command_interval='reference frame k command applied over [k/fps, (k+1)/fps]',
            config={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()})
        (args.output/'trial.json').write_text(json.dumps(report,indent=2)+'\n')
        sim.stop()


if __name__ == '__main__':
    main()
