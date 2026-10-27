#!/usr/bin/env python3
"""Deterministic shelf layouts for the bounded omni-PICK first stage.

CPU scene preparation and conservative full-outline layout checks only. The
manifest deliberately leaves physical, camera and trajectory acceptance open.
Run with the mesh311 Python environment. No simulator or GPU is started here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils
from scipy.spatial.transform import Rotation
from collision_scene import Body, Robot, STOCK, ROBOT, matrix, mesh_parts, points, pose, rigid_part, shape_mesh, traversal
from shapely.geometry import MultiPoint, Polygon, box
from shapely.ops import unary_union
from collision_initial_layout import _proxies, _separated
from validate_robot_initial_randomization import initial, recover_recorded_nominal, trace_nominal

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work/omni_pick_stage1_20260914/randomization"
BASELINE = ROOT / "assets/shelf_pick/current/scene.usda"
ORIGINAL_SHELF = Path("/mnt/agent-bench-data/lingjun/workspace/asset_imports/shelf_w120_from202_20260914.0QI4tz")
SHELF = ROOT / "work/omni_pick_stage1_20260914/assets/shelf_w120"
SHELF_PRIM = "/World/Assets/ShelvingUnit"
MESH_ROOT = SHELF_PRIM + "/o/Meshes/Sketchfab_model/root/GLTF_SceneRootNode"
BOARD_MESH = MESH_ROOT + "/shelf_boards_0/Object_4/Object_0"
UPRIGHTS = [MESH_ROOT + "/vertical_beams_2/Object_8/Object_2", MESH_ROOT + "/vertical_beams_2/Object_9/Object_3"]
SUPPORTS = [("bottom", SHELF_PRIM + "/BoardSupports/Bottom"),
            ("tier2", "/World/Assets/Tier2Overlay"),
            ("tier3", "/World/Assets/Tier3Overlay"),
            ("top", SHELF_PRIM + "/BoardSupports/Top")]
TASKS = {
    "benchmark_3_1": dict(target="/World/Assets/ShelfItems/cayman_jacked", shape="cylinder", support="tier3", source="episode_70004", label="Cayman Jacked Margarita", family="cayman_jacked_margarita"),
    "benchmark_3_21": dict(target="/World/Assets/Tier2Items/pring_bbq", shape="cylinder", support="tier2", source="episode_3128", label="Pringles BBQ", family="pringles_bbq"),
    "benchmark_3_25": dict(target="/World/Assets/Tier2Items/honeymaid", shape="box", support="tier2", source="episode_014", label="Honey Maid S'Mores Handi-Snacks carton", family="honey_maid_handi_snacks", identity_caveat="Sheet says peanut-butter crackers; mapped scene asset label is Honey Maid S'Mores. Both label and six-face carton mesh were inspected."),
    "benchmark_3_27": dict(target="/World/Assets/Tier2Items/zbar_t", shape="box", support="tier2", source="episode_1058", label="CLIF Zbar Protein Cookies 'n Creme carton", family="clif_zbar_protein"),
}
# Use genuine supplied products with simple stable bottoms, excluding each
# episode's target identity. Every instance retains its baseline physics.
POOL_NAMES = {"cayman_jacked", "coke_c3", "gatorade_m", "grapefruit_can", "starbucks", "good2grow", "pring_bbq", "honeymaid", "zbar_t", "skit_littles", "oreo_gf"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def set_pose(stage, path, xyz, quat):
    prim = stage.GetPrimAtPath(path)
    if not prim:
        raise ValueError("Missing pose root: " + path)
    # One full matrix removes ambiguity from inherited transform operations.
    x = UsdGeom.Xformable(prim)
    x.ClearXformOpOrder()
    op = x.GetPrim().GetAttribute("xformOp:transform")
    if not op:
        op = x.AddTransformOp().GetAttr()
    parent_world = matrix(UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetParent()))
    local = np.linalg.inv(parent_world) @ pose(xyz, quat)
    op.Set(Gf.Matrix4d(local.T.tolist()))
    x.SetXformOpOrder([UsdGeom.XformOp(op)])


def _world_mesh(stage, path):
    prim = stage.GetPrimAtPath(path)
    mesh = shape_mesh(prim)
    if mesh is None:
        raise ValueError("Missing mesh at " + path)
    mesh.remove_unreferenced_vertices()
    mesh.apply_transform(matrix(UsdGeom.XformCache().GetLocalToWorldTransform(prim)))
    return mesh


def snapshot_baseline(out):
    """Copy only the private layer; resolve dependencies back to their source."""
    # Resolve current once so saved episodes retain their selected release.
    source = Sdf.Layer.FindOrOpen(str(BASELINE.resolve(strict=True)))
    dest = Sdf.Layer.CreateNew(str(out / "baseline_snapshot.usda"))
    dest.TransferContent(source)
    UsdUtils.ModifyAssetPaths(dest, lambda p: Sdf.ComputeAssetPathRelativeToLayer(source, p))
    dest.Save()
    return Path(dest.realPath)


def ensure_mounted_shelf_bundle():
    """Copy the exact supplied bundle under the existing project bind mount."""
    SHELF.mkdir(parents=True,exist_ok=True)
    copied=[]
    for source in sorted(ORIGINAL_SHELF.iterdir()):
        if not source.is_file():continue
        destination=SHELF/source.name
        if destination.exists() and digest(destination)!=digest(source):
            raise ValueError('Preserving nonidentical project shelf asset: '+str(destination))
        if not destination.exists():shutil.copy2(source,destination)
        copied.append(dict(source=str(source),destination=str(destination),source_sha256=digest(source),
                           copied_sha256=digest(destination),byte_identical=source.read_bytes()==destination.read_bytes()))
    receipt=dict(schema='family3.exact_shelf_bundle_copy.v1',original_source=str(ORIGINAL_SHELF),
                 mounted_project_copy=str(SHELF),passed=bool(copied and all(x['byte_identical'] for x in copied)),files=copied,
                 reason='Original workspace/asset_imports path is not mounted in Isaac containers; USD silently fell back to the old shelf in physical002. Project paths are already mounted. Relative asset references are preserved unchanged.')
    write_json(SHELF/'BUNDLE_COPY.json',receipt)
    return receipt


def build_base_scene(out=OUT):
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_baseline(out)
    path = out / "scene_base.usda"
    stage = Usd.Stage.CreateNew(str(path))
    stage.GetRootLayer().subLayerPaths = [str(snapshot)]
    UsdGeom.SetStageMetersPerUnit(stage, 1.0); UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    # Imported collision patch authors an over at this parent. The private
    # baseline lacks its def, so child colliders otherwise remain undefined
    # and are omitted by runtime traversal despite collisionEnabled=true.
    UsdGeom.Xform.Define(stage, SHELF_PRIM + "/BoardSupports")
    # Preserve actual native mesh triangles; never convexify the whole shelf.
    for path_string in UPRIGHTS:
        prim = stage.GetPrimAtPath(path_string)
        if not prim or not prim.IsA(UsdGeom.Mesh):
            raise ValueError("Missing imported upright: " + path_string)
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")
    stage.GetRootLayer().Save()
    audit = audit_shelf(stage)
    selected = BASELINE.resolve(strict=True).parent
    audit.update(scene_path=str(path), baseline_source=str(BASELINE.resolve()),
                 baseline_snapshot=str(snapshot), asset_manifest=str(selected/'manifest.json'),
                 shelf_source=str(selected/'assets/shelf/asset.usda'), robot_asset=str(ROBOT))
    write_json(out / "ASSET_AUDIT.json", audit)
    if not audit["passed"]:
        raise ValueError("Imported shelf audit failed; see ASSET_AUDIT.json")
    return path, audit


def audit_shelf(stage):
    board_mesh = _world_mesh(stage, BOARD_MESH)
    triangles = board_mesh.triangles
    horizontal = np.ptp(triangles[:, :, 2], axis=1) < 1e-6
    supports = []
    for name, path in SUPPORTS:
        prim = stage.GetPrimAtPath(path)
        slab = _world_mesh(stage, path)
        top = float(slab.bounds[1, 2])
        faces = triangles[horizontal & (np.abs(triangles[:, :, 2].mean(1) - top) < 1e-5)]
        visible = unary_union([Polygon(t[:, :2]) for t in faces if Polygon(t[:, :2]).area > 1e-12])
        collision_fp = MultiPoint(slab.vertices[:, :2]).convex_hull
        match = not visible.is_empty and visible.symmetric_difference(collision_fp).area < 2e-6
        # The supplied boards have beveled edges: top triangles are inset
        # 0.356/0.458 mm from the full slab. Use their intersection for support
        # instead of pretending these two polygons are exactly identical.
        usable = not visible.is_empty and collision_fp.buffer(1e-7).covers(visible)
        usable_polygon = visible.intersection(collision_fp)
        supports.append(dict(id=name, prim_path=path, top_z_m=top,
                             bottom_z_m=float(slab.bounds[0, 2]),
                             polygon_xy_m=list(map(list, usable_polygon.convex_hull.exterior.coords[:-1])),
                             collision_polygon_xy_m=list(map(list, collision_fp.exterior.coords[:-1])),
                             bounds_world_m=slab.bounds.tolist(),
                             visible_collision_match=bool(match),
                             support_geometry_usable=bool(usable),
                             active_defined_loaded=bool(prim.IsActive() and prim.IsDefined() and prim.IsLoaded()),
                             visible_collision_boundary_difference_m=float(visible.hausdorff_distance(collision_fp)),
                             enabled=bool(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()),
                             visible_top_triangles=int(len(faces))))
    upright_reports = []
    for path in UPRIGHTS:
        prim = stage.GetPrimAtPath(path); mesh = _world_mesh(stage, path)
        # Connected components are reported separately; these two large meshes
        # contain all four physical corner posts and smaller fittings.
        clean = mesh.copy(); clean.merge_vertices()
        components = list(clean.split(only_watertight=False))
        tall = [m.bounds.tolist() for m in components if np.ptp(m.vertices[:, 2]) > .30]
        upright_reports.append(dict(path=path, enabled=bool(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()),
                                    active_defined_loaded=bool(prim.IsActive() and prim.IsDefined() and prim.IsLoaded()),
                                    approximation=UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get(),
                                    vertices=len(mesh.vertices), triangles=len(mesh.faces), bounds_world_m=mesh.bounds.tolist(),
                                    connected_components=len(components), post_segment_bounds_world_m=tall))
    shelf_meshes = [str(p.GetPath()) for p in traversal(stage.GetPrimAtPath(SHELF_PRIM)) if p.IsA(UsdGeom.Mesh) and UsdGeom.Imageable(p).ComputeVisibility() != "invisible"]
    full = np.concatenate([_world_mesh(stage, p).vertices for p in shelf_meshes])
    # Corner clustering identifies four posts even though each consists of
    # several disconnected height segments and small fittings.
    corners = {}
    for upright in upright_reports:
        for b in upright["post_segment_bounds_world_m"]:
            b = np.asarray(b); center = b.mean(0)
            key = ("left" if center[0] < -2.369 else "right") + "_" + ("front" if center[1] < 2.253 else "back")
            if key not in corners: corners[key] = b
            else: corners[key] = np.array([np.minimum(corners[key][0], b[0]), np.maximum(corners[key][1], b[1])])
    passed = all(s["enabled"] and s["support_geometry_usable"] and s["active_defined_loaded"] for s in supports) and all(u["enabled"] and u["active_defined_loaded"] and u["approximation"] == "none" for u in upright_reports)
    return dict(schema="family3.omni_shelf_audit.v1", passed=bool(passed),
                shelf_full_bounds_world_m=[full.min(0).tolist(), full.max(0).tolist()],
                supports=supports, upright_mesh_prim_count=len(upright_reports), uprights=upright_reports,
                physical_corner_post_count=len(corners), physical_corner_post_bounds_world_m={k:v.tolist() for k,v in corners.items()},
                note="Upright mesh count is not physical post count. Asset and layout checks do not establish PhysX runtime acceptance.")


def inspect_catalog(stage):
    result = {}; excluded = []
    for path in STOCK:
        prim = stage.GetPrimAtPath(path)
        if not prim or prim.GetName() not in POOL_NAMES:
            continue
        prim.SetActive(True)
        cache = UsdGeom.XformCache()
        world = rigid_part(matrix(cache.GetLocalToWorldTransform(prim)))
        parts = mesh_parts(prim, cache, reference=world)
        visual = trimesh.util.concatenate([m for _, m in parts])
        if not len(visual.vertices):
            raise ValueError("No visible product geometry: " + path)
        proxies = _proxies(stage, prim, cache)
        if not proxies:
            excluded.append(dict(path=path, reason="No inherited enabled product collider; excluded from first-stage distractor pool."))
            continue
        local_proxies = []
        for proxy in proxies:
            m = proxy.mesh.copy(); m.apply_transform(np.linalg.inv(world))
            local_proxies.append((proxy.name.removesuffix("#proxy"), m))
        all_points = np.concatenate([visual.vertices] + [m.vertices for _, m in local_proxies])
        references = [r for spec in prim.GetPrimStack() for r in spec.referenceList.GetAppliedItems() if r.assetPath]
        # The baseline product reference resolves from its own original layer.
        resolved = None
        for spec in prim.GetPrimStack():
            for ref in spec.referenceList.GetAppliedItems():
                if ref.assetPath and "newcans/" in ref.assetPath:
                    resolved = Sdf.ComputeAssetPathRelativeToLayer(spec.layer, ref.assetPath)
                    break
            if resolved: break
        if not resolved or not Path(resolved).is_file():
            raise ValueError("Unresolved real product asset: " + path)
        family = Path(resolved).parent.name
        textures = {}
        for child in traversal(prim):
            if child.IsA(UsdShade.Shader):
                for attr in child.GetAttributes():
                    if attr.GetTypeName() == Sdf.ValueTypeNames.Asset:
                        value = attr.Get()
                        if value and value.resolvedPath:
                            textures[value.resolvedPath] = digest(value.resolvedPath)
        result[path] = dict(asset=resolved, asset_sha256=digest(resolved), product_family=family,
                            visual=visual, proxies=local_proxies,
                            vertices=all_points, bounds_local_m=[all_points.min(0).tolist(), all_points.max(0).tolist()],
                            visual_bounds_local_m=visual.bounds.tolist(), texture_sha256=textures,
                            mass_kg=float(UsdPhysics.MassAPI(prim).GetMassAttr().Get()),
                            initial_rigid=bool(prim.HasAPI(UsdPhysics.RigidBodyAPI)),
                            source_root_world=world)
    write_json(OUT / "POOL_EXCLUSIONS.json", excluded)
    return result


def recover_verified_command_nominal(task, trace):
    """Recover intended arms/fixed posture from mutually verified commands.

    Keep the established base reference, but never call a deformed physical
    source arm state the canonical command. Only initial arrays are consumed.
    """
    canonical=recover_recorded_nominal(task,trace)
    if not canonical['passed']:return canonical
    source=Path(trace['nominal_episode'])
    metadata=json.loads((source/'inject_meta.json').read_text())
    with np.load(source/'action.npz',allow_pickle=False) as action, np.load(source/'recorded_playback_action.npz',allow_pickle=False) as playback:
        arm=np.asarray(action['arm'][0],float).copy()
        head=np.asarray(action['head'][0],float).copy();waist=np.asarray(action['waist'][0],float).copy()
        if (arm.shape!=(14,) or head.shape!=(2,) or waist.shape!=(2,)
                or not np.all(np.isfinite(np.r_[arm,head,waist]))
                or not np.array_equal(arm,playback['arm'][0])
                or not np.array_equal(arm,np.asarray(metadata['applied_arm_initial_state'],float))
                or not np.array_equal(head,playback['head'][0]) or not np.array_equal(waist,playback['waist'][0])):
            raise ValueError('Source initial action, recorded playback and applied-arm metadata disagree: '+str(source))
    delta=np.asarray(metadata['arm_delta'],float)
    if delta.shape!=(14,) or not np.all(np.isfinite(delta)) or not np.allclose(delta,np.radians(canonical['terminal_declared_draw']['arm_delta_deg']),rtol=0,atol=1e-15):
        raise ValueError('Verified command draw differs from canonical provenance: '+str(source))
    physical=initial(Path(trace['nominal_state']))
    indices=[physical['names'].index(n) for n in canonical['arm_dof_names']]
    fixed_names=['idx01_body_joint1','idx02_body_joint2','idx11_head_joint1','idx12_head_joint2']
    fixed_values=np.r_[waist[::-1],head]
    nominal=arm-delta
    canonical.update(method='verified source action arm[0] minus the same inject_meta arm_delta; fixed head/waist from source commands; established base reference retained',
                     source_command_verified=True,arm_joint_rad=nominal.tolist(),
                     fixed_command_dof_names=fixed_names,fixed_command_rad_or_m=fixed_values.tolist(),
                     source_command_audit=dict(source=str(source),action_sha256=digest(source/'action.npz'),
                         recorded_playback_sha256=digest(source/'recorded_playback_action.npz'),inject_meta_sha256=digest(source/'inject_meta.json'),
                         initial_arm_command=arm.tolist(),source_arm_delta=delta.tolist(),
                         physical_minus_command_arm_rad=(physical['joints'][indices]-arm).tolist(),
                         maximum_physical_minus_command_arm_deg=float(np.max(np.abs(np.degrees(physical['joints'][indices]-arm)))),
                         physical_minus_command_fixed_rad_or_m=(physical['joints'][[physical['names'].index(n) for n in fixed_names]]-fixed_values).tolist(),
                         initial_command_triple_agreement=True,source_motion_copied=False))
    return canonical


def canonical_robot(task, seed, provenance):
    trace = provenance[task]["trace"]; canonical = provenance[task]["canonical"]
    if not trace["passed"] or not canonical["passed"]:
        raise ValueError("Unproven canonical initial robot state")
    if not canonical.get('source_command_verified'):
        raise ValueError('Unverified physical-derived arm nominal; recover the mutually verified source command before generating a new episode')
    source = initial(Path(trace["nominal_state"]))
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 119]))
    delta_xy = rng.uniform(-.045, .045, 2)
    yaw_delta = float(rng.uniform(-4.5, 4.5))
    arm_delta = rng.uniform(-3.6, 3.6, 14)
    q = source["joints"].copy(); names = source["names"]
    arms = canonical["arm_dof_names"]
    arm_values = np.asarray(canonical["arm_joint_rad"]) + np.radians(arm_delta)
    for name, value in zip(arms, arm_values): q[names.index(name)] = value
    for name,value in zip(canonical['fixed_command_dof_names'],canonical['fixed_command_rad_or_m']):q[names.index(name)]=value
    root_quat = Rotation.from_quat(source["quat"][[1, 2, 3, 0]])
    # Undo oldest world-yaw draw before adding this episode's independent draw.
    oldest_yaw = canonical["terminal_declared_draw"]["base_yaw_deg"]
    rotation = Rotation.from_euler("z", yaw_delta-oldest_yaw, degrees=True) * root_quat
    base_xyz = np.asarray(canonical["base_xyz_m"]); base_xyz[:2] += delta_xy
    return dict(asset=str(ROBOT), canonical_source_episode=trace["nominal_episode"],
                canonical_provenance=str(OUT / "CANONICAL_PROVENANCE.json"),
                base_xyz_m=base_xyz.tolist(), base_wxyz=rotation.as_quat()[[3, 0, 1, 2]].tolist(),
                dof_names=names, joint_position_rad_or_m=q.tolist(), arm_dof_names=arms,
                arm_joint_rad=arm_values.tolist(),
                delta=dict(base_xy_m=delta_xy.tolist(), base_yaw_deg=yaw_delta, arm_joint_deg=arm_delta.tolist()),
                fixed_head_waist_names=[n for n in names if "head_joint" in n or "body_joint" in n])


def product_pose(path, product, xy, yaw_deg, support, role):
    quat = Rotation.from_euler("z", yaw_deg, degrees=True).as_quat()[[3, 0, 1, 2]]
    rotation = pose([0, 0, 0], quat)
    rotated = points(product["vertices"], rotation)
    xyz = [float(xy[0]), float(xy[1]), float(support["top_z_m"] + .0005 - rotated[:, 2].min())]
    world = pose(xyz, quat); all_points = points(product["vertices"], world)
    footprint = MultiPoint(all_points[:, :2]).convex_hull
    item = dict(role=role, prim_path=path, asset=product["asset"], asset_sha256=product["asset_sha256"],
                product_family=product["product_family"], xyz_m=xyz, wxyz=quat.tolist(),
                yaw_delta_deg=float(yaw_deg), support_id=support["id"],
                bounds_local_m=product["bounds_local_m"], visual_bounds_local_m=product["visual_bounds_local_m"],
                bounds_world_m=[all_points.min(0).tolist(), all_points.max(0).tolist()],
                footprint_world_xy_m=list(map(list, footprint.exterior.coords[:-1])),
                mass_kg=product["mass_kg"], collision_paths=[p for p, _ in product["proxies"]])
    bodies = [Body(path + "#visible", product["visual"], world)]
    bodies += [Body(p + "#proxy", m, world) for p, m in product["proxies"]]
    return item, bodies, footprint


def full_environment_bodies(stage):
    """Include invisible static proxies that the legacy Scene visual view skips."""
    cache = UsdGeom.XformCache(); bodies = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith("/World/Robot") or not path.startswith(("/World/Assets/", "/World/Background/")):
            continue
        owner = prim
        while owner and not owner.IsPseudoRoot() and not owner.HasAPI(UsdPhysics.RigidBodyAPI): owner = owner.GetParent()
        if owner and owner.HasAPI(UsdPhysics.RigidBodyAPI): continue
        shape = shape_mesh(prim)
        if shape is None: continue
        visible = UsdGeom.Imageable(prim).ComputeVisibility() != "invisible" and UsdGeom.Imageable(prim).ComputePurpose() not in ("proxy", "guide")
        collider = prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        if not (visible or collider): continue
        shape.apply_transform(matrix(cache.GetLocalToWorldTransform(prim)))
        bodies.append(Body(path, shape, metadata=dict(role="environment", visible=bool(visible), collision=bool(collider))))
    return bodies


def carton_nominal_heading(product,support):
    """Align the thin horizontal carton axis with the longest support axis."""
    bounds=np.asarray(product['visual_bounds_local_m']);thin=int(np.argmin(bounds[1,:2]-bounds[0,:2]))
    footprint=np.asarray(support['polygon_xy_m']);tangent=int(np.argmax(np.ptp(footprint,axis=0)))
    return float(((tangent-thin)*90)%180)


def sample_layout(task, seed, layout_index, base_scene, audit, catalog, robot, out, previous_layout=None):
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 271]))
    spec = TASKS[task]; supports = {s["id"]: s for s in audit["supports"]}
    support = supports[spec["support"]]; poly = Polygon(support["polygon_xy_m"])
    target_product = catalog[spec["target"]]
    # Front band is bounded by the real shelf front. Left/right stratification
    # makes the two canaries visibly different without moving fixed furniture.
    center_x = .5 * (poly.bounds[0] + poly.bounds[2])
    x_domain = [center_x-.20, center_x-.06] if layout_index % 2 == 0 else [center_x+.06, center_x+.20]
    y_domain = [poly.bounds[1]+.065, poly.bounds[1]+.13]
    target_yaw = float(rng.choice([-1, 1]) * rng.uniform(1, 3))
    nominal_heading=carton_nominal_heading(target_product,support) if spec['shape']=='box' else 0.
    xy = [rng.uniform(*x_domain), rng.uniform(*y_domain)]
    target, target_bodies, footprint = product_pose(spec["target"], target_product, xy, nominal_heading+target_yaw, support, "target")
    bounds = np.asarray(target["bounds_world_m"]); visual = np.asarray(target["visual_bounds_local_m"])
    target.update(shape=spec["shape"], label=spec["label"], identity_caveat=spec.get("identity_caveat"),
                  grasp_geometry=dict(radius_m=float(max(visual[1, :2]-visual[0, :2])/2), height_m=float(visual[1, 2]-visual[0, 2]), width_depth_m=(visual[1, :2]-visual[0, :2]).tolist()))
    if spec['shape']=='box':
        target.update(yaw_delta_deg=target_yaw,nominal_yaw_deg=nominal_heading,world_yaw_deg=nominal_heading+target_yaw,
                      heading_rule='Thin horizontal carton axis aligned with shelf tangent; fixed geometric nominal plus original small seeded yaw jitter.',
                      heading_evidence=str(OUT/'CARTON_HEADING_PROPOSAL.json'))
    # This reservation excludes distractors from a broad approach/lift/retreat
    # box. It does not certify arm geometry, reach, grasp or camera visibility.
    corridor_lo = [float(bounds[0, 0]-.16), float(poly.bounds[1]-.40), float(support["top_z_m"]+.001)]
    corridor_hi = [float(bounds[1, 0]+.16), float(bounds[1, 1]+.045), float(bounds[1, 2]+.155)]
    corridor = box(corridor_lo[0], corridor_lo[1], corridor_hi[0], corridor_hi[1])
    if not poly.buffer(-.018).covers(footprint):
        raise ValueError("Target full outline outside shelf margin")
    stage = Usd.Stage.Open(str(base_scene)); stage.SetEditTarget(stage.GetSessionLayer())
    for p in STOCK:
        prim = stage.GetPrimAtPath(p)
        if prim: prim.SetActive(False)
    fixtures = [b for b in full_environment_bodies(stage) if b.hi[0] > poly.bounds[0]-.3 and b.lo[0] < poly.bounds[2]+.3 and b.hi[1] > poly.bounds[1]-.5 and b.lo[1] < poly.bounds[3]+.2]
    robot_model = Robot(); robot_model.fk(robot["dof_names"], robot["joint_position_rad_or_m"], pose(robot["base_xyz_m"], robot["base_wxyz"]))
    # Initial robot intersections are reported to policy/root; reject item
    # placement next to an arm, but do not change the canonical posture.
    def fixture_clear(bodies, s):
        for body in bodies:
            for env in fixtures:
                if env.name == s["prim_path"]: continue
                if not _separated(body, env, .00015): return False
        return True
    if not fixture_clear(target_bodies, support):
        raise ValueError("Target intersects full shelf/environment geometry")
    items = [target]; placed = [(target, target_bodies, footprint)]
    choices = [p for p in catalog if catalog[p]["product_family"] != target_product["product_family"]]
    rng.shuffle(choices)
    # Two strata retain randomness and ensure paired item counts differ.
    count_domain = [6, 7] if layout_index % 2 == 0 else [8, 9]
    eligible_counts = [n for n in count_domain if n <= len(choices)]
    if not eligible_counts: raise ValueError("Insufficient real collidable products for count stratum")
    desired_count = int(rng.choice(eligible_counts))
    rejections = dict(support=0, corridor=0, overlap=0, fixture=0, robot=0)
    def permitted(item,bodies,fp,s):
        sp=Polygon(s['polygon_xy_m'])
        if not sp.buffer(-.014).covers(fp):rejections['support']+=1;return False
        ilo,ihi=np.asarray(item['bounds_world_m'])
        if ilo[2]<corridor_hi[2] and ihi[2]>corridor_lo[2] and fp.intersects(corridor):rejections['corridor']+=1;return False
        if any(not all(_separated(a,b,.012) for a in bodies for b in previous) for _,previous,_ in placed):rejections['overlap']+=1;return False
        if not fixture_clear(bodies,s):rejections['fixture']+=1;return False
        if any(not _separated(b,arm,.015) for b in bodies for arm in robot_model.bodies):rejections['robot']+=1;return False
        return True
    pending=choices[:desired_count];preserved=[];resampled=[]
    if previous_layout is not None:
        if previous_layout['task']!=task or previous_layout['seed']!=seed:raise ValueError('Prior layout identity mismatch')
        old_items=[x for x in previous_layout['objects'] if x['role']=='distractor']
        if {x['prim_path'] for x in old_items}!=set(pending):raise ValueError('Revision unexpectedly changed sampled product type/count')
        pending=[]
        for old in old_items:
            s=supports[old['support_id']]
            item,bodies,fp=product_pose(old['prim_path'],catalog[old['prim_path']],old['xyz_m'][:2],old['yaw_delta_deg'],s,'distractor')
            if permitted(item,bodies,fp,s):
                items.append(item);placed.append((item,bodies,fp));preserved.append(item['prim_path'])
            else:pending.append(item['prim_path'])
        # A separate stream makes conflict-only resampling independent of how
        # many prior candidates happened to be rejected in the old revision.
        rng=np.random.default_rng(np.random.SeedSequence([int(seed),941]))
    for path in pending:
        placed_ok = False
        for attempt in range(160):
            sid = str(rng.choice(["tier2", "tier3"]))
            s = supports[sid]; sp = Polygon(s["polygon_xy_m"])
            xx = rng.uniform(sp.bounds[0]+.06, sp.bounds[2]-.06)
            yy = rng.uniform(sp.bounds[1]+.06, sp.bounds[3]-.06)
            yaw = float(rng.uniform(-3, 3))
            item, bodies, fp = product_pose(path, catalog[path], [xx, yy], yaw, s, "distractor")
            if not permitted(item,bodies,fp,s):continue
            items.append(item); placed.append((item, bodies, fp)); placed_ok = True; break
        if not placed_ok:
            raise ValueError("Exhausted bounded layout sampling for " + path + ": " + json.dumps(rejections))
        if previous_layout is not None:resampled.append(path)
    if previous_layout is not None:
        order={x['prim_path']:i for i,x in enumerate(previous_layout['objects'])}
        items.sort(key=lambda x:order[x['prim_path']])
    clearance = dict(front_direction_world=[0, -1, 0], target_sampling_domain_xy_m=[[x_domain[0], y_domain[0]], [x_domain[1], y_domain[1]]],
                     approach_corridor_bounds_world_m=[corridor_lo, corridor_hi], reserved_lift_m=.15, reserved_retreat_m=.25,
                     purpose="Conservative layout reservation; policy must check real full robot and held-object geometry.")
    report = dict(status="CPU_CANDIDATE", passed=True, objects_count=len(items), distractor_count=len(items)-1,
                  unique_product_types=len({x["product_family"] for x in items}), rejections=rejections,
                  full_outline_support=True, visual_and_collision_pair_clearance_m=.012, distractor_initial_robot_clearance_m=.015,
                  target_support_margin_m=.018, distractor_support_margin_m=.014,
                  authored_bottom_gap_m=.0005,
                  unverified=["Actual scene initialization/settle", "Actual full-tick non-target stability", "Full robot initial/trajectory collisions", "Reach and camera visibility", "Physical grasp/lift/retreat", "Final-render initial-state image"],
                  geometry_method="All visible product triangles and circumscribed analytic/convex proxy meshes; exact support top polygon; conservative full footprint containment; FCL surface distance plus solid containment; complete nearby fixture geometry including invisible colliders.")
    if previous_layout is not None:
        report['heading_domain_revision']=dict(nominal_yaw_deg=nominal_heading,retained_original_jitter_deg=target_yaw,
                                               previous_episode=previous_layout['episode_id'],preserved_distractors=preserved,
                                               resampled_conflicting_distractors=resampled)
    return target, items, clearance, report


def author_episode_scene(path, base_scene, manifest):
    stage = Usd.Stage.CreateNew(str(path)); stage.GetRootLayer().subLayerPaths = [str(base_scene)]
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    active = {x["prim_path"] for x in manifest["objects"]}
    for stock in STOCK:
        prim = stage.GetPrimAtPath(stock)
        if prim: prim.SetActive(stock in active)
    for item in manifest["objects"]:
        prim = stage.GetPrimAtPath(item["prim_path"])
        body = UsdPhysics.RigidBodyAPI.Apply(prim)
        body.CreateRigidBodyEnabledAttr(True); body.CreateKinematicEnabledAttr(False)
        body.CreateVelocityAttr(Gf.Vec3f(0)); body.CreateAngularVelocityAttr(Gf.Vec3f(0))
        prim.AddAppliedSchema("PhysxRigidBodyAPI")
        prim.CreateAttribute("physxRigidBody:enableCCD", Sdf.ValueTypeNames.Bool).Set(True)
        set_pose(stage, item["prim_path"], item["xyz_m"], item["wxyz"])
    if stage.GetPrimAtPath("/World/Robot"):
        set_pose(stage, "/World/Robot", manifest["robot"]["base_xyz_m"], manifest["robot"]["base_wxyz"])
    stage.GetRootLayer().Save()
    recorded = sorted(str(p.GetPath()) for p in stage.Traverse()
                      if str(p.GetPath()).startswith(("/World/Assets/", "/World/EpisodeDistractors/"))
                      and p.HasAPI(UsdPhysics.RigidBodyAPI) and UsdPhysics.RigidBodyAPI(p).GetRigidBodyEnabledAttr().Get())
    if not active.issubset(recorded): raise ValueError("Recorder namespace coverage missing")
    return recorded


def audit_authored_world_poses(scene_path, manifest):
    """Reopen composed scene and compare actual USD world poses, not intent."""
    stage = Usd.Stage.Open(str(scene_path)); cache = UsdGeom.XformCache()
    entries = [(o["prim_path"], o["xyz_m"], o["wxyz"]) for o in manifest["objects"]]
    if stage.GetPrimAtPath("/World/Robot"):
        entries.append(("/World/Robot", manifest["robot"]["base_xyz_m"], manifest["robot"]["base_wxyz"]))
    rows = []
    for path, xyz, quat in entries:
        prim = stage.GetPrimAtPath(path)
        observed = matrix(cache.GetLocalToWorldTransform(prim))
        expected = pose(xyz, quat)
        error = float(np.max(np.abs(observed-expected)))
        rows.append(dict(prim_path=path, expected_world_matrix=expected.tolist(), actual_world_matrix=observed.tolist(),
                         maximum_matrix_error=error, passed=error < 1e-8))
    return dict(schema="family3.omni_authored_world_pose.v1", passed=all(r["passed"] for r in rows),
                checked_roots=len(rows), roots=rows)


def place_layout_interface(table_polygon_xy_m, tray_outer_polygon_local_xy_m, tray_inner_polygon_local_xy_m, tray_pose_xyz_wxyz, target_polygon_xy_m, robot_front_world_xy):
    """Validate caller-supplied PLACE geometry; does not create PLACE episodes.

    `robot_front_world_xy` points from the robot into the table. Every tray
    vertex must lie on the near side of the actual tabletop midpoint.
    """
    table = Polygon(table_polygon_xy_m); vertices = np.asarray(tray_outer_polygon_local_xy_m)
    xyz, quat = tray_pose_xyz_wxyz
    transformed = points(np.c_[vertices, np.zeros(len(vertices))], pose(xyz, quat))[:, :2]
    tray = Polygon(transformed)
    inner_vertices = np.asarray(tray_inner_polygon_local_xy_m)
    inner = Polygon(points(np.c_[inner_vertices, np.zeros(len(inner_vertices))], pose(xyz, quat))[:, :2])
    direction = np.asarray(robot_front_world_xy, float)
    direction /= np.linalg.norm(direction)
    table_v = np.asarray(table.exterior.coords[:-1]); projection = table_v @ direction
    midpoint = float((projection.min()+projection.max())/2)
    far_margin = float(midpoint-(transformed @ direction).max())
    return dict(schema="family3.omni_place_layout.v1", table_support=bool(table.covers(tray)),
                tray_near_half=far_margin >= 0, tray_far_edge_margin_m=far_margin,
                target_in_tray=bool(inner.covers(Polygon(target_polygon_xy_m))),
                full_tray_polygon_world_xy_m=transformed.tolist(),
                trajectory_status="UNVERIFIED", note="Caller must supply actual inner tray polygon for target containment, tray outer polygon for tabletop/near-half checks, and reserve carry/release/retreat corridors before sampling distractors.")


def generate(out=OUT / "revision4", tasks=None):
    start = time.monotonic(); out = Path(out).resolve()
    tasks = list(tasks or TASKS)
    provenance_path = out / "CANONICAL_PROVENANCE.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    if (out/'INDEX.json').exists() and any(not row['canonical'].get('source_command_verified') for row in provenance.values()):
        raise ValueError('Existing physical-derived provenance is immutable evidence; use a fresh revision directory for verified-command generation')
    for task in tasks:
        source = ROOT / f"incoming/l1_20260903/outputs/task_owned_50x4/{task}/trajectories/{TASKS[task]['source']}"
        trace = trace_nominal(source)
        canonical = recover_verified_command_nominal(task, trace) if trace["passed"] else {"passed": False}
        provenance[task] = dict(source=str(source), trace=trace, canonical=canonical)
    write_json(provenance_path, provenance)
    base_scene = out / "scene_base.usda"
    if base_scene.is_file() and (out / "ASSET_AUDIT.json").is_file():
        audit = json.loads((out / "ASSET_AUDIT.json").read_text())
        if not audit["passed"] or audit.get("baseline_source") != str(BASELINE.resolve(strict=True)):
            raise ValueError("Existing base scene is unverified or source changed; use a fresh output directory")
    else:
        base_scene, audit = build_base_scene(out)
    stage = Usd.Stage.Open(str(base_scene)); stage.SetEditTarget(stage.GetSessionLayer())
    catalog = inspect_catalog(stage)
    write_json(out / "PRODUCT_CATALOG.json", {p: {k: v for k, v in d.items() if k not in ("visual", "proxies", "vertices", "source_root_world")} for p, d in catalog.items()})
    index_path = out / "INDEX.json"
    entries = json.loads(index_path.read_text())["episodes"] if index_path.is_file() else []
    for task in tasks:
        for layout_index in range(2):
            tick = time.monotonic(); task_number = int(task.rsplit("_", 1)[1])
            seed = 2026091400 + task_number*10 + layout_index
            robot = canonical_robot(task, seed, provenance); robot["canonical_provenance"] = str(provenance_path)
            from omni_robot_support import registered_robot
            robot = registered_robot(robot, stage)
            directory = out / "episodes" / f"{task}_layout{layout_index}"; directory.mkdir(parents=True, exist_ok=True)
            if (directory / "episode.json").exists():
                previous = json.loads((directory / "episode.json").read_text())
                if not audit_authored_world_poses(previous["scene"]["path"], previous)["passed"]:
                    raise ValueError("Existing scene is rejected by world-pose composition gate; preserve it and use a fresh output directory")
                print(json.dumps(dict(task=task, seed=seed, status="EXISTING_UNCHANGED", manifest=str(directory / "episode.json"))), flush=True)
                continue
            previous_path=OUT/'revision3/episodes'/directory.name/'episode.json'
            previous=json.loads(previous_path.read_text()) if TASKS[task]['shape']=='box' and previous_path.is_file() else None
            target, items, clearance, report = sample_layout(task, seed, layout_index, base_scene, audit, catalog, robot, out, previous)
            manifest = dict(schema="family3.omni_episode.v1", episode_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"ziche/omni-stage1/v6/{task}/{seed}")),
                            task=task, mode="PICK", seed=seed, layout_index=layout_index,
                            scene=dict(path=str(directory / "scene.usda"), shelf_asset_root=str(SHELF), asset_audit=str(out / "ASSET_AUDIT.json")),
                            robot=robot, target=target, objects=items, supports=audit["supports"], clearance=clearance,
                            appearance=dict(task=task, seed=str(seed)+"-appearance", renderer="tools/family3/render_appearance.py:apply", status="NOT_RENDERED"),
                            recording=dict(prop_scope="all_active", required_namespaces=["/World/Assets", "/World/EpisodeDistractors"]),
                            precheck=dict(status="CPU_CANDIDATE", report=str(directory / "layout_audit.json")), place=None)
            all_rigid = author_episode_scene(directory / "scene.usda", base_scene, manifest)
            pose_audit = audit_authored_world_poses(directory / "scene.usda", manifest)
            write_json(directory / "authored_world_pose_audit.json", pose_audit)
            if not pose_audit["passed"]:
                raise ValueError("Authored world pose differs from manifest: " + str(directory))
            report["composed_world_pose_audit"] = dict(passed=True, path=str(directory / "authored_world_pose_audit.json"))
            manifest["scene"]["sha256"] = digest(directory / "scene.usda")
            manifest["recording"]["rigid_body_paths"] = all_rigid
            manifest["recording"]["randomized_rigid_body_paths"] = [x["prim_path"] for x in items]
            report["wall_seconds"] = time.monotonic()-tick
            write_json(directory / "layout_audit.json", report); write_json(directory / "episode.json", manifest)
            # Only initial arrays, deliberately separate from any fake full_state.
            np.savez(directory / "robot_initial.npz", robot_dof_names=np.asarray(robot["dof_names"]),
                     robot_initial_joint_position_rad_or_m=np.asarray(robot["joint_position_rad_or_m"]),
                     robot_root_xyz_m=np.asarray(robot["base_xyz_m"]), robot_root_wxyz=np.asarray(robot["base_wxyz"]))
            entry = dict(task=task, seed=seed, episode=manifest["episode_id"], manifest=str(directory / "episode.json"),
                         target_xyz_m=target["xyz_m"], objects_count=len(items), status="CPU_CANDIDATE", wall_seconds=report["wall_seconds"])
            entries.append(entry); write_json(out / "INDEX.json", dict(episodes=entries, wall_seconds=time.monotonic()-start))
            print(json.dumps(entry), flush=True)
    return entries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT / "revision4")
    parser.add_argument("--task", action="append", choices=list(TASKS))
    args = parser.parse_args()
    generate(args.out, args.task)
