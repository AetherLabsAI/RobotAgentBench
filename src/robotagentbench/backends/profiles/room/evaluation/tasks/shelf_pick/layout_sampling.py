"""Task-side layout randomization for shelf-pick (API 4-1 … 4-14), CPU only.

Ported rule-for-rule from ziche's generator
(`/mnt/agent-bench-data/lingjun/ziche/work/shelf_training_visible360_20260916`,
read 2026-09-23; audited in
`evaluation/batches/task3_fetch_snacks_20260922/pick_data_audit/pick_audit.md` §1):

    setup_bulk.py:138-146    64 scene proposals; seed = sha256(seed_key)[:6];
                             SeedSequence(seed).spawn(3) -> scene/selection/grasp;
                             layer_counts = {sid: integers(4, 11)} for tier3 then tier2
    snack_diversity.py:13-20 inventory = full shuffled rounds before any repeat
    random_shelf_scene.py:39 required edge gap 0.085 / 0.055 / 0.035 m by layer count
    random_shelf_scene.py:51 packing order: shuffle, then sort by footprint AABB area desc
    random_shelf_scene.py:14 yaw ~ U(-180, 180) deg, drawn once per item before XY
    random_shelf_scene.py:72 XY ~ U over the support-polygon bounds, 8 rounds x
                             12 batches x 64 candidates, shuffled, 12 kept per batch
    random_shelf_scene.py:96 reject: footprint not covered by the support; edge gap
                             below the requirement; 12 mm body separation;
                             0.15 mm fixture clearance
    random_shelf_scene.py:151 exhaustion raises; the constraint is never weakened

This module only produces object poses. It authors no prim, touches no physics
parameter, asset or robot configuration, and never writes into the read-only
dataset. The robot station comes from an accepted source episode and is
perturbed by the already-ported `episode_randomization.perturb`.

KNOWN, UNREMOVABLE DIFFERENCES vs. the generator (also emitted in the snapshot
`source.differences_vs_source` field and in the alignment report):

1. Instance ceiling. The 202 public baseline carries exactly one instance of
   each product: 8 drinks on tier3 and 6 snacks on tier2. The generator reaches
   4..10 per layer by cloning prims into `<prim>_copy_NN`. We do not create
   prims, so a layer count drawn above the ceiling is truncated to 8 (tier3) /
   6 (tier2). The raw draw is recorded next to the truncated count.
2. Target visibility. The generator gates the target on a head-camera render of
   the front label (visible_target_check.py). That cannot be reproduced on CPU.
   `front_clearance_m` below is a coarse geometric stand-in and is DISABLED by
   default; with it disabled the target yaw here is uniform over (-180, 180)
   while the accepted source target yaw spans only about -105..+86 deg.
3. Target selection. With one instance per product there is nothing to choose
   between, so `selection_rng` draws nothing. The generator's "random order over
   visible candidates with remaining quota" collapses to identity here.
4. Footprints. Measured from the 202 collision/visible shape extents (the
   `geometry.py` `body_points` rule, schema extents first), not from the source
   asset's raw mesh vertices. The hull of shape boxes is >= the hull of the
   mesh, so our placements are conservative.
5. Body separation / fixture clearance are checked in 2-D. Every required edge
   gap (>= 35 mm) exceeds the generator's 12 mm body separation, and a probe of
   the 202 rack shows no bracket or beam vertex strictly inside either board
   rectangle above the tier, so the "footprint inside the board, inset by
   0.15 mm" test implies the generator's fixture clearance on this rack.
6. Guided policies. `targeted_quota_v1` and `target_first_random_neighborhood_v1`
   (75 of the 210 episodes) steer the target pose from a reference episode. Only
   the unguided `scene_first_visible360_v1` rule is ported.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .episode_randomization import perturb
from .initial_state import BASE, NEWDATA, ROBOT, digest
from .source_episodes import CONTROLLED_JOINTS, SCHEMA, validate
from .targets import SHELF_ITEMS, TIER2_ITEMS, TIER3, TIER2, spec as target_spec

POLICY = 'arena.shelf_pick.layout_sampling.v1'
SOURCE_POLICY = 'scene_first_visible360_v1'
SOURCE_GENERATOR = ('/mnt/agent-bench-data/lingjun/ziche/work/'
                    'shelf_training_visible360_20260916')
# snack_diversity.py:7-8 — the 8 drinks of tier3 and the 6 snacks of tier2.
DRINKS = ('coke_c3', 'cayman_jacked', 'good2grow', 'gatorade_t', 'gatorade_m',
          'grapefruit_can', 'starbucks', 'sprite_c1')
SNACKS = ('chomps_r', 'pring_bbq', 'oreo_gf', 'honeymaid', 'zbar_t', 'skit_littles')
# tier -> (202 stock layer, 202 support slab, product families)
LAYERS = {'tier3': (SHELF_ITEMS, TIER3, DRINKS), 'tier2': (TIER2_ITEMS, TIER2, SNACKS)}
LAYER_ORDER = ('tier3', 'tier2')          # setup_bulk.py:145 draw order
COUNT_RANGE = (4, 11)                     # integers(4, 11) -> 4..10
PROPOSALS = 64                            # setup_bulk.py:138
PACKING_ROUNDS = 8                        # random_shelf_scene.py:41
BATCHES = 12                              # random_shelf_scene.py:73
CANDIDATES = 64                           # random_shelf_scene.py:75
SHORTLIST = 12                            # random_shelf_scene.py:101
SUPPORT_CLEARANCE_M = .0005               # setup_bulk.py:116
FIXTURE_CLEARANCE_M = .00015              # random_shelf_scene.py:105
BODY_SEPARATION_M = .012                  # random_shelf_scene.py:103
SPACING_TABLE = ((5, .085), (9, .055), (None, .035))   # random_shelf_scene.py:39


def spacing_for(count):
    """random_shelf_scene.py:39 — required edge gap from the layer count."""
    for limit, gap in SPACING_TABLE:
        if limit is None or count <= limit:
            return gap
    raise AssertionError


def product_of(api_id):
    """The product family the API asks for (targets.py binding)."""
    spec = target_spec(api_id)
    if spec.external_target:
        raise ValueError('Layout sampling covers the shelf stock APIs only: ' + api_id)
    name = spec.target.rsplit('/', 1)[-1]
    for tier, (_, _, kinds) in LAYERS.items():
        if name in kinds:
            return name, tier
    raise ValueError('API target is not a source release product: ' + api_id)


# --------------------------------------------------------------------------
# Convex polygon helpers (no shapely on 202; scipy.spatial.ConvexHull is fine).
# --------------------------------------------------------------------------
def convex_hull(points):
    """CCW hull vertices of a 2-D point set, degenerate input returned as-is."""
    from scipy.spatial import ConvexHull, QhullError
    p = np.asarray(points, dtype=float)[:, :2]
    if len(p) < 3:
        return p
    try:
        return p[ConvexHull(p).vertices]
    except QhullError:
        # Collinear/duplicate points: fall back to the extreme pair.
        order = np.lexsort((p[:, 1], p[:, 0]))
        return p[[order[0], order[-1]]]


def polygons_overlap(a, b):
    """Separating-axis test for two convex polygons (touching counts as apart)."""
    for poly in (a, b):
        for i in range(len(poly)):
            edge = poly[(i + 1) % len(poly)] - poly[i]
            norm = math.hypot(edge[0], edge[1])
            if norm <= 0:
                continue
            axis = np.array([-edge[1], edge[0]]) / norm
            pa, pb = a @ axis, b @ axis
            if pa.min() >= pb.max() or pb.min() >= pa.max():
                return False
    return True


def _vertex_edge_distance(a, b):
    """Minimum distance from the vertices of `a` to the edges of `b`."""
    best = math.inf
    rolled = np.roll(b, -1, axis=0)
    edges = rolled - b
    lengths = np.einsum('ij,ij->i', edges, edges)
    for point in a:
        delta = point - b
        with np.errstate(invalid='ignore', divide='ignore'):
            t = np.where(lengths > 0, np.einsum('ij,ij->i', delta, edges) / np.where(lengths > 0, lengths, 1), 0.)
        t = np.clip(t, 0., 1.)
        closest = b + edges * t[:, None]
        best = min(best, float(np.linalg.norm(point - closest, axis=1).min()))
    return best


def polygon_distance(a, b):
    """shapely `Polygon.distance` for convex polygons: 0 when they overlap."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        raise ValueError('Degenerate footprint')
    if polygons_overlap(a, b):
        return 0.
    return min(_vertex_edge_distance(a, b), _vertex_edge_distance(b, a))


def _aabb(polygon):
    return np.concatenate([polygon.min(0), polygon.max(0)])


def gap_at_least(polygon, neighbours, required, boxes=None):
    """`min(distance to neighbours) >= required`, with an AABB pre-filter.

    Only the predicate is needed while proposing candidates (the generator keeps
    the gap value but never ranks on it, random_shelf_scene.py:99-101); exact
    distances are recomputed for the accepted layout's spacing metrics.
    """
    box = _aabb(polygon)
    for index, other in enumerate(neighbours):
        other_box = boxes[index] if boxes is not None else _aabb(other)
        separation = max(other_box[0] - box[2], box[0] - other_box[2],
                         other_box[1] - box[3], box[1] - other_box[3])
        if separation >= required:
            continue
        if polygon_distance(polygon, other) < required:
            return False
    return True


def rect_covers(rect, polygon):
    """shapely `covers` against an axis-aligned support rectangle."""
    (x0, y0), (x1, y1) = rect
    p = np.asarray(polygon, dtype=float)
    return bool((p[:, 0] >= x0).all() and (p[:, 0] <= x1).all()
                and (p[:, 1] >= y0).all() and (p[:, 1] <= y1).all())


# --------------------------------------------------------------------------
# 202 scene geometry
# --------------------------------------------------------------------------
def measure_supports(stage=None):
    """Board rectangles and top z of both tiers, measured on the 202 scene."""
    from pxr import Usd, UsdGeom, UsdPhysics
    stage = stage if stage is not None else Usd.Stage.Open(str(BASE))
    cache = UsdGeom.BBoxCache(0, ['default', 'render', 'proxy', 'guide'], False, True)
    supports = {}
    for tier, (_, slab, _) in LAYERS.items():
        prim = stage.GetPrimAtPath(slab)
        if not prim or not prim.HasAPI(UsdPhysics.CollisionAPI):
            raise ValueError('Missing colliding support ' + slab)
        bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if bounds.IsEmpty():
            raise ValueError('Empty support geometry ' + slab)
        low, high = np.asarray(bounds.GetMin()), np.asarray(bounds.GetMax())
        supports[tier] = dict(prim_path=slab, top_z_m=float(high[2]),
                              rect_xy_m=[low[:2].tolist(), high[:2].tolist()])
    return supports


def measure_catalog(stage=None):
    """Local footprint hull and bottom z for the 14 release products.

    Shape extents follow `geometry.py body_points`: schema-derived extents first
    (`UsdGeom.Boundable.ComputeExtentFromPlugins`), because several analytic
    collision shapes in this asset carry a stale authored `extent`.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    stage = stage if stage is not None else Usd.Stage.Open(str(BASE))
    stage.SetEditTarget(stage.GetSessionLayer())
    catalog = {}
    for tier, (root, _, kinds) in LAYERS.items():
        for name in kinds:
            path = root + '/' + name
            body = stage.GetPrimAtPath(path)
            if not body:
                raise ValueError('Missing stock instance ' + path)
            body.SetActive(True)
            transform = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(0)
            scale = Gf.Transform(transform).GetScale()
            if any(abs(v - 1) > 1e-5 for v in scale):
                raise ValueError('Unexpected instance scale ' + path)
            inverse = transform.GetInverse()
            points = []
            for child in Usd.PrimRange(body):
                shape = UsdGeom.Boundable(child)
                if not shape:
                    continue
                collision = bool(child.HasAPI(UsdPhysics.CollisionAPI)
                                 and child.GetAttribute('physics:collisionEnabled').Get() is not False)
                visible = UsdGeom.Imageable(child).ComputeVisibility() != 'invisible'
                if not collision and not visible:
                    continue
                extent = UsdGeom.Boundable.ComputeExtentFromPlugins(shape, Usd.TimeCode.Default())
                if extent is None:
                    extent = shape.GetExtentAttr().Get()
                if extent is None or len(extent) != 2:
                    raise ValueError('Missing local geometry extent ' + str(child.GetPath()))
                box = Gf.Range3d(Gf.Vec3d(*extent[0]), Gf.Vec3d(*extent[1]))
                relative = UsdGeom.Xformable(child).ComputeLocalToWorldTransform(0) * inverse
                points.extend(list(relative.Transform(box.GetCorner(i))) for i in range(8))
            if not points:
                raise ValueError('Missing body geometry ' + path)
            vertices = np.asarray(points, dtype=float)
            hull = convex_hull(vertices[:, :2])
            if len(hull) < 3:
                raise ValueError('Degenerate footprint ' + path)
            catalog[name] = dict(
                prim_path=path, tier=tier, footprint_xy_m=hull,
                # random_shelf_scene.py:52 sorts on the local AABB area, not the hull area.
                aabb_area_m2=float(np.prod(np.ptp(vertices, axis=0)[:2])),
                local_z_min_m=float(vertices[:, 2].min()),
                local_z_max_m=float(vertices[:, 2].max()),
                shape_points=int(len(vertices)))
    return catalog


def scene_geometry(stage=None):
    """Both measurements from one stage open."""
    from pxr import Usd
    stage = stage if stage is not None else Usd.Stage.Open(str(BASE))
    return measure_catalog(stage), measure_supports(stage)


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------
def choose_without_early_repeats(rng, pool, count):
    """snack_diversity.py:13-20, value for value."""
    pool = list(pool)
    assert pool and len(pool) == len(set(pool)) and count >= 0
    chosen = []
    while len(chosen) < count:
        chosen.extend(rng.permutation(pool).tolist()[:count - len(chosen)])
    return chosen


def seed_chain(api_id, seed, proposal):
    """setup_bulk.py:141-144, keyed by API id and job seed instead of slot/attempt."""
    if not isinstance(api_id, str) or not api_id.strip():
        raise ValueError('Missing API identity for the layout key')
    if type(seed) is not int or isinstance(seed, bool) or not 0 <= seed < 2 ** 32:
        raise ValueError('Invalid episode seed')
    key = f'{POLICY}:{api_id}:{seed}:{proposal}'
    value = int.from_bytes(hashlib.sha256(key.encode()).digest()[:6], 'big')
    spawned = np.random.SeedSequence(value).spawn(3)
    return key, value, np.random.default_rng(spawned[0]), np.random.default_rng(spawned[1])


def _rotation(yaw_deg):
    angle = math.radians(yaw_deg)
    return np.array([[math.cos(angle), -math.sin(angle)],
                     [math.sin(angle), math.cos(angle)]]), angle


def _place(name, xy, yaw_deg, catalog, supports, tier):
    """setup_bulk.py:114-123 pose_item, restricted to what layout sampling needs."""
    entry = catalog[name]
    _, angle = _rotation(yaw_deg)
    z = supports[tier]['top_z_m'] + SUPPORT_CLEARANCE_M - entry['local_z_min_m']
    half = angle / 2
    return dict(prim_path=entry['prim_path'], product_family=name, tier=tier,
                support=supports[tier]['prim_path'],
                position_m=[float(xy[0]), float(xy[1]), float(z)],
                quaternion_wxyz=[math.cos(half), 0., 0., math.sin(half)],
                yaw_deg=float(yaw_deg))


def _front_clear(candidate_footprint, placed_same_tier, supports, tier, clearance_m):
    """COARSE geometric stand-in for the generator's rendered label-visibility gate.

    Not a visibility test: it only asserts that no same-tier footprint intrudes
    into the axis-aligned corridor between this item and the front board edge
    (-y is the robot side). Disabled unless `front_clearance_m` is given.
    """
    (_, y0), _ = supports[tier]['rect_xy_m']
    low, high = candidate_footprint.min(0), candidate_footprint.max(0)
    corridor = np.array([[low[0] - clearance_m, y0 - 1.], [high[0] + clearance_m, y0 - 1.],
                         [high[0] + clearance_m, low[1]], [low[0] - clearance_m, low[1]]])
    return not any(polygons_overlap(corridor, other) for other in placed_same_tier)


def sample_scene(scene_rng, catalog, supports, layer_counts, *, front_clearance_m=None):
    """random_shelf_scene.py:33-151 without shapely and without the guided branches."""
    spacing = {tier: spacing_for(count) for tier, count in layer_counts.items()}
    rejects = dict(support=0, object=0, fixture=0, target_visibility_or_corridor=0)
    history = []
    for packing_round in range(PACKING_ROUNDS):
        inventory = {tier: choose_without_early_repeats(scene_rng, LAYERS[tier][2], count)
                     for tier, count in layer_counts.items()}
        placed = []
        chosen = [(name, tier) for tier, names in inventory.items() for name in names]
        # Packing order depends only on physical size, never on a target role.
        scene_rng.shuffle(chosen)
        chosen.sort(key=lambda pair: -catalog[pair[0]]['aabb_area_m2'])
        for name, tier in chosen:
            rect = supports[tier]['rect_xy_m']
            # 0.15 mm fixture clearance: on this rack no bracket or beam vertex lies
            # inside the board rectangle above the tier, so insetting the rectangle
            # reproduces the generator's fixture test (see module docstring, item 5).
            inset = [[rect[0][0] + FIXTURE_CLEARANCE_M, rect[0][1] + FIXTURE_CLEARANCE_M],
                     [rect[1][0] - FIXTURE_CLEARANCE_M, rect[1][1] - FIXTURE_CLEARANCE_M]]
            neighbours = [fp for row, fp in placed if row['tier'] == tier]
            neighbour_boxes = [_aabb(fp) for fp in neighbours]
            placed_item = False
            # Orientation is drawn once per item, before any XY proposal.
            heading = float(scene_rng.uniform(-180., 180.))
            rot, _ = _rotation(heading)
            outline = catalog[name]['footprint_xy_m'] @ rot.T
            for _ in range(BATCHES):
                candidates = []
                for _ in range(CANDIDATES):
                    xy = scene_rng.uniform(rect[0], rect[1])
                    footprint = outline + xy
                    if not rect_covers(inset, footprint):
                        rejects['support'] += 1
                        continue
                    if not gap_at_least(footprint, neighbours, spacing[tier], neighbour_boxes):
                        rejects['object'] += 1
                        continue
                    candidates.append(xy)
                # Uniform proposal order with minimum spacing only; no largest-gap
                # ranking that would bias towards edges or particular XY.
                scene_rng.shuffle(candidates)
                for xy in candidates[:SHORTLIST]:
                    footprint = outline + xy
                    if front_clearance_m is not None and not _front_clear(
                            footprint, neighbours, supports, tier, front_clearance_m):
                        rejects['target_visibility_or_corridor'] += 1
                        continue
                    placed.append((_place(name, xy, heading, catalog, supports, tier), footprint))
                    placed_item = True
                    break
                if placed_item:
                    break
            if not placed_item:
                break
        complete = len(placed) == sum(layer_counts.values())
        history.append(dict(round=packing_round, inventory=inventory, placed=len(placed), complete=complete))
        if complete:
            metrics = {}
            for tier in layer_counts:
                footprints = [fp for row, fp in placed if row['tier'] == tier]
                nearest = [min(polygon_distance(fp, other) for j, other in enumerate(footprints) if i != j)
                           for i, fp in enumerate(footprints)]
                metrics[tier] = dict(objects=len(footprints), required_edge_gap_m=spacing[tier],
                                     minimum_edge_gap_m=float(min(nearest)),
                                     median_nearest_edge_gap_m=float(np.median(nearest)))
                assert min(nearest) >= spacing[tier] - 1e-8
            report = dict(policy=POLICY, ported_from=SOURCE_POLICY, packing_round=packing_round,
                          rounds=history, layer_counts=dict(layer_counts), spacing_metrics=metrics,
                          rejections=rejects, yaw_range_deg=[-180., 180.],
                          uses_target=False, uses_robot=False)
            return [row for row, _ in placed], report
    raise RuntimeError(('Independent shelf layout sampling exhausted', layer_counts, history, rejects))


def sample_layout(api_id, seed, catalog, supports, *, front_clearance_m=None):
    """One reproducible layout for `api_id` containing that API's product.

    Mirrors the generator's proposal loop (setup_bulk.py:138-231): each proposal
    redraws the layer counts and the whole scene from its own seed; the first
    proposal that actually contains the requested product is kept. The generator
    additionally requires the rendered label to be visible and a CPU grasp
    precheck to pass; neither is reproduced here (docstring, item 2).
    """
    product, tier = product_of(api_id)
    failures = []
    for proposal in range(1, PROPOSALS + 1):
        key, value, scene_rng, selection_rng = seed_chain(api_id, seed, proposal)
        raw_counts = {t: int(scene_rng.integers(*COUNT_RANGE)) for t in LAYER_ORDER}
        counts = {t: min(raw_counts[t], len(LAYERS[t][2])) for t in LAYER_ORDER}
        try:
            placed, report = sample_scene(scene_rng, catalog, supports, counts,
                                          front_clearance_m=front_clearance_m)
        except RuntimeError as error:
            failures.append(dict(proposal=proposal, reason='PACKING_FAILED', detail=str(error.args[0])))
            continue
        matches = [i for i, row in enumerate(placed) if row['product_family'] == product]
        if not matches:
            failures.append(dict(proposal=proposal, reason='TARGET_PRODUCT_ABSENT',
                                 detail=f'{product} not drawn into {tier}'))
            continue
        # One instance per product on the 202 baseline: the generator's random
        # order over visible candidates has a single element here.
        index = int(selection_rng.choice(matches)) if len(matches) > 1 else matches[0]
        report.update(seed=value, seed_key=key, proposal=proposal,
                      proposals_tried=proposal, failed_proposals=failures,
                      raw_layer_counts=raw_counts,
                      layer_count_ceiling={t: len(LAYERS[t][2]) for t in LAYER_ORDER},
                      layer_counts_truncated={t: raw_counts[t] > counts[t] for t in LAYER_ORDER},
                      target_index=index, target_product=product, target_tier=tier,
                      target_instances=len(matches),
                      front_clearance_m=front_clearance_m)
        return placed, report
    raise RuntimeError(('Layout proposals exhausted for ' + api_id, seed, failures))


# --------------------------------------------------------------------------
# Snapshot assembly
# --------------------------------------------------------------------------
def differences_vs_source():
    """The honest difference list, carried inside every snapshot we emit."""
    return [
        dict(item='layer_instance_ceiling', aligned=False,
             detail='202 public baseline holds one instance per product (8 drinks / 6 snacks). '
                    'The generator reaches 4..10 per layer with `_copy_NN` clones. We create no '
                    'prims, so a draw above the ceiling is truncated. Raw draw kept in the receipt.'),
        dict(item='target_label_visibility', aligned=False,
             detail='The generator gates the target on a head-camera render (visible_target_check.py). '
                    'Not reproducible on CPU. Disabled by default; `front_clearance_m` offers a coarse '
                    'geometric corridor test that is NOT a visibility test. Consequence: our target yaw '
                    'is uniform (-180, 180) while accepted source target yaw spans about -105..+86 deg.'),
        dict(item='cpu_grasp_precheck', aligned=False,
             detail='The generator also rejects scenes whose target fails an IK/grasp precheck. '
                    'Not ported; no reachability filtering is applied here.'),
        dict(item='target_selection', aligned=False,
             detail='With one instance per product the generator\'s random choice over eligible '
                    'visible instances is the identity; selection_rng draws nothing.'),
        dict(item='guided_placement_policies', aligned=False,
             detail='targeted_quota_v1 and target_first_random_neighborhood_v1 (75 of 210 episodes) '
                    'steer the target from a reference episode. Only scene_first_visible360_v1 is ported.'),
        dict(item='footprint_source', aligned='approximate',
             detail='Footprints are hulls of the 202 collision/visible shape extents '
                    '(geometry.py body_points rule), not of the source asset mesh vertices. '
                    'Conservative: the shape-box hull contains the mesh hull.'),
        dict(item='body_separation_and_fixture_clearance', aligned='approximate',
             detail='Checked in 2-D. Every required edge gap (>=35 mm) exceeds the generator\'s 12 mm '
                    'body separation, and a probe of the 202 rack found no bracket/beam vertex inside '
                    'either board rectangle above the tier, so the inset-rectangle containment test '
                    'implies the 0.15 mm fixture clearance on this rack.'),
        dict(item='robot_station', aligned=False,
             detail='Base/arm pose is the 202 accepted snapshot for this API plus '
                    'episode_randomization.perturb, not the single fixed source station.'),
    ]


def station_snapshot(api_id, station=None):
    """The accepted v2 snapshot whose robot station and joints we reuse."""
    path = Path(station) if station else NEWDATA / api_id / 'fixed.json'
    data = json.loads(Path(path).read_text())
    if data.get('schema') != SCHEMA or data.get('api_id') != api_id:
        raise ValueError('Station snapshot is not the v2 snapshot of ' + api_id)
    return data, str(path)


def build_snapshot(api_id, seed, catalog, supports, *, station=None,
                   front_clearance_m=None, perturb_robot=True, baseline=BASE):
    """A locally sampled `arena.shelf_pick.initial_state.v2` snapshot.

    Same schema as `source_episodes.extract`, so it feeds
    `initial_state.prepare(..., snapshot=...)` unchanged.
    """
    spec = target_spec(api_id)
    placed, report = sample_layout(api_id, seed, catalog, supports,
                                   front_clearance_m=front_clearance_m)
    target_row = placed[report['target_index']]
    if not spec.is_target(target_row['prim_path']):
        raise ValueError('Sampled target is not an accepted target of ' + api_id)
    station_data, station_path = station_snapshot(api_id, station)
    source = dict(
        origin='locally sampled layout (202 task side); NOT a recorded ziche episode',
        policy=POLICY, ported_from=SOURCE_POLICY, source_generator=SOURCE_GENERATOR,
        source_rules_audit='evaluation/batches/task3_fetch_snacks_20260922/pick_data_audit/pick_audit.md',
        seed=seed, seed_key=report['seed_key'], derived_seed=report['seed'],
        proposal=report['proposal'], proposals_tried=report['proposals_tried'],
        layer_counts=report['layer_counts'], raw_layer_counts=report['raw_layer_counts'],
        layer_count_ceiling=report['layer_count_ceiling'],
        layer_counts_truncated=report['layer_counts_truncated'],
        spacing_metrics=report['spacing_metrics'],
        station_snapshot=station_path,
        station_episode_id=station_data['source'].get('episode_id'),
        robot_frame=ROBOT,
        differences_vs_source=differences_vs_source(),
        scope='object layout sampled on 202; robot station reused from an accepted episode and '
              'perturbed by shelf_pick_episode_randomization_v1; source physics (60 Hz) and '
              'controller are NOT reproduced',
        physical_acceptance=False)
    items = [dict(prim_path=row['prim_path'], support=row['support'],
                  position_m=row['position_m'], quaternion_wxyz=row['quaternion_wxyz'])
             for row in placed if row is not target_row]
    snapshot = dict(schema=SCHEMA, api_id=api_id, source=source,
                    target_path=target_row['prim_path'], target_tier=target_row['tier'],
                    target=dict(position_m=target_row['position_m'],
                                quaternion_wxyz=target_row['quaternion_wxyz']),
                    robot=dict(station_data['robot']),
                    joint_positions=dict(station_data['joint_positions']),
                    items=items,
                    verification=dict(controlled_joints=list(CONTROLLED_JOINTS),
                                      max_joint_deviation_rad=0., joint_tolerance_rad=0.,
                                      max_pose_deviation_m=0., pose_tolerance_m=0.,
                                      joint_reference='reused verbatim from ' + station_path,
                                      pose_reference='sampled on 202 geometry; no dataset reference exists',
                                      baseline_sha256=digest(baseline), baseline=str(baseline)),
                    physical_acceptance=False,
                    layout_sampling=report)
    if perturb_robot:
        snapshot, receipt = perturb(snapshot, seed)
        snapshot['source'] = dict(snapshot['source'], robot_perturbation=receipt['policy'])
    validate(snapshot)
    return snapshot


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Sample one shelf-pick layout (CPU only).')
    parser.add_argument('--api', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--front-clearance-m', type=float, default=None)
    parser.add_argument('--no-perturb', action='store_true')
    args = parser.parse_args()
    catalog, supports = scene_geometry()
    snapshot = build_snapshot(args.api, args.seed, catalog, supports,
                              front_clearance_m=args.front_clearance_m,
                              perturb_robot=not args.no_perturb)
    Path(args.output).write_text(json.dumps(_jsonable(snapshot), indent=2, allow_nan=False) + '\n')
    print(json.dumps(_jsonable(snapshot['source']['layer_counts'])))


if __name__ == '__main__':
    main()
