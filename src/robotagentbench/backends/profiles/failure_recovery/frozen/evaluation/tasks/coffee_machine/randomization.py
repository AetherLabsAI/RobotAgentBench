"""Seeded initial layouts of the four APIs, sampled with the data producer's own code.

The samplers are the producer's (resources/shared_assets/coffee_v1/producer_sources_20260930, vendored byte-identically
in vendor_producer/ with hashes); only the authoring of the public prims is ours. One proposal per seed, no
resampling: a layout that fails initial acceptance is reported, never replaced.

1-1 放杯子    cup_randomize.py::scene_proposals (the producer's left_study_reference/randomize.py) with the config v8
              "large-only" its final 182 workers ran (workbench AGENTS.md, 2026-09-24): cup and machine XY within
              ±8 cm with horizontal displacement ≥ 5 cm, cup |yaw| 20–45°, machine |yaw| 7–10°, bag XY ±8 cm, bag yaw
              0–90°, three cup styles. Recomputes the recorded values of the 20 v8 episodes exactly. The 13 episodes of
              the earlier uniform config and the 15 left-arm episodes came from other samplers that were not delivered.
1-2 倾倒咖啡豆  generate_until_50.py: success_coverage.choose(root, state, 0.1, 20.0) picks the episode seed (advancing
              it until the bag-yaw stratum matches) and the machine pose; build_cycle_initial.py then draws from
              random.Random(seed): machine offset (2) and yaw (overridden by the chooser), bag offset (2), bag yaw,
              cup style. Evaluated with an empty success history (the producer's first-batch behaviour), so the layout
              depends on the seed only. The bag / style draws recompute all 5 packaged episodes exactly; the historical
              machine poses depended on the producer's success history and cannot be re-derived.
1-3 按按钮    button_batch_common.py::proposal, unchanged.
1-4 取杯子    randomize_retrieval_pose.py::retarget, line 13: numpy default_rng(seed) → machine offset uniform(±0.08 m)
              ×2, yaw uniform(±10°), applied about the nominal machine origin; the cup follows the machine. Recomputes
              the 19 常规 records exactly; the 18 大扰动 records fit the same form with ±0.15 m / ±30° but that code
              was not delivered.
"""
import ast
import hashlib
import json
import math
import random

from .specs import MACHINE, CUP, BAG, BEANS, CUP_STYLES, TASK
from . import vendor_producer

SUPPORTED = ('1-1', '1-2', '1-3', '1-4')
MACHINE_ORIGIN = (2.569096825, 0.166750471, 0.92)      # producer cup_place/scene_pose.py:30, nominal machine origin
V8_CONFIG = dict(xy_half_range_m=[.08, .08], xy_min_radius_m=.05, yaw_half_range_deg=45., yaw_min_abs_deg=20.,
                 machine_xy_half_range_m=[.08, .08], machine_xy_min_radius_m=.05, machine_yaw_half_range_deg=10.,
                 machine_yaw_min_abs_deg=7., bag_xy_half_range_m=[.08, .08], bag_yaw_range_deg=[0., 90.],
                 cup_styles=list(CUP_STYLES))
POUR_EXTENT_M, POUR_YAW_DEG = 0.1, 20.0                # generate_until_50.py:53  choose(ROOT, state, 0.1, 20.0)
RETRIEVE_XY_M, RETRIEVE_YAW_DEG = 0.08, 10.0           # randomize_retrieval_pose.py:13


def vendored(name):
    path = vendor_producer.HERE / name
    return dict(file='vendor_producer/' + name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def functions(name, wanted, namespace):
    """The named top-level functions of a vendored file, executed unchanged in `namespace`."""
    path = vendor_producer.HERE / name
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            exec(compile(ast.Module([node], []), str(path), 'exec'), namespace)
    missing = [w for w in wanted if w not in namespace]
    if missing:
        raise ValueError('Vendored sampler functions missing: ' + ', '.join(missing))
    return namespace


# ---- samplers (pure) ----
def sample_cup_place(seed):
    import numpy as np
    ns = functions('cup_randomize.py', ('large_xy', 'large_yaw', 'scene_proposals'), {'np': np})
    offset, yaw, style, machine, machine_yaw, bag, bag_yaw = next(ns['scene_proposals'](seed, V8_CONFIG, 1))
    return dict(sampler='vendor_producer/cup_randomize.py::scene_proposals (numpy default_rng(seed), first proposal)', config=V8_CONFIG,
                cup_offset_m=[float(v) for v in offset], cup_yaw_delta_deg=float(yaw), cup_style=str(style),
                machine_offset_m=[float(v) for v in machine], machine_yaw_deg=float(machine_yaw),
                bag_offset_m=[float(v) for v in bag], bag_yaw_deg=float(bag_yaw))


def sample_pour(seed):
    choose = vendor_producer.load('success_coverage').choose
    state = dict(batches=[], qualified=[], next_seed=seed)          # no success history: the producer's first-batch behaviour
    episode_seed, pose = choose(str(TASK / 'artifacts'), state, POUR_EXTENT_M, POUR_YAW_DEG)
    specs = json.loads((vendor_producer.HERE / 'cup_variants.json').read_text())
    rng = random.Random(episode_seed)                                # build_cycle_initial.py:31-37, verbatim draw order
    machine_offset = [rng.uniform(-.10, .10), rng.uniform(-.10, .10), 0.]
    machine_yaw = rng.uniform(-20, 20)
    bag_offset = [rng.uniform(-.05, .05), rng.uniform(-.05, .05), 0.]
    bag_yaw = rng.uniform(0, 90)
    style = rng.choice(list(specs))
    x, y, yaw = map(float, pose)                                     # build_cycle_initial.py:38-42, --machine-pose-json override
    if not (-.1 <= x <= .1 and -.1 <= y <= .1 and -20 <= yaw <= 20):
        raise ValueError('Machine sample outside approved range')
    return dict(sampler='vendor_producer/success_coverage.py::choose + build_cycle_initial.py:31-42 draws', requested_seed=seed,
                episode_seed=episode_seed, chooser=dict(extent_m=POUR_EXTENT_M, yaw_extent_deg=POUR_YAW_DEG, history='empty',
                                                       selection=state['coverage_selection']),
                machine_offset_m=[x, y, 0.], machine_yaw_deg=yaw, own_machine_draw_overridden=dict(offset_m=machine_offset, yaw_deg=machine_yaw),
                bag_offset_m=bag_offset, bag_yaw_deg=bag_yaw, cup_style=style)


def sample_retrieve(seed):
    import numpy as np
    rng = np.random.default_rng(seed)                                # randomize_retrieval_pose.py:13, verbatim draw order
    offset = [float(rng.uniform(-RETRIEVE_XY_M, RETRIEVE_XY_M)), float(rng.uniform(-RETRIEVE_XY_M, RETRIEVE_XY_M)), 0.]
    yaw = float(rng.uniform(-RETRIEVE_YAW_DEG, RETRIEVE_YAW_DEG))
    return dict(sampler='vendor_producer/randomize_retrieval_pose_reference.py:13 (numpy default_rng(seed))',
                machine_offset_m=offset, machine_yaw_deg=yaw)


# ---- authoring ----
def rigid_delta(origin, offset, yaw_deg):
    """Producer scene_pose.rigid_delta: rotate about `origin` around world Z by yaw_deg, then translate by `offset`."""
    from pxr import Gf
    o = Gf.Vec3d(*[float(v) for v in origin])
    rotation = Gf.Matrix4d(1).SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), float(yaw_deg)))
    return Gf.Matrix4d(1).SetTranslate(-o) * rotation * Gf.Matrix4d(1).SetTranslate(o + Gf.Vec3d(*[float(v) for v in offset]))


def move(stage, paths, delta):
    from pxr import Gf, UsdGeom, UsdPhysics
    cache = UsdGeom.XformCache()
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim:
            raise ValueError('Missing prim to move: ' + path)
        world = cache.GetLocalToWorldTransform(prim)
        parent = cache.GetParentToWorldTransform(prim)
        UsdGeom.Xformable(prim).MakeMatrixXform().Set(world * delta * parent.GetInverse())
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            body = UsdPhysics.RigidBodyAPI(prim)
            body.CreateVelocityAttr().Set(Gf.Vec3f(0)); body.CreateAngularVelocityAttr().Set(Gf.Vec3f(0))


def origin_of(stage, path):
    from pxr import UsdGeom
    return UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path)).ExtractTranslation()


def heading_deg(stage, path):
    from pxr import Gf, UsdGeom
    x = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path)).TransformDir(Gf.Vec3d(1, 0, 0))
    return math.degrees(math.atan2(x[1], x[0]))


def apply(stage, api, manifest, seed, set_cup_style=None):
    """Move the fixed layout (the packaged source episode) to the seeded proposal, the producer's way.

    The packaged episode's own recorded randomization is undone first, so every proposal is expressed relative to the
    producer's nominal layout, exactly as the producer authored its episodes.
    """
    from pxr import Gf
    if api not in SUPPORTED:
        raise ValueError(f'{api}: no reproducible sampler')
    if type(seed) is not int or seed < 0:
        raise ValueError('Random layouts need a non-negative integer seed')
    facts = manifest['facts']
    beans = [str(p.GetPath()) for p in stage.GetPrimAtPath(BEANS).GetChildren()] if stage.GetPrimAtPath(BEANS) and stage.GetPrimAtPath(BEANS).IsActive() else []
    before = dict(machine=list(origin_of(stage, MACHINE)), cup=list(origin_of(stage, CUP)), bag=list(origin_of(stage, BAG)))
    if api == '1-3':
        common = vendor_producer.load('button_batch_common')
        sample = common.proposal(seed)
        old = facts['randomization']
        reference = Gf.Vec3d(*origin_of(stage, MACHINE)) - Gf.Vec3d(old['machine_dx_m'], old['machine_dy_m'], 0)
        delta = (rigid_delta(reference, (old['machine_dx_m'], old['machine_dy_m'], 0), old['machine_dyaw_deg']).GetInverse()
                 * rigid_delta(reference, (sample['machine_dx_m'], sample['machine_dy_m'], 0), sample['machine_dyaw_deg']))
        move(stage, [MACHINE, CUP, *beans], delta)
        record = dict(sampler='vendor_producer/button_batch_common.py::proposal', proposal=sample, config=common.CONFIG,
                      reference_machine_origin=list(reference), packaged_episode_proposal=old, cup_style=facts.get('cup_style'),
                      layout_gate='cup completely inside the head camera frame; judged at initial acceptance on the live camera pose')
    elif api == '1-1':
        plan = json.loads((TASK / 'resources/source_episodes' / api / manifest['episode'] / 'producer_plan.json').read_text())
        old = plan['randomization']
        if 'offset_m' not in old:
            raise ValueError('The fixed 1-1 episode must be a right-arm episode with a recorded cup offset / yaw delta')
        new = sample_cup_place(seed)
        cup_origin = Gf.Vec3d(*origin_of(stage, CUP)) - Gf.Vec3d(*old['offset_m'])          # template cup origin
        bag_origin = Gf.Vec3d(*origin_of(stage, BAG)) - Gf.Vec3d(*old['bag_offset_m'])      # nominal bag origin
        move(stage, [CUP], rigid_delta(cup_origin, old['offset_m'], old['yaw_delta_deg']).GetInverse() * rigid_delta(cup_origin, new['cup_offset_m'], new['cup_yaw_delta_deg']))
        move(stage, [MACHINE], rigid_delta(MACHINE_ORIGIN, old['machine_offset_m'], old['machine_yaw_deg']).GetInverse() * rigid_delta(MACHINE_ORIGIN, new['machine_offset_m'], new['machine_yaw_deg']))
        move(stage, [BAG, *beans], rigid_delta(bag_origin, old['bag_offset_m'], old['bag_yaw_deg']).GetInverse() * rigid_delta(bag_origin, new['bag_offset_m'], new['bag_yaw_deg']))
        record = dict(new, packaged_episode_randomization=old, template_cup_origin=list(cup_origin), nominal_bag_origin=list(bag_origin),
                      nominal_machine_origin=list(MACHINE_ORIGIN))
    elif api == '1-2':
        old = facts['randomization']
        new = sample_pour(seed)
        # build_cycle_initial.py:49-53: rotate about the object's current origin by (new - old) yaw, translate by (new - old) offset
        dm = rigid_delta(origin_of(stage, MACHINE), [a - b for a, b in zip(new['machine_offset_m'], old['machine_offset_m'])], new['machine_yaw_deg'] - old['machine_yaw_deg'])
        db = rigid_delta(origin_of(stage, BAG), [a - b for a, b in zip(new['bag_offset_m'], old['bag_offset_m'])], new['bag_yaw_deg'] - old['bag_yaw_deg'])
        move(stage, [MACHINE, CUP], dm)
        move(stage, [BAG, *beans], db)
        record = dict(new, packaged_episode_randomization=old)
    else:   # 1-4
        old = dict(machine_offset_m=facts['machine_offset_m'], machine_yaw_deg=facts['machine_yaw_deg'])
        new = sample_retrieve(seed)
        move(stage, [MACHINE, CUP], rigid_delta(MACHINE_ORIGIN, old['machine_offset_m'], old['machine_yaw_deg']).GetInverse()
             * rigid_delta(MACHINE_ORIGIN, new['machine_offset_m'], new['machine_yaw_deg']))
        record = dict(new, packaged_episode_machine=old, nominal_machine_origin=list(MACHINE_ORIGIN), cup_style=facts.get('cup_style'),
                      bag='unchanged (the producer kept the bag of the placement run)')
    style = record.get('cup_style')
    if api in ('1-1', '1-2'):
        if style not in CUP_STYLES:
            raise ValueError('Sampled cup style unknown: ' + str(style))
        if set_cup_style is None:
            raise ValueError('A cup style callback is required for ' + api)
        set_cup_style(style)
    record.update(seed=seed, api=api, source_code=[vendored(n) for n in ('cup_randomize.py', 'success_coverage.py', 'button_batch_common.py', 'randomize_retrieval_pose_reference.py')],
                  moved_prims=dict(machine=True, cup=True, bag=api in ('1-1', '1-2'), beans=len(beans) if api in ('1-1', '1-2', '1-3') else 0),
                  before=before, after=dict(machine=list(origin_of(stage, MACHINE)), cup=list(origin_of(stage, CUP)), bag=list(origin_of(stage, BAG))),
                  machine_local_x_heading_deg=heading_deg(stage, MACHINE), resampled=False)
    return record
