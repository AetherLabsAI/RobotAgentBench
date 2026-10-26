"""Read-only USD measurements for a design preview; never edits the stage."""
import hashlib
import json
from pathlib import Path
from pxr import Usd, UsdGeom

ROOT = Path(__file__).resolve().parents[1]
SCENE = Path('/mnt/agent-bench-data/lingjun/releases/living50-candidate-20260919/layouts/ID_01_LIVING_STRAIGHT.usdc')
stage = Usd.Stage.Open(str(SCENE))
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy'])
prefix = '/World/Background/Shell/ConstructorKitchen/MetricScale/'

def measure(name, path, **extra):
    prim = stage.GetPrimAtPath(path)
    if not prim:
        raise ValueError(path)
    bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if bounds.IsEmpty():
        raise ValueError('Empty bound: ' + path)
    return dict(name=name, path=path, min=list(bounds.GetMin()), max=list(bounds.GetMax()), **extra)

surfaces = [measure('岛台', '/World/Background/Shell/CounterTop_17', group='island')]
for name, rel in [
    ('灶台南侧', 'RightWallBase150/Body/Worktop'),
    ('灶台南侧', 'RightWallBase400/Body/Worktop'),
    ('灶台南侧', 'RightEnd300/Module/Body/Worktop'),
    ('灶台南侧', 'BP_Kitchen36/Body/Worktop'),
    ('炉头台面', 'RightRange600/Module/Body/Worktop'),
    ('炉头与水槽之间', 'BP_Kitchen9/Body/Worktop'),
    ('水槽台面', 'BP_Kitchen10/Body/Worktop'),
    ('备餐台面', 'BP_Kitchen8/Body/Worktop'),
    ('备餐台面', 'CornerGapBase300_A/Body/Worktop'),
    ('备餐台面', 'CornerGapBase300_B/Body/Worktop'),
]:
    surfaces.append(measure(name, prefix + rel, group='stove'))
uncertain = [measure('转角台面：形状待细核', prefix + 'BP_Kitchen7/Body/Worktop')]
for letter in 'ABC':
    uncertain.append(measure('回转台面：与冰箱净空待核', prefix + 'LeftReturnBase150_' + letter + '/Body/Worktop'))
blockers = [
    measure('微波炉', '/World/Assets/DiningTableMicrowave', kind='occupied'),
    measure('现有炸锅', '/World/Assets/AirFryer', kind='occupied', fryer=True),
    measure('咖啡磨豆机', '/World/Assets/CoffeeGrinder', kind='occupied'),
    measure('水槽 · 不摆放', prefix + 'BP_Kitchen10/Body/Sink', kind='excluded'),
    measure('餐盘 / 食物托盘', '/World/FoodTray/Render', kind='occupied'),
    measure('餐具', '/World/Cutlery', kind='occupied'),
    measure('便当食物', '/World/MealDiverse/Main', kind='occupied'),
    measure('饮品', '/World/MealDiverse/Drink', kind='occupied'),
]
usable_features = [measure('炉头 · 可用于摆放', prefix + 'RightRange600/Module/Body/Stove',
    kind='user_approved_placement_candidate', physical_validation='pending', requires_stove_off=True)]
audit_path = ROOT / 'task_scene_registration_v1/task2_workspace_audit_v1.json'
audit = json.loads(audit_path.read_text())
# Derive the union from all audited stages; do not substitute a reach radius.
def union(key):
    values = [t[key] for t in audit['per_task'].values()]
    return dict(min=[min(v['min'][i] for v in values) for i in range(3)],
                max=[max(v['max'][i] for v in values) for i in range(3)])
data = dict(scene=str(SCENE), scene_sha256=hashlib.sha256(SCENE.read_bytes()).hexdigest(),
    audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
    source_method='Default-time visual aligned world bounds, metres. Candidate rectangles, not support/collision validation.',
    surfaces=surfaces, uncertain=uncertain, blockers=blockers, usable_features=usable_features,
    anchor=[2.810000011, -0.700000010], yaw_deg=180,
    object_bounds=union('object_bounds'), combined_bounds=union('combined_bounds'),
    episodes=audit['episodes'], frames=audit['frames'], sample_rate_hz=audit['sample_rate_hz'])
(ROOT / 'web/kitchen-workspace-data.js').write_text('window.KITCHEN_WORKSPACE = ' + json.dumps(data, ensure_ascii=False, indent=2) + ';\n')
print(json.dumps(dict(surfaces=len(surfaces), blockers=len(blockers), scene_sha256=data['scene_sha256'])))
