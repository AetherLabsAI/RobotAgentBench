"""外观随机化，规则取自数据方 place_tray_lerobot_20260918 的 APPEARANCE.json。

用户 2026-09-23 定：我方随机化要与数据方的**全部**随机化对齐，外观也要补齐——
"只影响渲染、不影响物理"不是跳过对齐的理由。数据方自己也把外观标为
`physics_effect: "none; applied to render-only overlay"`，本模块同样不碰任何物理量：
只写材质颜色/粗糙度/纹理与灯光强度/色温。

取值集合与范围是把数据方 700 条回合的 `APPEARANCE.json` 全量统计出来的：

  茶几材质 4 种（共用同一张纹理，差别在 rgb_scale / roughness / material_look）
  托盘颜色 3 种（linear_rgb，roughness 0.5，无纹理）
  地板纹理 3 种
  灯光 10 盏：吊顶 8 盏 intensity∈[0.65,1.35] / 色温∈[3600,6200] K，
              日光 2 盏 intensity∈[0.35,1.45] / 色温∈[4700,7500] K
  材质色调 11 处，各自 rgb_scale 区间见 TINTS

灯光与色调**不是均匀分布**：把 700 条回合归一化到各自区间后看分位点，是钟形
（10%/90% 落在 0.13/0.77 而不是 0.1/0.9，中位数约 0.41–0.50）。因此按实测拟合的
截断正态抽样，参数见 SHAPES；分类轴（茶几/托盘/地板）实测是均匀的，按均匀抽。

这些区间是**实测出来的经验范围**，不是数据方源码里的字面常数——他们的生成器不在
只读挂载上，无法逐值核对。因此本模块对齐的是分布，不是实现；对齐审计里按
APPROXIMATE 记录，不宣称逐值一致。
"""
import hashlib
from dataclasses import dataclass, asdict

import numpy as np

POLICY = 'arena.coffee_table_place.appearance.v1'
SOURCE_POLICY = 'place_l1_four_table_neutral_tray_v5'
TEXTURES = 'assets/source_appearance_20260918/textures'
TABLE_TEXTURE = 'T_Coffee_Table_01_Wood_01_Diffuse_4K.1001.png'

TABLE_LOOKS = {
    'warm_oak_linen':  dict(rgb_scale=(1.18, 1.05, .82), roughness=.58, metallic=0.,
                            material_look='oak_light_matte'),
    'light_scandi':    dict(rgb_scale=(1.22, 1.16, 1.02), roughness=.62, metallic=0.,
                            material_look='ash_light_matte'),
    'cool_gray_black': dict(rgb_scale=(.92, .95, 1.02), roughness=.55, metallic=0.,
                            material_look='gray_matte'),
    'dark_walnut':     dict(rgb_scale=(.78, .66, .52), roughness=.5, metallic=0.,
                            material_look='walnut_satin'),
}
TRAY_COLOURS = {
    'black': dict(linear_rgb=(.025, .025, .025), roughness=.5),
    'gray':  dict(linear_rgb=(.14, .14, .14), roughness=.5),
    'white': dict(linear_rgb=(.85, .85, .85), roughness=.5),
}
FLOOR_TEXTURES = ('floor_wood_dark_walnut_v1.png', 'floor_wood_cool_gray_black_v1.png',
                  'floor_wood_light_scandi_v1.png')
LIGHTS = {
    'ceiling': tuple(f'/World/Background/Lights/Ceil_{i}' for i in range(8)),
    'daylight': ('/World/Background/Lights/Daylight_Cube',
                 '/World/Background/Lights/Daylight_Sofa'),
}
# 区间与形状都来自 700 条回合的实测；mean/sd 是归一化到 [0,1] 之后拟合的截断正态。
SHAPES = {
    ('ceiling', 'intensity'): dict(low=.6501, high=1.3496, mean=.4269, sd=.2161),
    ('ceiling', 'kelvin'): dict(low=3600.1587, high=6198.6000, mean=.4260, sd=.2485),
    ('daylight', 'intensity'): dict(low=.3505, high=1.4494, mean=.4771, sd=.2571),
    ('daylight', 'kelvin'): dict(low=4706.8059, high=7499.0353, mean=.4737, sd=.2339),
}
TINT_SHAPE = dict(mean=.4823, sd=.2442)
TINTS = (
    ('walls', '/World/Background/Shell/WallMat', (.70, 1.25)),
    ('ceiling_shell', '/World/Background/Shell/CeilMat', (.75, 1.15)),
    ('floor', '/World/Background/Shell/FloorMat', (.70, 1.30)),
    ('skirt', '/World/Background/Shell/SkirtMat', (.70, 1.20)),
    ('door', '/World/Assets/Door', (.75, 1.25)),
    ('fireplace', '/World/Assets/Fireplace', (.80, 1.20)),
    ('night_stand', '/World/Assets/NightStand', (.75, 1.25)),
    ('shelving_unit', '/World/Assets/ShelvingUnit', (.75, 1.25)),
    ('small_coffee_table', '/World/Assets/SmallCoffeeTable', (.75, 1.25)),
    ('sofa', '/World/Assets/Sofa', (.75, 1.25)),
    ('tv_table', '/World/Assets/TvTable', (.75, 1.25)),
)
TRAY_MESH = ('/World/Assets/Tray/Obj/Meshes/Sketchfab_model/'
             '_99fa031bb144e19baa97558c9a1ba91_fbx/RootNode/OPM0047/'
             'OPM0047_OPM0047_0/OPM0047_OPM0047_0')
TABLE_MATERIAL = '/World/Assets/CoffeeTable/Looks/CoffeeTableMat'
TABLE_MESHES = ('/World/Assets/CoffeeTable/Mesh',)


@dataclass(frozen=True)
class Appearance:
    """一次外观抽样的完整记录。物理量一个都不含。"""
    schema: str
    source_policy: str
    seed_key: str
    table: dict
    tray: dict
    floor_texture: str
    lights: tuple
    tints: tuple
    physics_effect: str = 'none; render-only'


def _truncated_normal(rng, low, high, mean, sd):
    """归一化截断正态，再映射回 [low, high]。拒绝采样，落在区间外就重抽。"""
    for _ in range(64):
        value = rng.normal(mean, sd)
        if 0. <= value <= 1.:
            return float(low + value*(high-low))
    return float(low + min(max(mean, 0.), 1.)*(high-low))


def _rng(api_id, seed):
    key = f'{POLICY}:{api_id}:{seed}:appearance'
    return np.random.default_rng(
        int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big')), key


def sample(api_id, seed):
    """独立于回合流抽样，不打乱布局采样器已逐值对齐的抽取顺序。"""
    rng, key = _rng(api_id, seed)
    table_id = sorted(TABLE_LOOKS)[int(rng.integers(len(TABLE_LOOKS)))]
    tray_id = sorted(TRAY_COLOURS)[int(rng.integers(len(TRAY_COLOURS)))]
    floor = FLOOR_TEXTURES[int(rng.integers(len(FLOOR_TEXTURES)))]
    lights = []
    for group in sorted(LIGHTS):
        for prim in LIGHTS[group]:
            lights.append(dict(
                prim_path=prim, group=group,
                intensity_scale=_truncated_normal(rng, **SHAPES[(group, 'intensity')]),
                color_temperature_k=_truncated_normal(rng, **SHAPES[(group, 'kelvin')])))
    tints = tuple(dict(id=name, prim_path=prim,
                       rgb_scale=tuple(_truncated_normal(rng, span[0], span[1],
                                                         **TINT_SHAPE) for _ in range(3)))
                  for name, prim, span in TINTS)
    return Appearance(
        schema=POLICY, source_policy=SOURCE_POLICY, seed_key=key,
        table=dict(id=table_id, material_path=TABLE_MATERIAL, meshes=list(TABLE_MESHES),
                   texture=f'{TEXTURES}/{TABLE_TEXTURE}', **TABLE_LOOKS[table_id]),
        tray=dict(id=tray_id, bound_meshes=[TRAY_MESH], pattern='uniform_solid',
                  **TRAY_COLOURS[tray_id]),
        floor_texture=f'{TEXTURES}/{floor}',
        lights=tuple(lights), tints=tints)


def record(appearance):
    return asdict(appearance)
