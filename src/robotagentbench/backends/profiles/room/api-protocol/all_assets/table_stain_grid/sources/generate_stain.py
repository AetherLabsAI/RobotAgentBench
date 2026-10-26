#!/usr/bin/env python3
"""Seeded generator for the short-task18 wipeable stain grid (visual only).

Local frame: origin = centre of the region's bottom face (x,y centred), +Z up, z=0 = table top.
x spans the 0.342 m side ([-0.171, 0.171]), y spans the 0.51 m side ([-0.255, 0.255]) (same axes as
all_assets/tablecloth_dirty bounds). Geometry sits at z=+0.75 mm.
No physics schemas, no collision, no rigid body.

Representation (chosen after the RTX proof, see README.md):
  ONE Mesh "/Asset/Grid", NX x NY cells, each cell = SUB x SUB quads with its own
  (unshared) vertices. primvars:displayOpacity (vertex) is read by a
  UsdPrimvarReader_float into UsdPreviewSurface.opacity with a fixed cutout
  threshold T (opacityThreshold, uniform). Diffuse colour comes from a seeded
  RGBA texture (dirt colour + grain).

  base value per vertex: 0 outside blobs, T + (1-T)*u inside, where u is a
  spatially smooth noise rank-transformed to U(0,1) over the in-blob vertices.
  Runtime ("dissolve") for a cell with remaining fraction k = max(0, 1 - wipes/W):
      displayOpacity = base * s(k),  s(k) = T / (1 - k*(1-T))  (k>0),  s(0) = 0
  so the visible area of that cell is ~k of its full-stain area.

Layout = set of blobs, deterministic from --seed. Output is byte-deterministic
(text USD + PNG + JSON). Only numpy + Pillow are needed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

SCHEMA = "short_task18.stain_grid.v2"
REGION_X = 0.342   # m, tablecloth_dirty bounds size[0]
REGION_Y = 0.510   # m, tablecloth_dirty bounds size[1]
NX, NY = 14, 21    # 2.443 cm x 2.429 cm cells, exactly tiling the region (294 cells)
SUB = 6            # 6x6 quads per cell (~4 mm vertex pitch) -> smooth cutout edges
Z_CELL = 0.00075   # m above the table top
TEX_W, TEX_H = 256, 384   # ~1.34 mm / texel colour texture
T = 0.5            # cutout threshold (UsdPreviewSurface.opacityThreshold, uniform)
WIPES_TO_CLEAR = 3
STAINED_FRACTION_MIN = 0.05   # a cell counts as stained if >=5% of its vertices are in a blob
DIRT_RGB = np.array([0.30, 0.22, 0.14])


def scale_for_remaining(k):
    """Runtime multiplier on the base opacity for remaining fraction k in [0, 1]."""
    return 0.0 if k <= 0 else T / (1.0 - k * (1.0 - T))


def value_noise(rng, h, w, octaves):
    out = np.zeros((h, w))
    for cells, amp in octaves:
        gy, gx = cells * h // w + 2, cells + 2
        grid = rng.random((gy, gx))
        ys = np.linspace(0, gy - 1.001, h)
        xs = np.linspace(0, gx - 1.001, w)
        y0, x0 = ys.astype(int), xs.astype(int)
        fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]
        fy, fx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)
        a = grid[np.ix_(y0, x0)]; b = grid[np.ix_(y0, x0 + 1)]
        c = grid[np.ix_(y0 + 1, x0)]; d = grid[np.ix_(y0 + 1, x0 + 1)]
        out += amp * ((a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy)
    return out


def make_layout(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 5))  # 2..4 blobs
    blobs = []
    for _ in range(n):
        rx = float(rng.uniform(0.035, 0.085)); ry = float(rng.uniform(0.03, 0.07))
        margin = max(rx, ry) * 0.8 + 0.01
        blobs.append({
            "center_m": [round(float(rng.uniform(-REGION_X / 2 + margin, REGION_X / 2 - margin)), 5),
                         round(float(rng.uniform(-REGION_Y / 2 + margin, REGION_Y / 2 - margin)), 5)],
            "radii_m": [round(rx, 5), round(ry, 5)],
            "yaw_rad": round(float(rng.uniform(0, math.pi)), 5),
            "darkness": round(float(rng.uniform(0.75, 1.0)), 4),
        })
    return {"seed": int(seed), "blobs": blobs, "noise_seed": int(rng.integers(0, 2**31 - 1))}


def fields(layout):
    """Texture-resolution fields: blob mask in [0,1], dissolve noise, grain. Row 0 = max y."""
    rng = np.random.default_rng(layout["noise_seed"])
    xs = (np.arange(TEX_W) + 0.5) / TEX_W * REGION_X - REGION_X / 2
    ys = REGION_Y / 2 - (np.arange(TEX_H) + 0.5) / TEX_H * REGION_Y
    X, Y = np.meshgrid(xs, ys)
    edge = value_noise(rng, TEX_H, TEX_W, ((8, 0.5), (24, 0.3), (64, 0.2)))
    grain = value_noise(rng, TEX_H, TEX_W, ((48, 0.6), (128, 0.4)))
    dissolve = value_noise(rng, TEX_H, TEX_W, ((20, 0.6), (48, 0.4)))   # ~1 cm patches
    mask = np.zeros((TEX_H, TEX_W)); dark = np.zeros((TEX_H, TEX_W))
    for b in layout["blobs"]:
        cx, cy = b["center_m"]; rx, ry = b["radii_m"]; c, s = math.cos(b["yaw_rad"]), math.sin(b["yaw_rad"])
        dx, dy = X - cx, Y - cy
        u, v = (c * dx + s * dy) / rx, (-s * dx + c * dy) / ry
        r = np.sqrt(u * u + v * v) + 0.55 * (edge - 0.5)   # ragged edge
        a = np.clip((1.0 - r) / 0.25, 0, 1)
        a = a * a * (3 - 2 * a)
        dark = np.where(a > mask, b["darkness"], dark)
        mask = np.maximum(mask, a)
    return mask, dissolve, grain, dark


def colour_texture(mask, grain, dark):
    shade = (0.75 + 0.5 * (grain - 0.5)) * (1.25 - 0.35 * dark)   # darker blobs -> darker dirt
    rgb = np.clip(DIRT_RGB[None, None, :] * shade[..., None], 0, 1)
    rgba = np.dstack([rgb, mask])  # alpha = blob mask (preview only; renderer uses the primvar)
    return np.round(rgba * 255).astype(np.uint8)


def bilinear(A, x, y):
    col = (x + REGION_X / 2) / REGION_X * TEX_W - 0.5
    row = (REGION_Y / 2 - y) / REGION_Y * TEX_H - 0.5
    c0 = np.clip(np.floor(col).astype(int), 0, TEX_W - 2); r0 = np.clip(np.floor(row).astype(int), 0, TEX_H - 2)
    fc = np.clip(col - c0, 0, 1); fr = np.clip(row - r0, 0, 1)
    return ((A[r0, c0] * (1 - fc) + A[r0, c0 + 1] * fc) * (1 - fr)
            + (A[r0 + 1, c0] * (1 - fc) + A[r0 + 1, c0 + 1] * fc) * fr)


def grid_vertices():
    """Cell-major vertex positions: cell k = iy*NX+ix owns vertices [k*V, (k+1)*V)."""
    dx, dy = REGION_X / NX, REGION_Y / NY
    n1 = SUB + 1
    j, i = np.meshgrid(np.arange(n1), np.arange(n1), indexing="ij")
    xs, ys = [], []
    for iy in range(NY):
        for ix in range(NX):
            xs.append(-REGION_X / 2 + ix * dx + i.ravel() * dx / SUB)
            ys.append(-REGION_Y / 2 + iy * dy + j.ravel() * dy / SUB)
    return np.concatenate(xs), np.concatenate(ys)


def base_opacity(layout):
    mask, dissolve, grain, dark = fields(layout)
    x, y = grid_vertices()
    m = bilinear(mask, x, y)
    n = bilinear(dissolve, x, y)
    inside = m >= 0.5
    u = np.zeros_like(n)
    idx = np.flatnonzero(inside)
    order = idx[np.argsort(n[idx], kind="stable")]
    u[order] = (np.arange(len(order)) + 0.5) / max(len(order), 1)
    base = np.where(inside, T + (1 - T) * u, 0.0)
    return np.round(base, 4), x, y, colour_texture(mask, grain, dark)


def f6(v):
    return f"{v:.6f}"


def build_usda(layout, base, x, y, tex):
    V = (SUB + 1) ** 2
    pts = ", ".join(f"({f6(a)}, {f6(b)}, {f6(Z_CELL)})" for a, b in zip(x, y))
    st = ", ".join(f"({f6((a + REGION_X / 2) / REGION_X)}, {f6((b + REGION_Y / 2) / REGION_Y)})" for a, b in zip(x, y))
    ops = ", ".join(f"{v:.4f}".rstrip("0").rstrip(".") if v else "0" for v in base)
    idx = []
    n1 = SUB + 1
    for k in range(NX * NY):
        for jj in range(SUB):
            for ii in range(SUB):
                a = k * V + jj * n1 + ii
                idx += [a, a + 1, a + n1 + 1, a + n1]
    nf = NX * NY * SUB * SUB
    return f'''#usda 1.0
(
    defaultPrim = "Asset"
    metersPerUnit = 1
    upAxis = "Z"
    doc = "short_task18 wipeable stain grid (visual only, no physics); generated by generate_stain.py seed={layout["seed"]}"
)

def Xform "Asset" (
    kind = "component"
    customData = {{
        string schema = "{SCHEMA}"
        int seed = {layout["seed"]}
        int grid_nx = {NX}
        int grid_ny = {NY}
        int sub_quads_per_cell_edge = {SUB}
        int vertices_per_cell = {V}
        string cell_order = "k = iy*grid_nx + ix; cell k owns vertices [k*vertices_per_cell, (k+1)*vertices_per_cell)"
        double cell_size_x_m = {REGION_X / NX:.6f}
        double cell_size_y_m = {REGION_Y / NY:.6f}
        double region_x_m = {REGION_X}
        double region_y_m = {REGION_Y}
        double cell_z_m = {Z_CELL}
        double cutout_threshold = {T}
        int wipes_to_clear = {WIPES_TO_CLEAR}
        string runtime_rule = "displayOpacity = base * s(k), k = max(0, 1 - wipes/wipes_to_clear), s(k) = T/(1-k*(1-T)) for k>0 else 0"
    }}
)
{{
    float3 xformOp:scale = (1, 1, 1)
    uniform token[] xformOpOrder = ["xformOp:scale"]

    def Mesh "Grid" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {{
        uniform bool doubleSided = 0
        float3[] extent = [({f6(-REGION_X / 2)}, {f6(-REGION_Y / 2)}, {f6(Z_CELL)}), ({f6(REGION_X / 2)}, {f6(REGION_Y / 2)}, {f6(Z_CELL)})]
        int[] faceVertexCounts = [{", ".join(["4"] * nf)}]
        int[] faceVertexIndices = [{", ".join(map(str, idx))}]
        rel material:binding = </Asset/Looks/Stain>
        point3f[] points = [{pts}]
        texCoord2f[] primvars:st = [{st}] (interpolation = "vertex")
        float[] primvars:displayOpacity = [{ops}] (interpolation = "vertex")
        uniform token subdivisionScheme = "none"
    }}

    def Scope "Looks"
    {{
        def Material "Stain"
        {{
            token outputs:surface.connect = </Asset/Looks/Stain/PBR.outputs:surface>
            def Shader "PBR"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor.connect = </Asset/Looks/Stain/Tex.outputs:rgb>
                float inputs:opacity.connect = </Asset/Looks/Stain/Opacity.outputs:result>
                float inputs:opacityThreshold = {T}
                float inputs:roughness = 0.85
                float inputs:metallic = 0
                token outputs:surface
            }}
            def Shader "Tex"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @{tex}@
                token inputs:sourceColorSpace = "sRGB"
                float2 inputs:st.connect = </Asset/Looks/Stain/St.outputs:result>
                token inputs:wrapS = "clamp"
                token inputs:wrapT = "clamp"
                float3 outputs:rgb
            }}
            def Shader "St"
            {{
                uniform token info:id = "UsdPrimvarReader_float2"
                string inputs:varname = "st"
                float2 outputs:result
            }}
            def Shader "Opacity"
            {{
                uniform token info:id = "UsdPrimvarReader_float"
                string inputs:varname = "displayOpacity"
                float inputs:fallback = 0
                float outputs:result
            }}
        }}
    }}
}}
'''


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--name", default="asset.usda")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    layout = make_layout(a.seed)
    base, x, y, rgba = base_opacity(layout)
    tex_rel = f"textures/stain_seed{a.seed}.png"
    (a.out / "textures").mkdir(exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(a.out / tex_rel, optimize=False, compress_level=6)
    (a.out / a.name).write_text(build_usda(layout, base, x, y, "./" + tex_rel))
    V = (SUB + 1) ** 2
    cells = []
    for k in range(NX * NY):
        frac = float((base[k * V:(k + 1) * V] > 0).mean())
        cells.append({"k": k, "ix": k % NX, "iy": k // NX, "stain_fraction": round(frac, 4),
                      "stained": frac >= STAINED_FRACTION_MIN})
    stained = [c for c in cells if c["stained"]]
    mass = sum(c["stain_fraction"] for c in stained)
    layout_doc = {"schema": SCHEMA, "layout": layout,
                  "grid": {"nx": NX, "ny": NY, "sub": SUB, "vertices_per_cell": V,
                           "cell_size_m": [REGION_X / NX, REGION_Y / NY], "region_size_m": [REGION_X, REGION_Y],
                           "cell_z_m": Z_CELL, "cutout_threshold": T, "wipes_to_clear": WIPES_TO_CLEAR,
                           "stained_fraction_min": STAINED_FRACTION_MIN,
                           "scale_for_wipes": {str(w): round(scale_for_remaining(max(0.0, 1 - w / WIPES_TO_CLEAR)), 6)
                                               for w in range(WIPES_TO_CLEAR + 1)}},
                  "coverage_definition": "coverage = sum_stained(stain_fraction_i * k_i) / sum_stained(stain_fraction_i), k_i = max(0, 1 - wipes_i/wipes_to_clear)",
                  "summary": {"cells": len(cells), "stained_cells": len(stained),
                              "stain_area_m2": round(mass * REGION_X * REGION_Y / (NX * NY), 6)},
                  "cells": cells}
    lay = a.out / a.name.replace(".usda", "_layout.json")
    lay.write_text(json.dumps(layout_doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps({str(p.relative_to(a.out)): sha(p) for p in (a.out / a.name, a.out / tex_rel, lay)}, indent=1))


if __name__ == "__main__":
    main()
