"""Hard gate for the kinematic arm: robot-furniture interpenetration per frame.

The Newton pipeline does not resolve robot/furniture contact (the arm is
prescribed), so a command sequence could pass through the table or basket.
This audit measures it in both directions:
  (a) robot surface samples (gripper CAD vertices, primitive surface samples)
      inside furniture meshes (signed distance by parity query),
  (b) furniture mesh vertices inside robot primitives (analytic box/cylinder/sphere).
Reports per-frame maximum penetration and the offending link/furniture.

usage: robot_furniture_audit.py <trial_dir> [--stride 2] [--limit-mm 2]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation

REPO = Path("/home/aether/workspace/agent_bench")
sys.path.insert(0, str(REPO / "experiments/physics_fluid"))


@wp.kernel
def signed_inside(mesh: wp.uint64, points: wp.array(dtype=wp.vec3), depth: wp.array(dtype=float)):
    i = wp.tid()
    q = wp.mesh_query_point_sign_parity(mesh, points[i], 0.2)
    if q.result and q.sign < 0.0:
        p = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
        depth[i] = wp.length(points[i] - p)
    else:
        depth[i] = 0.0


def primitive_samples(prim, n=12):
    t = prim["type"]
    if t == "box":
        h = np.asarray(prim["half_extents"])
        g = np.linspace(-1, 1, n)
        pts = []
        for ax in range(3):
            o = [i for i in range(3) if i != ax]
            A, B = np.meshgrid(g, g)
            for s in (-1, 1):
                p = np.zeros((A.size, 3)); p[:, ax] = s; p[:, o[0]] = A.ravel(); p[:, o[1]] = B.ravel()
                pts.append(p * h)
        pts = np.vstack(pts)
    elif t == "cylinder":
        r, hh = prim["radius"], prim["half_height"]
        a = np.linspace(0, 2 * np.pi, 3 * n, endpoint=False)
        z = np.linspace(-hh, hh, n)
        A, Z = np.meshgrid(a, z)
        side = np.c_[r * np.cos(A.ravel()), r * np.sin(A.ravel()), Z.ravel()]
        rr = np.linspace(0, r, n // 2)
        RR, AA = np.meshgrid(rr, a)
        caps = [np.c_[RR.ravel() * np.cos(AA.ravel()), RR.ravel() * np.sin(AA.ravel()), np.full(RR.size, s * hh)] for s in (-1, 1)]
        pts = np.vstack([side] + caps)
    else:
        r = prim["radius"]
        u, v = np.meshgrid(np.linspace(0, np.pi, n), np.linspace(0, 2 * np.pi, 2 * n))
        pts = r * np.c_[np.sin(u.ravel()) * np.cos(v.ravel()), np.sin(u.ravel()) * np.sin(v.ravel()), np.cos(u.ravel())]
    R = Rotation.from_quat(prim["quaternion_xyzw"]).as_matrix()
    return pts @ R.T + np.asarray(prim["position"])


def inside_primitive(prim, local_pts):
    R = Rotation.from_quat(prim["quaternion_xyzw"]).as_matrix()
    p = (local_pts - np.asarray(prim["position"])) @ R
    t = prim["type"]
    if t == "box":
        h = np.asarray(prim["half_extents"])
        d = h - np.abs(p)
        return np.where((d > 0).all(1), d.min(1), 0.0)
    if t == "cylinder":
        dr = prim["radius"] - np.linalg.norm(p[:, :2], axis=1)
        dz = prim["half_height"] - np.abs(p[:, 2])
        return np.where((dr > 0) & (dz > 0), np.minimum(dr, dz), 0.0)
    d = prim["radius"] - np.linalg.norm(p, axis=1)
    return np.maximum(d, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trial")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--limit-mm", type=float, default=2.0)
    a = ap.parse_args()
    from tablecloth_scene_binding import load_scene_collisions
    trial = Path(a.trial)
    s = np.load(trial / "full_state.npz")
    meta = json.loads((trial / "trial.json").read_text())
    cfg = meta["config"]
    rpath = Path(cfg["robot"]) if cfg["robot"].startswith("/") else REPO / cfg["robot"]
    robot = json.loads(rpath.read_text())
    cal = np.load(rpath.with_suffix(".npz"))
    meshes = load_scene_collisions(np.load(cfg["geometry"], allow_pickle=False))
    wp.init()
    fur = []
    for e, v, f in meshes:
        fur.append((e["group"] + ":" + e["prim_path"].split("/")[-1], wp.Mesh(points=wp.array(v, dtype=wp.vec3, device="cpu"),
                    indices=wp.array(f.reshape(-1), dtype=wp.int32, device="cpu")), v))
    bn = list(s["body_names"])
    samples = {}
    for shape in robot.get("nonhand_collision_shapes", []):
        if shape.get("primitive") and shape["link"] in bn:
            samples.setdefault(shape["link"], []).append(("prim", shape["primitive"], primitive_samples(shape["primitive"])))
    for link in cal["links"].tolist():
        if link in bn and f"points_{link}" in cal.files:
            pts = cal[f"points_{link}"]
            samples.setdefault(link, []).append(("cad", None, pts[:: max(1, len(pts) // 400)]))
    rows = []
    frames = range(0, len(s["body_q"]), a.stride)
    for fi in frames:
        worst = (0.0, None, None)
        for link, items in samples.items():
            x = s["body_q"][fi, bn.index(link)]
            R = Rotation.from_quat(x[3:]).as_matrix(); t = x[:3]
            for kind, prim, pts in items:
                world = pts @ R.T + t
                for name, mesh, verts in fur:
                    d = wp.zeros(len(world), dtype=float, device="cpu")
                    wp.launch(signed_inside, dim=len(world), inputs=[mesh.id, wp.array(world.astype(np.float32), dtype=wp.vec3, device="cpu"), d], device="cpu")
                    m = float(d.numpy().max())
                    if m > worst[0]:
                        worst = (m, link, name)
                    if kind == "prim":
                        lo, hi = world.min(0) - .05, world.max(0) + .05
                        cand = verts[np.all((verts >= lo) & (verts <= hi), 1)]
                        if len(cand):
                            dd = float(inside_primitive(prim, (cand - t) @ R).max())
                            if dd > worst[0]:
                                worst = (dd, link, name)
        rows.append(dict(frame=fi, max_penetration_mm=round(worst[0] * 1e3, 2), link=worst[1], furniture=worst[2]))
    bad = [r for r in rows if r["max_penetration_mm"] > a.limit_mm]
    summary = dict(frames_checked=len(rows), limit_mm=a.limit_mm, frames_over_limit=len(bad),
                   worst=max(rows, key=lambda r: r["max_penetration_mm"]) if rows else None,
                   first_over_limit=bad[0] if bad else None,
                   gate_pass=not bad)
    print(json.dumps(summary, indent=1))
    (trial / "robot_furniture_audit.json").write_text(json.dumps(dict(summary=summary, rows=rows), indent=1))


if __name__ == "__main__":
    main()
