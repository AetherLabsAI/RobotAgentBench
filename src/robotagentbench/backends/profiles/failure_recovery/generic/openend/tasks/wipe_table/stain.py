"""Deterministic wipeable-stain state for task st18-1 (pure numpy, no Isaac).

Pass-based counting (Supervisor decision Q1, 2026-10-13):
* A cell is *covered* in one control step when, at some physics substep of
  that step, (a) the cell centre lies inside the horizontal projection of the
  tool wiping face, (b) the wiping face is <= 5 mm above the table top and
  (c) the tool touches the table.
* A pass is counted only on an uncovered -> covered transition, and only if
  the cell was uncovered for >= 4 consecutive control steps before (boundary
  jitter guard). 3 passes clear a cell.
* remaining k = max(0, 1 - passes/3) -> 1, 2/3, 1/3, 0.
* coverage = sum_stained(f_i * k_i) / sum_stained(f_i)  (asset_layout.json
  coverage_definition, f_i = stain_fraction).
* Rendered opacity: displayOpacity = base * s(k), s(k) = T/(1-k(1-T)), s(0)=0
  (asset runtime_rule; with the 0.5 cutout this makes the visible stained area
  of a cell ~= k). Table s = 1, 0.75, 0.6, 0 for 0..3 passes.
No random numbers, no wall clock: identical inputs give identical states.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = 'arena.wipe_table.stain_state.v1'
PASSES_TO_CLEAR = 3
REARM_UNCOVERED_STEPS = 4
FACE_HEIGHT_TOLERANCE_M = 0.005
# Bristle collider (all_assets/wipe_tool_hand_brush, Collision/BristleBlock):
# x +-0.075, y +-0.029, z 0 in the tool root frame (origin = bottom centre).
WIPING_FACE_HALF_EXTENTS_M = (0.075, 0.029)


def opacity_scale(remaining, threshold):
    remaining = np.asarray(remaining, dtype=float)
    out = np.zeros_like(remaining)
    positive = remaining > 0
    out[positive] = threshold / (1.0 - remaining[positive] * (1.0 - threshold))
    return out


class StainLayout:
    """Grid facts from asset_layout.json plus the authored world placement."""

    def __init__(self, layout, origin_world_xy, table_top_z_m):
        grid = layout['grid']
        self.nx, self.ny = int(grid['nx']), int(grid['ny'])
        self.count = self.nx * self.ny
        self.vertices_per_cell = int(grid['vertices_per_cell'])
        self.cell_size = np.asarray(grid['cell_size_m'], dtype=float)
        self.region = np.asarray(grid['region_size_m'], dtype=float)
        self.threshold = float(grid['cutout_threshold'])
        if int(grid['wipes_to_clear']) != PASSES_TO_CLEAR:
            raise ValueError('Asset wipes_to_clear differs from the approved 3 passes')
        cells = sorted(layout['cells'], key=lambda c: c['k'])
        if [c['k'] for c in cells] != list(range(self.count)):
            raise ValueError('Incomplete stain cell table')
        if any(c['k'] != c['iy'] * self.nx + c['ix'] for c in cells):
            raise ValueError('Unexpected stain cell order')
        self.fraction = np.asarray([c['stain_fraction'] for c in cells], dtype=float)
        self.stained = np.asarray([bool(c['stained']) for c in cells])
        if not self.stained.any() or (self.fraction[self.stained] <= 0).any():
            raise ValueError('Stain layout has no positive stained cells')
        ix = np.asarray([c['ix'] for c in cells], dtype=float)
        iy = np.asarray([c['iy'] for c in cells], dtype=float)
        local = np.stack([-self.region[0] / 2 + (ix + .5) * self.cell_size[0],
                          -self.region[1] / 2 + (iy + .5) * self.cell_size[1]], axis=1)
        self.origin = np.asarray(origin_world_xy, dtype=float).reshape(2)
        self.centers = local + self.origin  # placement is a pure translation (checked by reader)
        self.table_top = float(table_top_z_m)
        expected = {str(i): float(v) for i, v in enumerate(opacity_scale(
            [1 - i / PASSES_TO_CLEAR for i in range(PASSES_TO_CLEAR + 1)], self.threshold))}
        table = {k: float(v) for k, v in grid['scale_for_wipes'].items()}
        if any(abs(table[k] - expected[k]) > 1e-6 for k in expected):
            raise ValueError('Asset scale_for_wipes differs from runtime_rule')

    @classmethod
    def from_file(cls, path, origin_world_xy, table_top_z_m):
        return cls(json.loads(Path(path).read_text()), origin_world_xy, table_top_z_m)

    def description(self):
        return dict(nx=self.nx, ny=self.ny, cells=self.count, stained_cells=int(self.stained.sum()),
                    origin_world_xy=self.origin.tolist(), table_top_z_m=self.table_top,
                    cell_size_m=self.cell_size.tolist(), cutout_threshold=self.threshold)


def face_corners(position, rotation_matrix, half=WIPING_FACE_HALF_EXTENTS_M):
    """World corners (4,3) of the wiping face from the tool root pose."""
    hx, hy = half
    local = np.asarray([[-hx, -hy, 0.], [hx, -hy, 0.], [hx, hy, 0.], [-hx, hy, 0.]])
    return np.asarray(position, dtype=float) + local @ np.asarray(rotation_matrix, dtype=float).T


def points_in_quad(points, quad):
    """Points (N,2) inside a convex quadrilateral (4,2) given in order; edges inclusive."""
    points = np.asarray(points, dtype=float)
    quad = np.asarray(quad, dtype=float)
    a = quad
    b = np.roll(quad, -1, axis=0)
    cross = ((b[:, 0] - a[:, 0])[None, :] * (points[:, 1:2] - a[:, 1][None, :])
             - (b[:, 1] - a[:, 1])[None, :] * (points[:, 0:1] - a[:, 0][None, :]))
    eps = 1e-12
    return np.all(cross >= -eps, axis=1) | np.all(cross <= eps, axis=1)


def covered_cells(layout, corners_world, tool_table_contact):
    """One physics sample -> (bool[N] covered, face height above table)."""
    corners = np.asarray(corners_world, dtype=float)
    height = float(corners[:, 2].min() - layout.table_top)
    if not tool_table_contact or height > FACE_HEIGHT_TOLERANCE_M:
        return np.zeros(layout.count, dtype=bool), height
    return points_in_quad(layout.centers, corners[:, :2]), height


class StainState:
    """Pass counter. Feed exactly one covered mask per control step."""

    def __init__(self, layout):
        self.layout = layout
        self.passes = np.zeros(layout.count, dtype=np.int64)
        self.covered = np.zeros(layout.count, dtype=bool)
        # Treat the pre-episode history as 'long uncovered' so the first cover counts.
        self.uncovered_run = np.full(layout.count, REARM_UNCOVERED_STEPS, dtype=np.int64)
        self.steps = 0
        self.initial_coverage = self.coverage()

    def remaining(self):
        return np.maximum(0.0, 1.0 - self.passes / PASSES_TO_CLEAR)

    def coverage(self):
        f = self.layout.fraction[self.layout.stained]
        k = self.remaining()[self.layout.stained]
        return float(np.sum(f * k) / np.sum(f))

    def scale(self):
        return opacity_scale(self.remaining(), self.layout.threshold)

    def update(self, covered):
        covered = np.asarray(covered, dtype=bool)
        if covered.shape != (self.layout.count,):
            raise ValueError('Covered mask has the wrong shape')
        rising = covered & ~self.covered & (self.uncovered_run >= REARM_UNCOVERED_STEPS)
        newly = np.flatnonzero(rising & (self.passes < PASSES_TO_CLEAR))
        self.passes[newly] += 1
        self.uncovered_run = np.where(covered, 0, self.uncovered_run + 1)
        self.covered = covered
        self.steps += 1
        return newly

    def force(self, cells, passes):
        """Smoke-test only (recorded in the reset contract): set pass counts."""
        cells = np.asarray(cells, dtype=np.int64)
        if ((cells < 0) | (cells >= self.layout.count)).any() or not 0 <= passes <= PASSES_TO_CLEAR:
            raise ValueError('Invalid forced stain state')
        self.passes[cells] = passes

    def vertex_opacity(self, base):
        base = np.asarray(base, dtype=np.float32)
        if base.shape != (self.layout.count * self.layout.vertices_per_cell,):
            raise ValueError('Base opacity does not match the grid')
        return (base * np.repeat(self.scale(), self.layout.vertices_per_cell)).astype(np.float32)

    def record(self, step, time_s, newly, tool_table_contact, face_height_m, covered_any):
        """Deterministic stain_state.jsonl row (simulation time only, no wall clock)."""
        return dict(schema=SCHEMA, step=int(step), sim_time_s=round(float(time_s), 9),
                    passes=self.passes.tolist(),
                    remaining=[round(float(v), 9) for v in self.remaining()],
                    opacity_scale=[round(float(v), 9) for v in self.scale()],
                    coverage=round(self.coverage(), 12),
                    newly_passed_cells=[int(i) for i in newly],
                    covered_cells=[int(i) for i in np.flatnonzero(self.covered)],
                    tool_table_contact=bool(tool_table_contact),
                    face_height_m=None if face_height_m is None else round(float(face_height_m), 9),
                    covered_any_substep=bool(covered_any))


def state_digest(rows):
    """sha256 over deterministic rows (for replay comparison)."""
    h = hashlib.sha256()
    for row in rows:
        h.update(json.dumps(row, sort_keys=True, allow_nan=False).encode())
        h.update(b'\n')
    return h.hexdigest()
