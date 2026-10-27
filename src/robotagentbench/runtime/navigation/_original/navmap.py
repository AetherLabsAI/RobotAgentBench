"""Occupancy queries for the kinematic base: footprint collision, ray clearance, reachability."""
import math
from collections import deque
from pathlib import Path

import numpy as np


def dilate(mask, radius_cells):
    """Binary dilation with a disk (numpy only)."""
    out = mask.copy()
    r = int(math.ceil(radius_cells))
    padded = np.pad(mask, r)
    rows, cols = mask.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if (dx or dy) and dx * dx + dy * dy <= radius_cells * radius_cells:
                out |= padded[r + dy:r + dy + rows, r + dx:r + dx + cols]
    return out


class NavMap:
    def __init__(self, scene_dir, radius=None, margin=0.02):
        scene_dir = Path(scene_dir)
        import json
        self.scene = json.loads((scene_dir / 'scene.json').read_text())
        data = np.load(scene_dir / 'maps.npz')
        self.origin = np.array(self.scene['map']['origin'], dtype=float)
        self.res = float(self.scene['map']['resolution'])
        self.blocked = data['blocked']
        self.height = data['height']
        self.shape = self.blocked.shape
        self.radius = float(radius if radius is not None else self.scene['robot_radius_m'])
        # a cell is free for the base centre when no blocked cell lies within the footprint radius
        self.occupied = dilate(self.blocked, (self.radius + margin) / self.res)
        self.voxel_size = float(data['voxel_size'])
        vshape = tuple(int(v) for v in data['voxel_shape'])
        self.voxels = np.unpackbits(data['voxels'])[:vshape[0] * vshape[1] * vshape[2]].reshape(vshape).astype(bool)
        self.reachable = self._flood(self.scene['spawn'][:2])
        # planning keeps two extra cells of clearance where possible, so executed paths do not graze obstacles
        self.roomy = self.reachable & ~dilate(self.occupied, 2.0)
        self.snug = self.reachable & ~dilate(self.occupied, 1.0)      # one cell of clearance: enough for a straight segment

    # -- grid helpers ------------------------------------------------------
    def cell(self, x, y):
        return int((y - self.origin[1]) / self.res), int((x - self.origin[0]) / self.res)

    def point(self, row, col):
        return self.origin[0] + (col + 0.5) * self.res, self.origin[1] + (row + 0.5) * self.res

    def inside(self, row, col):
        return 0 <= row < self.shape[0] and 0 <= col < self.shape[1]

    def base_free(self, x, y):
        row, col = self.cell(x, y)
        return self.inside(row, col) and not self.occupied[row, col]

    def segment_snug(self, x0, y0, x1, y1):
        """True when the straight segment keeps one cell of clearance all the way (safe to cut a corner along it)."""
        n = int(math.hypot(x1 - x0, y1 - y0) / (self.res / 2)) + 1
        for i in range(n + 1):
            row, col = self.cell(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n)
            if not (self.inside(row, col) and self.snug[row, col]):
                return False
        return True

    def point_free(self, x, y, z):
        """True when the point (head camera) is not inside a voxel that contains scene geometry."""
        v = self.voxel_size
        k, j, i = int(z / v), int((y - self.origin[1]) / v), int((x - self.origin[0]) / v)
        if not (0 <= k < self.voxels.shape[0] and 0 <= j < self.voxels.shape[1] and 0 <= i < self.voxels.shape[2]):
            return True
        return not self.voxels[k, j, i]

    def clearance(self, x, y, heading, max_range=5.0):
        """Distance the base centre can travel along heading before its footprint is blocked."""
        step = self.res / 2
        dx, dy = math.cos(heading) * step, math.sin(heading) * step
        d = 0.0
        while d < max_range:
            if not self.base_free(x + dx * (d / step + 1), y + dy * (d / step + 1)):
                return d
            d += step
        return max_range

    def scan(self, x, y, yaw, rays=24, max_range=5.0):
        """Planar range scan from the base centre to the nearest obstacle surface (not footprint-inflated)."""
        out = []
        step = self.res / 2
        for k in range(rays):
            angle = yaw + 2 * math.pi * k / rays
            dx, dy = math.cos(angle) * step, math.sin(angle) * step
            d, px, py = 0.0, x, y
            while d < max_range:
                px += dx
                py += dy
                d += step
                row, col = self.cell(px, py)
                if not self.inside(row, col) or self.blocked[row, col]:
                    break
            out.append(round(min(d, max_range), 2))
        return out

    def _flood(self, start):
        reach = np.zeros(self.shape, dtype=bool)
        row, col = self.cell(*start)
        if not self.inside(row, col) or self.occupied[row, col]:
            free = np.argwhere(~self.occupied)
            if not len(free):
                return reach
            row, col = free[np.argmin(((free - np.array([row, col])) ** 2).sum(axis=1))]
        queue = deque([(int(row), int(col))])
        reach[row, col] = True
        while queue:
            r, c = queue.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if self.inside(rr, cc) and not reach[rr, cc] and not self.occupied[rr, cc]:
                    reach[rr, cc] = True
                    queue.append((rr, cc))
        return reach

    def nearest_free(self, x, y):
        row, col = self.cell(x, y)
        free = np.argwhere(self.reachable)
        best = free[np.argmin(((free - np.array([row, col])) ** 2).sum(axis=1))]
        return self.point(int(best[0]), int(best[1]))

    def path(self, start, goal, allowed=None):
        """World waypoints from start to goal, or None. allowed: optional boolean grid the path must stay inside
        (the robot's explored map).

        Passages with less than one cell (4 cm) of spare clearance are avoided: executed motion is never exact, and
        a base that grazes an obstacle gets its motion refused. Such cells are only used right at the start and at
        the goal, or when there is no other way at all.
        """
        for strict in (True, False):
            found = self._path(start, goal, allowed, strict)
            if found:
                return found
        return None

    def _near(self, cell, radius_cells=8):
        mask = np.zeros(self.shape, dtype=bool)
        r0, r1 = max(0, cell[0] - radius_cells), min(self.shape[0], cell[0] + radius_cells + 1)
        c0, c1 = max(0, cell[1] - radius_cells), min(self.shape[1], cell[1] + radius_cells + 1)
        mask[r0:r1, c0:c1] = True
        return mask

    def _path(self, start, goal, allowed, strict):
        import heapq
        reachable = self.reachable if allowed is None else (self.reachable & allowed)
        s, g = self.cell(*start), self.cell(*goal)
        if not (self.inside(*s) and self.inside(*g)) or not reachable[g]:
            return None
        if not reachable[s]:
            free = np.argwhere(reachable)
            if not len(free):
                return None
            s = tuple(int(v) for v in free[np.argmin(((free - np.array(s)) ** 2).sum(axis=1))])
        roomy, snug = self.roomy & reachable, self.snug & reachable
        if strict:
            reachable = snug | (reachable & (self._near(s) | self._near(g)))
        queue, cost, parent = [(0.0, s)], {s: 0.0}, {}
        moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, 1.4142), (1, -1, 1.4142),
                 (-1, 1, 1.4142), (-1, -1, 1.4142)]
        while queue:
            _, node = heapq.heappop(queue)
            if node == g:
                break
            for dr, dc, w in moves:
                nxt = (node[0] + dr, node[1] + dc)
                if not self.inside(*nxt) or not reachable[nxt]:
                    continue
                if dr and dc and not (reachable[node[0] + dr, node[1]] and reachable[node[0], node[1] + dc]):
                    continue
                c = cost[node] + w * (1.0 if roomy[nxt] else 3.0 if snug[nxt] else 15.0)
                if c < cost.get(nxt, float('inf')):
                    cost[nxt], parent[nxt] = c, node
                    heapq.heappush(queue, (c + math.hypot(nxt[0] - g[0], nxt[1] - g[1]), nxt))
        if g not in parent and g != s:
            return None
        cells = [g]
        while cells[-1] != s:
            cells.append(parent[cells[-1]])
        cells.reverse()
        # line-of-sight shortcutting: long straight segments through roomy cells; where the passage is narrower,
        # through cells with one cell of clearance; single steps only where there is no clearance to spare
        simple, i = [cells[0]], 0
        while i < len(cells) - 1:
            for mask in (roomy, snug):
                j = len(cells) - 1
                while j > i + 1 and not self._line_free(cells[i], cells[j], mask):
                    j -= 1
                if j > i + 1:
                    break
            else:
                j = i + 1                                     # tight cell: follow the searched cells exactly
            simple.append(cells[j])
            i = j
        return [(float(x), float(y)) for x, y in (self.point(*c) for c in simple)]

    def _line_free(self, a, b, mask=None):
        mask = self.reachable if mask is None else mask
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2) + 1
        for t in np.linspace(0, 1, n + 1):
            r, c = int(round(a[0] + (b[0] - a[0]) * t)), int(round(a[1] + (b[1] - a[1]) * t))
            if not mask[r, c]:
                return False
        return True

    def render(self, path=None, robot=None, points=None, scale=3):
        """Schematic top-down image (PIL): height shading, blocked cells, reachable area, trajectory."""
        from PIL import Image, ImageDraw
        h = np.clip(self.height, 0, 2.2) / 2.2
        img = np.stack([40 + 150 * h, 44 + 150 * h, 52 + 150 * h], axis=-1)
        img[self.height < 0] = (18, 18, 22)
        img[self.reachable] = (232, 226, 205)
        img[self.blocked & ~self.reachable] = np.minimum(img[self.blocked & ~self.reachable] * 0.75 + 20, 255)
        image = Image.fromarray(img.astype(np.uint8)[::-1]).resize((self.shape[1] * scale, self.shape[0] * scale), Image.NEAREST)
        draw = ImageDraw.Draw(image)

        def px(x, y):
            return ((x - self.origin[0]) / self.res * scale, (self.shape[0] - (y - self.origin[1]) / self.res) * scale)

        if path and len(path) > 1:
            draw.line([px(x, y) for x, y in path], fill=(30, 110, 220), width=2)
        for x, y, color in points or []:
            u, v = px(x, y)
            draw.ellipse([u - 4, v - 4, u + 4, v + 4], fill=color)
        if robot:
            x, y, yaw = robot
            u, v = px(x, y)
            r = self.radius / self.res * scale
            draw.ellipse([u - r, v - r, u + r, v + r], outline=(220, 40, 40), width=2)
            draw.line([u, v, u + math.cos(yaw) * r * 1.6, v - math.sin(yaw) * r * 1.6], fill=(220, 40, 40), width=2)
        return image
