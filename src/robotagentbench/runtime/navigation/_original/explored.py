"""The robot's own map: the part of the floor plan that has been inside the head camera's view.

This is what an on-board mapping stack would give a real robot. It holds no task information, only
free / blocked floor cells, and it grows only where the camera has looked. The navigation commands
(goto, approach) plan inside this map, so they cannot be used to drive through space that was never seen.
"""
import math

import numpy as np

from .navmap import dilate

MAX_RANGE = 5.0      # m, mapping range of the head camera
FILL = 0.3           # m, unseen floor this close to seen floor still counts as mapped
SELF_RADIUS = 0.55   # m, the floor right around the robot is always known
LOW_HEIGHT = 1.4     # m, an obstacle lower than this is furniture the camera can look down onto


class ExploredMap:
    def __init__(self, nav):
        self.nav = nav
        self.known = np.zeros(nav.shape, dtype=bool)
        self._tall_cache = {}
        self._drivable = None

    def _tall(self, camera_z):
        """Cells that block the camera's line of sight at this camera height (walls, tall furniture)."""
        key = round(camera_z * 10)
        if key not in self._tall_cache:
            self._tall_cache[key] = self.nav.height >= (key / 10.0 - 0.05)
        return self._tall_cache[key]

    def reveal(self, camera_pose, hfov_deg, vfov_deg, base_xy):
        nav = self.nav
        position, forward = camera_pose[:3, 3], -camera_pose[:3, 2]
        heading = math.atan2(forward[1], forward[0])
        pitch_down = math.degrees(math.asin(max(-1.0, min(1.0, -float(forward[2])))))
        z = max(0.3, float(position[2]))
        top = pitch_down - vfov_deg / 2          # elevation of the top image row below the horizon
        far = MAX_RANGE if top <= 2.0 else min(MAX_RANGE, z / math.tan(math.radians(top)))
        # The floor between the robot and the lower image edge is not in the picture, but it is counted as mapped
        # (a real base has a short-range sensor for it); otherwise every plan would start with a gap.
        near = 0.0
        if far > near:
            angles = heading + np.radians(np.linspace(-hfov_deg / 2, hfov_deg / 2, 121))
            dist = np.arange(nav.res / 2, far, nav.res / 2)
            xs = position[0] + np.cos(angles)[:, None] * dist[None, :]
            ys = position[1] + np.sin(angles)[:, None] * dist[None, :]
            rows = ((ys - nav.origin[1]) / nav.res).astype(int)
            cols = ((xs - nav.origin[0]) / nav.res).astype(int)
            inside = (rows >= 0) & (rows < nav.shape[0]) & (cols >= 0) & (cols < nav.shape[1])
            r, c = np.clip(rows, 0, nav.shape[0] - 1), np.clip(cols, 0, nav.shape[1] - 1)
            wall = self._tall(z)[r, c] | ~inside
            before_wall = (np.cumsum(wall, axis=1) - wall) == 0      # up to and including the first blocking cell
            seen = before_wall & inside & (dist[None, :] >= near)
            self.known[r[seen], c[seen]] = True
        row, col = nav.cell(base_xy[0], base_xy[1])
        k = int(SELF_RADIUS / nav.res)
        r0, r1, c0, c1 = max(0, row - k), min(nav.shape[0], row + k + 1), max(0, col - k), min(nav.shape[1], col + k + 1)
        yy, xx = np.ogrid[r0:r1, c0:c1]
        self.known[r0:r1, c0:c1] |= (yy - row) ** 2 + (xx - col) ** 2 <= k * k

    # -- queries -------------------------------------------------------------
    def drivable(self):
        """Cells the base centre may be sent to: seen (or within 0.3 m of seen floor), and free for the footprint.

        The 0.3 m fill closes the thin unseen slivers behind table legs, poles and shelf posts; without it those
        shadows cut the map into pieces that the planner cannot connect.
        """
        count = int(self.known.sum())
        if self._drivable is None or self._drivable[0] != count:
            self._drivable = (count, dilate(self.known, FILL / self.nav.res) & self.nav.reachable)
        return self._drivable[1]

    def fraction(self):
        total = int(self.nav.reachable.sum())
        return float((self.known & self.nav.reachable).sum()) / total if total else 0.0

    def goals(self):
        """Cells a navigation command may stop at: drivable, with a little clearance to spare."""
        return self.drivable() & self.nav.snug

    def nearest_drivable(self, x, y, max_distance):
        mask = self.goals()
        cells = np.argwhere(mask)
        if not len(cells):
            return None
        row, col = self.nav.cell(x, y)
        d2 = ((cells - np.array([row, col])) ** 2).sum(axis=1)
        i = int(np.argmin(d2))
        if math.sqrt(float(d2[i])) * self.nav.res > max_distance:
            return None
        return self.nav.point(int(cells[i][0]), int(cells[i][1]))

    def standoff(self, hit_xy, robot_xy, distance, band=(0.45, 1.6)):
        """A seen, free floor cell about `distance` away from a point of interest, on the robot's side of it."""
        nav = self.nav
        cells = np.argwhere(self.goals())
        if not len(cells):
            return None
        xs = nav.origin[0] + (cells[:, 1] + 0.5) * nav.res
        ys = nav.origin[1] + (cells[:, 0] + 0.5) * nav.res
        to_hit = np.hypot(xs - hit_xy[0], ys - hit_xy[1])
        keep = (to_hit >= band[0]) & (to_hit <= band[1])
        if not keep.any():
            return None
        xs, ys, to_hit = xs[keep], ys[keep], to_hit[keep]
        cost = np.abs(to_hit - distance) + 0.12 * np.hypot(xs - robot_xy[0], ys - robot_xy[1])
        tall = nav.height >= 1.2
        for i in np.argsort(cost)[:40]:          # first candidate with a clear line of sight to the point
            n = max(2, int(to_hit[i] / (nav.res / 2)))
            lx = np.linspace(xs[i], hit_xy[0], n)[:-4]
            ly = np.linspace(ys[i], hit_xy[1], n)[:-4]
            rr = np.clip(((ly - nav.origin[1]) / nav.res).astype(int), 0, nav.shape[0] - 1)
            cc = np.clip(((lx - nav.origin[0]) / nav.res).astype(int), 0, nav.shape[1] - 1)
            if not tall[rr, cc].any():
                return float(xs[i]), float(ys[i])
        i = int(np.argmin(cost))
        return float(xs[i]), float(ys[i])

    def low(self):
        """Obstacle cells lower than the robot's eye level (tables, sofas, counters): things to look at from above."""
        return self.nav.blocked & (self.nav.height < LOW_HEIGHT)

    def closest_goal(self, x, y):
        """The seen, free floor cell closest to a point, at any distance: ((x, y), distance) or None."""
        cells = np.argwhere(self.goals())
        if not len(cells):
            return None
        xs = self.nav.origin[0] + (cells[:, 1] + 0.5) * self.nav.res
        ys = self.nav.origin[1] + (cells[:, 0] + 0.5) * self.nav.res
        d = np.hypot(xs - x, ys - y)
        i = int(np.argmin(d))
        return (float(xs[i]), float(ys[i])), float(d[i])

    def ray_hit(self, origin, direction, max_range=10.0, step=0.02):
        """First scene surface along a camera ray: (point, 'floor' | 'object') or None."""
        nav = self.nav
        d = np.arange(0.15, max_range, step)
        pts = origin[None, :] + d[:, None] * direction[None, :]
        v = nav.voxel_size
        k = (pts[:, 2] / v).astype(int)
        j = ((pts[:, 1] - nav.origin[1]) / v).astype(int)
        i = ((pts[:, 0] - nav.origin[0]) / v).astype(int)
        inside = (k >= 0) & (k < nav.voxels.shape[0]) & (j >= 0) & (j < nav.voxels.shape[1]) & (i >= 0) & (i < nav.voxels.shape[2])
        solid = np.zeros(len(d), dtype=bool)
        solid[inside] = nav.voxels[k[inside], j[inside], i[inside]]
        floor = pts[:, 2] <= 0.03
        outside = np.zeros(len(d), dtype=bool)          # leaving the mapped volume sideways = the outer wall
        outside[1:] = ~inside[1:] & inside[:-1] & (pts[1:, 2] < nav.voxels.shape[0] * v)
        hits = np.nonzero(solid | floor | outside)[0]
        if not len(hits):
            return None
        n = int(hits[0])
        return pts[n], ('floor' if pts[n][2] <= 0.08 else 'object')

    # -- picture -------------------------------------------------------------
    def render(self, robot, trajectory=None, scale=3, grid_m=1.0):
        """Top-down picture of the explored map with a metric grid, for agents and for the web viewer."""
        from PIL import Image, ImageDraw
        nav = self.nav
        img = np.full(nav.shape + (3,), 118, dtype=np.uint8)                 # unknown
        free = self.known & ~nav.blocked
        img[free] = (244, 241, 232)
        img[self.known & nav.occupied & ~nav.blocked] = (226, 220, 204)      # too close to an obstacle for the base
        # one colour for every obstacle. Marking low furniture separately was tried and withdrawn: a strong VLM then
        # steered by the coloured blobs of the map instead of the camera and ran in circles (see docs/RESULTS.md).
        # The class is still in the cell grid for programs (GET .../map?cells=1, 'o').
        img[self.known & nav.blocked] = (52, 56, 66)
        image = Image.fromarray(img[::-1]).resize((nav.shape[1] * scale, nav.shape[0] * scale), Image.NEAREST)
        draw = ImageDraw.Draw(image)
        width, height = image.size

        def px(x, y):
            return ((x - nav.origin[0]) / nav.res * scale, (nav.shape[0] - (y - nav.origin[1]) / nav.res) * scale)

        x0, y0 = nav.origin
        x1, y1 = x0 + nav.shape[1] * nav.res, y0 + nav.shape[0] * nav.res
        gx = math.ceil(x0 / grid_m) * grid_m
        while gx < x1:
            u = px(gx, 0)[0]
            draw.line([u, 0, u, height], fill=(150, 160, 175), width=1)
            draw.text((u + 2, 2), f'x={gx:g}', fill=(20, 60, 140))
            gx += grid_m
        gy = math.ceil(y0 / grid_m) * grid_m
        while gy < y1:
            v = px(0, gy)[1]
            draw.line([0, v, width, v], fill=(150, 160, 175), width=1)
            draw.text((2, v + 2), f'y={gy:g}', fill=(20, 60, 140))
            gy += grid_m
        if trajectory and len(trajectory) > 1:
            draw.line([px(x, y) for x, y in trajectory], fill=(30, 110, 220), width=2)
        x, y, yaw = robot
        u, v = px(x, y)
        r = nav.radius / nav.res * scale
        draw.ellipse([u - r, v - r, u + r, v + r], outline=(215, 40, 40), width=2)
        draw.line([u, v, u + math.cos(yaw) * r * 2.0, v - math.sin(yaw) * r * 2.0], fill=(215, 40, 40), width=3)
        return image

    def meta(self, scale=3):
        nav = self.nav
        return dict(origin=[float(nav.origin[0]), float(nav.origin[1])], resolution_m=nav.res,
                    px_per_m=round(scale / nav.res, 3), size=[nav.shape[1] * scale, nav.shape[0] * scale],
                    bounds=[float(nav.origin[0]), float(nav.origin[1]), float(nav.origin[0] + nav.shape[1] * nav.res),
                            float(nav.origin[1] + nav.shape[0] * nav.res)],
                    note='pixel (u, v) -> x = origin_x + u / px_per_m, y = bounds_y_max - v / px_per_m',
                    explored_fraction=round(self.fraction(), 3))
