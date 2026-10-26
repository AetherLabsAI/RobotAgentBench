"""Read-only robot/obstacle contact evidence, not a collision penalty/scorer."""
import json
import math
from functools import partial
from . import scene_obstacles
from .contact_identity import ContactIdentityCache


def classify_pair(colliders, actor_roots=("/World/Robot",)):
    for i in (0, 1):
        kind = scene_obstacles.category(colliders[i])
        other = colliders[1 - i]
        if kind and any(other == root or other.startswith(root + "/") for root in actor_roots):
            return kind
    return None


class ContactLog:
    def __init__(self, stage, output, clock, actor_roots=("/World/Robot",)):
        from pxr import PhysxSchema, PhysicsSchemaTools, Usd
        from omni.physx import get_physx_simulation_interface
        self.actor_roots = tuple(actor_roots)
        self.clock = clock
        self.file = open(output, "x")
        self.counts = {"reports": 0, "reports_with_positive_impulse": 0, "points": 0}
        self._identity = ContactIdentityCache(PhysicsSchemaTools.intToSdfPath,
            partial(classify_pair, actor_roots=self.actor_roots))
        # Rows are serialized synchronously before reuse, never exposed to a
        # caller. Cache only storage, not previous-step contact measurements.
        self._point_pool = []
        self._points = []
        # Contact reporting only. Do not change mass, collision shape, offsets,
        # filters, restitution, drives, or the robot's self-collision policy.
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            for path in self.actor_roots:
                prim = stage.GetPrimAtPath(path)
                if not prim:
                    raise ValueError("Contact report actor missing: " + path)
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
        self.subscription = get_physx_simulation_interface().subscribe_contact_report_events(self.on_contacts)

    def on_contacts(self, headers, data):
        for header in headers:
            paths, _, kind = self._identity.get(header.stage_id, header.collider0, header.collider1)
            if kind is None:
                continue
            points = self._points
            points.clear()
            for row_index, index in enumerate(range(header.contact_data_offset, header.contact_data_offset + header.num_contact_data)):
                p = data[index]
                if row_index < len(self._point_pool):
                    row = self._point_pool[row_index]
                else:
                    row = dict(position_m=[], normal=[], impulse_Ns=[], separation_m=0.0)
                    if row_index < 1024:
                        self._point_pool.append(row)
                row["position_m"][:] = p.position
                row["normal"][:] = p.normal
                row["impulse_Ns"][:] = p.impulse
                row["separation_m"] = float(p.separation)
                points.append(row)
            positive = any(math.sqrt(sum(v*v for v in p["impulse_Ns"])) > 0 for p in points)
            self.file.write(json.dumps(dict(**self.clock(), colliders=paths, category=kind,
                event_type=str(header.type), points=points, positive_impulse=positive,
                scoring_implemented=False), allow_nan=False) + "\n")
            self.counts["reports"] += 1
            self.counts["points"] += len(points)
            self.counts["reports_with_positive_impulse"] += int(positive)
        self.file.flush()

    def close(self):
        self.subscription = None
        self._identity.clear()
        self._points.clear()
        self._point_pool.clear()
        if not self.file.closed:
            self.file.close()
