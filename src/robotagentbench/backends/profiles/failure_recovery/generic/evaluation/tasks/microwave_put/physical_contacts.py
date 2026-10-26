"""Selected-pair load reduction; never changes collision or report settings."""
from dataclasses import dataclass
import logging
import numpy as np

from arena_api.contact_identity import ContactIdentityCache
from .physical_geometry import ROBOT, PAYLOAD, PLATE, STAND, PADS, under, inverse

log = logging.getLogger(__name__)
# Observed in shared_entry.p6bbd_3l (fridge_to_microwave, call 19): PhysX gave a
# zero-impulse FOUND and a zero-point LOST for the SAME ordered actor/collider
# pair in one report. Only exactly that shape is merged; anything else with a
# repeated pair (loaded, reordered, FOUND/FOUND, LOST/FOUND, 3+ headers) is
# still rejected because its semantics are unconfirmed.
ZERO_LOAD_TRANSIENT_RULE = "zero_impulse_found_or_persist_then_zero_point_lost.v1"

@dataclass(frozen=True)
class PairLoad:
    paths: tuple
    actors: tuple
    stage_id: int
    normal_sum_on_target: tuple
    loaded: bool
    wall_impact: bool


@dataclass(frozen=True)
class ContactFacts:
    held_by_right: bool
    held_by_robot: bool
    released: bool
    plate_load: bool
    wall_impact: bool
    active_pairs: int
    sleeping_pairs: int
    selected_points_read: int
    plate_supported_by_stand: bool


def vector(v):
    result = np.array([v.x, v.y, v.z], dtype=float)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite contact vector")
    return result


class PutContactStream:
    def __init__(self, geometry, resolve, is_sleeping, event_types, *, payload_variant="beef_box"):
        from .payload_binding import payload_for
        self.payload = payload_for(payload_variant)
        self.geometry, self.is_sleeping = geometry, is_sleeping
        self.found, self.persist, self.lost = event_types
        self.identity = ContactIdentityCache(resolve, self.classify)
        self.active = {}
        # Append-only; committed only by a successful read. Rare by construction.
        self.zero_load_transients = []
        self.read_calls = 0

    def classify(self, paths):
        if any(not isinstance(p, str) or not p.startswith("/") for p in paths):
            raise ValueError("Invalid physical collider identity")
        a, b = paths
        payload = under(a, self.payload) or under(b, self.payload)
        wall = next((w for w, actor in ((a,b),(b,a)) if w in self.geometry.walls
                     and (under(actor, ROBOT) or under(actor, self.payload))), None)
        stand = ((under(a, PLATE) and under(b, STAND))
                 or (under(b, PLATE) and under(a, STAND)))
        return payload, wall, stand

    def zero_load_transient(self, rows, data):
        """True only for [FOUND|PERSIST, LOST] of one ordered pair with zero impulse."""
        if len(rows) != 2:
            return False
        (_, first, _), (_, last, _) = rows
        if first.type not in (self.found, self.persist) or last.type != self.lost:
            return False
        if int(last.num_contact_data) != 0:
            return False
        ids = lambda h: (int(h.actor0), int(h.actor1), int(h.collider0), int(h.collider1))
        if ids(first) != ids(last):
            return False
        start, count = int(first.contact_data_offset), int(first.num_contact_data)
        if start < 0 or count <= 0 or start+count > len(data):
            return False
        for i in range(start, start+count):
            try:
                impulse, normal, _ = (vector(getattr(data[i], k))
                                      for k in ("impulse", "normal", "position"))
            except ValueError:
                return False
            if np.any(impulse != 0) or not np.any(normal != 0):
                return False
        return True

    @staticmethod
    def header_record(index, h, paths, data):
        start, count = int(h.contact_data_offset), int(h.num_contact_data)
        row = {k: int(getattr(h, k)) for k in ("stage_id", "actor0", "actor1", "collider0",
               "collider1", "contact_data_offset", "num_contact_data")}
        row.update(header_index=index, type=str(h.type), paths=list(paths),
                   points=[{k: vector(getattr(data[i], k)).tolist()
                            for k in ("position", "normal", "impulse")}
                           for i in range(start, start+count)])
        return row

    def read(self, report, plate_up):
        self.read_calls += 1
        if not isinstance(report, tuple) or len(report) < 2 or report[0] is None or report[1] is None:
            raise RuntimeError("Current PhysX contact report unavailable")
        headers, data = report[:2]
        seen, slept = set(), set()
        current = dict(self.active)  # An invalid report must not partially commit.
        points_read = 0
        selected, groups = [], {}
        for index, h in enumerate(headers):
            paths, pair, classes = self.identity.get(h.stage_id, h.collider0, h.collider1)
            payload, wall, stand = classes
            if not payload and wall is None and not stand:
                continue
            key = (int(h.stage_id), pair)
            selected.append((index, h, paths, key, classes))
            groups.setdefault(key, []).append((index, h, paths))
        merged, transients = set(), []
        for key, rows in groups.items():
            if len(rows) == 1:
                continue
            if not self.zero_load_transient(rows, data):
                raise RuntimeError("Ambiguous duplicate contact header")
            merged.add(rows[0][0])
            transients.append(dict(rule=ZERO_LOAD_TRANSIENT_RULE, read_call=self.read_calls,
                stage_id=key[0], pair=list(key[1]), merged_header_index=rows[0][0],
                terminal_event="LOST", physical_values_modified=False,
                headers=[self.header_record(i, h, p, data) for i, h, p in rows]))
        for index, h, paths, key, (payload, wall, stand) in selected:
            if index in merged:
                continue  # Its zero-load points are recorded above, not read as load.
            if key in seen:
                raise RuntimeError("Ambiguous duplicate contact header")
            seen.add(key)
            if h.type == self.lost:
                current.pop(key, None)
                continue
            if h.type not in (self.found, self.persist):
                raise RuntimeError("Unknown PhysX contact event")
            start, count = int(h.contact_data_offset), int(h.num_contact_data)
            if start < 0 or count <= 0 or start+count > len(data):
                raise RuntimeError("Incomplete selected contact pair")
            normal_sum, loaded, impact = np.zeros(3), False, False
            target = self.payload if payload else PLATE
            sign = 1 if under(paths[0], target) else -1
            for i in range(start, start+count):
                p = data[i]
                impulse = vector(p.impulse)
                if payload or stand:
                    normal = vector(p.normal)
                    length = np.linalg.norm(normal)
                    if length == 0:
                        raise ValueError("Zero contact normal")
                    normal /= length
                    # Actual normal load, not contactOffset candidate presence.
                    load = abs(float(np.dot(impulse, normal)))
                    if load > 0:
                        loaded = True
                        normal_sum += sign * normal * load
                if wall:
                    position = vector(p.position)
                    local = inverse([position], self.geometry.microwave_pose)[0]
                    impact |= bool(np.dot(impulse, impulse) > 0 and
                                   self.geometry.walls[wall].inner_face(local))
                points_read += 1
            current[key] = PairLoad(paths, (int(h.actor0),int(h.actor1)), int(h.stage_id),
                                    tuple(normal_sum), loaded, impact)
        sleeping = {}
        for key, state in current.items():
            if key in seen:
                continue
            for actor in state.actors:
                token = (state.stage_id, actor)
                if token not in sleeping:
                    sleeping[token] = self.is_sleeping(*token)
                if sleeping[token] is not True:
                    raise RuntimeError("Missing contact report for an awake pair: " + str(key))
            slept.add(key)
        pads = {h: [np.zeros(3),np.zeros(3)] for h in PADS}
        robot_load, plate_load, wall_impact = False, False, False
        plate_supported = False
        for key, state in current.items():
            wall_impact |= state.wall_impact and key not in slept
            if not state.loaded:
                continue
            a, b = state.paths
            force_direction = np.asarray(state.normal_sum_on_target)
            if self.classify(state.paths)[2]:
                plate_supported |= bool(force_direction[2] > 0)
            if not (under(a, self.payload) or under(b, self.payload)):
                continue
            other = b if under(a, self.payload) else a
            robot_load |= under(other, ROBOT)
            plate_load |= bool(under(other, PLATE) and np.dot(force_direction, plate_up) > 0
                               and force_direction[2] > 0)
            for hand, paths in PADS.items():
                if other in paths:
                    pads[hand][paths.index(other)] += force_direction
        opposed = {h: bool(np.dot(v[0],v[1]) < 0) for h,v in pads.items()}
        self.active = current
        for record in transients:
            self.zero_load_transients.append(record)
            log.warning("Merged zero-load transient contact pair (%s): read_call=%d stage=%d "
                        "pair=%s header=%d; full record in zero_load_transients",
                        record["rule"], record["read_call"], record["stage_id"],
                        record["pair"], record["merged_header_index"])
        return ContactFacts(opposed["r"], any(opposed.values()), not robot_load,
                            plate_load, bool(wall_impact), len(current), len(slept), points_read,
                            plate_supported)

    def close(self):
        self.active.clear()
        self.identity.clear()
