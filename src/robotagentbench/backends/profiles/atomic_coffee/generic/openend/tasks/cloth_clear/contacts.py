"""Shared native contact cache, with distinct touch and load-bearing evidence."""
from evaluation.tasks.shelf_pick.contact_stream import ContactStream, under, ROBOT, PADS
from evaluation.tasks.tabletop_pick.book_cloth.evidence_geometry import on_cloth_face
from openend.tasks.remote_transfer.geometry import inside


class ClothContacts(ContactStream):
    def classify(self, paths):
        return any(under(p, target) for p in paths for target in self.targets)

    def facts(self, geometry):
        bilateral = self.summarize()['bilateral_contacts']
        out = {p: dict(supports=set(), support_contact=False, robot_contact=False,
                      finger_contact=False, external_contact=False, on_floor=False,
                      bilateral=bilateral[p]) for p in self.targets}
        for paths, _, _, samples in self.active.values():
            for target, f in out.items():
                i = 0 if under(paths[0], target) else 1 if under(paths[1], target) else None
                if i is None:
                    continue
                other, sign = paths[1-i], 1 if i == 0 else -1
                robot = under(other, ROBOT)
                table, cloth = under(other, geometry.table), under(other, geometry.cloth)
                f['robot_contact'] |= robot
                f['finger_contact'] |= any(under(other, pad) for pads in PADS.values() for pad in pads)
                f['support_contact'] |= table or cloth
                if not samples:
                    continue
                f['on_floor'] |= any(under(other, p) for p in self.floor_paths)
                f['external_contact'] |= not (robot or table or cloth)
                for normal, point, _ in samples:
                    normal = tuple(sign*x for x in normal)
                    if (table and normal[2] > .5 and abs(point[2]-geometry.top['z']) <= self.tolerance
                            and inside(point[:2], geometry.top['polygon'])):
                        f['supports'].add(geometry.table)
                    if cloth and on_cloth_face(point, normal, geometry.surface, self.tolerance):
                        f['supports'].add(geometry.cloth)
        for f in out.values():
            f['supports'] = frozenset(f['supports'])
        return out
