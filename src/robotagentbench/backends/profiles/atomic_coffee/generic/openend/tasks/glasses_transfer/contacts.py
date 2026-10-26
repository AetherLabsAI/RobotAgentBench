"""Reduce actual nonzero-impulse native pairs, not AABB or command contacts."""
def under(path, root):
    return path == root or path.startswith(root + '/')


def reduce_pairs(pairs, *, target, source, bag, cabinet, floors, robot, pads):
    supports = set(); on_floor = False; robot_contact = False; finger = False; external = False
    normals = {h: [[0., 0., 0.], [0., 0., 0.]] for h in pads}
    for paths, samples in pairs:
        for i in (0, 1):
            subject, other = paths[i], paths[1-i]
            if not under(subject, target): continue
            sign = 1 if i == 0 else -1
            if under(other, robot):
                robot_contact = True
                for h, links in pads.items():
                    for k, link in enumerate(links):
                        if under(other, link):
                            finger = True
                            for n, _, _ in samples:
                                for axis in range(3): normals[h][k][axis] += sign * n[axis]
            else:
                external = True
                on_floor |= any(under(other, p) for p in floors)
                if any(sign*n[2] > .5 for n, _, _ in samples):
                    support = next((root for root in (bag, cabinet, source) if root is not None and under(other, root)), other)
                    supports.add(support)
    return dict(supports=frozenset(supports), on_floor=bool(on_floor), robot_contact=robot_contact,
        finger_contact=finger, external_contact=external,
        bilateral={target: {h: sum(a*b for a, b in zip(*v)) < 0 for h, v in normals.items()}})
