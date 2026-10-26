"""Wrist camera contact evidence, independent of optional diagnostic logging."""
CAMERA_ROOTS = tuple(f'/World/Robot/gripper_{side}_base_link/Cam' for side in ('l', 'r'))
CAMERA_COLLIDERS = tuple(root + '/Cam/' + part for root in CAMERA_ROOTS for part in ('NONE', 'cam'))

def is_camera(path):
    return any(path == root or path.startswith(root + '/') for root in CAMERA_ROOTS)

def camera_contacts(pairs):
    rows = []
    for paths, samples in pairs:
        if not any(is_camera(p) for p in paths):
            continue
        # Contacts.pairs() has already removed speculative, zero-force samples.
        impulses = [float(impulse) for _, impulse in samples if impulse > 0]
        if impulses:
            rows.append(dict(a=paths[0], b=paths[1], impulse=max(impulses)))
    return rows
