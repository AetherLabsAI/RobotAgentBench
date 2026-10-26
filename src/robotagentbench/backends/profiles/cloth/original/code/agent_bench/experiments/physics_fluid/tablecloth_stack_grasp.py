"""Measured opposing-pad contact through a connected folded cloth patch."""
import numpy as np
from trimesh.triangles import closest_point


def stack_contact_pair(q, uv, faces, edges, eligible, pad_distances, shell, self_radius, target, anchor_eligible=None):
    """Find two material-separated outer contacts connected inside the jaws.

    Mesh edges connect a fold's continuous tip; measured nonincident triangle
    contact connects layers farther from the crease. No free-space bridge is
    inferred from an AABB or a large particle-distance tolerance.
    """
    ids = np.flatnonzero(eligible)
    if len(ids) < 2:
        return None
    parent = {int(i): int(i) for i in ids}
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def union(i, j):
        parent[root(int(i))] = root(int(j))
    for i, j in edges[np.all(eligible[edges], axis=1)]:
        union(i, j)
    contact_bridges = []
    face_ids = np.flatnonzero(np.all(eligible[faces], axis=1))
    for i in ids:
        candidates = face_ids[np.linalg.norm(uv[faces[face_ids]].mean(1) - uv[i], axis=1) > .025]
        if not len(candidates):
            continue
        triangles = q[faces[candidates]]
        distance = np.linalg.norm(closest_point(triangles, np.broadcast_to(q[i], (len(triangles), 3)).copy()) - q[i], axis=1)
        for f, d in zip(candidates[distance <= self_radius + 1e-5], distance[distance <= self_radius + 1e-5]):
            union(i, faces[f, 0]);contact_bridges.append((int(i), int(f), float(d)))
    anchor_eligible = eligible if anchor_eligible is None else anchor_eligible
    left = ids[(pad_distances[ids, 0] <= shell) & anchor_eligible[ids]]
    right = ids[(pad_distances[ids, 1] <= shell) & anchor_eligible[ids]]
    pairs = []
    for i in left:
        for j in right:
            material = float(np.linalg.norm(uv[i]-uv[j]))
            spatial = float(np.linalg.norm(q[i]-q[j]))
            if i == j or root(int(i)) != root(int(j)) or material < max(.020, spatial*2.):
                continue
            score = np.linalg.norm(q[i]-q[target]) + np.linalg.norm(q[j]-q[target])
            pairs.append((score, int(i), int(j), material, spatial))
    if not pairs:
        return None
    _, i, j, material, spatial = min(pairs)
    return dict(particles=[i,j], material_separation_m=material, spatial_separation_m=spatial,
                inside_jaw_connected_vertices=[int(k) for k in ids if root(int(k)) == root(i)],
                measured_layer_contacts=contact_bridges, layer_contact_numeric_tolerance_m=1e-5, layer_contact_radius_m=float(self_radius))


def try_stack_clamp(sim, side, target, *, grip, event_time, substep):
    poses = sim.actual_hand_poses(side)
    _, axis, width, palm, low, high, bounds = sim.hands.mouth_local(side, poses)
    if high < low or high-low > .012:
        return False
    q = sim.a.particle_q.numpy()
    pose = poses[f'gripper_{side.lower()}_base_link']
    local = (q - pose[:3,3]) @ pose[:3,:3]
    depth = local @ palm - bounds[1,0]
    across = local @ width
    selected = np.flatnonzero((sim.cloth_ids == sim.cloth_ids[target]) &
                             (np.linalg.norm(q-q[target],axis=1) < .060) &
                             (depth >= 0.) & (depth <= np.ptp(bounds[1])) &
                             (across >= bounds[0,0]) & (across <= bounds[0,1]))
    if len(selected) < 2:
        return False
    d, vectors = sim.hands.pad_surfaces(q[selected], side, poses)
    signed_sides = vectors @ (pose[:3,:3] @ axis)
    valid = (d >= -.0005).all(1) & (d.sum(1) <= .012) & (signed_sides[:,0] <= 0.) & (signed_sides[:,1] >= 0.)
    eligible = np.zeros(sim.n, dtype=bool);eligible[selected[valid]] = True
    distances = np.full((sim.n,2),np.inf);distances[selected] = d
    minimum = sim.policy['deep_grasp']['minimum_depth_m']
    shell = sim.args.radius + .0005
    result = stack_contact_pair(q,sim.uv,sim.faces,sim.edges,eligible,distances,shell,
                                sim.args.self_radius,target,eligible & (depth >= minimum))
    if result is None:
        return False
    ids = np.asarray(result['particles'],dtype=int)
    # Contact-network observation only: no owner, anchor, mass or velocity writes.
    sim.stack_clamps[side] = dict(outer_pad_indices=[0,1])
    sim.record_pad_contact(side, ids, event_time=event_time, substep=substep,
        requested_target_particle=int(target), actual_grip_fraction=float(grip),
        opposing_pad_gap_m=float(high-low), insertion_depths_m=depth[ids].tolist(),
        minimum_required_insertion_m=minimum, contact_shell_m=shell,
        extra_contact_tolerance_m=0., opposing_surface_distances_m=distances[ids].tolist(),
        outer_pad_indices=[0,1], model='opposing_pad_folded_stack_observation',
        contact_network=result)
    return True
