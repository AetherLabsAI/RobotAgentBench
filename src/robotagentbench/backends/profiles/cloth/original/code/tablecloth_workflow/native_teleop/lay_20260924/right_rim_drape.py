"""Pre-episode folded cloth draped over the measured basket's right rim."""
from __future__ import annotations

import numpy as np


def right_rim_drape(uv: np.ndarray, *, high: bool = False, front_fold: bool = False) -> tuple[np.ndarray, dict]:
    uv = np.asarray(uv, dtype=float)
    u = uv[:, 0] - uv[:, 0].min()
    v = uv[:, 1] - uv[:, 1].min()
    width = float(np.ptp(u))
    length = float(np.ptp(v))
    if not .33 < width < .36 or not .50 < length < .53:
        raise ValueError('Unexpected clean cloth dimensions')

    # The sheet lies folded on the basket floor and travels up the right wall.
    # Its outer strip hangs beyond the rim. Both curves are parameterized by
    # material arc length, so this is a pre-episode fold, not a stretched mesh.
    shoulder = .25 if high else .245
    arc = np.linspace(0., width, 2049)
    start_cosine = .25 if high else .55
    start_angle = np.arctan2(np.sqrt(1 - start_cosine ** 2), start_cosine)
    end_angle = 0. if high else np.arctan2(-np.sqrt(1 - .45 ** 2), .45)
    blend = np.clip((arc - (.19 if high else .175)) / (.06 if high else .140), 0., 1.)
    angle = start_angle + (end_angle - start_angle) * blend
    dx, dz = np.cos(angle), np.sin(angle)
    arc_x = np.r_[0., np.cumsum(np.diff(arc) * (dx[1:] + dx[:-1]) / 2)]
    arc_z = np.r_[0., np.cumsum(np.diff(arc) * (dz[1:] + dz[:-1]) / 2)]
    start_x = -2.16 if high else -2.20
    x = start_x + np.interp(u, arc, arc_x)
    z = .290 + np.interp(u, arc, arc_z)

    # Two floor layers fold around a physical 6 mm bend. The existing native
    # cloth material's 170 degree preferred fold is centered at v=length/2.
    radius = .002
    signed = v - length / 2
    distance = np.abs(signed)
    theta = np.minimum(distance / radius, np.pi / 2)
    folded_offset = radius * np.sin(theta) + np.maximum(distance - np.pi * radius / 2, 0.)
    y = -.60 - folded_offset if front_fold else -.995 + folded_offset
    h = np.sign(signed) * radius * (1 - np.cos(theta))
    # Correct sampled bend chords to exact material row lengths.
    levels, first, inverse = np.unique(v, return_index=True, return_inverse=True)
    curve = np.column_stack((y[first], h[first]))
    segments = np.diff(curve, axis=0)
    segments *= (np.diff(levels) / np.linalg.norm(segments, axis=1))[:, None]
    corrected = np.vstack((np.zeros(2), np.cumsum(segments, axis=0)))
    anchor = np.argmin(np.abs(levels - length / 2))
    corrected += curve[anchor] - corrected[anchor]
    y, h = corrected[inverse].T
    # A bend normal to the rising sheet avoids shear between the two folds.
    local_angle = np.interp(u, arc, angle)
    nx = -np.sin(local_angle)
    nz = np.cos(local_angle)
    q = np.column_stack((x + nx * h, y, z + nz * h))
    return q, dict(model='right_rim_drape_front' if front_fold else 'right_rim_drape_high' if high else 'right_rim_drape', initialization_only=True,
                   right_wall_inner_x_m=-2.061, rim_z_m=.450,
                   shoulder_u_m=shoulder, shoulder_xyz_m=[float(start_x + np.interp(shoulder, arc, arc_x)), -.60 if front_fold else -.995, float(.290 + np.interp(shoulder, arc, arc_z))],
                   material_arc_length_curves=True, post_start_pose_writes=False)
