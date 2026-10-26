"""Make the existing basket rim fold isometric before the first physics step.

The Newton rim curve is retained as a direction field. Its sampled row chords
are stretched to their material UV lengths, with the basket-floor crease fixed.
No episode state is modified by this module.
"""
import numpy as np


def long_midpoint_fold(uv, size, bounds, particle_radius=.0015, overhang_extra=0.):
    """Original real-rim fold with only its outer oval radius exposed for diagnosis."""
    uv = np.asarray(uv, dtype=float)
    width, length = map(float, size)
    s = uv[:, 1]-length/2
    distance, sign = np.abs(s), np.sign(s)
    rb = .002
    floor = float(bounds[0, 2])+.024
    z0 = floor+particle_radius+.004
    rim_y, rim_z = float(bounds[0, 1])+.010, float(bounds[1, 2])-.020
    crease_y = rim_y+.030
    angle = np.minimum(distance/rb, np.pi/2)
    y = crease_y+sign*rb*np.sin(angle)
    z = z0+rb*(1-np.cos(angle))
    after = distance-np.pi*rb/2
    for side in (-1, 1):
        ids = np.flatnonzero((sign == side)&(after > 0))
        ry, rz = .014+overhang_extra+side*rb, .027+side*rb
        start_y, start_z = crease_y+side*rb, z0+rb
        end_y, rise = rim_y+ry, rim_z-start_z
        dy = end_y-start_y
        leg = np.hypot(rise, dy)
        d = after[ids]
        ratio = np.minimum(d/leg, 1)
        y[ids] = start_y+ratio*dy
        z[ids] = start_z+ratio*rise
        over = ids[d > leg]
        if len(over):
            phi = np.linspace(0, np.pi, 2049)
            speed = np.sqrt((ry*np.sin(phi))**2+(rz*np.cos(phi))**2)
            arc = np.r_[0, np.cumsum((speed[1:]+speed[:-1])*.5*np.diff(phi))]
            travel = after[over]-leg
            t = np.interp(np.minimum(travel, arc[-1]), arc, phi)
            y[over] = rim_y+ry*np.cos(t)
            z[over] = rim_z+rz*np.sin(t)-np.maximum(travel-arc[-1], 0)
    q = np.column_stack((bounds[:, 0].mean()+uv[:, 0]-width/2, y, z))
    return q, dict(original_rim_horizontal_radius_m=.014,
                   rim_horizontal_radius_m=.014+overhang_extra,
                   basket_edge='y_min', initialization_only=True)


def isometric_rows(q, uv, crease_v):
    q = np.asarray(q, dtype=float)
    uv = np.asarray(uv, dtype=float)
    levels, first, inverse = np.unique(uv[:, 1], return_index=True, return_inverse=True)
    if q.shape != (len(uv), 3) or len(levels) < 3:
        raise ValueError('Invalid rim fold mesh')
    curve = q[first, 1:3]
    if np.max(np.linalg.norm(q[:, 1:3] - curve[inverse], axis=1)) > 1e-6:
        raise ValueError('Fold rows must share one material curve')
    segments = np.diff(curve, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    if np.any(lengths <= 0):
        raise ValueError('Degenerate rim fold segment')
    segments *= (np.diff(levels)/lengths)[:, None]
    corrected = np.vstack((np.zeros(2), np.cumsum(segments, axis=0)))
    anchor = int(np.argmin(np.abs(levels-crease_v)))
    corrected += curve[anchor] - corrected[anchor]
    result = q.copy()
    result[:, 1:3] = corrected[inverse]
    return result, dict(crease_v_m=float(levels[anchor]),
                        maximum_initial_correction_m=float(np.linalg.norm(result-q, axis=1).max()),
                        material_row_lengths_preserved=True)


def photo_basket_fold(uv, bounds, particle_radius=.0015):
    """Pre-episode upright U fold suggested by the real basket photographs.

    The row curve is parameterized by material arc length. It uses physical
    basket contact only; no pinned nodes or post-start pose writes.
    """
    uv = np.asarray(uv, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    width, length = np.ptp(uv, axis=0)
    # The visible inner wall is about 20 mm inboard of the outer mesh bounds.
    # Keep the whole folded sheet inside that actual cavity at frame zero.
    center_y = float(bounds[0, 1] + .43 * (bounds[1, 1] - bounds[0, 1]))
    floor_z = float(bounds[0, 2] + .024 + particle_radius + .004)
    radius = .012
    distance = np.abs(uv[:, 1] - length / 2)
    sign = np.where(uv[:, 1] < length / 2, -1., 1.)
    arc = np.pi * radius / 2
    near = distance <= arc
    theta = np.minimum(distance / radius, np.pi / 2)
    y = center_y + sign * radius * np.sin(theta)
    z = floor_z + radius * (1 - np.cos(theta))
    beyond = np.maximum(distance - arc, 0)
    beta = np.deg2rad(np.where(sign < 0, 25., 20.))
    y[~near] += sign[~near] * np.sin(beta[~near]) * beyond[~near]
    z[~near] += np.cos(beta[~near]) * beyond[~near]
    x = bounds[:, 0].mean() + uv[:, 0] - width / 2
    q = np.column_stack((x, y, z))
    inner_margin = .02
    if (np.any(q[:, 1] < bounds[0, 1] + inner_margin + particle_radius) or
            np.any(q[:, 1] > bounds[1, 1] - inner_margin - particle_radius) or
            np.any(q[:, 0] < bounds[0, 0] + inner_margin + particle_radius) or
            np.any(q[:, 0] > bounds[1, 0] - inner_margin - particle_radius)):
        raise ValueError('Photo fold intersects the measured basket inner cavity at initialization')
    return q, dict(model='photo_basket_u_fold', crease_radius_m=radius,
                   center_y_m=center_y, floor_z_m=floor_z, inner_wall_inset_m=inner_margin,
                   left_lean_deg=25., right_lean_deg=20.,
                   source_images=['picture/house/107042459a9b0c3b5531bca64cfdb626.jpg',
                                  'picture/house/58b1a49b90ea7860e6339c7c14437818.jpg'],
                   initial_only=True, material_arc_length_preserved=True)


def photo_wall_fold(uv, bounds, particle_radius=.0015):
    """Pre-episode narrow U fold with BOTH panels leaning against the y-min wall."""
    uv=np.asarray(uv,dtype=float); bounds=np.asarray(bounds,dtype=float)
    width,length=np.ptp(uv,axis=0); radius=.006
    floor=float(bounds[0,2]+.024+particle_radius+.004)
    # Measured existing inner wall; no new support or changed collider.
    wall=float(bounds[0,1]+.021); rim=float(bounds[1,2])
    beta=np.deg2rad(12.)
    center=wall+radius+.003+np.tan(beta)*(rim-floor)
    s=uv[:,1]-length/2; distance=np.abs(s); sign=np.sign(s)
    theta=np.minimum(distance/radius,np.pi/2)
    y=sign*radius*np.sin(theta); z=radius*(1-np.cos(theta))
    extra=np.maximum(distance-np.pi*radius/2,0)
    z+=extra
    # Rotate the entire folded cross section toward the same wall.
    yy=y*np.cos(beta)-z*np.sin(beta)+center
    zz=y*np.sin(beta)+z*np.cos(beta)+floor
    q=np.column_stack((bounds[:,0].mean()+uv[:,0]-width/2,yy,zz))
    return q,dict(model='photo_wall_fold',crease_radius_m=radius,lean_deg=12.,
                  wall_y_m=wall,center_y_m=center,initial_only=True,
                  source_images=['picture/house/107042459a9b0c3b5531bca64cfdb626.jpg','picture/house/58b1a49b90ea7860e6339c7c14437818.jpg'])
