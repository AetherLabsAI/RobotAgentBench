"""Approved per-object placement posture. No physical/control parameters.

2026-09-17: 5-1..5-8 and 5-14 use upright within 15 degrees as quality
criteria (-1 if placed but not upright), not an eligibility gate for +3.
Local +Z was checked against the current asset geometry and shelf transforms;
do not infer an up direction from a moving object's longest world-box axis.
"""
import math

UPRIGHT_LIMIT_DEG = 15.0
UPRIGHT_AXES = {**{f'5-{i}': (0., 0., 1.) for i in range(1, 9)},
                '5-14': (0., 0., 1.)}
UNRESTRICTED_IDS = frozenset(('5-9','5-10','5-11','5-12','5-13','30-1','30-2','30-3'))


def placement_orientation(task_id, quaternion_wxyz):
    if task_id in UNRESTRICTED_IDS:
        return dict(required=False, satisfied=True, tilt_deg=None, limit_deg=None,
                    local_up_axis=None)
    if task_id not in UPRIGHT_AXES:
        raise ValueError('Unknown placement orientation policy: '+str(task_id))
    if (quaternion_wxyz is None or len(quaternion_wxyz) != 4 or
            any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x)
                for x in quaternion_wxyz)):
        raise ValueError('Upright placement requires a measured wxyz quaternion')
    norm = math.hypot(*quaternion_wxyz)
    if norm == 0:
        raise ValueError('Zero quaternion is not an orientation')
    w,x,y,z = (v/norm for v in quaternion_wxyz)
    a,b,c = UPRIGHT_AXES[task_id]
    # Z component of R(q) * local_up, preserving sign (upside-down is not upright).
    up_z = 2*(x*z-w*y)*a + 2*(y*z+w*x)*b + (1-2*(x*x+y*y))*c
    tilt = math.degrees(math.acos(max(-1., min(1.,up_z))))
    # Numerical roundoff only, not an added angular tolerance.
    ok = up_z >= math.cos(math.radians(UPRIGHT_LIMIT_DEG))-1e-12
    return dict(required=True, satisfied=bool(ok), tilt_deg=tilt,
                limit_deg=UPRIGHT_LIMIT_DEG, local_up_axis=list(UPRIGHT_AXES[task_id]))
