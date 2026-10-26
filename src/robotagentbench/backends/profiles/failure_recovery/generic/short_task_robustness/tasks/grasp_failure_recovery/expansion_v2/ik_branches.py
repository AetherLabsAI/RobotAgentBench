"""Numerical IK branch search; never physical reset/jitter or new joint limits.

The same fixed TCP pose is solved from multiple legal numerical starting points.
Every solve uses the existing Kinematics.solve limits, 200 evaluations and
original tolerances. The stage, target, base, orientation and body stay fixed.
"""
from copy import copy


def seeds(model):
    original=dict(model.initial)
    midpoint={n:(model.bounds[n][0]+model.bounds[n][1])/2 for n in model.names}
    seen={tuple(original[n] for n in model.names)}
    rows=[('native_limit_midpoint',dict(original,**midpoint))]
    for name in model.names:
        lo,hi=model.bounds[name]
        for fraction in (.25,.5,.75):
            q=dict(original);q[name]=lo+(hi-lo)*fraction
            rows.append((f'{name}:legal_numerical_seed:{fraction}',q))
    for label,q in rows:
        key=tuple(q[n] for n in model.names)
        if key in seen:continue
        seen.add(key)
        if any(not model.bounds[n][0]<=q[n]<=model.bounds[n][1] for n in model.names):
            raise ValueError('Numerical IK seed outside unchanged asset limits')
        if any(q[n]!=v for n,v in original.items() if n not in model.names):
            raise ValueError('Numerical seed changed a non-active coordinate')
        yield label,q


def solutions(model,goal):
    for label,q in seeds(model):
        numerical=copy(model)
        numerical.initial=q
        coordinates,receipt=numerical.solve(goal)
        yield coordinates,dict(receipt,numerical_start=label,
            numerical_seed_is_robot_reset=False,physical_state_written=False)
