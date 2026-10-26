"""CPU-only one-shot motion envelope. Geometry is supplied by scene admission.

No target transform commands: output is only the human wrist pose/closure.
Phase boundaries allocate the approved ~3 s motion, not scoring thresholds.
Native contact, support and swept-clearance checks remain mandatory.
"""
from dataclasses import dataclass
import math
import random

MIN_HORIZONTAL_RELOCATION_M = 0.10


def horizontal_relocation_allowed(source, destination):
    """Strict endpoint XY displacement; lift/path length cannot satisfy it."""
    a, b = tuple(source[:2]), tuple(destination[:2])
    if len(a) != 2 or len(b) != 2 or not all(math.isfinite(v) for v in (*a, *b)):
        return False
    distance = math.dist(a, b)
    # Treat floating-point roundoff at exactly 10 cm as the boundary, not a pass.
    return (distance > MIN_HORIZONTAL_RELOCATION_M
            and not math.isclose(distance, MIN_HORIZONTAL_RELOCATION_M,
                                 rel_tol=0., abs_tol=1e-12))


def smooth(t):
    t = min(1., max(0., t))
    return t*t*t*(10+t*(-15+6*t))


def closure_fraction(t):
    # The quintic can round a few ULPs beyond [0, 1] near its endpoints.
    # Bound only the produced hand closure; keep wrist interpolation and the
    # independent ContactClosure validation unchanged.
    return min(1., max(0., smooth(t)))


def interpolate(a,b,t):
    return tuple(x+(y-x)*smooth(t) for x,y in zip(a,b))


@dataclass(frozen=True)
class MotionSample:
    phase: str
    wrist_position: tuple
    closure: float


class RelocationMotion:
    def __init__(self, entry, grasp, raised, destination_raised, destination, exit_pose):
        self.points = tuple(tuple(float(v) for v in p) for p in
                            (entry,grasp,raised,destination_raised,destination,exit_pose))
        if any(len(p)!=3 or not all(math.isfinite(v) for v in p) for p in self.points):
            raise ValueError('Finite admitted 3D waypoints required')
        if self.points[1] == self.points[4]:
            raise ValueError('Relocation needs a different destination')
        self.duration_s = 3.0

    def sample(self, elapsed):
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError('Invalid elapsed simulation time')
        t = elapsed/self.duration_s
        entry,grasp,raised,other_raised,destination,exit_pose = self.points
        if t < .2:
            return MotionSample('approach',interpolate(entry,grasp,t/.2),0.)
        if t < .3:
            return MotionSample('approach',grasp,closure_fraction((t-.2)/.1))
        if t < .4:
            return MotionSample('carry',interpolate(grasp,raised,(t-.3)/.1),1.)
        if t < .6:
            return MotionSample('carry',interpolate(raised,other_raised,(t-.4)/.2),1.)
        if t < .7:
            return MotionSample('carry',interpolate(other_raised,destination,(t-.6)/.1),1.)
        if t < .8:
            return MotionSample('release',destination,1-closure_fraction((t-.7)/.1))
        if t < 1:
            return MotionSample('retreat',interpolate(destination,exit_pose,(t-.8)/.2),0.)
        return MotionSample('done',exit_pose,0.)


def footprint_candidates(bounds, target, obstacles):
    """Choose a distinct same-level empty AABB footprint, no invented clearance.

    Uses the actual object footprint as grid spacing. This only proposes a
    destination; it is not a physical/swept-hand collision certificate.
    All inputs are shelf-local XY bounds, so rotated layouts are supported.
    """
    (lo,hi),(a,b) = bounds,target
    size = [b[i]-a[i] for i in (0,1)]
    if any(s<=0 for s in size):
        raise ValueError('Positive target footprint required')
    center = [(a[i]+b[i])/2 for i in (0,1)]
    candidates = []
    for axis in (0,1):
        count = int((hi[axis]-lo[axis])/size[axis])
        if count<1:
            raise ValueError('Target does not fit shelf')
        candidates.append([lo[axis]+size[axis]/2+j*(hi[axis]-lo[axis]-size[axis])/max(1,count-1)
                           for j in range(count)])
    # Same-depth relocation avoids dragging the forearm behind the front row.
    # It also prevents grid quantization from eliminating valid lateral slots.
    candidates[1].append(center[1])
    valid = []
    for x in candidates[0]:
        for y in candidates[1]:
            p = (x,y)
            if not horizontal_relocation_allowed(center, p):
                continue
            dest = ([p[i]-size[i]/2 for i in (0,1)], [p[i]+size[i]/2 for i in (0,1)])
            # Never propose the original occupied footprint, even partially.
            if any(all(dest[0][i] < obj[1][i] and obj[0][i] < dest[1][i] for i in (0,1))
                   for obj in (target,*obstacles)):
                continue
            valid.append(p)
    if not valid:
        raise ValueError('No separate empty same-level destination with horizontal displacement > 10 cm')
    return list(dict.fromkeys(valid))


def seeded_destinations(bounds, target, obstacles, seed):
    """Random order of same-level empty footprints, reproducible per seed.

    These are placement candidates, NOT approved transport paths. The caller
    must reject candidates whose approach/carry/retreat geometry is blocked.
    """
    if type(seed) is not int:raise ValueError('Integer relocation seed required')
    candidates=footprint_candidates(bounds,target,obstacles)
    random.Random(seed).shuffle(candidates)
    return candidates


def footprint_destination(bounds, target, obstacles):
    """Legacy nearest-empty diagnostic, preserved for old receipts/tests."""
    center=[(target[0][i]+target[1][i])/2 for i in (0,1)]
    return min(footprint_candidates(bounds,target,obstacles),
               key=lambda p:(abs(p[1]-center[1]),sum((p[i]-center[i])**2 for i in (0,1))))
