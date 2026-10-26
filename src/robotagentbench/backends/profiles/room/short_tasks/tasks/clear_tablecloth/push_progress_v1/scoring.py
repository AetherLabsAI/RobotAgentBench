"""Task-local clear-cloth area progress. No scene or physics writes."""
import math
from openend.tasks.cloth_clear.scoring import ClothClear
from openend.tasks.remote_transfer.geometry import polygon, cross, fully_inside

VERSION = 'short_tasks.clear_tablecloth.area_progress.v2'
COMPLETION_OUTSIDE_FRACTION = 0.99


def area(p):
    if len(p) < 3:
        return 0.0
    # Translate first to reduce cancellation for distant world coordinates.
    x, y = p[0]
    return abs(sum((a[0]-x)*(b[1]-y)-(b[0]-x)*(a[1]-y)
                   for a, b in zip(p, p[1:]+p[:1]))) / 2


def outside_fraction(footprint, cloth):
    subject, clip = polygon(footprint), polygon(cloth)
    original_area = area(subject)
    if original_area <= 0:
        raise ValueError('Nondegenerate footprint required')
    for a, b in zip(clip, clip[1:]+clip[:1]):
        if not subject:
            break
        out = []
        prev = subject[-1]
        dprev = cross(a, b, prev)
        for point in subject:
            dpoint = cross(a, b, point)
            if (dprev >= 0) != (dpoint >= 0):
                t = dprev / (dprev-dpoint)
                out.append((prev[0]+t*(point[0]-prev[0]), prev[1]+t*(point[1]-prev[1])))
            if dpoint >= 0:
                out.append(point)
            prev, dprev = point, dpoint
        subject = tuple(out)
    return min(1.0, max(0.0, 1-area(subject)/original_area))


class AreaProgressClothClear(ClothClear):
    def __init__(self, *args, **kwargs):
        self.progress_level = 0
        self.fraction_outside = 0.0
        self.progress_events = []
        super().__init__(*args, **kwargs)

    def observe(self, f, *, final_sample=True):
        initial = self.last is None
        # Preserve original typed evidence, identity, sequence, clock and initial
        # admission checks. Grasp/lift remain diagnostic, never scoring gates.
        super().observe(f, final_sample=final_sample)
        self.fraction_outside = outside_fraction(f.footprint, f.cloth)
        legal_xy = fully_inside(f.footprint, f.tabletop, f.exclusions)
        allowed_supports = frozenset((self.binding.cloth_path, self.binding.table_path))
        supported = bool(f.supports) and f.supports <= allowed_supports
        legal_progress = (not initial and legal_xy and not f.on_floor
                          and not f.external_contact and (supported or f.held))
        if legal_progress:
            level = min(9, int(math.floor(self.fraction_outside*10 + 1e-9)))
            for new_level in range(self.progress_level+1, level+1):
                self.progress_events.append(dict(level=new_level, points=0.1,
                    sample_index=f.sample_index, time_s=f.time_s,
                    fraction_outside=self.fraction_outside))
            self.progress_level = max(self.progress_level, level)
        # User-approved area tolerance applies only to the cloth boundary.
        # Keep the original table/exclusion geometry checks and real support.
        destination = legal_xy and self.fraction_outside >= COMPLETION_OUTSIDE_FRACTION
        self.current_placed = bool(not initial and destination and not f.held
            and not f.on_floor and not f.external_contact
            and self.binding.table_path in f.supports and supported)
        if self.current_placed:
            self.awards.add('place')
        else:
            self.awards.discard('place')
        return self.summary()

    def summary(self):
        process = self.progress_level / 10
        final = round(3.0-process, 10) if self.current_placed else 0.0
        return dict(score=3.0 if self.current_placed else process,
            raw_score=3.0 if self.current_placed else process, maximum_score=3,
            process_points=process, final_state_points=final,
            success=self.current_placed, operation_complete=self.current_placed,
            current_placed=self.current_placed, ended=self.ended,
            finalized=self.finalized, reason=self.reason, failure=None,
            awards=sorted(self.awards), diagnostic_grasp_lift_only=True,
            rule_version=VERSION, fraction_outside=self.fraction_outside,
            completion_outside_fraction=COMPLETION_OUTSIDE_FRACTION,
            progress_level=self.progress_level, progress_events=list(self.progress_events),
            timeout_owner='room', evidence_status='external_facts_not_native_verified')
