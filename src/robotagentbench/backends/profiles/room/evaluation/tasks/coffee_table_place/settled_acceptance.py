"""Stable initial state; designated bilateral pad contact is diagnostic only."""
import math
from evaluation.tasks.dining_place.physical_detection import DetectionConfig

HISTORY_STEPS = 120
SETTLE_SECONDS = 5.0
ACCEPTANCE_SCHEMA = 'coffee_table_place.initial_acceptance.v5'
HELD_POSITION_SPAN_MAX_M = .005
PAYLOAD_RELATIVE_SPAN_LIMIT_M = .001
DETECTION = DetectionConfig()


def relative_position(row):
    """Box origin in the current unscaled plate frame (wxyz quaternion)."""
    q = row['pose']['quaternion_wxyz']
    p, b = row['pose']['position_m'], row['payload_pose']['position_m']
    if len(q) != 4 or len(p) != 3 or len(b) != 3 or not all(math.isfinite(v) for v in (*q, *p, *b)):
        return None
    norm = math.hypot(*q)
    if not norm:
        return None
    w, x, y, z = (v/norm for v in q)
    u = (-x, -y, -z)
    v = [b[i]-p[i] for i in range(3)]
    def cross(a, b):
        return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
    uv = cross(u, v); uuv = cross(u, uv)
    return [v[i]+2*w*uv[i]+2*uuv[i] for i in range(3)]


def assess(rows, carries_payload):
    checks = {'complete_window': len(rows) == HISTORY_STEPS}
    if not checks['complete_window']:
        return dict(accepted=False, checks=checks, measured_samples=len(rows))

    def span(role):
        positions = [row[role]['position_m'] for row in rows]
        if any(len(p) != 3 or not all(math.isfinite(v) for v in p) for p in positions):
            return None
        return max(max(p[i] for p in positions)-min(p[i] for p in positions) for i in range(3))

    held_span = span('pose')
    checks.update(held_free_of_external_support=not any(r['contacts']['held_on_external_support'] for r in rows),
        held_position_stable=held_span is not None and held_span <= HELD_POSITION_SPAN_MAX_M)
    ratio = payload_span = relative_span = max_unloaded_s = None
    if carries_payload:
        facts = [r['payload_facts'] for r in rows]
        supported = lambda f: f['on_held_body'] and f['inside_held_body'] and f['above_held_body']
        ratio = sum(bool(supported(f)) for f in facts)/len(facts)
        payload_span = span('payload_pose')
        relative = [relative_position(r) for r in rows]
        if all(p is not None for p in relative):
            relative_span = max(max(p[i] for p in relative)-min(p[i] for p in relative) for i in range(3))
        run = longest = 0
        for f in facts:
            run = 0 if supported(f) else run+1
            longest = max(longest, run)
        max_unloaded_s = longest*DETECTION.physics_dt_s
        checks.update(payload_stays_on_plate=all(f['inside_held_body'] and f['above_held_body'] for f in facts),
            payload_has_measured_support=any(supported(f) for f in facts),
            payload_no_sustained_unloading=max_unloaded_s < DETECTION.unsupported_confirm_s,
            payload_free_of_other_support=not any(f['on_other_support'] or f['grasped'] for f in facts),
            payload_relative_position_stable=relative_span is not None and relative_span < PAYLOAD_RELATIVE_SPAN_LIMIT_M)
    return dict(accepted=all(checks.values()), checks=checks, measured_samples=len(rows),
        held_position_span_m=held_span, payload_position_span_m=payload_span,
        payload_support_ratio=ratio,
        payload_relative_position_span_m=relative_span, payload_max_unloaded_s=max_unloaded_s,
        diagnostics=dict(bilateral_pad_contact_fraction=sum(bool(r['contacts']['held_grasped']) for r in rows)/len(rows),
                         bilateral_pad_contact_required=False))
