"""User-approved initial-only numeric zero normalization; never edits a pose."""
import math

def validate(angle, swing, door):
    angle, swing, door = map(float, (angle, swing, door))
    finite = all(math.isfinite(v) for v in (angle, swing, door))
    normalized = 0.0 if finite and abs(angle) <= 0.001 else angle
    evidence = dict(raw_knob_angle_deg=angle, normalized_knob_angle_deg=normalized,
                    zero_tolerance_deg=0.001, swing_deg=swing, door_angle_deg=door,
                    scope='initial_validation_only_no_pose_or_process_change')
    if not finite or not 0.0 <= normalized <= 5.0 or swing > 2.0 or abs(door) > 2.0:
        raise ValueError('Measured initial knob/door outside startup contract: '+repr(evidence))
    return evidence
