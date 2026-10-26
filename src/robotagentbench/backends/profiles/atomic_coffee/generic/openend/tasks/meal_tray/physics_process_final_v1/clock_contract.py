"""Read-only actual-workspace sampling validation. Never advances physics."""
import math


def validate_interval(rows,hz,now):
    if hz not in (60,240) or len(rows)!=hz//20:
        raise RuntimeError('Incomplete 20 Hz native interval')
    dt=1/hz
    previous=now-.05
    for row in rows:
        t=row['time_s'];actual=row['dt']
        if (not math.isfinite(t) or not math.isfinite(actual) or
                abs(actual-dt)>1e-10 or abs(t-previous-dt)>1e-6):
            raise RuntimeError('Repeated, skipped or wrong-rate native substep')
        previous=t
    if abs(previous-now)>1e-6:raise RuntimeError('Stale native interval')
