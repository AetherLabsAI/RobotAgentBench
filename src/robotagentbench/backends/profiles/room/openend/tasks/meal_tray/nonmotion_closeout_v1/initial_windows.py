"""Initial evidence windows in real simulation seconds, never fake 240 Hz rows."""
import math

def initial_window(history, window_s, physics_dt_s):
    if not math.isfinite(window_s) or window_s <= 0:
        raise ValueError('Positive existing history duration required')
    if not any(abs(physics_dt_s-dt)<1e-10 for dt in (1/60,1/240)):
        raise ValueError('Unsupported initial physics clock')
    count=math.ceil(window_s/physics_dt_s-1e-9)+1
    rows=list(history)[-count:]
    complete=(len(rows)==count and
        all(math.isfinite(r['time_s']) for r in rows) and
        all(abs(r.get('physics_dt_s', physics_dt_s)-physics_dt_s)<1e-10 for r in rows) and
        all(abs(b['time_s']-a['time_s']-physics_dt_s)<1e-6 for a,b in zip(rows,rows[1:])) and
        rows[-1]['time_s']-rows[0]['time_s']+1e-6>=window_s)
    return rows,bool(complete)
