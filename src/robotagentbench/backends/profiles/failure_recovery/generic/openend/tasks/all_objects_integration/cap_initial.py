"""User-approved cap-only initial rocking policy; no scoring/physics edits."""
import numpy as np

def evaluate(rows,centres,tolerance_m):
    """Same native window/tolerance, but test centre drift rather than oscillation.

    Medians of the two chronological halves suppress back-and-forth rocking.
    This is a finite-window drift screen, not proof of future static stability.
    Large sustained translations, loss of support, floor/robot contact remain
    blocked. Caller retains all original history/support/unheld gates.
    """
    p=np.asarray(centres,dtype=float)
    if p.shape!=(len(rows),3) or len(rows)<2 or not np.isfinite(p).all():
        raise ValueError('Complete measured centre history required')
    origins=np.asarray([r['pose'].position_m for r in rows])
    n=len(rows)//2
    first=np.median(p[:n],axis=0);last=np.median(p[-n:],axis=0)
    drift=np.abs(last-first)
    return dict(policy='cap_supported_rocking_v1',
        no_sustained_drift=bool(drift.max()<=tolerance_m),
        drift_m=drift.tolist(),centre_span_m=np.ptp(p,axis=0).tolist(),
        rigid_origin_span_m=np.ptp(origins,axis=0).tolist(),
        existing_drift_tolerance_m=float(tolerance_m),
        original_strict_stable=bool(np.ptp(origins,axis=0).max()<=tolerance_m),
        rocking_is_initial_warning_only=True,
        future_stability_certified=False,
        sample_times_s=[float(r['time_s']) for r in rows],centre_history_m=p.tolist())
