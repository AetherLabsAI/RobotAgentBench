"""Derive target-specific mean forces from every recorded PhysX substep.

A sensor sampled only after the final substep can report zero while earlier
substeps carried the object. This module aggregates observations only; it never
moves bodies, changes contacts, or substitutes geometry for measured impulses.
"""
from __future__ import annotations
from collections.abc import Mapping, Sequence
import numpy as np


def aggregate_target_impulses(events: Sequence[Mapping], *, target_path: str,
                              pad_paths: Sequence[str], frame_durations_s):
    durations = np.asarray(frame_durations_s, dtype=float)
    if durations.ndim != 1 or not len(durations) or not np.isfinite(durations).all() or np.any(durations <= 0):
        raise ValueError('frame durations must be finite, positive and one-dimensional')
    pads = tuple(map(str, pad_paths))
    if not target_path or not pads or any(not pad for pad in pads) or len(set(pads)) != len(pads) or target_path in pads:
        raise ValueError('target and pads must be distinct nonempty body paths')
    impulses = np.zeros((len(durations), len(pads), 3), dtype=float)
    counts = np.zeros((len(durations), len(pads)), dtype=np.int64)
    for event in events:
        frame = event['frame']
        if isinstance(frame, bool) or not isinstance(frame, (int, np.integer)):
            raise ValueError('contact frame must be an integer')
        if frame < 0:  # Pre-recording settling remains provenance, not action contact.
            continue
        if frame >= len(durations):
            raise ValueError('contact frame exceeds captured duration array')
        pair = (str(event['actor0']), str(event['actor1']))
        for index, pad in enumerate(pads):
            sign = 1 if pair == (pad, target_path) else -1 if pair == (target_path, pad) else 0
            if not sign:
                continue
            impulse = np.asarray(event['impulse_ns'], dtype=float)
            count = event['contact_points']
            if impulse.shape != (3,) or not np.isfinite(impulse).all():
                raise ValueError('contact impulse must be a finite three-vector')
            if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
                raise ValueError('contact point count must be a nonnegative integer')
            impulses[frame, index] += sign * impulse
            counts[frame, index] += count
    return {'mean_force_vectors_n': impulses / durations[:, None, None],
            'impulse_vectors_ns': impulses, 'contact_point_counts': counts,
            'frame_durations_s': durations.copy()}
