"""Robot commands for the original fixed-base source; no physical setters.

The source command-replay accepts all its existing driven joints, including
head and torso. This is NOT its arm-only repair_request interface. A native34
source vector is preserved exactly; a public20 command is expanded by the
existing NativeAdapter. Nonzero base commands are rejected before execution.
"""
import numpy as np


class SourceControls:
    def __init__(self, *, command_names, driven, initial_targets, adapter):
        self.names = tuple(command_names)
        self.driven = tuple(driven)
        self.adapter = adapter
        self.initial = np.array(initial_targets, dtype=float, copy=True)
        if (len(set(self.names)) != len(self.names)
                or self.initial.shape != (len(self.names),)
                or not np.isfinite(self.initial).all()
                or not set(self.driven) <= set(self.names)
                or not set(adapter.driven) <= set(self.driven)):
            raise ValueError('Exact native command and drive identity required')
        self.columns = np.array([self.names.index(n) for n in self.driven])

    @staticmethod
    def reject_moving_base(velocity):
        if (set(velocity) != {'base_forward_m_s', 'base_yaw_rate_rad_s'}
                or any(type(v) not in (int, float) or not np.isfinite(v) or v != 0
                       for v in velocity.values())):
            raise ValueError('Original tablecloth robot has a fixed world anchor; base must remain zero')

    def public_targets(self, targets, velocity):
        self.reject_moving_base(velocity)
        expanded = self.adapter.expand(targets)
        full = dict(zip(self.names, self.initial))
        full.update(expanded)
        # Preserve the original bilateral command convention, including the
        # passive inner coordinate (the unchanged native Mimic still owns it).
        for inner, outer in self.adapter.followers.items():
            full[inner] = -targets[outer]
        return np.array([full[n] for n in self.driven], dtype=float)

    def source_targets(self, row):
        row = np.asarray(row)
        if row.shape != self.initial.shape or not np.isfinite(row).all():
            raise ValueError('Full finite original native34 command required')
        # Diagnostic source vectors are already native actuator commands.
        # Do not normalize, calibrate, interpolate or re-expand these rows.
        for name, value in zip(self.names, row):
            if name in self.adapter.limits:
                low, high = self.adapter.limits[name]
                if not low <= value <= high:
                    raise ValueError('Source command outside original native limits')
        return np.array(row[self.columns], dtype=float, copy=True)
