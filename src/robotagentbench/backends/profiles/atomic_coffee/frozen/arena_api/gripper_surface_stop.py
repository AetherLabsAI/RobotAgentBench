"""Shared, per-episode contact-stop controller. Position targets only.

Same implementation for either action source and both primary grippers. No
task IDs, force/PD/material edits, body pose writes, attachments or success gates.
"""
from dataclasses import dataclass, asdict
import math

PROFILE = "arena.gripper.surface_stop.v1"


@dataclass(frozen=True)
class Config:
    latch_force_n: float = 5.0
    hold_force_n: float = 10.0
    preload_rad: float = .004
    max_follow_rad: float = .03
    # L1/L2's .002/.005 rad per 20 Hz frame, converted to time-based rates.
    low_force_rate_rad_s: float = .04
    one_sided_rate_rad_s: float = .10
    closed_fraction: float = .5
    lower_margin_rad: float = .001


def specification():
    return dict(profile_id=PROFILE, parameters=asdict(Config()),
        activation="requested opening <= half USD range; opening half releases latch",
        contact_scope="opposing pad loads on the same external dynamic rigid body",
        force_measurement="net normal-impulse vector per pad divided by physics dt",
        missing_contact="no stale-force reuse; no blind extra tightening after both pads lose contact",
        interpolation="latched target bypasses gripper interpolation; native lead bound retained",
        initialization="native reset unchanged; latch starts at first execution step",
        physics_configuration_changed=False, task_success_implemented=False)


def norm(v):
    return math.sqrt(sum(x*x for x in v))


class SurfaceStop:
    def __init__(self, joints, config=Config()):
        self.config = config
        values = asdict(config)
        if any(not math.isfinite(v) or v < 0 for v in values.values()):
            raise ValueError("Invalid contact-stop configuration")
        if config.closed_fraction > 1 or config.hold_force_n < config.latch_force_n:
            raise ValueError("Invalid contact-stop thresholds")
        self.state = {name: None for name in joints}
        self.last_sample = None

    def apply(self, requested, candidate, measured, limits, contacts, *, sample, dt):
        """Once per physical substep, using the immediately preceding sample.

        contacts[joint][rigid_body] is [inner force vector, outer force vector].
        Latch only on opposing loads. Lost contact never changes object state.
        """
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Invalid physics dt")
        if self.last_sample is not None and sample <= self.last_sample:
            raise RuntimeError("Stale/reused gripper contact sample")
        for groups in contacts.values():
            for forces in groups.values():
                if len(forces) != 2 or any(len(v) != 3 for v in forces):
                    raise ValueError("Invalid pad forces")
                if not all(math.isfinite(x) for v in forces for x in v):
                    raise ValueError("Nonfinite pad force")
        self.last_sample = sample
        result, evidence = dict(candidate), []
        c = self.config
        for name, state in self.state.items():
            lo, hi = limits[name]
            actual, wanted = measured[name], requested[name]
            if not all(math.isfinite(v) for v in (lo, hi, actual, wanted)) or hi <= lo:
                raise ValueError("Invalid gripper input")
            event = "unlatched"
            groups = contacts.get(name, {})
            if wanted > lo + c.closed_fraction*(hi-lo):
                event = "release" if state else "open"
                state = None
            elif state is None:
                eligible = []
                for body, (a, b) in groups.items():
                    loads = (norm(a), norm(b))
                    if min(loads) >= c.latch_force_n and sum(x*y for x,y in zip(a,b)) < 0:
                        eligible.append((min(loads), body, loads))
                if eligible:
                    _, body, loads = max(eligible)
                    target = max(lo+c.lower_margin_rad, min(hi, actual-c.preload_rad))
                    state = dict(body=body, target=target, latched_target=target,
                                 lower=max(lo+c.lower_margin_rad, target-c.max_follow_rad))
                    event = "latch"
            else:
                forces = groups.get(state["body"], ((0.,0.,0.), (0.,0.,0.)))
                loads = [norm(v) for v in forces]
                positive = [v > 1e-9 for v in loads]
                if any(positive):
                    opposing = sum(x*y for x,y in zip(*forces)) < 0
                    rate = (c.one_sided_rate_rad_s if not all(positive) else
                            c.low_force_rate_rad_s if opposing and min(loads) < c.hold_force_n else 0.)
                    target = max(state["lower"], state["target"]-rate*dt)
                    event = "follow" if target < state["target"] else "hold"
                    state["target"] = target
                else:
                    event = "contact_lost_hold_bounded_target"
            self.state[name] = state
            if state is not None:
                result[name] = state["target"]
            evidence.append(dict(joint=name, sample=sample, event=event,
                requested_rad=wanted, measured_rad=actual, target_rad=result[name],
                state=None if state is None else dict(state),
                loads_n={body:[norm(v) for v in forces] for body,forces in groups.items()}))
        return result, evidence
