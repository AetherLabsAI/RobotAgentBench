"""One-shot, episode-local approach/contact rewards; no simulator writes."""
from copy import deepcopy
from dataclasses import dataclass
import math

VERSION = "short_tasks.interaction_progress.v2"
APPROACH_TIERS = ((0.30, 0.1), (0.20, 0.2), (0.10, 0.2))
APPROACH_POINTS = 0.5
TOUCH_POINTS = 1.0


@dataclass(frozen=True)
class Hand:
    distance_m: float
    # Current hand measured against the previous target geometry. This prevents
    # a falling/moving object, without a reaching hand, from earning approach.
    previous_geometry_distance_m: float
    touching: bool


@dataclass(frozen=True)
class Sample:
    active: bool
    left: Hand
    right: Hand


class Progress:
    def __init__(self, episode_id, target_ids):
        self.ids = tuple(target_ids)
        if not isinstance(episode_id,str) or not episode_id or not self.ids or len(set(self.ids)) != len(self.ids):
            raise ValueError("Unique episode and operation target identities required")
        if any(not isinstance(x, str) or not x for x in self.ids):
            raise ValueError("Nonempty target identity required")
        self.episode_id = episode_id
        self.last = None
        self.awards = {x: set() for x in self.ids}
        self.skipped = {x: set() for x in self.ids}
        self.events = []

    @property
    def maximum(self):
        return len(self.ids) * (APPROACH_POINTS + TOUCH_POINTS)

    def set_active_flags(self, flags):
        """Stage transition at a control boundary; retain the measured baseline."""
        if self.last is None or set(flags)!=set(self.ids) or any(type(v) is not bool for v in flags.values()):
            raise ValueError('Complete stage flags after native initial admission required')
        step,time_s,samples=self.last
        for key, sample in samples.items():
            if flags[key] and not sample.active:
                self._skip_satisfied(key, sample)
        self.last=(step,time_s,{k:Sample(flags[k],v.left,v.right) for k,v in samples.items()})

    def _skip_satisfied(self, key, sample):
        # Already-close initial/activation/resume poses do not create an
        # allowance that can be farmed by backing away and approaching again.
        distance = min(sample.left.distance_m, sample.right.distance_m)
        for threshold, _ in APPROACH_TIERS:
            if distance <= threshold and threshold not in self.awards[key]:
                self.skipped[key].add(threshold)

    def observe(self, episode_id, step, time_s, samples, *, baseline=False):
        if episode_id != self.episode_id or set(samples) != set(self.ids):
            raise ValueError("Wrong episode or incomplete operation target evidence")
        if type(step) is not int or step < 0 or type(time_s) not in (int, float) or not math.isfinite(time_s) or time_s < 0:
            raise ValueError("Finite measured clock required")
        for sample in samples.values():
            if not isinstance(sample, Sample) or type(sample.active) is not bool:
                raise ValueError("Typed measured samples required")
            for hand in (sample.left, sample.right):
                if not isinstance(hand, Hand) or type(hand.touching) is not bool:
                    raise ValueError("Typed physical contact required")
                if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
                       for v in (hand.distance_m, hand.previous_geometry_distance_m)):
                    raise ValueError("Finite nonnegative measured distances required")
        frame = (step, float(time_s), deepcopy(samples))
        if self.last == frame:
            return self.summary()
        if self.last is None:
            if not baseline:
                raise ValueError("Initial admission baseline required")
        elif step != self.last[0] + 1 or time_s <= self.last[1]:
            raise ValueError("Missing, repeated or stale measured frame")
        if baseline:
            for key, sample in samples.items():
                self._skip_satisfied(key, sample)
        if self.last is not None and not baseline:
            for key, sample in samples.items():
                old = self.last[2][key]
                if not sample.active:
                    continue
                if not old.active:
                    self._skip_satisfied(key, sample)
                for name in ("left", "right"):
                    hand, previous = getattr(sample, name), getattr(old, name)
                    for threshold, points in APPROACH_TIERS:
                        if (old.active and previous.distance_m > threshold
                                and hand.distance_m <= threshold
                                and hand.previous_geometry_distance_m <= threshold
                                and threshold not in self.skipped[key]):
                            self._award(key, "approach", points, name, step, time_s,
                                        threshold_m=threshold)
                    if hand.touching and not previous.touching:
                        self._award(key, "touch", TOUCH_POINTS, name, step, time_s)
        self.last = frame
        return self.summary()

    def _award(self, key, kind, points, hand, step, time_s, *, threshold_m=None):
        token = threshold_m if kind == "approach" else kind
        if token in self.awards[key]:
            return
        self.awards[key].add(token)
        event = dict(target_id=key, kind=kind, points=points,
                     hand=hand, step=step, time_s=float(time_s))
        if threshold_m is not None:
            event['threshold_m'] = threshold_m
        self.events.append(event)

    def summary(self):
        return dict(version=VERSION, points=round(sum(e["points"] for e in self.events), 10),
                    maximum=self.maximum, events=deepcopy(self.events),
                    approach_tiers=[dict(distance_m=d, points=p) for d,p in APPROACH_TIERS],
                    items=[dict(target_id=k,
                                approach=round(sum(p for d,p in APPROACH_TIERS if d in a), 10),
                                touch=TOUCH_POINTS if "touch" in a else 0.,
                                skipped_approach_thresholds_m=sorted(self.skipped[k], reverse=True))
                           for k, a in self.awards.items()])

    def augment(self, snapshot, *, arithmetic="gross_minus_deductions"):
        """Do not alter underlying scores or their independent success test."""
        out = deepcopy(snapshot)
        base = deepcopy(out["scores"])
        if "interaction_progress" in out:
            raise ValueError("Interaction rewards cannot be added twice")
        bonus = self.summary()
        scores = out["scores"]
        scores["process"] += bonus["points"]
        scores["maximum"] += bonus["maximum"]
        if arithmetic == "component_defined":
            scores["total"] += bonus["points"]
        elif arithmetic == "gross_minus_deductions":
            scores["total"] = max(0., scores["process"] + scores["current_final"] - scores["deductions"])
        else:
            raise ValueError("Unknown score arithmetic")
        out["interaction_progress"] = dict(bonus, base_scores=base)
        out.setdefault("stages", []).extend(dict(name="interaction:" + v["target_id"],
            score=v["approach"]+v["touch"], maximum=1.5,
            approach=v["approach"], touch=v["touch"], supplementary=True,
            rule_version=VERSION,
            events=[e for e in bonus['events'] if e['target_id']==v['target_id']]) for v in bonus["items"])
        return out
