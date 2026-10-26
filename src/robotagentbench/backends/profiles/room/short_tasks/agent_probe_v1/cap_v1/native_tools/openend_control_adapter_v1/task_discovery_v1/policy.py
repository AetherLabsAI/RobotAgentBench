"""Object-level coverage from the existing native finger-contact stream.

This reducer does not query or modify physics. Its trusted caller owns scene
acceptance, the frozen private object catalogue and the unchanged native clock.
"""
from copy import deepcopy
import math

RULE_VERSION = "openend.task_discovery.object_coverage.v1"


def _path(value):
    if (not isinstance(value, str) or not value.startswith("/World/")
            or any(s in ("", ".", "..") for s in value.split("/")[1:])):
        raise ValueError("Canonical absolute prim path required")
    return value


def _within(path, root):
    return path == root or path.startswith(root + "/")


def _time(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Finite nonnegative native simulation time required")
    return value


class CoverageLedger:
    def __init__(self, *, episode_id, targets, non_targets, fingers):
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("Episode identity required")
        if not targets:
            raise ValueError("Empty-target scenes are not discovery evaluations")
        if set(targets) & set(non_targets):
            raise ValueError("One physical object cannot be both target and background")
        self.episode_id = episode_id
        self.targets = self._catalogue(targets)
        self.non_targets = self._catalogue(non_targets)
        self.fingers = tuple(_path(p) for p in fingers)
        if not self.fingers or any(not p.startswith("/World/Robot/gripper_") for p in self.fingers):
            raise ValueError("Existing native finger body roots required, not arms")
        roots = [(key, root) for key, paths in {**self.targets, **self.non_targets}.items() for root in paths]
        for i, (key, root) in enumerate(roots):
            for other_key, other in roots[i + 1:]:
                if key != other_key and (_within(root, other) or _within(other, root)):
                    raise ValueError("Ambiguous overlapping physical-object bindings")
        self.index = roots
        self.origin = self.last_time = None
        self.sample_index = 0
        self.target_events = {}
        self.wrong_events = {}
        self.unmapped_pairs = set()
        self.final_result = None

    @staticmethod
    def _catalogue(objects):
        result = {}
        for key, paths in objects.items():
            if not isinstance(key, str) or not key.strip() or isinstance(paths, str):
                raise ValueError("Explicit physical object ID and root list required")
            roots = tuple(_path(p) for p in paths)
            if not roots or len(set(roots)) != len(roots):
                raise ValueError("Nonempty unique roots required for each object")
            if any(_within(p, "/World/Robot") for p in roots):
                raise ValueError("Robot cannot be an evaluated scene object")
            result[key] = roots
        return result

    def arm(self, time_s):
        if self.origin is not None:
            raise RuntimeError("Arm only once, after accepted initialization")
        self.origin = self.last_time = _time(time_s)

    def observe(self, pairs, *, episode_id, time_s, accepting_contacts):
        if self.origin is None or self.final_result is not None:
            raise RuntimeError("A live accepted episode is required")
        if episode_id != self.episode_id:
            raise ValueError("Cross-episode contact evidence rejected")
        now = _time(time_s)
        if now <= self.last_time or type(accepting_contacts) is not bool:
            raise ValueError("Fresh native substep and explicit host deadline decision required")
        found = {}
        unmapped = set()
        for paths, samples in pairs:
            if len(paths) != 2:
                raise ValueError("Native collider pair required")
            a, b = (_path(p) for p in paths)
            if not samples:
                continue
            hands = [any(_within(p, root) for root in self.fingers) for p in (a, b)]
            if hands[0] == hands[1]:
                continue
            other = b if hands[0] else a
            # Robot self-contact is not discovery or a scene-object mistake.
            if _within(other, "/World/Robot"):
                continue
            matched = [key for key, root in self.index if _within(other, root)]
            if matched:
                found.setdefault(matched[0], dict(collider_paths=[a, b]))
            else:
                unmapped.add(other)
        self.last_time = now
        self.sample_index += 1
        new = []
        if not accepting_contacts:
            return new
        self.unmapped_pairs.update(unmapped)
        for key, evidence in sorted(found.items()):
            ledger = self.target_events if key in self.targets else self.wrong_events
            if key in ledger:
                continue
            event = dict(episode_id=self.episode_id, object_id=key, time_s=now,
                         sample_index=self.sample_index,
                         is_target=key in self.targets,
                         source="existing_native_physx_finger_contact_stream",
                         **evidence)
            ledger[key] = event
            new.append(deepcopy(event))
        return new

    def report(self):
        if self.final_result is not None:
            return deepcopy(self.final_result)
        hits, total = len(self.target_events), len(self.targets)
        return dict(rule_version=RULE_VERSION, episode_id=self.episode_id,
                    status="running", evaluation_valid=None,
                    score=hits, maximum_score=total, coverage=hits / total,
                    coverage_percent=100 * hits / total,
                    all_targets_contacted=hits == total,
                    target_object_count=total, contacted_target_count=hits,
                    missed_target_ids=sorted(set(self.targets) - set(self.target_events)),
                    wrong_object_count=len(self.wrong_events),
                    wrong_object_ids=sorted(self.wrong_events),
                    unmapped_collider_count=len(self.unmapped_pairs),
                    non_target_penalty=0, wrong_contact_terminates=False)

    def finalize(self, *, evaluation_valid, termination_reason):
        if type(evaluation_valid) is not bool or not isinstance(termination_reason, str) or not termination_reason:
            raise ValueError("Explicit native validity and termination reason required")
        if self.origin is None:
            raise RuntimeError("Cannot finalize an unaccepted scene")
        if self.final_result is not None:
            if (self.final_result["evaluation_valid"] != evaluation_valid
                    or self.final_result["termination_reason"] != termination_reason):
                raise ValueError("Conflicting finalization")
            return deepcopy(self.final_result)
        result = self.report()
        result.update(status="completed" if evaluation_valid else "invalid",
                      evaluation_valid=evaluation_valid,
                      termination_reason=termination_reason,
                      success=evaluation_valid and result["all_targets_contacted"],
                      events=deepcopy(self.target_events),
                      wrong_events=deepcopy(self.wrong_events))
        if not evaluation_valid:
            result.update(observed_coverage=result["coverage"], score=None,
                          coverage=None, coverage_percent=None)
        self.final_result = result
        return deepcopy(result)
