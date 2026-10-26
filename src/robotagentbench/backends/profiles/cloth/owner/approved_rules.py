"""Approved cloth rules over explicit facts; no simulator, actions or timeout loop.

Only a trusted task-owned native reader may supply physical facts in evaluation.
This engine does not measure folds/grasp/support, certify native evidence, or
register an evaluator. Incomplete evidence is an error, not zero-score failure.
"""
from copy import deepcopy
from dataclasses import dataclass, fields
import json
import math
from pathlib import Path

POLICY = json.loads((Path(__file__).with_name('SCORING_POLICY.json')).read_text())


@dataclass(frozen=True)
class Binding:
    task_name: str
    episode_id: str
    target_path: str
    source_path: str
    destination_path: str

    def __post_init__(self):
        if self.task_name not in POLICY['stages'] or not isinstance(self.episode_id, str) or not self.episode_id:
            raise ValueError('Explicit cloth task and episode identity required')
        paths = (self.target_path, self.source_path, self.destination_path)
        if len(set(paths)) != 3 or any(not isinstance(p, str) or not p.startswith('/World/')
                or '//' in p or any(x in ('.', '..') for x in p.split('/')) for p in paths):
            raise ValueError('Distinct canonical scene-bound role paths required')


@dataclass(frozen=True)
class Facts:
    binding: Binding
    sample_index: int
    time_s: float
    physics_dt_s: float
    complete: bool
    finger_contact: bool
    held: bool
    grasp_lift_verified: bool
    whole_clear_source: bool
    whole_inside_destination: bool
    destination_support_verified: bool
    source_supported: bool
    folded_at_least_once_verified: bool
    unfolded_verified: bool
    coverage: float
    stable_verified: bool
    on_floor: bool


class ClothProcessScore:
    """Per-episode numerical scoring only; generic owner must bind and close it."""

    def __init__(self, binding):
        if not isinstance(binding, Binding):
            raise ValueError('Typed scene binding required')
        self.binding = binding
        self.history = set()
        self.dropped = False
        self.hold_start = None
        self.placed = self.released = False
        self.last = self.cached = self.termination = None

    def _validate(self, f):
        if not isinstance(f, Facts) or f.binding != self.binding:
            raise ValueError('Wrong task, episode or object binding')
        if type(f.sample_index) is not int or f.sample_index < 0:
            raise ValueError('Nonnegative measured sample index required')
        for key in ('time_s', 'physics_dt_s', 'coverage'):
            value = getattr(f, key)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('Finite numerical evidence required')
        if f.time_s < 0 or f.physics_dt_s != 1 / 60 or not 0 <= f.coverage <= 1:
            raise ValueError('Original 60 Hz substeps and fractional area coverage required')
        excluded = {'binding', 'sample_index', 'time_s', 'physics_dt_s', 'coverage'}
        if any(type(getattr(f, field.name)) is not bool for field in fields(f) if field.name not in excluded):
            raise ValueError('Explicit boolean physical evidence required')
        if not f.complete:
            raise ValueError('Missing physical evidence; not a scored failure')
        if f.grasp_lift_verified and (not f.held or not f.finger_contact or f.on_floor):
            raise ValueError('Grasp/lift facts contradict contact, hold or floor state')
        if f.folded_at_least_once_verified and f.unfolded_verified:
            raise ValueError('Contradictory current fold state')
        if self.last is not None and f == self.last:
            return
        expected = 0 if self.last is None else self.last.sample_index + 1
        if f.sample_index != expected:
            raise ValueError('Missing or stale measured substep')
        if self.last is not None and not math.isclose(f.time_s - self.last.time_s,
                f.physics_dt_s, rel_tol=0, abs_tol=1e-9):
            raise ValueError('Continuous simulation-time evidence required')

    def observe(self, f):
        if self.termination is not None:
            raise RuntimeError('Finalized scorer cannot accept further evidence')
        self._validate(f)
        if self.last is not None and f == self.last:
            return deepcopy(self.cached)  # Cached read, no repeated awards or time.
        if self.last is None:
            if (f.finger_contact or f.held or f.on_floor or not f.source_supported
                    or f.whole_clear_source or f.whole_inside_destination):
                raise ValueError('Initial target must be unheld and supported at its source')
            folded = f.folded_at_least_once_verified
            if self.binding.task_name == 'spread_tablecloth' and not folded:
                raise ValueError('Spreading must start with folded cloth')
            if self.binding.task_name == 'collect_tablecloth' and (folded or not f.unfolded_verified):
                raise ValueError('Collection must start with unfolded cloth on the table')
        else:
            if f.on_floor:
                self.dropped = True
            else:
                if f.finger_contact:
                    self.history.add('contact')
                if 'contact' in self.history and f.grasp_lift_verified:
                    self.history.add('grasp_lift')
                if self.binding.task_name == 'spread_tablecloth':
                    if 'grasp_lift' in self.history and f.whole_clear_source:
                        self.history.add('clear_source')
                elif ('contact' in self.history and f.folded_at_least_once_verified
                      and not f.whole_inside_destination):
                    # A fold on the tabletop before lifting is allowed. A fold
                    # first observed after basket containment earns no fold stage.
                    self.history.add('folded')
            ready = len(self.history) == 3
            condition = (f.unfolded_verified and f.coverage >= POLICY['spread_designated_region_coverage_minimum']
                         if self.binding.task_name == 'spread_tablecloth'
                         else f.folded_at_least_once_verified)
            # Spreading is judged by unfolded surface coverage and real table
            # support, not an extra unapproved whole-cloth containment gate.
            contained = (True if self.binding.task_name == 'spread_tablecloth'
                         else f.whole_inside_destination)
            self.placed = bool(ready and condition and f.whole_clear_source
                and contained and f.destination_support_verified and not f.on_floor)
            holding = self.placed and not f.held and not f.finger_contact and f.stable_verified
            if not holding:
                self.hold_start = None
            elif self.hold_start is None:
                self.hold_start = f.time_s
            self.released = bool(holding and f.time_s - self.hold_start + 1e-9 >=
                                 POLICY['released_stability_duration_simulation_s'])
        self.last = f
        self.cached = self._summary()
        return deepcopy(self.cached)

    def _summary(self):
        current = int(self.placed) + int(self.released)
        deductions = POLICY['floor_drop_penalty_once_per_episode'] if self.dropped else 0
        return dict(task_name=self.binding.task_name, episode_id=self.binding.episode_id,
            rule_version='cloth.process_score.v1',
            scores=dict(process=len(self.history), current_final=current,
                deductions=deductions, total=max(0, len(self.history) + current - deductions),
                maximum=POLICY['maximum_per_task']),
            success=self.released, stages=sorted(self.history) +
                (['placed'] if self.placed else []) + (['released'] if self.released else []),
            evidence_step=self.last.sample_index, evidence_time_s=self.last.time_s,
            evidence_status='external_facts_not_native_verified', native_admission=False,
            benchmark_registered=False, termination=self.termination,
            timeout_owner='shared_episode', no_progress_timeout_s=120,
            floor_drop_is_terminal_failure=False)

    def finalize(self, reason):
        if self.last is None:
            raise ValueError('Cannot finalize absent evidence')
        if reason not in ('success', 'client_closed', 'no_new_net_progress_timeout'):
            raise ValueError('Supported shared lifecycle termination required')
        if self.termination is not None:
            if self.termination != reason:
                raise ValueError('Contradictory settlement')
            return deepcopy(self.cached)
        if (reason == 'success') != self.released:
            raise ValueError('Termination must match current measured completion')
        self.termination = reason
        self.cached = self._summary()
        return deepcopy(self.cached)
