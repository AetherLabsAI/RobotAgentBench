"""Trusted environment lifecycle coordinator with task-owned native backend.

Backend must read real PhysX facts and command only its declared human actor.
The user approved a temporary native fixed joint after human contact for this
scripted interference (not a friction-only human grasp and never robot help).
This module neither teleports a target nor fabricates contact/grasp facts.
"""
from copy import deepcopy
from .intervention_rules import HumanCollisionLedger, InterventionAudit, Phase
from .reservation import slot_is_clear
from .proximity import ProximityTrigger
import warnings


class EnvironmentActor:
    def __init__(self, backend):
        if (getattr(backend, 'physics_backend', None) != 'cpu_physx'
                or getattr(backend, 'motion_backend', None) != 'cpu'):
            raise ValueError('Human actor requires CPU PhysX and CPU motion control')
        self.backend = backend
        self.audit = InterventionAudit(backend.target_path, backend.shelf_level_path)
        self.proximity = ProximityTrigger(backend.target_path)
        self.ledger = HumanCollisionLedger(backend.robot_body_ids, backend.human_body_ids)
        self.armed = self.closed = False
        self.index = 0
        self.started_at = None
        self.release_gate_pending = False
        self.contact_gap_pending = False
        self.intervention_evidence_unknown = False

    def initialize(self, clock, view, ports):
        if len(ports) != 1:
            raise ValueError('Exactly one drink component required')
        self.clock = clock
        self.port = next(iter(ports.values()))
        if (not callable(getattr(self.port, 'set_external_control', None))
                or not callable(getattr(self.port, 'unlock_pickup_after_intervention', None))):
            raise ValueError('An intervention-aware target port is required')
        bind_port = getattr(self.backend, 'bind_port', None)
        if bind_port is not None:
            bind_port(self.port)
        self.backend.initialize(clock, view)
        self.reservation = deepcopy(getattr(self.backend, 'initialization_reservation', {}))
        if (self.reservation.get('target_path') != self.audit.target
                or self.reservation.get('shelf_level_path') != self.audit.shelf_level):
            raise ValueError('Exact target/shelf initialization reservation required')
        if not callable(getattr(self.backend, 'reservation_obstacle_bounds', None)):
            raise ValueError('Read-only reserved-slot occupancy check required')
        if not slot_is_clear(self.reservation, self.backend.reservation_obstacle_bounds()):
            raise ValueError('Initialization reserved destination is occupied')
        if 'bilateral_routes' in self.reservation:
            from .bilateral_routes import reservations
            for route in reservations(self.reservation):
                if not slot_is_clear(route,self.backend.reservation_obstacle_bounds()):
                    raise ValueError('Initialization bilateral destination is occupied')

    def arm(self):
        if self.armed or self.closed:
            raise RuntimeError('Arm once after initial admission')
        self.armed = True

    def before_physics(self, dt):
        if not self.armed or self.closed:
            raise RuntimeError('Actor has not been admitted')
        if self.release_gate_pending:
            # Previous native substep has now also been read by the target port.
            if self.audit.phase != Phase.DONE:
                raise RuntimeError('Cannot unlock pickup before completed relocation and exit')
            self.port.set_external_control(False)
            self.port.unlock_pickup_after_intervention(self.audit)
            self.release_gate_pending = False
        if self.audit.phase == Phase.WAITING:
            state = self.backend.trigger_facts()
            if state['simulation_time_s'] != float(self.clock.current_time):
                raise RuntimeError('Stale proximity evidence')
            # Recheck the already fixed slot. Never select another point or
            # move background stock in response to a runtime obstruction.
            clear = slot_is_clear(self.reservation, self.backend.reservation_obstacle_bounds())
            selected = None
            if 'bilateral_routes' in self.reservation:
                selected = self.backend.choose_reservation(state)
                clear = selected is not None
            decision = self.proximity.observe(state, route_clear=clear)
            if decision['start'] and self.audit.trigger(
                    robot_holding=state['robot_holding_target'],
                    robot_contacting=state['robot_contacting_target']):
                if selected is not None:
                    self.reservation = deepcopy(selected)
                self.port.set_external_control(True)
                self.started_at = float(self.clock.current_time)
                self.backend.start(duration_s=3.0, reservation=deepcopy(self.reservation))
        if self.audit.phase not in (Phase.WAITING, Phase.DONE) and getattr(self.backend,'phase',None) != 'done':
            self.backend.before_physics(dt)

    def after_physics(self, report):
        # Native helper report is never rewritten or stripped.
        facts = self.backend.read_native_facts()
        now = float(self.clock.current_time)
        if facts['simulation_time_s'] != now:
            raise RuntimeError('Stale external-actor evidence')
        if (facts['target_path'] != self.audit.target
                or facts['shelf_level_path'] != self.audit.shelf_level):
            raise RuntimeError('External actor changed target or shelf identity')
        self.ledger.observe(self.index, now, facts['contact_pairs'],
                            complete=facts['contacts_complete'])
        self.index += 1
        phase = Phase(facts['phase'])
        if not facts['contacts_complete']:
            self.contact_gap_pending = True
            return  # unknown is neither collision-free nor supported release
        if facts.get('intervention_evidence_unknown'):
            self.intervention_evidence_unknown = True
        if self.intervention_evidence_unknown:
            return
        phases = list(Phase)
        if self.contact_gap_pending and phases.index(phase) > phases.index(self.audit.phase) + 1:
            warnings.warn('Contact gap spanned an interference transition; continuing '
                          'with transition evidence unknown and pickup credit locked.', RuntimeWarning)
            self.intervention_evidence_unknown = True
            return
        self.contact_gap_pending = False
        if phase != self.audit.phase:
            if phase == Phase.DONE and (
                    self.started_at is None or now - self.started_at < 3.0 - 1e-6):
                raise RuntimeError('Actor completed before its approved 3-second motion')
            self.audit.advance(
                phase, target_path=facts['target_path'],
                shelf_level_path=facts['shelf_level_path'],
                human_released=facts['human_released'],
                native_supported=facts['target_supported'],
                destination_valid=facts['destination_valid'],
                actor_clear=facts['actor_clear'])
            if phase == Phase.DONE:
                self.release_gate_pending = True

    def score_snapshot(self, snapshot):
        value = deepcopy(snapshot)
        if self.intervention_evidence_unknown:
            value['scoring_status'] = 'waiting_for_contact_evidence'
            value['success'] = False
        # Also protect public results from legacy/incorrect pre-relocation facts.
        # Collision monitoring continues; only robot pickup awards are gated.
        if self.audit.phase != Phase.DONE:
            for channel in ('process', 'current_final', 'total'):
                if channel in value['scores']: value['scores'][channel] = 0
            value['success'] = False
            for row in value.get('targets', []):
                for key in ('process', 'current_final', 'component_total'):
                    if key in row: row[key] = 0
                row.update(success=False, process_complete=False)
            for row in value.get('stages', []):
                if 'process_points' in row: row['process_points'] = 0
                if 'process_complete' in row: row['process_complete'] = False
        deduction = self.ledger.deduction
        value['scores']['deductions'] += deduction
        # Preserve the existing ShortTask nonnegative-score convention.
        value['scores']['total'] = max(0, value['scores']['total'] - deduction)
        value['success'] = bool(value['success'] and self.audit.phase == Phase.DONE)
        # Existing result transport preserves target rows, not arbitrary root fields.
        for target in value['targets']:
            target['interference'] = dict(
                phase=self.audit.phase.value,
                exercised=self.audit.phase == Phase.DONE,
                contact_episodes=self.ledger.contact_episodes,
                human_collision_deduction=deduction)
            target['interference']['pickup_scoring_unlocked'] = bool(
                self.audit.phase == Phase.DONE and getattr(self.port,'pickup_unlocked',False))
            target['interference']['proximity_trigger'] = deepcopy(self.proximity.latest)
            target['interference']['contact_gap_steps'] = self.ledger.gap_steps
            target['interference']['intervention_evidence_unknown'] = self.intervention_evidence_unknown
        return value

    def close(self):
        if not self.closed:
            try:
                self.backend.close()
            finally:
                self.closed = True
