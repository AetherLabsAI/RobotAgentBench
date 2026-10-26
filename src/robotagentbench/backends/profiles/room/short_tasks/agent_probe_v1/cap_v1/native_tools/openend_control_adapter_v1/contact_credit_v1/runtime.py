"""Owned OpenEnd factory extension. Original ports/dispatcher/physics are reused."""
import json
from openend.room import MultiTaskRoom
from .policy import ContactLedger, KINDS, RULE_VERSION


def target_roots(kind, port):
    if kind not in KINDS:
        raise ValueError('Contact-credit binding is not approved for '+kind)
    if kind == 'coffee':
        # Actual manipulands, not an incidental collision with the machine body.
        r = port.owner.service.roles
        return tuple(r[k] for k in ('cup', 'bag', 'button'))
    if kind == 'fryer':
        # The existing native tray body includes its basket handle. Neither the
        # appliance chassis nor the unselected cake can earn this point.
        return (port.source.paths['tray'],)
    if kind == 'fridge_to_microwave':
        from openend.tasks.meal_tray.task_contact_routing import roots_for
        return roots_for(kind, port, None)
    if kind == 'microwave_box_to_meal_tray':
        from openend.tasks.meal_tray.gripper_wake_v1.host import bindings_for
        return bindings_for(kind, port, None, None)
    if kind == 'snack_drink_to_tray':
        # SnackTrayPort exposes its exact selected rigid body through SceneBinding,
        # not PortablePort.geometry. Exclude stock peers, shelf and shared tray.
        return tuple(port.scene.identity.candidates)
    # Never bind a stock shelf, destination surface, trash bin or task workspace.
    return (port.geometry.target,)


def permits_contact(kind, port, now):
    if port.ended:
        return False
    if kind in ('coffee_to_meal_tray', 'bread_to_meal_tray'):
        owner = port.device_owner
        return bool(owner is not None and owner.composite.success)
    if kind == 'microwave_box_to_meal_tray':
        h = port.handoff
        return bool(h is not None and h.ready and now > h.time_s + 1e-9)
    return True


class ContactDispatcher:
    """Score-only composition; no change to awake set, timeouts or Hz policy."""
    def __init__(self, native, ledger):
        self.native, self.ledger = native, ledger

    def __getattr__(self, name):
        return getattr(self.native, name)

    @property
    def registrations(self):
        return self.native.registrations

    @registrations.setter
    def registrations(self, value):
        self.native.registrations = value

    def record_scores(self, results):
        return self.native.record_scores({k: self.ledger.augment(k, r) for k, r in results.items()})

    def record_score(self, instance_id, result):
        if instance_id != self.native.current:
            raise ValueError('Only current task may use single-score submission')
        return self.record_scores({instance_id: result})

    def settle_terminal_scores(self, scores):
        return self.native.settle_terminal_scores({k: self.ledger.augment(k, r) for k, r in scores.items()})


class ContactRoom(MultiTaskRoom):
    def __init__(self, stage, manifest, output, episode_id, *, request, sensor, workspace_selector=None):
        super().__init__(stage, manifest, output, episode_id, request=request,
                         sensor=sensor, workspace_selector=workspace_selector)
        try:
            if self.contact_routes is None:
                raise ValueError('Native central contact stream required')
            # This is the original constructor's actual expanded registration;
            # in particular one selected garbage family can resolve to N items.
            registration = json.loads((self.output/'registration.json').read_text())
            kinds = {t['instance_id']: t['type'] for t in registration['request']['tasks']}
            if set(kinds) != set(self.ports):
                raise ValueError('Expanded task identity mismatch')
            from ..bag_support_v2.policy import enable_for_ports
            packed_support = enable_for_ports(kinds, self.ports)
            self.contact_ledger = ContactLedger(episode_id=episode_id,
                kinds=kinds, bindings={k: target_roots(kind, self.ports[k]) for k, kind in kinds.items()},
                fingers=self.contact_routes.native.fingers)
            self.dispatch = ContactDispatcher(self.dispatch, self.contact_ledger)
            (self.output/'contact_rule.json').write_text(json.dumps(dict(
                rule_version=RULE_VERSION, kinds=kinds, bindings=self.contact_ledger.bindings,
                points_per_registered_instance=1, original_points_removed=0,
                underlying_ports_and_physics_unchanged=not any(packed_support['enabled'].values()),
                physics_parameters_changed=False,
                bag_support_policy=packed_support), indent=2)+'\n')
            self.contact_trace = (self.output/'contact_events.jsonl').open('x')
        except BaseException:
            super().close()
            raise

    def accept_initial(self):
        receipt = super().accept_initial()
        self.contact_ledger.arm(self.clock.current_time)
        (self.output/'contact_initial.json').write_text(json.dumps(dict(
            rule_version=RULE_VERSION, accepted_at_simulation_s=self.clock.current_time,
            scoreboard={k: self.contact_ledger.augment(k, p.score)
                        for k, p in self.ports.items()},
            initial_contact_points=0), indent=2)+'\n')
        return receipt

    def _after_physics(self, helper_report):
        super()._after_physics(helper_report)
        if not self.started:
            return
        now = self.clock.current_time
        # Do not bank a late contact and resurrect an expired episode during
        # finalization. The original dispatcher still owns expiry and deadline.
        on_time = now <= self.dispatch.last_progress_time + self.dispatch.timeout + 1e-9
        eligible = {k: bool(on_time and permits_contact(kind, self.ports[k], now))
                    for k, kind in self.contact_ledger.kinds.items()}
        # Reuse the already ingested current cache; no second report/pose query.
        events = self.contact_ledger.observe(self.contact_routes.native.contacts.pairs(),
                                            time_s=now, eligible=eligible)
        for event in events:
            self.contact_trace.write(json.dumps(event)+'\n')
        if events:
            self.contact_trace.flush()

    def _settle_terminal(self):
        if hasattr(self, 'terminal_result'):
            return
        super()._settle_terminal()
        (self.output/'contact_final.json').write_text(json.dumps(
            self.contact_ledger.finalize(), indent=2)+'\n')

    def close(self):
        try:
            super().close()
        finally:
            stream = getattr(self, 'contact_trace', None)
            if stream is not None:
                stream.close()
