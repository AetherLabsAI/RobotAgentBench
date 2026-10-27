"""Ignore anonymous contact events; retain every identifiable raw contact.

User policy: missing actor/collider IDs never terminate this short task, at
any phase or rate. These headers cannot establish contact/load/holding and
are not given guessed identities. Other malformed evidence still reaches
the original reader's validation. No physics or scores are written here.
"""

RULE = 'ignore_anonymous_contact_event.v2'
IDENTITY_FIELDS = ('stage_id', 'actor0', 'actor1', 'collider0', 'collider1')


def pair_key(h):
    return (int(h.stage_id), *sorted(((int(h.actor0), int(h.collider0)),
                                    (int(h.actor1), int(h.collider1)))))


class ContactLifecycle:
    def __init__(self, source, decode, clock, fridge_root):
        self.source, self.decode, self.clock = source, decode, clock
        self.fridge_root = fridge_root.rstrip('/')
        self.active = set()
        self.last_read_time = None
        self.transition = None

    def note_transition(self, before, after):
        if before != after:
            self.transition = (before, after, float(self.clock.current_time), int(self.clock.hz))

    def normalize(self, report):
        headers, data = report[:2]
        valid, events = [], []
        for i, header in enumerate(headers):
            missing = [key for key in IDENTITY_FIELDS if int(getattr(header, key)) <= 0]
            if not missing:
                valid.append(header)
                continue
            events.append(dict(rule=RULE, time_s=float(self.clock.current_time),
                removed_header_index=i, event_type=str(header.type), missing_identity_fields=missing,
                raw_header={key: int(getattr(header, key)) for key in
                    (*IDENTITY_FIELDS, 'contact_data_offset', 'num_contact_data')},
                physical_values_modified=False, ignored_for_scoring=True))
        if not events:
            return report, []
        # Preserve data/offsets and header objects exactly; never compact points,
        # invent a LOST/FOUND, or infer a grasp from the omitted event.
        return (valid, data, *report[2:]), events

    def commit(self, report):
        for h in report[0]:
            key = pair_key(h)
            if h.type == self.source.lost:
                self.active.discard(key)
            elif h.type in (self.source.found, self.source.persist):
                self.active.add(key)
        self.last_read_time = float(self.clock.current_time)
