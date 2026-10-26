"""Consume actual opposed contact during closure, not a later timer phase."""


def acquisition_ready(*,phase,already_acquired,measured_contact,opposed_contact,closure_values):
    if any(type(v) is not bool for v in (already_acquired,measured_contact,opposed_contact)):
        raise ValueError('Explicit native contact/acquisition facts required')
    if not closure_values or any(not 0<=v<=1 for v in closure_values.values()):
        raise ValueError('Current bounded finger closure values required')
    return (phase in ('approach','carry') and not already_acquired and measured_contact and opposed_contact
            and any(value>0 for value in closure_values.values()))
