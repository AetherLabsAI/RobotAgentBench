"""Adapt the original instantaneous book-surface rule to room suspension.

CurrentSurface has no hold timer, pending contact window or historical awards.
Its sole state is the last observed current_final predicate. Suspending evidence
must preserve that state; only a new measured observe call may change its three
current-region points. The surrounding original SuspensibleEvidence still
pauses Pickup, clearing its unfinished hold/contact windows and advancing the
validated inactive clock. No scorer, native reader or physics method is replaced.
"""


class CurrentSurfaceLifecycle:
    """A narrow instance adapter; observe/summary remain the original methods."""
    def __init__(self, surface):
        from short_tasks.tasks.sofa_to_drawer.transfer import CurrentSurface
        if type(surface) is not CurrentSurface or set(vars(surface)) != {'current_final'}:
            raise ValueError('Original stateless CurrentSurface required')
        self.source = surface

    @property
    def current_final(self):
        return self.source.current_final

    def observe(self, *args, **kwargs):
        return self.source.observe(*args, **kwargs)

    def summary(self):
        return self.source.summary()

    def pause(self):
        # There is no pending surface hold to interrupt. Return the unchanged
        # original measured-state summary; do not clear points or fabricate facts.
        # Pickup.pause is separately called by the original suspend_substep.
        return self.source.summary()


def factory_override(plan, declaration):
    """Keep the trusted Short port and attach only the missing lifecycle API."""
    from short_tasks.tasks.sofa_to_coffee_table.adapter import factory_override as original_factory
    if plan.get('asset') != 'book_red' or declaration.get('type') != 'book_transfer':
        raise ValueError('Long book surface adapter requires the original book task')
    original = original_factory(plan, declaration)

    class LongBookSurfacePort(original):
        def create_composite(self, row):
            super().create_composite(row)
            self.composite.place = CurrentSurfaceLifecycle(self.composite.place)

    return LongBookSurfacePort
