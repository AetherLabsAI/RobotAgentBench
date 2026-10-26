"""Expose the original wallet pickup pause to the original room lifecycle.

LegacyPickup owns a RecoverablePick which already implements pause. Preserve
that exact scorer and its original hold, awards and finalization semantics.
The subclass identity is essential: Short Transfer.finalize uses isinstance
LegacyPickup to apply its existing incomplete-hold deduction.
"""
from short_tasks.tasks.sofa_to_drawer.transfer import LegacyPickup
from openend.tasks.wallet_transfer.components import RecoverablePick


class WalletPickupLifecycle(LegacyPickup):
    def __init__(self, source):
        if type(source) is not LegacyPickup or type(source.core) is not RecoverablePick:
            raise ValueError('Original wallet LegacyPickup and RecoverablePick required')
        self.source = source

    def __getattr__(self, name):
        return getattr(self.source, name)

    def observe(self, *args, **kwargs):
        return self.source.observe(*args, **kwargs)

    def summary(self):
        return self.source.summary()

    def pause(self):
        return self.source.core.pause()


def wallet_port():
    from short_tasks.tasks.sofa_to_drawer.native_port import WalletDrawerPort

    class LongWalletDrawerPort(WalletDrawerPort):
        def create_composite(self, row):
            super().create_composite(row)
            self.composite.pick = WalletPickupLifecycle(self.composite.pick)

    return LongWalletDrawerPort
