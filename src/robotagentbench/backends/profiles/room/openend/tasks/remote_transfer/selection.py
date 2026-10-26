"""Task selection only; never awards points or substitutes for grasp scoring."""
class ContactSelection:
    def __init__(self):
        self.locked=False
        self.finger_contact=False

    def observe(self,*,finger_contact,supports,external_contact,on_floor):
        if any(type(v) is not bool for v in (finger_contact,external_contact,on_floor)):
            raise ValueError('Measured contact booleans required')
        self.finger_contact=finger_contact
        # Hand contact while completely clear of external surfaces is enough
        # to retain task selection. This does NOT assert a scored stable grasp.
        if finger_contact and not supports and not external_contact and not on_floor:
            self.locked=True

    def request(self,*,ended):
        return dict(active=not ended and (self.finger_contact or self.locked),
                    locked=not ended and self.locked)
