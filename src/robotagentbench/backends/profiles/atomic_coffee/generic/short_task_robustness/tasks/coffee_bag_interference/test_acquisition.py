import unittest
from .acquisition import acquisition_ready


class AcquisitionTests(unittest.TestCase):
    def facts(self,**change):
        return dict(dict(phase='approach',already_acquired=False,measured_contact=True,
                         opposed_contact=True,closure_values={'index':.135,'thumb':0.}),**change)

    def test_native_opposition_is_consumed_before_the_carry_timer(self):
        self.assertTrue(acquisition_ready(**self.facts()))
        self.assertFalse(acquisition_ready(**self.facts(already_acquired=True)))

    def test_command_or_single_side_contact_is_not_enough(self):
        self.assertFalse(acquisition_ready(**self.facts(measured_contact=False)))
        self.assertFalse(acquisition_ready(**self.facts(opposed_contact=False)))
        self.assertFalse(acquisition_ready(**self.facts(closure_values={'index':0.,'thumb':0.})))

    def test_no_attachment_during_release_or_retreat(self):
        for phase in ('waiting','release','retreat','done'):
            self.assertFalse(acquisition_ready(**self.facts(phase=phase)))


if __name__=='__main__':unittest.main()
