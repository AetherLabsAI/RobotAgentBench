"""CPU-only fixed source-alias migration boundaries, with actual policy pins."""
import unittest
from robotagentbench.simulation import dedicated_delivery as delivery
from robotagentbench.simulation.specialized import camera_release


class SourceAliasPolicy(unittest.TestCase):
    def test_exact_old_alias_resolves_only_to_fixed_reviewed_source(self):
        for family in ('atomic_coffee', 'failure_recovery'):
            for target, row in camera_release.profile(family)['files'].items():
                self.assertEqual(delivery.source_alias_pin(target, row['before']['sha256'], family),
                                 row['after']['sha256'])
                for mismatch in ('f' * 64, row['after']['sha256']):
                    with self.assertRaisesRegex(ValueError, 'exact reviewed release predecessor'):
                        delivery.source_alias_pin(target, mismatch, family)

    def test_other_aliases_and_cross_family_names_are_not_rebound(self):
        for family in ('atomic_coffee', 'failure_recovery'):
            other = 'atomic_coffee' if family == 'failure_recovery' else 'failure_recovery'
            target, row = next(iter(camera_release.profile(other)['files'].items()))
            self.assertEqual(delivery.source_alias_pin(target, row['before']['sha256'], family),
                             row['before']['sha256'])
            self.assertEqual(delivery.source_alias_pin('/unlisted/scorer.py', 'a' * 64, family), 'a' * 64)


if __name__ == '__main__':
    unittest.main(verbosity=2)
