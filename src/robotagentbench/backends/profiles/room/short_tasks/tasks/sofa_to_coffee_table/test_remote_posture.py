import unittest
from .initial_station import sample_candidates,sample_reset_candidates,posture_policy

class RemotePostureTests(unittest.TestCase):
    def rows(self,seed=42,count=100):
        return list(sample_reset_candidates(seed,[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dict(asset='remote_tv',source='sofa'),count))

    def test_ranges_and_spread(self):
        rows=self.rows(count=2000)
        for key,(lo,hi) in posture_policy()['ranges'].items():
            vals=[r['targets_rad'][key] for r in rows]
            self.assertTrue(all(lo<=v<=hi for v in vals))
            if hi>lo:
                self.assertLess(min(vals),lo+.01*(hi-lo))
                self.assertGreater(max(vals),hi-.01*(hi-lo))

    def test_seeded_not_episode_replay(self):
        self.assertEqual(self.rows(),self.rows())
        self.assertNotEqual(self.rows(),self.rows(43))
        evidence=posture_policy()['evidence']
        for row in self.rows(count=5):
            self.assertFalse(any(all(row['targets_rad'][k]==e['joints'][k] for k in row['targets_rad']) for e in evidence))

    def test_station_preserved(self):
        expected=sample_candidates(42,[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dict(asset='remote_tv',source='sofa'),count=100)
        for old,new in zip(expected,self.rows()):
            self.assertEqual(old,{k:new[k] for k in old})

    def test_no_generic_posture(self):
        for r in self.rows():
            q=r['targets_rad']
            self.assertLess(q['idx01_body_joint1'],.020001)
            self.assertAlmostEqual(q['idx02_body_joint2'],.523599,places=5)
            self.assertEqual(r['posture_source'],'remote_task119_first_state_continuous_ranges')

if __name__=='__main__':unittest.main()
