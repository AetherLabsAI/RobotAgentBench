import math
import unittest
import numpy as np
from .operating_station import with_fixed_heading,with_source_heading
from .catalog import definition


class OperatingStation(unittest.TestCase):
    def test_only_confirmed_cases_use_operating_stations(self):
        for name in ('sofa','table_waste_empty_grasp'):
            self.assertEqual(definition(name)['robot_station_policy'],'post_failure_operating_station.v1')
        for name in ('shelf_drink','coffee_bag','table_waste_tipped_cup'):
            self.assertNotIn('robot_station_policy',definition(name))

    def test_reuses_original_xy_and_keeps_original_heading(self):
        xy=np.array([1.,2.]);rows=[(.4,xy,0.),(.6,np.array([1.1,2.]),1.)]
        result=with_fixed_heading(rows,.3)
        self.assertEqual(result[0]['initial_base_xyyaw'],[1.,2.,.3])
        self.assertEqual(result[1]['initial_base_xyyaw'],[1.1,2.,.3])
        self.assertEqual(result[0]['candidate_index'],0)
        result[0]['initial_base_xyyaw'][0]=99
        np.testing.assert_array_equal(xy,[1.,2.])

    def test_bad_sector_candidate_refused_not_clipped(self):
        for yaw in (float('nan'),float('inf')):
            with self.assertRaises(ValueError):with_fixed_heading([],yaw)
        for row in ((0.,[1.,2.],0.),(-1.,[1.,2.],0.),(.5,[1.],0.),(.5,[float('nan'),2.],0.)):
            with self.assertRaises(ValueError):with_fixed_heading([row],0.)

    def test_existing_look_at_headings_are_reused_verbatim(self):
        rows=[(.4,np.array([1.,2.]),-.8),(.6,np.array([1.1,2.]),-.3)]
        values=with_source_heading(rows)
        self.assertEqual([r['initial_base_xyyaw'][2] for r in values],[-.8,-.3])
        self.assertEqual([r['candidate_index'] for r in values],[0,1])


if __name__=='__main__':unittest.main()
