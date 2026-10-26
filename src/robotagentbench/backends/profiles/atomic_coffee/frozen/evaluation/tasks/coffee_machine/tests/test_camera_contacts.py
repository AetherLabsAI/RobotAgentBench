import unittest
from ..camera_contacts import CAMERA_ROOTS, CAMERA_COLLIDERS, camera_contacts, is_camera

class CameraContactTests(unittest.TestCase):
    def test_both_sides(self):
        self.assertEqual(len(CAMERA_COLLIDERS), 4)
        for camera in CAMERA_COLLIDERS:
            rows = camera_contacts([((camera, '/World/Background/Shell/CounterTop_17'), [((0,0,1), .4)])])
            self.assertEqual(rows[0]['impulse'], .4)
    def test_no_speculative_force(self):
        self.assertEqual(camera_contacts([((CAMERA_COLLIDERS[0], '/table'), [((0,0,1), 0)])]), [])
    def test_other_contacts_not_camera(self):
        self.assertEqual(camera_contacts([(('/World/Robot/base_link', '/floor'), [((0,0,1), 5)])]), [])
        self.assertFalse(is_camera(CAMERA_ROOTS[0] + 'Wrong/mesh'))
    def test_order_independent(self):
        self.assertEqual(len(camera_contacts([(('/table', CAMERA_COLLIDERS[3]), [((0,0,1), 2)])])), 1)
