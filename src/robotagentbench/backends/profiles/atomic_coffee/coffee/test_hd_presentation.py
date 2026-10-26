"""Delivery-resolution tests only; no scoring or physical changes."""
import io
import unittest
from unittest.mock import patch, Mock
import numpy as np
from .atomic_replay import capture_resolution, StreamingVideo


class PresentationTests(unittest.TestCase):
    def test_default_unchanged_and_hd_same_aspect(self):
        self.assertEqual(capture_resolution(), (640, 480))
        self.assertEqual(capture_resolution(True), (1440, 1080))
        self.assertEqual(640 * 1080, 1440 * 480)
        with self.assertRaises(ValueError):
            capture_resolution('HD')

    def test_encoder_consumes_native_hd_without_scale_filter(self):
        process = Mock(stdin=io.BytesIO())
        process.wait.return_value = 0
        with patch('subprocess.Popen', return_value=process) as launch:
            encoder = StreamingVideo('presentation.mp4', capture_resolution(True))
            args = launch.call_args.args[0]
            self.assertEqual(args[args.index('-s') + 1], '1440x1080')
            self.assertEqual(args[args.index('-r') + 1], '20')
            self.assertNotIn('-vf', args)
            encoder.write(np.zeros((1080, 1440, 3), dtype=np.uint8))
            self.assertEqual(len(process.stdin.getvalue()), 1440 * 1080 * 3)
            with self.assertRaises(ValueError):
                encoder.write(np.zeros((480, 640, 3), dtype=np.uint8))
            encoder.close()

    def test_default_encoder_remains_original_quality_and_dimensions(self):
        with patch('subprocess.Popen') as launch:
            StreamingVideo('diagnostic.mp4')
            args = launch.call_args.args[0]
            self.assertEqual(args[args.index('-s') + 1], '640x480')
            self.assertNotIn('-crf', args)


if __name__ == '__main__':
    unittest.main()
