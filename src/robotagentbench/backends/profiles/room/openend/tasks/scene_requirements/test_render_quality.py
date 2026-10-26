import tempfile
import unittest
from pathlib import Path
import numpy as np
from render_quality import record_view, completion_fields


class QualityTests(unittest.TestCase):
    def test_bad_view_does_not_stop_next_view(self):
        with tempfile.TemporaryDirectory() as d:
            images, warnings = [], []
            good = np.indices((800,1280))[1].astype(np.uint8)
            good = np.repeat(good[:,:,None],3,axis=2)
            for name, raw in [('area_fridge',np.zeros((800,1280,4),dtype=np.uint8)),('area_microwave',good)]:
                item, warning = record_view(raw,dict(name=name,label=name),Path(d))
                if item: images.append(item)
                if warning: warnings.append(warning)
            receipt=completion_fields(images,warnings,2,2)
            self.assertEqual(receipt['status'],'completed_with_warnings')
            self.assertEqual(receipt['valid_images'],1)
            self.assertEqual(receipt['attempted_views'],2)
            self.assertEqual(images[0]['name'],'area_microwave')
            self.assertEqual(warnings[0]['pixel_std'],0)
            self.assertTrue((Path(d)/warnings[0]['diagnostic_file']).is_file())

    def test_empty_or_wrong_shape_warns(self):
        with tempfile.TemporaryDirectory() as d:
            for raw in [None, np.array([]), np.zeros((4,4,3)), np.zeros((800,1280,2))]:
                item,w=record_view(raw,dict(name='overview'),Path(d))
                self.assertIsNone(item)
                self.assertEqual(w['reason'],'图片尺寸异常')

    def test_nonfinite_warns(self):
        with tempfile.TemporaryDirectory() as d:
            item,w=record_view(np.full((800,1280,3),np.nan),dict(name='overview'),Path(d))
            self.assertIsNone(item)
            self.assertIsNone(w['pixel_std'])

    def test_zero_valid_remains_failure(self):
        with self.assertRaises(ValueError):completion_fields([],[],2,2)

    def test_healthy_and_missing_overview(self):
        self.assertEqual(completion_fields([{'name':'kitchen'}],[],1,1)['status'],'complete')


if __name__=='__main__':unittest.main()
