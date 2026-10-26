import hashlib,json,tempfile,unittest
from pathlib import Path
from render_reuse import reuse_images

class ReuseTests(unittest.TestCase):
    def test_same_scene_retained_with_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'renders_v3').mkdir();(p/'renders_v4').mkdir()
            (p/'room.usda').write_text('same scene')
            (p/'renders_v3/overview.png').write_bytes(b'previous validated image')
            sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
            receipt={'scene_sha256':sha(p/'room.usda'),'images':[{'name':'overview','file':'overview.png','sha256':sha(p/'renders_v3/overview.png')}]}
            (p/'renders_v3/manifest.json').write_text(json.dumps(receipt))
            views=[{'name':'overview'},{'name':'area_fridge'}]
            images,source=reuse_images(p,p/'renders_v4','renders_v3',['area_fridge'],views)
            self.assertEqual(len(images),1)
            self.assertEqual(source['retained_images'],1)
            self.assertEqual(sha(p/'renders_v4/overview.png'),images[0]['sha256'])
            with self.assertRaises(ValueError):reuse_images(p,p/'renders_v4','renders_v3',['not_real'],views)
            with self.assertRaises(ValueError):reuse_images(p,p/'renders_v4','renders_v3',['area_fridge'],views+[{'name':'area_microwave'}])
            (p/'renders_v3/overview.png').write_bytes(b'changed')
            with self.assertRaises(ValueError):reuse_images(p,p/'renders_v4','renders_v3',['area_fridge'],views)
            (p/'room.usda').write_text('changed scene')
            with self.assertRaises(ValueError):reuse_images(p,p/'renders_v4','renders_v3',['area_fridge'],views)

if __name__=='__main__':unittest.main()
