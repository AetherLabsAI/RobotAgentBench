"""CPU-only native camera routing; no model loading, simulation or CUDA."""
import ast
import base64
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from robotagentbench.interfaces.action_contract import ORDER
from robotagentbench.model_examples.internvla import server_policy as policy
from robotagentbench.model_examples.internvla.__main__ import main
from robotagentbench.model_examples.internvla.operator import model_service,configuration
from robotagentbench.model_examples.internvla import test_configure as fixtures
from robotagentbench.model_examples.internvla.configure import build
from robotagentbench.model_examples.native_receipts import ATOMIC,CLOTH,RECOVERY


def binding(family,task):
    return dict(family=family,task_id=task)


class BeforeTransform(Exception):
    pass


class Tensor:
    """Only the CPU tensor operations before the unchanged image transform."""
    def __init__(self,value):self.value=np.asarray(value)
    def permute(self,*axes):return Tensor(self.value.transpose(axes))
    def float(self):return Tensor(self.value.astype(np.float32))
    def __truediv__(self,value):return Tensor(self.value/value)


class CPUOnlyTorch:
    from_numpy=staticmethod(Tensor)
    zeros=staticmethod(lambda *shape:Tensor(np.zeros(shape)))


def request(shapes):
    images={}
    for camera,shape in shapes:
        image=Image.new('RGB',(shape[1],shape[0]),(0,0,0))
        # Distinct corners reveal any crop or stretch before the old transform.
        image.putpixel((0,0),(255,0,0));image.putpixel((shape[1]-1,shape[0]-1),(0,255,0))
        output=io.BytesIO();image.save(output,format='PNG')
        images[camera]=base64.b64encode(output.getvalue()).decode()
    return dict(op='infer',action_order=ORDER,training_task_index='public:test',
        observation=dict(instruction='Pick up the target',joints=dict.fromkeys(ORDER[:20],0.),
            base_state=dict(position_world_m=[0.,0.,0.],yaw_world_unwrapped_rad=0.),images=images))


class CameraProfiles(unittest.TestCase):
    def test_exact_owner_profiles_and_ordinary_default(self):
        self.assertEqual(policy.camera_profile(),policy.STANDARD_CAMERA_PROFILE)
        self.assertEqual(policy.camera_profile(binding('room','prepare_dinner')),policy.STANDARD_CAMERA_PROFILE)
        for task in ATOMIC:
            self.assertEqual(policy.camera_profile(binding('atomic_coffee',task)),policy.STANDARD_CAMERA_PROFILE)
        for task in RECOVERY:
            self.assertEqual(policy.camera_profile(binding('failure_recovery',task)),policy.STANDARD_CAMERA_PROFILE)
        for task in CLOTH:
            self.assertEqual(policy.camera_profile(binding('cloth',task)),policy.STANDARD_CAMERA_PROFILE)

    def test_unknown_cross_family_or_mismatched_binding_rejected(self):
        for value in ([],{},binding('failure_recovery','unknown'),binding('cloth','prepare_dinner'),
                binding('atomic_coffee','failure_recovery_sofa'),binding('room','coffee_bag_interference'),
                dict(binding('atomic_coffee','coffee_bag_interference'),camera_profile='anything')):
            with self.subTest(value=value),self.assertRaises(ValueError):policy.camera_profile(value)
        with self.assertRaises(ValueError):
            policy.camera_profile(binding('atomic_coffee','coffee_bag_interference'),task_id='prepare_dinner')

    def test_service_profile_comes_only_from_operator_binding(self):
        task='coffee_bag_interference';receipt=binding('atomic_coffee',task)
        row=dict(native_receipt=receipt,camera_profile='untrusted extra is not forwarded')
        result=model_service({'model':'fixture'},{'public:'+task:'Pick up'},task,row)
        self.assertEqual(set(result),{'model','prompts','native_receipt'})
        self.assertEqual(result['native_receipt'],receipt)
        self.assertIsNot(result['native_receipt'],receipt)
        self.assertEqual(set(model_service({}, {},'prepare_dinner',{})),{'model','prompts'})

    def test_operator_configuration_validates_family_before_start(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory).resolve()
            args=fixtures.ConfigurationSeam().fixture(folder)
            value,_=build(**args);task=args['task'];row=value['tasks'].pop(task)
            task='coffee_bag_interference';value['tasks'][task]=row
            row['native_receipt']=binding('atomic_coffee',task)
            path=folder/'operator.json';path.write_text(json.dumps(value))
            self.assertEqual(configuration(path,task)[1]['native_receipt'],row['native_receipt'])
            row['native_receipt']['family']='cloth';path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):configuration(path,task)

    def test_tested_service_cli_forwards_binding_not_custom_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'service.json'
            for receipt in (None,binding('atomic_coffee','coffee_bag_interference'),
                    binding('failure_recovery','failure_recovery_sofa')):
                value=dict(model={},prompts={})
                if receipt is not None:value['native_receipt']=receipt
                path.write_text(json.dumps(value))
                with patch.object(policy,'serve_stdio') as serve:
                    self.assertEqual(main(['tested-service','--service-config',str(path)]),0)
                    serve.assert_called_once_with({}, {},native_receipt=receipt)
            value['camera_profile']='native_three640x480';path.write_text(json.dumps(value))
            with patch.object(policy,'serve_stdio') as serve,self.assertRaises(ValueError):
                main(['tested-service','--service-config',str(path)])
            serve.assert_not_called()

    def test_owner_public_prompt_must_match_before_loading_model(self):
        with patch.object(policy,'model_config',return_value=({},{})):
            with self.assertRaisesRegex(ValueError,'public task prompt'):
                policy.TestedPolicy({}, {'public:other':'Pick up'},
                    native_receipt=binding('atomic_coffee','coffee_bag_interference'))

    def probe(self,profile,req):
        tested=policy.TestedPolicy.__new__(policy.TestedPolicy)
        tested.camera_profile=profile;tested.np=np;tested.torch=CPUOnlyTorch
        tested.order=ORDER;tested.prompts={'public:test':'Pick up the target'}
        captured=[]
        def transform(sample):
            captured.append(sample)
            raise BeforeTransform()
        tested.transform=transform
        with self.assertRaises(BeforeTransform):tested.infer(req)
        return captured[0]

    def test_full_native_frames_reach_transform_without_extra_crop_or_resize(self):
        for profile,shapes in policy.CAMERA_SHAPES.items():
            sample=self.probe(profile,request(shapes))
            for camera,(height,width,_) in shapes:
                tensor=sample['observation.images.'+camera].value
                self.assertEqual(tensor.shape,(3,height,width))
                np.testing.assert_array_equal(tensor[:,0,0],[1.,0.,0.])
                np.testing.assert_array_equal(tensor[:,-1,-1],[0.,1.,0.])
            self.assertEqual(sample['observation.state'].value.shape,(23,))
            self.assertEqual(sample['action'].value.shape,(50,22))

    def test_640_or_mixed_dimensions_rejected_even_for_dedicated_owners(self):
        standard=policy.CAMERA_SHAPES[policy.STANDARD_CAMERA_PROFILE]
        legacy=tuple((name,(480,640,3)) for name in ('head','left','right'))
        receipts=[None,binding('room','prepare_dinner')]
        receipts += [binding(family,task) for family,tasks in
            (('atomic_coffee',ATOMIC),('failure_recovery',RECOVERY),('cloth',CLOTH)) for task in tasks]
        for receipt in receipts:
            profile=policy.camera_profile(receipt)
            for shapes in (legacy,standard[:2]+legacy[2:],legacy[:1]+standard[1:]):
                req=request(shapes)
                req['camera_profile']=req['observation']['camera_profile']='native_three640x480'
                with self.subTest(receipt=receipt,shapes=shapes),self.assertRaisesRegex(ValueError,'requires'):
                    self.probe(profile,req)

    def test_original_resize224_is_only_resize_and_decode_is_unchanged(self):
        tree=ast.parse(Path(policy.__file__).read_text())
        resize=[node for node in ast.walk(tree) if isinstance(node,ast.Call)
            and isinstance(node.func,ast.Name) and node.func.id=='ResizeImagesWithPadFn']
        self.assertEqual(len(resize),1)
        dimensions={item.arg:item.value.value for item in resize[0].keywords if item.arg in ('height','width')}
        self.assertEqual(dimensions,{'height':224,'width':224})
        self.assertFalse(any(isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
            and node.func.attr in ('resize','crop','thumbnail') for node in ast.walk(tree)))
        for profile in policy.CAMERA_SHAPES:
            self.assertEqual(len(policy.CAMERA_SHAPES[profile]),3)


if __name__=='__main__':unittest.main()
