import json
from pathlib import Path
import tempfile
import unittest
from robotagentbench.model_examples.internvla.operator import native_cleanup


class NativeReceiptTests(unittest.TestCase):
    def test_generic_original_cleanup_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'native_report.json'
            path.write_text(json.dumps(dict(status='completed',cleanup_errors=[],
                source_layers_unchanged=True,native_shutdown_complete=True,task_result={'score':0})))
            accepted,result=native_cleanup({},path,{'closed_ack':True,'exit_code':0},'fresh')
            self.assertTrue(accepted)
            self.assertEqual(result,{'score':0})

    def test_atomic_requires_same_episode_and_actual_exit(self):
        task='coffee_bag_interference'
        row={'native_receipt':{'family':'atomic_coffee','task_id':task}}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory).resolve()/'native_report.json'
            path.write_text(json.dumps(dict(status='completed',cleanup_errors=[],task_name=task,
                native_shutdown_complete=True,result=dict(schema='short_tasks.atomic_coffee.result.v1',
                    task_name=task,episode_id='fresh',status='completed',evaluation_valid=True,success=False,
                    source_evaluation=dict(task_id='1-2',episode_id='fresh',valid=True,finalized=True)))))
            self.assertTrue(native_cleanup(row,path,{'closed_ack':True,'exit_code':0},'fresh')[0])
            self.assertFalse(native_cleanup(row,path,{'closed_ack':True,'exit_code':1},'fresh')[0])
            self.assertFalse(native_cleanup(row,path,{'closed_ack':True,'exit_code':0},'previous')[0])


if __name__=='__main__':unittest.main()
