"""Bounded file I/O only; retain complete original SHA/path admission."""
import hashlib
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from robotagentbench.simulation.cloth import config
from robotagentbench.simulation.workspace import NativeWorkspace


class ClothHashIO(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()
        self.inputs=self.root/'inputs';self.inputs.mkdir()
        self.output=self.root/'output';self.output.mkdir()
        self.locks=self.root/'locks';self.locks.mkdir()
        self.workspace=NativeWorkspace((self.inputs,),self.output,self.locks)
        self.pins={}
        for index in range(12):
            path=self.inputs/str(index);raw=('original bytes '+str(index)).encode()
            path.write_bytes(raw)
            self.pins[str(path)]=hashlib.sha256(raw).hexdigest()

    def test_all_bytes_rehashed_on_each_call_without_cache(self):
        calls=[];original=config.sha
        def recorded(path):
            calls.append(str(path));return original(path)
        with patch.object(config,'sha',side_effect=recorded):
            config._verify_pins(self.pins,self.workspace)
            config._verify_pins(self.pins,self.workspace)
        self.assertEqual(len(calls),24)
        self.assertEqual(set(calls),set(self.pins))
        first=Path(next(iter(self.pins)));first.write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError,'Pinned original input changed: '+str(first)):
            config._verify_pins(self.pins,self.workspace)

    def test_missing_and_symlink_inputs_still_rejected(self):
        path=Path(next(iter(self.pins)));raw=path.read_bytes();path.unlink()
        with self.assertRaisesRegex(ValueError,'Native input must be a real file'):
            config._verify_pins(self.pins,self.workspace)
        target=self.inputs/'same';target.write_bytes(raw);path.symlink_to(target)
        with self.assertRaisesRegex(ValueError,'Native input must be a real file'):
            config._verify_pins(self.pins,self.workspace)

    def test_bounded_four_readers_and_four_pending_items(self):
        original=config.sha;gate=threading.Event();entered=threading.Event();lock=threading.Lock()
        active=0;maximum=0;calls=[];errors=[]
        def delayed(path):
            nonlocal active,maximum
            with lock:
                active+=1;maximum=max(maximum,active);calls.append(str(path))
                if active==4:entered.set()
            self.assertTrue(gate.wait(5),'CPU test gate timed out')
            try:return original(path)
            finally:
                with lock:active-=1
        def run():
            try:config._verify_pins(self.pins,self.workspace)
            except BaseException as exc:errors.append(exc)
        with patch.object(config,'sha',side_effect=delayed):
            thread=threading.Thread(target=run);thread.start()
            try:
                self.assertTrue(entered.wait(5))
                self.assertEqual(len(calls),4)
            finally:
                gate.set();thread.join(5)
            self.assertFalse(thread.is_alive())
        self.assertEqual(errors,[])
        self.assertLessEqual(maximum,4)
        self.assertEqual(len(calls),len(self.pins))

    def test_earlier_hash_error_precedes_later_path_or_format_error(self):
        paths=list(self.pins);Path(paths[0]).write_bytes(b'changed first')
        for later in ('format','missing'):
            pins=dict(self.pins)
            if later=='format':pins[paths[1]]='not a digest'
            else:Path(paths[1]).unlink()
            with self.subTest(later=later),self.assertRaisesRegex(
                    ValueError,'Pinned original input changed: '+paths[0]):
                config._verify_pins(pins,self.workspace)

    def test_later_admission_error_retained_after_successful_prior_hashes(self):
        pins=dict(self.pins);pins[list(pins)[2]]='invalid'
        with self.assertRaisesRegex(ValueError,'Exact SHA256 required'):
            config._verify_pins(pins,self.workspace)

    def test_out_of_order_hash_failures_report_first_pin(self):
        paths=list(self.pins);second_done=threading.Event();original=config.sha
        def failures(path):
            if str(path)==paths[0]:
                self.assertTrue(second_done.wait(5));return '0'*64
            if str(path)==paths[1]:
                second_done.set();return '1'*64
            return original(path)
        with patch.object(config,'sha',side_effect=failures),self.assertRaisesRegex(
                ValueError,'Pinned original input changed: '+paths[0]):
            config._verify_pins(self.pins,self.workspace)


if __name__=='__main__':unittest.main()
