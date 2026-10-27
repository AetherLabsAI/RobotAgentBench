"""CPU checks of prepared-launch routing only, not native execution."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import native_worker


class PreparedWorkerTests(unittest.TestCase):
    def setUp(self):
        private=Path(__file__).parent/'private';private.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=private);self.base=Path(self.temp.name).resolve()
        self.root=self.base/'source';self.root.mkdir()
        self.lock=self.base/'locks';self.lock.mkdir()
        (self.base/'runtime.json').write_text(json.dumps(dict(
            schema='robotagentbench.local_installation.v1',profile='room',
            runtime=str(self.base),source_root=str(self.root),source_files={'unused.py':'a'*64})))
        self.launch=self.root/'launch.json'
        self.launch.write_text(json.dumps(dict(schema='openend.short_task.launch.v1',task_name='prepare_coffee')))
        self.output=self.root/'new-run'

    def tearDown(self):self.temp.cleanup()

    def test_describe_does_not_execute(self):
        args=['--runtime',str(self.base),'--sdk-site',str(self.base/'not-installed-sdk'),
              '--launch',str(self.launch),'--output',str(self.output),
              '--lock-root',str(self.lock),'--gpu','0']
        text=io.StringIO()
        with patch.object(native_worker,'execute')as execute,contextlib.redirect_stdout(text):
            self.assertEqual(native_worker.main(args),0)
            execute.assert_not_called()
        self.assertFalse(json.loads(text.getvalue())['native_started'])
        self.assertFalse(self.output.exists())

    def test_fresh_output_required(self):
        self.output.mkdir()
        with self.assertRaises(ValueError):
            native_worker.configuration(self.base,self.launch,self.output,self.lock,0)

    def test_external_launch_cannot_select_unbound_namespace(self):
        launch=self.base/'outside.json';launch.write_bytes(self.launch.read_bytes())
        with self.assertRaises(ValueError):
            native_worker.configuration(self.base,launch,self.output,self.lock,0)

    def test_explicit_saved_input_root_preserves_launch_bytes(self):
        inputs=self.base/'saved';inputs.mkdir()
        launch=inputs/'launch.json';launch.write_bytes(self.launch.read_bytes())
        before=launch.read_bytes()
        native_worker.configuration(self.base,launch,self.output,self.lock,0,input_roots=[inputs])
        self.assertEqual(launch.read_bytes(),before)
        self.assertFalse(self.output.exists())

    def test_no_cloth_fallback(self):
        self.launch.write_text(json.dumps(dict(schema='robotagentbench.native_cloth.launch.v1')))
        with self.assertRaises(ValueError):
            native_worker.configuration(self.base,self.launch,self.output,self.lock,0)

    def public_binding(self):
        from robotagentbench.simulation.public_input import SCHEMA
        self.launch.write_text(json.dumps(dict(schema='openend.coding_room.launch.v1',task_name='openend')))
        public=self.root/'public_input.json';public.write_text('{"prompt":"Current request","context":"Current memory"}')
        binding=self.root/'public_binding.json'
        digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
        binding.write_text(json.dumps(dict(schema=SCHEMA,source=str(public),sha256=digest(public),launch_sha256=digest(self.launch))))
        return binding

    def test_public_input_survives_raw_fresh_interpreter_handoff(self):
        binding=self.public_binding();site=self.base/'sdk';site.mkdir()
        usd=site/'usd';usd.mkdir();physx=site/'physx';physx.mkdir()
        with patch('robotagentbench.example.sdk_paths',return_value=(site,usd,physx)),\
             patch.dict(native_worker.os.environ,{},clear=True),\
             patch.object(native_worker.os,'execve',side_effect=SystemExit(0))as handoff:
            with self.assertRaises(SystemExit):
                native_worker.execute(self.base,site,self.launch,self.output,self.lock,0,public_input_binding=binding)
        args=handoff.call_args.args[1]
        self.assertEqual(args[args.index('--public-input-binding')+1],str(binding))
        self.assertFalse(self.output.exists())

    def test_coding_public_input_is_validated_without_loading_sdk(self):
        from .codex import worker
        binding=self.public_binding()
        value=dict(schema='robotagentbench.codex_worker.v1',runtime=str(self.base),sdk_site=str(self.base/'absent-sdk'),
            launch=str(self.launch),output=str(self.output),lock_root=str(self.lock),gpu=0,coding_files=['unused.py'],
            policy_image=worker.POLICY_IMAGE,docker_host='unix:///run/docker.sock',public_input_binding=str(binding))
        path=self.base/'worker.json';path.write_text(json.dumps(value))
        with patch.object(worker,'selected_image',return_value=worker.POLICY_IMAGE),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(worker.main([str(path)]),0)
            changed=json.loads(binding.read_text());changed['launch_sha256']='0'*64
            binding.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'different saved launch'):worker.main([str(path)])
        self.assertFalse(self.output.exists())


if __name__=='__main__':unittest.main()
