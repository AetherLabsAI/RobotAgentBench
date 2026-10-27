"""Focused CPU-only tests for guided client/container wiring, not inference."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from robotagentbench.model_examples import docker_workflow as flow
from robotagentbench.model_examples.internvla import test_configure as fixtures

IMAGE = 'sha256:'+'a'*64
POLICY = 'sha256:'+'b'*64


class GuidedWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.fixture = fixtures.ConfigurationSeam().fixture(self.root)
        self.base = self.fixture['runtime']
        self.source = self.base/'source'
        self.weights = self.root/'model'
        stats = json.loads((self.weights/'stats.json').read_text())
        (self.weights/'stats.json').write_text(json.dumps({'G1_omnipicker': stats['fixture']}))
        (self.weights/'vlm').mkdir()
        self.receipts = self.root/'Restored'
        self.receipts.mkdir()
        self.selected = self.receipts/'selection.json'
        self.row = dict(schema='robotagentbench.restored_compact_initial_state.v1',
            task_id='prepare_coffee', runtime=str(self.base), source_root=str(self.source),
            launch=str(self.fixture['launch']), launch_sha256=flow.sha(self.fixture['launch']),
            public_input_binding=None, readonly_resource_aliases=[])
        flow.write(self.selected, self.row)
        self.dest = self.root/'config'

    def args(self, agent='internvla'):
        return argparse.Namespace(selection=self.selected, name='one', gpu=0,
            lock_root=self.root/'locks', destination=self.dest, agent=agent,
            model_python=Path(sys.executable), model_dir=self.weights, model_gpu=1,
            max_calls=1, image='example:native', docker_host='unix:///var/run/docker.sock',
            codex_executable=Path(sys.executable))

    def configure(self, args=None):
        with patch.object(flow, 'image_id', return_value=IMAGE), \
                patch.object(flow.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)), \
                patch.object(flow.deployment, 'validate', return_value={}):
            return flow.configure(args or self.args())

    def test_internvla_uses_download_root_and_preserves_venv_python(self):
        venv = self.root/'venv/bin'
        venv.mkdir(parents=True)
        interpreter = venv/'python'
        interpreter.symlink_to(sys.executable)
        args = self.args()
        args.model_python = interpreter
        result = self.configure(args)
        config = json.loads((self.dest/'operator.json').read_text())
        self.assertEqual(config['model']['python'], str(interpreter))
        self.assertEqual(config['model']['checkpoint'], str(self.weights))
        self.assertEqual(config['model']['vlm'], str(self.weights/'vlm'))
        self.assertEqual(config['tasks']['prepare_coffee']['worker_argv'][-1], str(self.dest/'run.json'))
        self.assertFalse(result['model_called'])
        self.assertFalse(Path(result['output']).exists())
        flow.checked(self.dest/'run.json')

    def test_top_level_public_binding_is_forwarded_and_hash_checked(self):
        launch = self.fixture['launch']
        launch.write_text(json.dumps(dict(schema='openend.coding_room.launch.v1', task_name='openend')))
        public = self.source/'public.json'
        flow.write(public, dict(instruction='Host guests', context='Public memory'))
        binding = self.source/'binding.json'
        flow.write(binding, dict(schema='robotagentbench.public_input_binding.v1',
            source=str(public), sha256=flow.sha(public), launch_sha256=flow.sha(launch)))
        self.row.update(launch_sha256=flow.sha(launch), public_input_binding=str(binding))
        self.selected.write_text(json.dumps(self.row))
        self.configure()
        _, value = flow.checked(self.dest/'run.json')
        self.assertEqual(value['worker']['public_input_binding'], str(binding))
        expected=self.source/'short_tasks/agent_probe_v1/cap_v1/native_tools/openend_control_adapter_v1/model_runs/one'
        self.assertEqual(Path(value['worker']['output']),expected)
        public.write_text('{}')
        with self.assertRaises(ValueError):
            flow.checked(self.dest/'run.json')

    def test_openend_rejects_short_task_output_namespace(self):
        launch=self.fixture['launch']
        launch.write_text(json.dumps(dict(schema='openend.coding_room.launch.v1',task_name='openend')))
        self.row['launch_sha256']=flow.sha(launch)
        self.selected.write_text(json.dumps(self.row))
        self.configure()
        path,value=flow.checked(self.dest/'run.json')
        value['worker']['output']=str(self.source/'short_tasks/agent_probe_v1/cap_v1/model_runs/one')
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'task-owned workspace'):
            flow.checked(path)

    def test_worker_has_same_path_runtime_selection_mounts_and_no_tty(self):
        self.configure()
        path, value = flow.checked(self.dest/'run.json')
        with patch.object(flow.deployment, 'validate', return_value={'mounts': (), 'icd': self.selected, 'sha256': 'x'}), \
                patch.dict(os.environ, OMNI_KIT_ACCEPT_EULA='YES'):
            cmd = flow.worker_command(path, value)
        self.assertIn('--interactive', cmd)
        self.assertNotIn('--tty', cmd)
        self.assertIn('type=bind,src='+str(self.base)+',dst='+str(self.base), cmd)
        self.assertIn('type=bind,src='+str(self.receipts)+',dst='+str(self.receipts)+',readonly', cmd)
        self.assertNotIn('/var/run/docker.sock', cmd)  # raw policy has no host Docker socket
        self.assertEqual(cmd[-2:], ['enter', str(path)])

    def test_dispatch_is_single_use_even_after_worker_failure(self):
        self.configure()
        with patch.object(flow, 'worker_command', return_value=['docker','run','--name','owned',IMAGE]), \
                patch.object(flow.subprocess, 'run', return_value=subprocess.CompletedProcess([], 7)) as dispatch:
            self.assertEqual(flow.worker(self.dest/'run.json'), 7)
            with self.assertRaises(FileExistsError):
                flow.worker(self.dest/'run.json')
            self.assertEqual(dispatch.call_count, 1)

    def test_inspect_does_not_dispatch_or_call_model(self):
        self.configure()
        with patch.object(flow.subprocess, 'run') as run, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(flow.main(['run',str(self.dest/'run.json')]), 0)
        run.assert_not_called()

    def test_container_entry_rechecks_baseline_then_calls_original_worker(self):
        self.configure()
        cache = self.root/'cache'
        cache.mkdir()
        flow.write(cache/'image_baseline.json', {'original': 'baseline'})
        with patch.object(flow.deployment, 'validate', return_value={'root':cache}), \
                patch.object(flow.deployment, 'verify_image_baseline', return_value={'original':'baseline'}), \
                patch('robotagentbench.model_examples.native_worker.execute', return_value=0) as execute:
            self.assertEqual(flow.enter(self.dest/'run.json'), 0)
        self.assertEqual(execute.call_args.args[1], '/opt/robotagentbench/native')
        self.assertEqual(execute.call_args.args[2], self.fixture['launch'])
        self.assertEqual(execute.call_args.args[4], Path('/locks'))
        self.assertEqual(execute.call_args.kwargs['input_roots'], (self.source,))
        with patch.object(flow.deployment, 'validate', return_value={'root':cache}), \
                patch.object(flow.deployment, 'verify_image_baseline', return_value={'changed':True}), \
                patch('robotagentbench.model_examples.native_worker.execute') as execute:
            with self.assertRaises(ValueError):
                flow.enter(self.dest/'run.json')
            execute.assert_not_called()

    def test_launch_change_and_invalid_gpu_rejected_before_prepare(self):
        args = self.args()
        args.model_gpu = 0
        with patch.object(flow.subprocess, 'run') as run:
            with self.assertRaises(ValueError):
                flow.configure(args)
            self.fixture['launch'].write_text('{}')
            with self.assertRaises(ValueError):
                flow.configure(self.args())
        run.assert_not_called()
        self.assertFalse(self.dest.exists())

    def test_codex_config_is_staged_and_uses_original_client(self):
        from robotagentbench.model_examples.codex.server_profile import deployment_payloads
        metadata = json.loads((self.base/'runtime.json').read_text())
        metadata.update(coding_policy_image=POLICY, resources={})
        for relative, data in deployment_payloads(POLICY).items():
            item = self.source/relative
            item.parent.mkdir(parents=True, exist_ok=True)
            item.write_bytes(data)
            metadata['source_files'][relative] = flow.sha(item)
        (self.base/'runtime.json').write_text(json.dumps(metadata))
        with patch('robotagentbench.model_examples.codex.sandbox_build.inspect_image', return_value=POLICY), \
                patch('robotagentbench.model_examples.codex.namespace.docker_socket_group', return_value=77), \
                patch.object(flow.shutil, 'which', return_value=sys.executable), \
                patch.object(flow.subprocess, 'check_output', return_value='codex-cli fixture\n'):
            self.configure(self.args('codex'))
        operator = json.loads((self.dest/'operator.json').read_text())
        self.assertEqual(operator['model'], 'gpt-6-astra')
        self.assertEqual(operator['expected_cli_version'], 'codex-cli fixture')
        native = json.loads((self.dest/'native.json').read_text())
        self.assertEqual(native['sdk_site'], '/opt/robotagentbench/native')
        self.assertEqual(native['lock_root'], '/locks')
        self.assertEqual(native['policy_image'], POLICY)
        self.assertEqual(native['input_roots'], [str(self.source)])
        path, value = flow.checked(self.dest/'run.json')
        # Validate daemon access is mounted only in the trusted native owner.
        socket = self.root/'docker.sock'
        socket.touch()
        value['worker']['docker_host'] = 'unix://'+str(socket)
        with patch.object(flow.deployment, 'validate', return_value={'mounts': (), 'icd': self.selected, 'sha256':'x'}), \
                patch.dict(os.environ, OMNI_KIT_ACCEPT_EULA='YES'), \
                patch('robotagentbench.model_examples.codex.namespace.docker_socket_group', return_value=77):
            cmd = flow.worker_command(path, value)
        self.assertIn('type=bind,src='+str(socket)+',dst='+str(socket), cmd)
        self.assertIn('--group-add',cmd)
        self.assertNotIn('--privileged',cmd)

    def test_remote_codex_preparation_keeps_cli_on_client(self):
        from robotagentbench.model_examples.codex.server_profile import deployment_payloads
        metadata = json.loads((self.base/'runtime.json').read_text())
        metadata.update(coding_policy_image=POLICY, resources={})
        for relative, data in deployment_payloads(POLICY).items():
            item = self.source/relative
            item.parent.mkdir(parents=True, exist_ok=True)
            item.write_bytes(data)
            metadata['source_files'][relative] = flow.sha(item)
        (self.base/'runtime.json').write_text(json.dumps(metadata))
        args = self.args('codex')
        args.codex_executable = Path('/client-only/bin/codex')
        args.codex_client_version = 'codex-cli fixture'
        with patch('robotagentbench.model_examples.codex.sandbox_build.inspect_image', return_value=POLICY), \
                patch('robotagentbench.model_examples.codex.namespace.docker_socket_group', return_value=77), \
                patch.object(flow.shutil, 'which', return_value=sys.executable), \
                patch.object(flow.subprocess, 'check_output') as cli:
            self.configure(args)
        cli.assert_not_called()
        operator = json.loads((self.dest/'operator.json').read_text())
        self.assertEqual(operator['codex_executable'], '/client-only/bin/codex')
        self.assertEqual(operator['expected_cli_version'], 'codex-cli fixture')

    def test_restoration_container_is_cpu_only_and_uses_selected_episode(self):
        bank = self.root/'Bank'
        bank.mkdir()
        release = bank/'release.json'
        release.write_text('{}')
        args = argparse.Namespace(runtime=self.base, release=release, task='prepare_coffee',
            episode=3, output=self.receipts/'new', image='example:native', docker_host='unix:///var/run/docker.sock')
        with patch('robotagentbench.simulation.saved_instance_release.restore.describe', return_value={'mode':'ordinary'}), \
                patch.object(flow, 'image_id', return_value=IMAGE), \
                patch.object(flow.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)) as run:
            result = flow.restore(args)
        cmd = run.call_args.args[0]
        self.assertNotIn('--gpus', cmd)
        self.assertIn('NVIDIA_VISIBLE_DEVICES=void', cmd)
        self.assertEqual(cmd[cmd.index('--episode')+1], '3')
        self.assertFalse(result['model_called'])
        self.assertIn('type=bind,src='+str(bank)+',dst='+str(bank)+',readonly',cmd)


if __name__ == '__main__':
    unittest.main()
