"""Build selection, source identity and sandbox isolation; no native/model calls."""
import ast
import hashlib
import io
import json
import shutil
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from robotagentbench.model_examples.codex import sandbox_build as build
from robotagentbench.model_examples.codex import server_profile as profile
from robotagentbench.model_examples.codex.worker import selected_image

ID = 'sha256:' + 'a' * 64


class SandboxBuildTests(unittest.TestCase):
    def metadata(self):
        return {'Id': ID, 'Os': 'linux', 'Architecture': 'amd64',
                'Config': {'Labels': dict(build.LABELS), 'Volumes': None, 'OnBuild': None}}

    def test_local_tag_resolves_without_pull_or_container(self):
        with patch.object(build.subprocess, 'run', return_value=SimpleNamespace(stdout=json.dumps([self.metadata()]))) as run:
            self.assertEqual(build.inspect_image('rab-sandbox:local', 'unix:///var/run/docker.sock'), ID)
        self.assertEqual(run.call_args.args[0], ['docker', '--host', 'unix:///var/run/docker.sock', 'image', 'inspect', 'rab-sandbox:local'])

    def test_wrong_platform_labels_volumes_or_identity_rejected(self):
        for change in ('platform', 'label', 'volumes', 'hooks', 'identity'):
            row = self.metadata()
            if change == 'platform': row['Architecture'] = 'arm64'
            elif change == 'label': row['Config']['Labels'] = {}
            elif change == 'volumes': row['Config']['Volumes'] = {'/secret': {}}
            elif change == 'hooks': row['Config']['OnBuild'] = ['RUN true']
            else: row['Id'] = 'sha256:' + 'b' * 64
            with patch.object(build.subprocess, 'run', return_value=SimpleNamespace(stdout=json.dumps([row]))):
                with self.assertRaises(ValueError): build.inspect_image(ID, 'unix:///var/run/docker.sock')
        with patch.object(build.subprocess, 'run') as run:
            for ref, endpoint in [('--privileged', 'unix:///tmp/docker.sock'), ('tag', 'tcp://remote:2375')]:
                with self.assertRaises(ValueError): build.inspect_image(ref, endpoint)
            run.assert_not_called()

    def test_export_is_complete_exact_lock_and_no_clobber(self):
        with tempfile.TemporaryDirectory() as temp:
            root = build.export(Path(temp) / 'build')
            self.assertEqual(hashlib.sha256((root / 'requirements.lock').read_bytes()).hexdigest(), build.LOCK_SHA256)
            self.assertIn('-r /opt/robotagentbench/requirements.lock', (root / 'Dockerfile').read_text())
            with self.assertRaises(FileExistsError): build.export(root)

    def test_only_image_literal_changes_in_original_launcher(self):
        original = profile.payloads()
        deployed = profile.deployment_payloads(ID)
        key = profile.PREFIX + 'sandbox_launcher.py'
        self.assertEqual({k: v for k, v in original.items() if k != key}, {k: v for k, v in deployed.items() if k != key})
        def scrub(raw):
            tree = ast.parse(raw)
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'IMAGE' for t in node.targets):
                    node.value = ast.Constant('IMAGE')
            return ast.dump(tree)
        self.assertEqual(scrub(original[key]), scrub(deployed[key]))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); path = root / key; path.parent.mkdir(parents=True); path.write_bytes(deployed[key])
            pins = {key: hashlib.sha256(deployed[key]).hexdigest()}
            self.assertEqual(selected_image(root, pins), ID)
            path.write_bytes(deployed[key].replace(b"'--network', 'none'", b"'--network', 'host'"))
            pins[key] = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(ValueError): selected_image(root, pins)

    def test_configuration_uses_staged_id_not_current_tag(self):
        from robotagentbench.model_examples.codex import configure
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = profile.deployment_payloads(ID)
            for name in ('broker.py','sandbox.py','sandbox_boot.py','contract.py'):
                data[profile.PREFIX+name] = b'# test fixture\n'
            for name, raw in data.items():
                path = root/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(raw)
            pins = {name:hashlib.sha256(raw).hexdigest() for name,raw in data.items()}
            value = {'source_files':pins,'coding_policy_image':ID}
            for name in ('sdk','locks'):(root/name).mkdir()
            for name in ('launch','codex'):(root/name).write_text('{}')
            args = ['--runtime',str(root),'--sdk-site',str(root/'sdk'),'--launch',str(root/'launch'),
                    '--lock-root',str(root/'locks'),'--native-output',str(root/'out'),'--agent-output',str(root/'agent'),
                    '--destination',str(root/'cfg'),'--codex-executable',str(root/'codex'),
                    '--gpu','0','--docker-host','unix:///tmp/docker.sock','--cli-version','mock']
            with patch('robotagentbench.example.runtime',return_value=(root,root,value)), patch.object(build,'inspect_image',return_value=ID) as inspect, patch('sys.stdout',new_callable=io.StringIO):
                self.assertEqual(configure.main(args),0)
                inspect.assert_called_once_with(ID,'unix:///tmp/docker.sock')
            self.assertEqual(json.loads((root/'cfg/worker.json').read_text())['policy_image'],ID)

    def test_staging_binds_build_without_mutating_provider(self):
        from robotagentbench.model_examples.codex import stage_runtime
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve(); provider = root/'provider'; source = provider/'source'
            source.mkdir(parents=True)
            original = profile.payloads()
            for name, raw in original.items():
                path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
            config={'schema':'robotagentbench.local_installation.v1','profile':'room',
                    'runtime':str(provider),'source_root':str(source),'resources':{},
                    'source_files':{name:hashlib.sha256(raw).hexdigest()for name,raw in original.items()}}
            data=json.dumps(config).encode();(provider/'runtime.json').write_bytes(data)
            def copy(args,**kwargs):
                self.assertEqual(args[:3],['cp','-a','--reflink=auto'])
                shutil.copytree(args[3],args[4]);return SimpleNamespace(returncode=0)
            with patch.object(build,'inspect_image',return_value=ID),patch.object(stage_runtime.subprocess,'run',side_effect=copy):
                result=stage_runtime.stage(provider,root/'selected',policy_image='custom:local',docker_host='unix:///tmp/docker.sock')
            new=json.loads((Path(result['runtime'])/'runtime.json').read_text())
            self.assertEqual(new['coding_policy_image'],ID)
            self.assertEqual(selected_image(Path(new['source_root']),new['source_files']),ID)
            self.assertEqual((provider/'runtime.json').read_bytes(),data)
            for name,raw in original.items():self.assertEqual((source/name).read_bytes(),raw)


if __name__ == '__main__': unittest.main()
