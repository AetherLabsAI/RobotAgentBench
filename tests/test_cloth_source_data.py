"""Original registry closure, no simulator, model, sampling, or score edits."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from robotagentbench.backends import sources
from robotagentbench.installation import configure
from robotagentbench.simulation.cloth import source_data


class ClothSourceData(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.pins = {}
        for relative, record in source_data.FILES.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(sources.root('cloth') / 'original' / relative, path)
            self.pins[str(path)] = record['sha256']
        self.original = {'files': {'code/original.py': {'sha256': 'a' * 64, 'bytes': 19}}}

    def extend(self):
        return source_data.extend_manifest(self.original, self.root, self.pins,
                                           upstream_sha256='b' * 64)

    def test_original_data_is_exact_and_embedded(self):
        packaged = sources._manifest()['profiles']['cloth']['files']
        for relative, row in source_data.FILES.items():
            self.assertEqual(packaged['original/' + relative], row)
            raw = (self.root / relative).read_bytes()
            self.assertEqual(len(raw), row['bytes'])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), row['sha256'])
            self.assertEqual(json.loads(raw).get('read_only', True), True)

    def test_additive_manifest_retains_original_and_provenance(self):
        before = deepcopy(self.original)
        derived = self.extend()
        self.assertEqual(self.original, before)
        self.assertEqual(derived['files']['code/original.py'], before['files']['code/original.py'])
        self.assertEqual(len(derived['files']), len(before['files']) + 2)
        meta = derived['static_dependency_delivery']
        self.assertEqual(meta['upstream_manifest_sha256'], 'b' * 64)
        self.assertFalse(meta['provenance']['original_source_edited'])
        self.assertFalse(meta['provenance']['physics_scoring_modified'])

    def test_missing_tampered_alias_and_unpinned_inputs_fail_closed(self):
        relative, record = next(iter(source_data.FILES.items()))
        path = self.root / relative
        raw = path.read_bytes()
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'canonical'):
            self.extend()
        path.write_bytes(raw + b' ')
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.extend()
        target = self.root / 'same_bytes.json'; target.write_bytes(raw)
        path.unlink(); path.symlink_to(target)
        with self.assertRaisesRegex(ValueError, 'canonical'):
            self.extend()
        path.unlink(); path.write_bytes(raw)
        self.pins.pop(str(path))
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.extend()

    def test_conflicting_source_manifest_or_missing_guard_entry_rejected(self):
        relative = next(iter(source_data.FILES))
        self.original['files'][relative] = {'sha256': '0' * 64, 'bytes': 0}
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            self.extend()
        with self.assertRaisesRegex(ValueError, 'differs from source manifest'):
            source_data.require_inputs(self.root, self.pins, {})

    def test_original_import_chain_reads_only_the_two_added_json_documents(self):
        # Execute the exact original station call in a fresh CPU interpreter.
        # Recording reads does not replace any task module or return value.
        original = sources.root('cloth') / 'original/code/agent_bench'
        script = '''import json, sys
from pathlib import Path
root = Path(sys.argv[1]); reads = []
read_text = Path.read_text
def recording(path, *args, **kwargs):
    if path.suffix == '.json': reads.append(str(path.relative_to(root)))
    return read_text(path, *args, **kwargs)
Path.read_text = recording
sys.path[:0] = [str(root / 'experiments/physics_fluid'), str(root)]
from tablecloth_scene_binding import station_xyyaw
poses = {task: station_xyyaw(task).tolist() for task in ('8.8', '8.9')}
assert not any(n.split('.')[0] in {'isaacsim', 'omni', 'carb', 'pxr'} for n in sys.modules)
print(json.dumps({'reads': reads, 'poses': poses}))
'''
        result = subprocess.run([sys.executable, '-B', '-c', script, str(original)],
                                text=True, capture_output=True, check=True)
        report = json.loads(result.stdout)
        self.assertEqual(set(report['reads']), {
            'real2sim/production/kitchen_task_components_v1.json',
            'real2sim/production/google_sheet_tasks_current.json',
        })
        self.assertEqual(report['poses']['8.8'], report['poses']['8.9'])
        self.assertEqual(len(report['poses']['8.8']), 3)

    def test_installation_pins_data_at_original_import_paths(self):
        selected = {'original/' + path: row for path, row in source_data.FILES.items()}
        profile = {'profiles': {'cloth': {'files': selected, 'compatibility_namespaces': []}}}
        assets = self.root / 'Assets'; assets.mkdir()
        index = {'schema': 'robotagentbench.asset_index.v1', 'version': 'test',
                 'objects': {}, 'locations': {}}
        raw = json.dumps(index).encode(); (assets / 'resource-index.json').write_bytes(raw)
        manifest = {'version': 'test', 'archives': {'empty.tar.gz': {'members': {
            'resource-index.json': {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}}}}}
        with patch('robotagentbench.installation.sources._manifest', return_value=profile):
            result = configure(manifest, assets, self.root / 'Runtime', 'cloth')
        record = json.loads((self.root / 'Runtime/runtime.json').read_text())
        for relative, row in selected.items():
            path = Path(result['source_root']) / relative
            self.assertTrue(path.is_file())
            self.assertEqual(record['source_files'][relative], row['sha256'])


if __name__ == '__main__':
    unittest.main()
