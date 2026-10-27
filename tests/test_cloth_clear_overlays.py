"""The small original rigid-object overlays are required by the scene recipe."""
import hashlib
import json
import re
import unittest

from robotagentbench.backends import sources


class ClothClearOverlayTests(unittest.TestCase):
    def test_all_original_report_entries_are_delivered_without_parameter_changes(self):
        root = sources.root('room')
        folder = root/'openend/tasks/cloth_clear/dynamic.f_hyv0g1'
        report = json.loads((folder/'report.json').read_text())
        pins = sources._manifest()['profiles']['room']['files']
        self.assertEqual(report['count'], 20)
        for row in report['assets']:
            name = row['entry'].removeprefix('/home/lingjun/robot_agent_arena/')
            raw = (root/name).read_bytes()
            with self.subTest(asset=row['id']):
                self.assertEqual(pins[name], dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
                text = raw.decode()
                self.assertEqual(float(re.search(r'float physics:mass = ([0-9.]+)',text)[1]), row['mass_kg'])
                self.assertIn('bool physics:kinematicEnabled = 0', text)
                self.assertIn('bool physics:rigidBodyEnabled = 1', text)
                self.assertEqual(re.findall(r'@([^@]+)@',text),
                                 [p for p in row['source_layers'] if p.endswith('/asset.usda')])


if __name__ == '__main__':
    unittest.main()
