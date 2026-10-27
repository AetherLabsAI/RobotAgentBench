"""The release must retain original helpers, rather than substitute sampling."""
import hashlib
import unittest
from robotagentbench.backends.sources import root,_manifest
from robotagentbench.simulation.composite_instance_bank.resource_contract import CONTRACT


class ShelfGenerationDeliveryTests(unittest.TestCase):
    def test_original_python_sources_and_licenses_are_bundled(self):
        selected=_manifest()['profiles']['room']['files']
        checked=0
        for name,record in CONTRACT['files'].items():
            relative='vendor/shelf_generation/'+name.removeprefix('generation_dependencies/ziche/')
            if relative not in selected:
                # Native extension payloads are delivered in the asset release.
                self.assertIn('.so',name)
                continue
            data=(root('room')/relative).read_bytes()
            self.assertEqual(len(data),record['bytes'])
            self.assertEqual(hashlib.sha256(data).hexdigest(),record['sha256'])
            self.assertEqual(selected[relative],record)
            checked+=1
        self.assertGreater(checked,200)


if __name__=='__main__':unittest.main()
