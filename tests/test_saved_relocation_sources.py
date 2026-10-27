"""CPU-only frozen geometry-key rebinding; no scene/physics/model imports."""
from copy import deepcopy
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from robotagentbench.simulation.saved_instance_release import relocation_sources as sources


class FrozenRelocationSourceTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.runtime = Path(temporary.name).resolve() / 'Runtime'
        self.source = self.runtime / 'source'; self.source.mkdir(parents=True)
        self.metadata = dict(schema='robotagentbench.local_installation.v1', profile='room',
                             runtime=str(self.runtime), source_root=str(self.source), source_files={})
        records = []
        for index, (relative, _) in enumerate(sources._GEOMETRY):
            data = ('original geometry fixture ' + str(index)).encode()
            pin = hashlib.sha256(data).hexdigest()
            path = self.source / relative; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            self.metadata['source_files'][relative] = pin
            records.append((relative, pin))
        item = patch.object(sources, '_GEOMETRY', tuple(records)); item.start(); self.addCleanup(item.stop)
        self.original = {str(sources._ORIGINAL_SOURCE / relative): pin for relative, pin in records}
        row = dict(schema='arena.human_relocation_reservation.v1',
                   producer=dict(task_name='stock_wooden_tray_interference', version='reserve_first.v1'),
                   geometry_source_hashes=deepcopy(self.original), hand_asset_sha256=records[0][1],
                   seed=19, target_path='/World/Target', destination_xy=[0.2, 0.1],
                   waypoints=[[0, 0, 0], [1, 0, 0]],
                   source_product_range=dict(bounds=[-0.3, 0.3]))
        root = deepcopy(row)
        root['bilateral_routes'] = dict(schema='shelf_interference.bilateral_routes.v1',
            coordinate_frame='shelf_local_m', selection_at='native_proximity', runtime_replanning=False,
            routes=[dict(side=side, reservation=deepcopy(row)) for side in ('left', 'right')])
        self.manifest = dict(environment_reservations=[root], source_hashes={'untouched': 'unchanged'})

    def restore(self):
        return sources.restore_geometry_sources(self.manifest, source=self.source,
                                                runtime=self.runtime, metadata=self.metadata)

    def rows(self, manifest):
        root = manifest['environment_reservations'][0]
        return [root, *[r['reservation'] for r in root['bilateral_routes']['routes']]]

    def test_only_three_geometry_map_keys_change_and_input_is_immutable(self):
        before = deepcopy(self.manifest)
        result, evidence = self.restore()
        self.assertEqual(self.manifest, before)
        self.assertEqual(evidence['reservations_rebound'], 3)
        self.assertFalse(evidence['reservation_geometry_modified'])
        self.assertFalse(evidence['physics_scoring_modified'])
        for row in self.rows(result):
            # Same actual path lookup/hash equality used by native validate_frozen.
            self.assertEqual(set(row['geometry_source_hashes']),
                             {str(self.source / relative) for relative, _ in sources._GEOMETRY})
            for relative, pin in sources._GEOMETRY:
                path = self.source / relative
                self.assertEqual(row['geometry_source_hashes'][str(path)],
                                 hashlib.sha256(path.read_bytes()).hexdigest())
            row['geometry_source_hashes'] = deepcopy(self.original)
        self.assertEqual(result, before)

    def test_original_single_route_and_relocated_runtime_metadata(self):
        self.manifest['environment_reservations'][0].pop('bilateral_routes')
        self.metadata.update(runtime='/previous/Runtime', source_root='/previous/Runtime/source')
        result, evidence = self.restore()
        self.assertEqual(evidence['reservations_rebound'], 1)

    def test_each_route_rejects_changed_missing_or_extra_original_identity(self):
        before = deepcopy(self.manifest)
        for index in range(3):
            for mutation in ('pin', 'missing', 'extra', 'path', 'collision', 'hand'):
                with self.subTest(index=index, mutation=mutation):
                    self.manifest = deepcopy(before)
                    row = self.rows(self.manifest)[index]; mapping = row['geometry_source_hashes']
                    key = next(iter(mapping))
                    if mutation == 'pin': mapping[key] = '0' * 64
                    elif mutation == 'missing': mapping.pop(key)
                    elif mutation == 'extra': mapping['/unapproved/source.json'] = mapping[key]
                    elif mutation == 'path': mapping['/unknown/' + key.rsplit('/', 1)[1]] = mapping.pop(key)
                    elif mutation == 'collision': mapping[str(self.source / sources._GEOMETRY[0][0])] = mapping[key]
                    else: row['hand_asset_sha256'] = '0' * 64
                    with self.assertRaisesRegex(ValueError, 'identities differ'): self.restore()

    def test_source_pin_and_bytes_are_independently_checked(self):
        relative, pin = sources._GEOMETRY[0]
        self.metadata['source_files'][relative] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'source pin differs'): self.restore()
        self.metadata['source_files'][relative] = pin
        (self.source / relative).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'bytes differ'): self.restore()

    def test_missing_and_symlinked_geometry_are_not_repaired(self):
        path = self.source / sources._GEOMETRY[0][0]
        saved = path.with_name('same-bytes.json'); path.rename(saved)
        with self.assertRaises(FileNotFoundError): self.restore()
        path.symlink_to(saved)
        with self.assertRaisesRegex(ValueError, 'Canonical installed'): self.restore()

    def test_unknown_task_or_partial_nested_bilateral_routes_rejected(self):
        before = deepcopy(self.manifest)
        for mutation in ('task', 'partial', 'nested', 'runtime_replanning', 'duplicate_root'):
            with self.subTest(mutation=mutation):
                self.manifest = deepcopy(before)
                root = self.manifest['environment_reservations'][0]
                if mutation == 'task': root['producer']['task_name'] = 'unrelated_task'
                elif mutation == 'partial': root['bilateral_routes']['routes'].pop()
                elif mutation == 'nested': self.rows(self.manifest)[1]['bilateral_routes'] = {}
                elif mutation == 'runtime_replanning': root['bilateral_routes']['runtime_replanning'] = True
                else: self.manifest['environment_reservations'].append(deepcopy(root))
                with self.assertRaises(ValueError): self.restore()

    def test_runtime_source_binding_cannot_redirect(self):
        self.metadata['source_root'] = str(self.runtime / 'elsewhere')
        with self.assertRaisesRegex(ValueError, 'source-root binding'): self.restore()


if __name__ == '__main__': unittest.main()
