from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from robotagentbench.model_examples import relocation


class RelocationRuntimeTests(TestCase):
    def test_other_tasks_keep_runtime_and_no_profile_export(self):
        with patch('robotagentbench.simulation.diagnostic_instance_bank.source_install.export_original_relocation_profile') as export:
            self.assertEqual(relocation.runtime_for_restore(Path('/runtime'), 'prepare_coffee',
                Path('/runs/one'), sdk_site='/sdk'), (Path('/runtime'), []))
        export.assert_not_called()

    def test_relocation_uses_fresh_profile_and_original_sdk_contract(self):
        with patch('robotagentbench.example.runtime', return_value=(Path('/runtime'), Path('/runtime/source'), {})), \
             patch('robotagentbench.simulation.diagnostic_instance_bank.source_install.export_original_relocation_profile') as export, \
             patch.object(relocation, 'aliases', return_value=['verified']) as aliases:
            result = relocation.runtime_for_restore(Path('/runtime'), relocation.TASK,
                Path('/runs/one'), sdk_site='/sdk')
        export.assert_called_once_with(Path('/runtime'), Path('/runs/one-relocation-runtime'),
            sdk_site='/sdk', logical_source_root=Path('/runs/one-relocation-runtime/source'))
        aliases.assert_called_once_with(Path('/runs/one-relocation-runtime'), relocation.TASK)
        self.assertEqual(result, (Path('/runs/one-relocation-runtime'), ['verified']))

    def test_extra_mount_cannot_be_injected(self):
        with self.assertRaisesRegex(ValueError, 'Unexpected resource'):
            relocation.aliases('/runtime', 'prepare_coffee', [{'target': '/unsafe'}])
        with patch('robotagentbench.simulation.diagnostic_instance_bank.relocation_resources.original_resource_aliases', return_value=[]):
            with self.assertRaisesRegex(ValueError, 'differ'):
                relocation.aliases('/runtime', relocation.TASK, [{'target': '/unsafe'}])

    def test_exact_original_readonly_aliases_are_retained(self):
        expected = [dict(source='/runtime/original', target='/original', read_only=True)]
        with patch('robotagentbench.simulation.diagnostic_instance_bank.relocation_resources.original_resource_aliases', return_value=expected):
            self.assertEqual(relocation.aliases('/runtime', relocation.TASK, expected), expected)
