"""CPU-only guards for the native image's original CUDA wheel selection."""
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class NativeTorchRecipeTests(unittest.TestCase):
    def setUp(self):
        self.requirements = [line for line in (ROOT/'docker/requirements-native.txt').read_text().splitlines()
                             if line and not line.startswith('#')]
        self.recipe = (ROOT/'docker/Dockerfile').read_text()

    def test_original_cuda_variants_are_exact_not_default_pypi_wheels(self):
        expected = ['torch==2.7.0+cu128', 'torchvision==0.22.0+cu128', 'torchaudio==2.7.0+cu128']
        self.assertEqual([line for line in self.requirements if line.startswith(('torch==', 'torchvision==', 'torchaudio=='))], expected)

    def test_original_isaac_and_numeric_versions_remain_pinned(self):
        self.assertIn('isaacsim[all,extscache]==5.1.0.0', self.requirements)
        self.assertIn('numpy==1.26.0', self.requirements)
        self.assertIn('scipy==1.15.3', self.requirements)

    def test_cuda_index_and_metadata_assertion_are_present(self):
        self.assertIn('--extra-index-url https://download.pytorch.org/whl/cu128', self.recipe)
        self.assertIn('--extra-index-url https://pypi.nvidia.com', self.recipe)
        self.assertIn("from importlib.metadata import version; assert", self.recipe)
        self.assertIn("['2.7.0+cu128', '0.22.0+cu128', '2.7.0+cu128']", self.recipe)
        self.assertNotIn('import torch', self.recipe)
        self.assertNotIn('import isaacsim', self.recipe)


if __name__ == '__main__':
    unittest.main()
