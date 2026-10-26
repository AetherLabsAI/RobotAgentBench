"""Read-only namespace for the current robot22 contract/base executor.

Loaded under this independent package name, never replacing or monkey-patching
the frozen arena_api used by the atomic scorer and physics. All dependencies
are pinned by the task integration source manifest.
"""
from ..adapter import ROOT
__path__ = [str(ROOT / 'arena_api')]
