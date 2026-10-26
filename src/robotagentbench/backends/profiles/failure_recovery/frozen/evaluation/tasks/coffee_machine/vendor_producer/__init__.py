"""Vendored data-producer device logic (see MANIFEST.json). Import through vendor_producer.load()."""
import importlib, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    """Import one vendored module by its original top-level name (they import each other that way)."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    return importlib.import_module(name)
