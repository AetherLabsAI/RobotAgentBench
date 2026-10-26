"""Serialize diagnostic evidence consistently before and after native shutdown."""
import json
from pathlib import Path


def native_value(value):
    # Imports stay local: ordinary registry loading must not initialize Isaac.
    import numpy as np
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError('Unsupported diagnostic evidence: '+type(value).__name__)


def report_json(report):
    return json.dumps(report, default=native_value, allow_nan=False, indent=2)
