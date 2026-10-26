"""Small explicit wall-time counters; no simulation scheduling changes."""
from collections import defaultdict
from contextlib import contextmanager
import json
from time import perf_counter


class PhaseTiming:
    def __init__(self):
        self.rows = defaultdict(lambda: {'calls': 0, 'seconds': 0., 'max_seconds': 0.})

    @contextmanager
    def measure(self, name):
        start = perf_counter()
        try:
            yield
        finally:
            elapsed = perf_counter() - start
            row = self.rows[name]
            row['calls'] += 1
            row['seconds'] += elapsed
            row['max_seconds'] = max(row['max_seconds'], elapsed)

    def save(self, path, **metadata):
        path.write_text(json.dumps(dict(phases=dict(self.rows), **metadata), indent=2) + '\n')
