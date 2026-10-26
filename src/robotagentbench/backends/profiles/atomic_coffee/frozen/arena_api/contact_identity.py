"""Bounded, per-consumer cache of immutable contact identities, not contacts.

Native collider IDs resolve to Sdf paths. A stage ID is part of every key;
each consumer creates a fresh cache for its stage/episode. Classification must
depend only on paths and the consumer's fixed configuration. Never cache event
types, point counts, impulses, positions, or whether contact still exists here.
"""


class ContactIdentityCache:
    def __init__(self, resolve, classify, *, max_pairs=16384):
        if max_pairs < 1:
            raise ValueError("max_pairs must be positive")
        self.resolve = resolve
        self.classify = classify
        self.max_pairs = max_pairs
        self.paths = {}
        self.pairs = {}

    def get(self, stage_id, collider0, collider1):
        # Preserve reported order separately from the canonical active-pair key.
        token = (stage_id, collider0, collider1)
        identity = self.pairs.get(token)
        if identity is None:
            if len(self.pairs) >= self.max_pairs:
                self.clear()  # Bounded memory; eviction affects speed only.
            paths = []
            for collider in (collider0, collider1):
                key = (stage_id, collider)
                path = self.paths.get(key)
                if path is None:
                    path = str(self.resolve(collider))
                    self.paths[key] = path
                paths.append(path)
            paths = tuple(paths)
            identity = (paths, tuple(sorted(paths)), self.classify(paths))
            # Cache negative classifications too: ignored pairs are frequent.
            self.pairs[token] = identity
        return identity

    def clear(self):
        self.paths.clear()
        self.pairs.clear()
