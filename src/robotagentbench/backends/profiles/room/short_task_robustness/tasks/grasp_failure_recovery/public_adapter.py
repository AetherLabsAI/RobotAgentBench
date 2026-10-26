"""Trusted full-tree router for the five approved failure recoveries."""
from pathlib import Path
from .common import HERE, CURRENT, COFFEE_SOURCE, digest
from .unified_five_v3.entry import (
    NAMES, RULE_VERSION, request_for, resolve, verify_sources_for_plan,
    bind, prepare_registered, admit_launch, exec_native,
)


def verify_pins(record, source_root, task_root, frozen_root=COFFEE_SOURCE):
    """No path escape or source-record rewrite can bless a changed dependency."""
    count = 0
    for label, root in (('shared_sources', Path(source_root)), ('task_sources', Path(task_root)),
                        ('frozen_sources', Path(frozen_root))):
        root = root.resolve(strict=True)
        values = record.get(label)
        if not isinstance(values, dict) or not values:
            raise ValueError('Nonempty pinned sources required: '+label)
        for relative, expected in values.items():
            path = (root/relative).resolve(strict=True)
            if not path.is_relative_to(root) or not path.is_file() or digest(path) != expected:
                raise ValueError('Pinned recovery source mismatch: '+relative)
            count += 1
    return dict(verified=True, file_count=count)
