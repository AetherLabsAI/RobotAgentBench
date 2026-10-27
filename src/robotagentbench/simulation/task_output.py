"""Route evidence into an original task's existing output scope, not its rules."""
import hashlib
from pathlib import Path

LIGHTING_TASK = 'stock_wooden_tray_lighting'
LIGHTING_MODULE = 'short_tasks.tasks.stock_wooden_tray_lighting.lighting'
LIGHTING_ROOT = Path('short_tasks/tasks/stock_wooden_tray_lighting')
RELOCATION_TASK = 'stock_wooden_tray_interference'
RELOCATION_MODULE = 'short_tasks.tasks.stock_wooden_tray_interference.native_backend'
RELOCATION_ROOT = Path('short_tasks/tasks/stock_wooden_tray_interference')


def scoring_output(sources, task_name, native_output):
    """Keep the model/sandbox output unchanged and respect original owners.

    The shelf-lighting and human-relocation owners require audit evidence
    beneath their own task directories. Generic native output belongs to the
    code sandbox's separate cap_v1 scope. Only their evidence directories are
    routed here; no original allowlist, seed, schedule, score, or source byte
    is replaced. The original runtime path guards still execute unchanged.
    """
    output = Path(native_output)
    if task_name not in (LIGHTING_TASK, RELOCATION_TASK):
        return output / 'scoring'
    root = sources.root
    if (not output.is_absolute() or output.resolve(strict=True) != output
            or not output.is_dir() or output == root or not output.is_relative_to(root)):
        raise ValueError('Existing task-owned native output in the selected Runtime is required')
    sources.verify()
    is_lighting = task_name == LIGHTING_TASK
    here = root / (LIGHTING_ROOT if is_lighting else RELOCATION_ROOT)
    if is_lighting:
        owner = sources.import_module(LIGHTING_MODULE)
        owner_here = Path(owner.HERE)
    else:
        # Resolve and verify the original backend without importing its native
        # dependencies before Kit initialization. HERE is that file's parent;
        # the unchanged constructor enforces its own output guard later.
        relative, package = sources.module_file(RELOCATION_MODULE)
        if package or relative != (RELOCATION_ROOT / 'native_backend.py').as_posix():
            raise ValueError('Evidence owner differs from the pinned Runtime task directory')
        sources.read_verified(relative)
        owner_here = (root / relative).parent
    if (here.resolve(strict=True) != here or not here.is_dir() or owner_here != here):
        raise ValueError('Evidence owner differs from the pinned Runtime task directory')
    digest = hashlib.sha256(str(output).encode('utf-8')).hexdigest()
    parent = here / 'model_evidence' / digest
    scoring = parent / 'scoring'
    if (parent.resolve() != parent or parent.exists()
            or not scoring.is_relative_to(here) or scoring == here):
        raise ValueError('Fresh evidence within the unchanged task owner scope is required')
    if is_lighting and not owner.permitted_audit_output(scoring / 'lighting.json'):
        raise ValueError('Fresh evidence within the unchanged lighting owner scope is required')
    parent.mkdir(parents=True, exist_ok=False)
    # The original build_session owns creation of the fresh scoring directory.
    return scoring
