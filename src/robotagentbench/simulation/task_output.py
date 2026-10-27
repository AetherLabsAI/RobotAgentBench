"""Route evidence into an original task's existing output scope, not its rules."""
import hashlib
from pathlib import Path

LIGHTING_TASK = 'stock_wooden_tray_lighting'
LIGHTING_MODULE = 'short_tasks.tasks.stock_wooden_tray_lighting.lighting'
LIGHTING_ROOT = Path('short_tasks/tasks/stock_wooden_tray_lighting')


def scoring_output(sources, task_name, native_output):
    """Keep the model/sandbox output unchanged and respect the lighting owner.

    The original shelf-lighting controller requires its audit JSON beneath its
    own task directory. Generic native output belongs to the code sandbox's
    separate cap_v1 scope. Only this task's evidence directory is routed here;
    no original allowlist, seed, schedule, score, or source byte is replaced.
    """
    output = Path(native_output)
    if task_name != LIGHTING_TASK:
        return output / 'scoring'
    root = sources.root
    if (not output.is_absolute() or output.resolve(strict=True) != output
            or not output.is_dir() or output == root or not output.is_relative_to(root)):
        raise ValueError('Existing task-owned native output in the selected Runtime is required')
    sources.verify()
    lighting = sources.import_module(LIGHTING_MODULE)
    here = root / LIGHTING_ROOT
    if (here.resolve(strict=True) != here or not here.is_dir()
            or Path(lighting.HERE) != here):
        raise ValueError('Lighting owner differs from the pinned Runtime task directory')
    digest = hashlib.sha256(str(output).encode('utf-8')).hexdigest()
    parent = here / 'model_evidence' / digest
    scoring = parent / 'scoring'
    if (parent.resolve() != parent or parent.exists()
            or not lighting.permitted_audit_output(scoring / 'lighting.json')):
        raise ValueError('Fresh evidence within the unchanged lighting owner scope is required')
    parent.mkdir(parents=True, exist_ok=False)
    # The original build_session owns creation of the fresh scoring directory.
    return scoring
