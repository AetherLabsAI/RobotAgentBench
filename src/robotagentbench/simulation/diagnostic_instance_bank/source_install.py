"""Install a byte-identical original preparer in this bank's owned source view."""
import hashlib
import json
from pathlib import Path

LIGHTING_SHA256 = 'eef50872329f8389a266366f1240db58f2a9595d0b414bd26fa1b07c2b3c9cac'
LIGHTING_RELATIVE = 'short_tasks/tasks/stock_wooden_tray_lighting/prepare_existing.py'


def export_original_relocation_profile(runtime_root, output_root, *, sdk_site, logical_source_root):
    """Export a fresh standard relocation Runtime, preserving the old Runtime.

    Source patches include the audited baseline, SDK-read seam, measured
    contact-offset data, formal request and their new explicit pins. The SDK
    and source paths come from the caller's operator namespace contract.
    ``output_root`` can be passed directly to the standard ``--runtime`` option.
    """
    from .relocation_export import export_profile
    return export_profile(runtime_root, output_root, sdk_site=sdk_site,
                          logical_source_root=logical_source_root)


def _owned_source(bank_root, source_root):
    root, source = Path(bank_root), Path(source_root)
    if (root.resolve() != root or source.resolve() != source or source == root
            or not source.is_relative_to(root)
            or json.loads((root/'OWNER.json').read_text()) != {
                'owner': 'RobotAgentBench-diagnostic-instance-bank-v31'}):
        raise ValueError('Canonical diagnostic-bank-owned source view required')
    return source


def install_original_lighting_preparer(bank_root, source_root):
    source = _owned_source(bank_root, source_root)
    raw = (Path(__file__).parent/'original_sources/lighting_prepare_existing.py').read_bytes()
    if hashlib.sha256(raw).hexdigest() != LIGHTING_SHA256:
        raise ValueError('Original lighting preparer bytes changed')
    target = source/LIGHTING_RELATIVE
    if target.resolve() != target:
        raise ValueError('Original preparation target must not escape through a symlink')
    if target.exists():
        if target.read_bytes() != raw:
            raise ValueError('Never replace a different existing preparation source')
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(raw)
    return dict(relative=LIGHTING_RELATIVE, sha256=LIGHTING_SHA256,
                original_bytes=True, scoring_or_physics_modified=False)


def install_original_recovery_dependencies(bank_root, source_root):
    """Install exact original missing files, never overwrite an existing version.

    Preflight the complete archive before writing. This operates only on a new
    bank-owned source view, not a shared checkout or an active Runtime. It does
    not replace zero-byte namespace mountpoints with different contents.
    """
    from .recovery_sources import original_dependencies
    source = _owned_source(bank_root, source_root)
    dependencies = original_dependencies()
    targets = []
    for record, raw in dependencies:
        relative = Path(record['relative'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Relative original dependency path required')
        target = source/relative
        if target.resolve() != target:
            raise ValueError('Original dependency must not escape through a symlink')
        if target.exists() and (not target.is_file() or target.read_bytes() != raw):
            raise ValueError('Never replace a different existing Recovery source: '
                             + record['relative'])
        targets.append((record, raw, target))
    installed = []
    for record, raw, target in targets:
        existed = target.exists()
        if not existed:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(raw)
        installed.append(dict(relative=record['relative'], sha256=record['sha256'],
                              bytes=record['bytes'], existing=existed))
    return dict(files=installed, original_bytes=True,
                scoring_or_physics_modified=False, original_modules_executed=False)
