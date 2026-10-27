"""Use the already reviewed shelf-relocation source profile for saved cases.

The prepared configuration bank uses this profile, including its measured SDK
read seam. Restoring it with the older generic generator changes the source
identity. Export a separate standard Runtime; never edit the caller's Runtime.
"""
from pathlib import Path

TASK = 'stock_wooden_tray_interference'


def runtime_for_restore(base, task, output, *, sdk_site):
    if task != TASK:
        return Path(base), []
    from robotagentbench.example import runtime
    from robotagentbench.simulation.diagnostic_instance_bank.source_install import export_original_relocation_profile
    original, source, _ = runtime(base)
    target = output.parent / (output.name + '-relocation-runtime')
    logical = target / source.relative_to(original)
    export_original_relocation_profile(original, target, sdk_site=sdk_site,
                                       logical_source_root=logical)
    return target, aliases(target, task)


def aliases(base, task, recorded=None):
    if task != TASK:
        if recorded:
            raise ValueError('Unexpected resource aliases for an ordinary task')
        return []
    from robotagentbench.simulation.diagnostic_instance_bank.relocation_resources import original_resource_aliases
    expected = original_resource_aliases(base)
    if recorded is not None and recorded != expected:
        raise ValueError('Saved shelf-relocation resource aliases differ from original bindings')
    return expected
