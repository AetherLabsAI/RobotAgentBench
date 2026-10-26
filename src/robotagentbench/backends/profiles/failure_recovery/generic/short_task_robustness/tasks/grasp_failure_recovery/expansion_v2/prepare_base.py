"""Call the unmodified full-tree preparation entry under isolated output mounts.

No source/parameter monkey patches, no reduced-release import fallback. The
container runner must bind the two declared output namespaces to this task.
"""
import argparse
import hashlib
import json
from pathlib import Path
from ..common import HERE, CURRENT, owned, write_json
from .catalog import base_request

# Use the existing output mountpoint: readonly project mounts must not create
# a new child directory in the shared source just to hold a Docker mount.
NAMESPACE = CURRENT/'short_tasks/agent_probe_v1/runs'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=('sofa', 'table_waste_tipped_cup', 'table_waste_empty_grasp'), required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--sofa-asset', choices=('remote_tv', 'book_red', 'wallet', 'glasses_case'))
    ap.add_argument('--waste-category', choices=('paper', 'snack', 'can', 'cup'), action='append')
    ap.add_argument('--key', required=True)
    a = ap.parse_args()
    if not a.key.replace('_', '').isalnum():
        raise ValueError('Simple fresh key required')
    own = owned(HERE/'expansion_v2/runs'/a.key, fresh=True)
    own.mkdir(parents=True)
    out = NAMESPACE/a.key/'base'
    request = base_request(a.variant, a.seed, sofa_asset=a.sofa_asset, waste_categories=a.waste_category)
    from short_tasks import registry
    from short_tasks.agent_probe_v1 import prepare_task
    from short_tasks.scene import verify_scene_sources
    from openend.tasks.layout_repair_v1 import ready_envelope
    modules = (registry, prepare_task)
    before = {str(Path(m.__file__).resolve()): sha(m.__file__) for m in modules}
    if any(not Path(p).is_relative_to(CURRENT) for p in before):
        raise ValueError('Only the full migrated source may supply this preparation')
    # Exact existing calibration and measured generic envelope, not new values.
    reference = ready_envelope.REFERENCE/'navigation_result.json'
    calibration = CURRENT/'short_tasks/agent_probe_v1/completion_trash43_v1/native/launch.json'
    ready = json.loads(reference.read_text())['navigation_posture']
    sensor = json.loads(calibration.read_text())['sensor']
    write_json(own/'ready_reference.json', ready)
    write_json(own/'sensor_reference.json', dict(sensor=sensor))
    write_json(own/'request.json', request)
    report = dict(variant=a.variant, seed=a.seed, full_tree=str(CURRENT), request=request,
        shared_sources=before, calibration_sources={str(reference):sha(reference), str(calibration):sha(calibration)},
        original_source_validation=verify_scene_sources(),
        namespace=str(NAMESPACE), shared_sources_read_only=True,
        phase='original_task_random_scene_only_not_failed_state', native_accepted=False,
        recovery_initial_generated=False, model_called=False)
    try:
        report['base'] = prepare_task.prepare(request, own/'ready_reference.json', own/'sensor_reference.json', out)
        report['status'] = 'base_prepared_not_native_accepted'
    except BaseException as e:
        report.update(status='base_rejected', error=type(e).__name__+': '+str(e))
        raise
    finally:
        report['source_bytes_unchanged'] = all(sha(p) == h for p, h in before.items())
        write_json(own/'base_report.json', report)


if __name__ == '__main__':
    main()
