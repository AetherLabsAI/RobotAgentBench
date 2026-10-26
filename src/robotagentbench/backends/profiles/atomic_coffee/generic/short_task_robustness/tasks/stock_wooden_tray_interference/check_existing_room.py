"""Read-only scene admission diagnostic; does not constitute an episode run."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--product',required=True,choices=['sprite_c1','gatorade_t'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    if not output.is_relative_to(HERE) or output.exists():
        raise ValueError('Fresh task-owned output directory required')
    from pxr import Usd
    from short_tasks.tasks.stock_wooden_tray_interference.diagnostic_variant import bind_existing_drink
    from short_tasks.tasks.stock_wooden_tray_interference.prepare_scene import prepare_reserved,TASK
    original=json.loads(args.manifest.read_text())
    output.mkdir()
    report=dict(admitted=False,source_manifest=str(args.manifest),product=args.product,
                physical_validation=False,gpu_started=False,diagnostic_only=True)
    sources={}
    try:
        manifest,identity,_=bind_existing_drink(original,args.product)
        stage=Usd.Stage.Open(manifest['scene'])
        sources={l.realPath:hashlib.sha256(Path(l.realPath).read_bytes()).hexdigest()
                 for l in stage.GetUsedLayers() if l.realPath}
        plan=dict(task_name=TASK,runtime_components=[dict(instance_id=identity,
                 parameters=dict(product=args.product))])
        result=prepare_reserved(plan,stage,manifest)
        (output/'reservation.json').write_text(json.dumps(result['environment_reservations'],indent=2)+'\n')
        report.update(admitted=True,robot_pose_evidence='authored_geometry_only',
                      native_ready_pose_recheck_required=True)
    except Exception as exc:
        report.update(error_type=type(exc).__name__,reason=str(exc))
    report['source_hashes']=sources
    report['source_unchanged']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in sources.items())
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_hashes'}))
    if not report['source_unchanged']:raise RuntimeError('Source unexpectedly changed')


if __name__=='__main__':main()
