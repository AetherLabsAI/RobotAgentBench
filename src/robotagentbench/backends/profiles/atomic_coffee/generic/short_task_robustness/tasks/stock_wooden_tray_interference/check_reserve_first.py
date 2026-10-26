"""Exercise candidate shelf composition against an existing room, CPU/USD only.

Uses a private room layer. Does not deploy shared code or launch a robot.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SCENE=ROOT/'openend/tasks/scene_requirements'
REG=ROOT/'layout_task_activation_v1/task_scene_registration_v1'
sys.path[:0]=[str(ROOT),str(SCENE),str(SCENE/'cpu_dependencies'),str(REG),str(ROOT/'api-protocol')]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--mode',choices=['before','ordinary','interference'],required=True)
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--product',required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();out=args.output.resolve()
    if out==HERE or not out.is_relative_to(HERE) or out.exists():raise ValueError('Fresh task-owned output required')
    from pxr import Usd,UsdGeom
    from short_tasks.tasks.stock_wooden_tray_interference.adapter import resolve
    from short_tasks.tasks.stock_wooden_tray_interference.generation import POLICY
    from short_tasks.tasks.stock_wooden_tray_interference.prepare_scene import prepare_reserved
    source=json.loads(args.manifest.read_text());source_hash=sha(source['scene'])
    candidate=HERE/'integration/reserve_first_v1'/('before' if args.mode=='before' else 'candidate')/'build_shelf.py'
    spec=importlib.util.spec_from_file_location('shelf_candidate_'+args.mode,candidate)
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    out.mkdir(parents=True)
    stage=Usd.Stage.CreateNew(str(out/'room.usda'))
    stage.GetRootLayer().subLayerPaths=[source['scene']]
    UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    old_stock=[o for o in source['objects'] if o.get('source')=='snack_shelf' and o['id']!='snack_shelf']
    for o in old_stock:stage.OverridePrim(o['path']).SetActive(False)
    data=dict(request=dict(seed=args.seed),requirements=dict(shelf_task_requirements=[
        dict(id='stock_item_1',type='snack_drink_to_tray',product=args.product)]))
    if args.mode=='interference':data['task_scene_policy']=POLICY
    receipt=dict(mode=args.mode,seed=args.seed,product=args.product,physical_validation=False,
                 gpu_started=False,candidate_sha256=sha(candidate))
    start=time.monotonic();hashes={}
    try:
        rows,tasks,extra,_=builder.compose(stage,data,ROOT,REG,hashes,sha)
        stage.GetRootLayer().Save()
        (out/'raw_composition.json').write_text(json.dumps(dict(objects=rows,**extra),indent=2)+'\n')
        # Retain only original infrastructure, never stale stock/task aliases.
        fresh={o['path']:o for o in source['objects'] if o not in old_stock}
        fresh.update({o['path']:o for o in rows})
        manifest=dict(scene=str(out/'room.usda'),seed=args.seed,objects=list(fresh.values()),tasks=tasks,**extra)
        if args.mode=='interference':
            plan=resolve(dict(task_name='stock_wooden_tray_interference',seed=args.seed,
                              targets=[dict(category=args.product,quantity=1)]))
            frozen=json.dumps(manifest['environment_reservations'],sort_keys=True)
            manifest=prepare_reserved(plan,stage,manifest)
            if frozen!=json.dumps(manifest['environment_reservations'],sort_keys=True):
                raise ValueError('Preparation changed frozen reservation')
        stage.GetRootLayer().Save()
        (out/'composition.json').write_text(json.dumps(manifest,indent=2)+'\n')
        receipt.update(success=True,objects=len(rows),shelf_initialization=extra['shelf_initialization'])
    except Exception as exc:
        receipt.update(success=False,error=type(exc).__name__+': '+str(exc))
    receipt.update(elapsed_s=time.monotonic()-start,source_unchanged=source_hash==sha(source['scene']))
    (out/'report.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='shelf_initialization'}),flush=True)
    if not receipt['success'] or not receipt['source_unchanged']:raise SystemExit(1)


if __name__=='__main__':main()
