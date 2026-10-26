"""Run one real generation with a hard deny-list for motion-data reads."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
from .common import owned, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant',choices=('shelf_drink','coffee_bag'),required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    out = owned(args.output,fresh=True)
    started = time.time()
    opened, denied = [], []
    def guard(event, values):
        if event != 'open' or not values or not isinstance(values[0],(str,bytes)):
            return
        path = str(values[0])
        if path.endswith('.npz') or Path(path).name == 'producer_plan.json':
            denied.append(path)
            raise PermissionError('Motion-data access prohibited in procedural generation')
        if len(opened) < 20000:
            opened.append(path)
    sys.addaudithook(guard)
    receipt = dict(variant=args.variant,seed=args.seed,started=started,
                   motion_reads_denied_by_audit_hook=True)
    try:
        from .procedural_initial import build
        value = build(out,variant=args.variant,seed=args.seed)
        receipt.update(status='cpu_proposal_accepted',scene=value['scene'],
            scene_sha256=value['scene_sha256'],procedural_grasp=value['procedural_grasp'],
            physical_settle_verified=False,model_evaluated=False)
        print(json.dumps(receipt))
    except Exception as error:
        receipt.update(status='cpu_proposal_rejected',error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        receipt.update(ended=time.time(),opened_python_paths=opened,denied_motion_paths=denied)
        write_json(out.parent/(out.name+'.dependency.json'),receipt)


if __name__ == '__main__':
    main()
