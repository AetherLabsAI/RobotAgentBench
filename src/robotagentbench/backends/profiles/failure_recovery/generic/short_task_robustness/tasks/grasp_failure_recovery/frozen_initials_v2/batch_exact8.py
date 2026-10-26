"""Actual CPU generation for the approved exact-eight-only entry."""
import argparse
import json
import sys
from ..common import owned,write_json
from .exact8 import build


def main():
    p=argparse.ArgumentParser();p.add_argument('--base-key',required=True);p.add_argument('--output',required=True)
    p.add_argument('--bank');p.add_argument('--seeds',type=int,nargs='+',required=True);a=p.parse_args()
    out=owned(a.output,fresh=True);out.mkdir(parents=True);rows=[];calls=[]
    def audit(frame,event,arg):
        if event=='call' and ((frame.f_code.co_name=='solve' and frame.f_code.co_filename.endswith('procedural_grasp.py'))
            or (frame.f_code.co_name=='least_squares' and 'scipy' in frame.f_code.co_filename)):
            calls.append(dict(function=frame.f_code.co_name,file=frame.f_code.co_filename))
    sys.setprofile(audit)
    try:
        for seed in a.seeds:
            try:
                v=build(a.base_key,out/str(seed),seed=seed,bank_path=a.bank)
                rows.append(dict(seed=seed,accepted=True,hand=v['active_hand'],profile=v['frozen_profile_id'],
                    gap_m=v['procedural_grasp']['fingertip_envelope_signed_gap_m'],target_world=v['proposed_target_world'],
                    generation_wall_seconds=v['generation_wall_seconds'],native_accepted=False))
            except ValueError as exc:rows.append(dict(seed=seed,accepted=False,error=str(exc)))
            write_json(out/'progress.json',dict(rows=rows,optimizer_calls=calls))
    finally:sys.setprofile(None)
    write_json(out/'summary.json',dict(requested=len(rows),accepted=sum(r['accepted'] for r in rows),rows=rows,
        optimizer_calls=calls,training_dataset_mounted=False,physical_settle_tested=False,model_calls=0))
    if calls or not all(r['accepted'] for r in rows):raise SystemExit(1)


if __name__=='__main__':main()
