"""Real CPU scene generations, with an observing-only optimizer call audit."""
import argparse
import json
import statistics
import sys
from ..common import owned,write_json
from .cup import build


def main():
    p=argparse.ArgumentParser();p.add_argument('--base-key',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seeds',type=int,nargs='+',required=True);a=p.parse_args()
    out=owned(a.output,fresh=True);out.mkdir(parents=True);rows=[];optimizer_calls=[]
    def audit(frame,event,arg):
        if event=='call' and ((frame.f_code.co_name=='solve' and frame.f_code.co_filename.endswith('procedural_grasp.py'))
            or (frame.f_code.co_name=='least_squares' and 'scipy' in frame.f_code.co_filename)):
            optimizer_calls.append(dict(function=frame.f_code.co_name,file=frame.f_code.co_filename))
    sys.setprofile(audit)
    try:
        for seed in a.seeds:
            try:
                value=build(a.base_key,out/str(seed),seed=seed)
                rows.append(dict(seed=seed,accepted=True,hand=value['active_hand'],profile=value['frozen_profile_id'],
                    gap_m=value['procedural_grasp']['fingertip_envelope_signed_gap_m'],
                    target_world=value['proposed_target_world'],room_seed=value['room_seed'],
                    generation_wall_seconds=value['generation_wall_seconds'],native_settle_verified=False))
            except ValueError as exc:rows.append(dict(seed=seed,accepted=False,error=str(exc)))
            write_json(out/'progress.json',dict(rows=rows,optimizer_calls=optimizer_calls))
    finally:sys.setprofile(None)
    good=[r for r in rows if r['accepted']];times=[r['generation_wall_seconds'] for r in good]
    result=dict(requested=len(a.seeds),accepted=len(good),rows=rows,optimizer_calls=optimizer_calls,
        timing_seconds=dict(min=min(times),median=statistics.median(times),max=max(times)) if times else None,
        distinct_gaps=len({r['gap_m'] for r in good}),distinct_target_poses=len({json.dumps(r['target_world']) for r in good}),
        hands=sorted({r['hand'] for r in good}),scene_scope='same_explicit_full_source_room_fresh_continuous_cup_draws',
        fresh_whole_rooms_generated=False,physical_settle_tested=False,model_calls=0)
    write_json(out/'summary.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
    if optimizer_calls or len(good)!=len(a.seeds):raise SystemExit(1)


if __name__=='__main__':main()
