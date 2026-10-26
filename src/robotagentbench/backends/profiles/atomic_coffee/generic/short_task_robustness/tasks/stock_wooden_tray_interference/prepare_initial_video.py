"""Use the existing shelf station/recorded arms for a task-local video launch."""
import argparse
import hashlib
import json
from pathlib import Path
from pxr import Usd
from short_tasks.tasks.stock_wooden_tray import prepare_initial

HERE=Path(__file__).resolve().parent
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();out=args.output.resolve()
    if out==HERE or not out.is_relative_to(HERE) or out.exists():raise ValueError('Fresh task output required')
    m=json.loads(args.manifest.read_text());stage=Usd.Stage.Open(m['scene'])
    station=prepare_initial.choose(stage,m,seed=m['seed'])
    out.mkdir(parents=True)
    (out/'station.json').write_text(json.dumps(station,indent=2)+'\n')
    (out/'ready_pose.json').write_text(json.dumps(dict(targets_rad=station['targets_rad']),indent=2)+'\n')
    launch=dict(manifest=str(args.manifest.resolve()),ready_pose=str(out/'ready_pose.json'),
        initial_base_xyyaw=[*station['position'][:2],station['yaw_rad']],
        scene_sha256=hashlib.sha256(Path(m['scene']).read_bytes()).hexdigest(),
        native_physics_validated=False,initialization_video_only=True)
    (out/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    print(json.dumps(launch))

if __name__=='__main__':main()
