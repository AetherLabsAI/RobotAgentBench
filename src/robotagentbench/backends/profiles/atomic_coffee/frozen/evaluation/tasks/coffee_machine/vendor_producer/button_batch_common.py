from pathlib import Path
import argparse,json,random
ROOT=Path(__file__).resolve().parent.parent
CONFIG=dict(version=1,machine_xy_m=.15,machine_yaw_deg=30.,first_seed=5000,attempts=10,reference='button_horizontal_visible_2000_v1',physics_hz=240,control_hz=20,capture_fps=30,appearance='api_shared_lights_materials_v1',proposal_policy='one uniform proposal per seed; retain all rejections; no resampling')
def proposal(seed):
 rng=random.Random(seed)
 return dict(seed=seed,machine_dx_m=rng.uniform(-CONFIG['machine_xy_m'],CONFIG['machine_xy_m']),machine_dy_m=rng.uniform(-CONFIG['machine_xy_m'],CONFIG['machine_xy_m']),machine_dyaw_deg=rng.uniform(-CONFIG['machine_yaw_deg'],CONFIG['machine_yaw_deg']))
def episode_directory():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();out=a.out.resolve();out.relative_to((ROOT/'runs').resolve());return out
