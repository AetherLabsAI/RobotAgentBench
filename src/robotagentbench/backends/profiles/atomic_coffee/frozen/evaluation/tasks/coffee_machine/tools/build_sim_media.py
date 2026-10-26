"""Transcode the selected source-rendered simulation videos into preview/<key>/sim-<label>-{head,left,right}.mp4
(+ poster / 5-frame keyframe strip for the head camera) and write references/source_sim_videos/<key>/sim-provenance.json
(202 path, producer-registered source path, data_list.json identity, sha256 of source and preview files). CPU/ffmpeg only.

These are videos rendered by the data producer for its own existing trajectories; nothing is replayed or re-rendered here."""
from pathlib import Path
import hashlib, json, subprocess, sys, time

TASK = Path(__file__).resolve().parents[1]
REPO = TASK.parents[2]
REF = TASK / 'references/source_sim_videos'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()


def probe(p):
    s = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-count_packets', '-select_streams', 'v:0', '-show_entries', 'stream=nb_read_packets,width,height,r_frame_rate,duration', '-of', 'json', str(p)],
                                  capture_output=True, text=True, check=True).stdout)['streams'][0]
    return dict(frames=int(s['nb_read_packets']), resolution=[int(s['width']), int(s['height'])], fps=s['r_frame_rate'], duration_s=round(float(s.get('duration') or 0), 3))


def main(keys=None):
    selection = json.loads((REF / 'selection.json').read_text())
    for key, sel in selection.items():
        if keys and key not in keys: continue
        out_dir = TASK / 'preview' / key; out_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for ep in sel['episodes']:
            label = ep['label']; files = []; head = None
            for cam in ('head', 'left', 'right'):
                src = Path(ep['videos'][cam])
                assert src.is_file(), src
                mp4 = out_dir / f'sim-{label}-{cam}.mp4'
                if not mp4.exists():
                    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(src), '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '26', '-vf', "scale='min(960,iw)':-2", '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(mp4)], check=True)
                info = probe(src)
                files.append(dict(cam=cam, source=str(src), source_sha256=sha(src), source_bytes=src.stat().st_size, **{'source_' + k: v for k, v in info.items()},
                                  preview=str(mp4.relative_to(REPO)), preview_sha256=sha(mp4)))
                if cam == 'head':
                    head = info
                    n = probe(mp4)['frames']
                    poster = out_dir / f'sim-{label}-poster.jpg'; keysheet = out_dir / f'sim-{label}-keyframes.jpg'
                    if not poster.exists():
                        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(mp4), '-frames:v', '1', '-vf', 'scale=640:-2', '-q:v', '4', str(poster)], check=True)
                    if not keysheet.exists():
                        idx = [0, n // 4, n // 2, 3 * n // 4, n - 1]
                        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(mp4), '-vf', "select='" + '+'.join(f'eq(n\\,{i})' for i in idx) + "',scale=480:-2,tile=5x1", '-frames:v', '1', '-q:v', '4', str(keysheet)], check=True)
            records.append(dict(label=label, host='202', episode_dir=ep['episode_dir'], head_frames=head['frames'], head_fps=head['fps'], head_duration_s=head['duration_s'], head_resolution=head['resolution'],
                                data_list=ep['data_list'], features=ep['features'], categories=ep['categories'], appearance=ep['appearance'], producer=ep['producer'], files=files))
            print(key, label, head['frames'], 'frames', head['duration_s'], 's', flush=True)
        (REF / key).mkdir(parents=True, exist_ok=True)
        (REF / key / 'sim-provenance.json').write_text(json.dumps(dict(
            page_key=key, kind='source_rendered_simulation_video', captured_at=time.strftime('%Y-%m-%d %H:%M:%S'),
            note='数据方（lukun 咖啡机工程）对其现有轨迹的渲染视频，取自 202 的网站数据目录；不是 202 回放，也不代表 202 验收。source_path 是数据方登记的产出路径（1.1/1.4 在 182，其中 /opt/dlami/nvme 为临时盘），不是 202 下载地址；source_status 为数据清单原样标记，未重新判定。',
            pool=sel['pool'], selection_method=sel['method'], feature_ranges_in_batch=sel['feature_ranges_in_batch'], count=len(records), episodes=records), indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main(sys.argv[1:] or None)
