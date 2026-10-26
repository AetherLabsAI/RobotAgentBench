"""Transcode the selected L2 real-robot head videos for the coffee_machine review pages (CPU/ffmpeg only).

Input : artifacts/real-batch/<uuid>/{meta_info.json,camera/head_color/head_color.h265,head_color.txt}
        references/l2_real_batch/selection.json  (API id -> groups -> episodes; written after first-frame identity review)
Output: preview/<key>/real-head-<uuid8>.mp4, -poster.jpg, -keyframes.jpg
        references/l2_real_batch/<key>/<uuid>/{meta_info.json,head_color.txt} and media-provenance.json
Page keys without a real recording get a media-provenance.json with zero episodes and the searched scope, so the page shows 0/3."""
from pathlib import Path
import hashlib, json, shutil, subprocess, sys, time

TASK = Path(__file__).resolve().parents[1]
BATCH = TASK / 'artifacts/real-batch'
REF = TASK / 'references/l2_real_batch'
REMOTE = '/data/admin/work/livingroom_data/record'
SEARCHED = ('L2 /data/admin/work/livingroom_data/record_index.tsv（5108 条，2026-09-29 读取）按 task_name 查 coffee/cup/button/machine/bean/pour：'
            '只有 task 75 pour_coffee_bean（75 条）属于咖啡机家族；29 trash_cup、33 sofa_pick_cup、77 atom_pick_cup 是其他家族的杯子任务')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def frames(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-count_packets', '-select_streams', 'v:0', '-show_entries', 'stream=nb_read_packets,width,height', '-of', 'json', str(path)], capture_output=True, text=True, check=True).stdout
    s = json.loads(out)['streams'][0]
    return int(s['nb_read_packets']), int(s['width']), int(s['height'])


def main(keys=None):
    selection = json.loads((REF / 'selection.json').read_text())
    for key, groups in selection.items():
        if keys and key not in keys: continue
        out_dir = TASK / 'preview' / key; out_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for grp in groups:
            for ep in grp['episodes']:
                uuid = ep['uuid']; short = uuid[:8]
                src = BATCH / uuid / 'camera/head_color/head_color.h265'
                ts = BATCH / uuid / 'camera/head_color/head_color.txt'
                meta_path = BATCH / uuid / 'meta_info.json'
                meta = json.loads(meta_path.read_text())
                assert str(meta['task_id']) == str(ep['task_id']), (uuid, meta['task_id'], ep['task_id'])
                fps = float(meta['fps']['head_color'])
                n, w, h = frames(src)
                stamps = [float(l.split()[0]) for l in ts.read_text().split('\n') if l.strip()]
                span = (stamps[-1] - stamps[0]) if len(stamps) > 1 else None
                if span and span > 1e6: span /= 1e9   # nanoseconds
                mp4 = out_dir / f'real-head-{short}.mp4'; poster = out_dir / f'real-head-{short}-poster.jpg'; keysheet = out_dir / f'real-head-{short}-keyframes.jpg'
                if not mp4.exists():
                    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-r', f'{fps}', '-i', str(src), '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '26', '-vf', 'scale=960:-2', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(mp4)], check=True)
                m = frames(mp4)[0]
                if not poster.exists():
                    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(mp4), '-frames:v', '1', '-vf', 'scale=640:-2', '-q:v', '4', str(poster)], check=True)
                idx = [0, m // 4, m // 2, 3 * m // 4, m - 1]
                if not keysheet.exists():
                    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(mp4), '-vf', "select='" + '+'.join(f'eq(n\\,{i})' for i in idx) + "',scale=480:-2,tile=5x1", '-frames:v', '1', '-q:v', '4', str(keysheet)], check=True)
                ref_dir = REF / key / uuid; ref_dir.mkdir(parents=True, exist_ok=True)
                for f in (meta_path, ts): shutil.copy2(f, ref_dir / f.name)
                records.append(dict(page_key=key, kind=grp['kind'], group_label=grp['label'], note=ep.get('note'), small_task_id=int(meta['task_id']), episode_uuid=uuid, host='arena-l2',
                                    remote_episode=f'{REMOTE}/{uuid}', remote_video=f'{REMOTE}/{uuid}/camera/head_color/head_color.h265',
                                    create_time=meta['create_time'], meta_duration_s=meta['duration'], validate=meta['data_validate']['validate'],
                                    original_resolution=[w, h], decoded_frames=n, preview_frames=m, source_timestamp_span_s=span, fps_from_meta=fps, keyframe_indices=idx,
                                    preview_encoding='H.264 MP4 960px wide, CRF 26, fps from meta; original timestamps kept in head_color.txt',
                                    files=[dict(local_path=str(p.relative_to(TASK.parents[2])), bytes=p.stat().st_size, sha256=sha(p)) for p in (src, mp4, poster, keysheet, ref_dir / 'meta_info.json', ref_dir / 'head_color.txt')]))
                print(key, short, n, 'frames', f'{span:.2f}s' if span else '', flush=True)
        (REF / key).mkdir(parents=True, exist_ok=True)
        (REF / key / 'media-provenance.json').write_text(json.dumps(dict(
            page_key=key, kind='real_robot_head_camera', searched=SEARCHED,
            groups=[dict(kind=g['kind'], l2_task_id=g['l2_task_id'], label=g['label'], count=len(g['episodes']), rejected=g.get('rejected', []), review_sheet=g.get('review_sheet')) for g in groups],
            captured_at=time.strftime('%Y-%m-%d %H:%M:%S'), episodes=records), indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main(sys.argv[1:] or None)
