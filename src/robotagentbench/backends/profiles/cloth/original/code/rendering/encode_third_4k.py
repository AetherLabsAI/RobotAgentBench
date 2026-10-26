"""Encode the rendered native head camera and check every decoded frame."""
from pathlib import Path
import argparse
import datetime
from fractions import Fraction
import json
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('episode', type=Path)
args = parser.parse_args()
episode = args.episode.resolve()
root = episode.parent
trial = json.loads((episode / 'trial.json').read_text())
render = json.loads((episode / 'render_4k/render.json').read_text())
camera = json.loads((episode / 'render_4k/native_camera_readback.json').read_text())['overview']
fps, count = int(trial['fps']), int(trial['frames'])
assert render['frames'] == list(range(count)) and render['stride'] == 1
assert render['views'] == ['overview']
assert camera['resolution'] == [3840, 2560] and not camera['mount_and_intrinsics_modified']
reference = json.loads((episode / 'REFERENCE_OVERVIEW_CAMERA.json').read_text())
for field in ['path', 'parent', 'local_transform_row_major', 'attributes']:
    assert camera[field] == reference[field], 'Camera calibration differs from original HD render: ' + field
assert [int(p.stem) for p in sorted((episode / 'render_4k/overview').glob('*.jpg'))] == list(range(count))


def call(executable, argv):
    return subprocess.run([
        'docker', 'run', '--rm', '--user', '1000:1234', '--entrypoint', executable,
        '-v', str(root) + ':/io:rw', 'registry.agibot.com/genie-sim/open_source:latest',
        *argv,
    ], capture_output=True, text=True, check=True)


outputs = [(episode.name + '_third_person_full_4K.mp4', count)]
rows = []
for name, frames in outputs:
    output = root / name
    assert not output.exists(), output
    call('/usr/local/bin/ffmpeg', [
        '-nostdin', '-v', 'error', '-framerate', str(fps), '-start_number', '0',
        '-i', f'/io/{episode.name}/render_4k/overview/%06d.jpg', '-frames:v', str(frames),
        '-vf', 'scale=3240:2160:flags=lanczos:in_range=pc:out_range=tv:out_color_matrix=bt709,pad=3840:2160:300:0:black',
        '-an', '-c:v', 'libx264', '-threads', '8', '-preset', 'fast', '-crf', '16', '-pix_fmt', 'yuv420p',
        '-color_range', 'tv', '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-movflags', '+faststart', '/io/' + name,
    ])
    probe = json.loads(call('/usr/local/bin/ffprobe', [
        '-v', 'error', '-select_streams', 'v:0', '-count_frames',
        '-show_entries', 'stream=codec_name,pix_fmt,color_range,color_space,width,height,avg_frame_rate,duration,nb_read_frames:frame=best_effort_timestamp_time',
        '-of', 'json', '/io/' + name,
    ]).stdout)
    stream = probe['streams'][0]
    times = [float(f['best_effort_timestamp_time']) for f in probe['frames']]
    assert int(stream['nb_read_frames']) == frames and len(times) == frames
    assert Fraction(stream['avg_frame_rate']) == fps
    assert (stream['width'], stream['height']) == (3840, 2160)
    assert stream['codec_name'] == 'h264' and stream['pix_fmt'] == 'yuv420p'
    assert stream['color_range'] == 'tv' and stream['color_space'] == 'bt709'
    assert abs(float(stream['duration']) - frames / fps) < 1e-5
    assert max(abs(t - i / fps) for i, t in enumerate(times)) < 1e-5
    call('/usr/local/bin/ffmpeg', ['-nostdin', '-v', 'error', '-i', '/io/' + name, '-f', 'null', '-'])
    atoms = {}
    with output.open('rb') as f:
        total = output.stat().st_size
        while f.tell() < total:
            offset = f.tell()
            header = f.read(8)
            assert len(header) == 8
            size = int.from_bytes(header[:4], 'big')
            kind = header[4:].decode('ascii', errors='replace')
            if size == 1:
                size = int.from_bytes(f.read(8), 'big')
            elif size == 0:
                size = total - offset
            assert size >= 8 and offset + size <= total
            atoms.setdefault(kind, offset)
            f.seek(offset + size)
    assert atoms['moov'] < atoms['mdat']
    row = {
        'path': str(output), 'camera': 'TableclothOverview', 'view': 'third_person',
        'frames': frames, 'fps': fps, 'duration_s': frames / fps,
        'native_render_resolution': [3840, 2560], 'delivery_resolution': [3840, 2160],
        'content_resolution': [3240, 2160], 'color_range': 'tv', 'color_space': 'bt709', 'camera_crop': False,
        'camera_mount_and_intrinsics_modified': False,
        'bytes': output.stat().st_size, 'full_decode_pass': True,
        'timestamps_pass': True, 'faststart_verified': True, 'true_4k_rerender': True,
    }
    rows.append(row)
    print(json.dumps(row), flush=True)

report = {
    'episode': episode.name, 'source_physics': trial['source'],
    'historical_physical_success': trial['physical_success'],
    'source_state': json.loads((episode / 'SOURCE_STATE.json').read_text()),
    'true_4k_rerender': True, 'video_qa_pass': True, 'new_physics_execution': False,
    'production_count': 0, 'owner_approved': False, 'videos': rows,
    'updated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
(root / (episode.name + '_4K_VIDEO_QA.json')).write_text(json.dumps(report, indent=2) + '\n')
