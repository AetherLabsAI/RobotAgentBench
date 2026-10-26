"""Encode all audited camera frames of one replay into three videos and one key-frame sheet.

One frame per 20 Hz control step, encoded at 20 fps, so the videos run in real simulation time.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw

from evaluation.tasks.coffee_machine.tools.audit_replay import AUDIT_SCHEMA, CAMERAS, RULE_VERSION

FPS = 20


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job', type=Path, required=True)
    p.add_argument('--steps', type=int, nargs='+', required=True)
    a = p.parse_args(argv)
    audit = json.loads((a.job / 'run/physical-audit.json').read_text())
    assert audit['passed'] and audit['rule_version'] == RULE_VERSION and audit['schema'] == AUDIT_SCHEMA
    count = audit['control_steps'] + 1
    if any(i < 0 or i >= count for i in a.steps):
        raise ValueError('Review frame out of range')
    ep = a.job / 'episodes' / audit['episode_id']
    out = a.job / 'videos'
    out.mkdir(exist_ok=False)
    videos = []
    for camera in CAMERAS:
        target = out / (camera + '.mp4')
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-n', '-threads', '2',
                        '-framerate', str(FPS), '-i', str(ep / f'step%04d_{camera}.png'), '-c:v', 'libx264',
                        '-threads', '2', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(target)], check=True)
        info = json.loads(subprocess.check_output(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=nb_frames,r_frame_rate',
             '-of', 'json', str(target)], text=True))['streams'][0]
        assert int(info['nb_frames']) == count and info['r_frame_rate'] == f'{FPS}/1'
        videos.append(dict(camera=camera, file=target.name, frames=count, fps=FPS,
                           sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    canvas = Image.new('RGB', (960, len(a.steps) * 260), 'white')
    draw = ImageDraw.Draw(canvas)
    for row, step in enumerate(a.steps):
        for col, camera in enumerate(CAMERAS):
            with Image.open(ep / f'step{step:04d}_{camera}.png') as im:
                im.thumbnail((320, 230))
                canvas.paste(im, (col * 320, row * 260 + 25))
            draw.text((col * 320, row * 260), f'{step} {camera}', fill='black')
    canvas.save(a.job / 'keyframes.jpg')
    result = dict(rule_version=RULE_VERSION, source_episode=ep.name, videos=videos, terminal_included=True, keyframe_steps=a.steps,
                  scope='Complete encoded camera evidence; visual review must be recorded separately')
    (out / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
