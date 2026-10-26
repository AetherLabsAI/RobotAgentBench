"""Local, read-only native tablecloth state viewer with frame-anchored feedback.

Geometry is an explicitly selected export for a canonical diagnostic station.
It is not a House renderer or a task checker. Feedback never changes acceptance.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from zipfile import ZipFile, BadZipFile
from urllib.parse import parse_qs, unquote, urlsplit
from urllib.request import urlopen
from uuid import uuid4
from datetime import datetime, timezone

import numpy as np

WEB = Path(__file__).with_name('tablecloth_review_web')
THREE_VERSION = '0.170.0'


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def within(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Path outside review root')
    return path


class Review:
    def __init__(self, root, geometry, robot, assets):
        self.root, self.assets = root.resolve(), assets.resolve()
        self.geometry, self.robot = geometry.resolve(), robot.resolve()
        self.lock = threading.RLock()
        self.feedback_path = self.root / 'review_feedback.jsonl'
        self.static = None

    def episode(self, name):
        path = within(self.root, name)
        if path.parent != self.root or not (path / 'trial.json').is_file():
            raise ValueError('Unknown trial')
        trial = read_json(path / 'trial.json', {})
        if trial.get('particle_state_readback') != 'native GPU tensors':
            raise ValueError('Only recorded native physics states are supported')
        return path, trial

    def revision(self, path, trial):
        s = (path / 'full_state.npz').stat()
        # JavaScript numbers cannot exactly preserve nanosecond timestamps.
        return dict(size=s.st_size, mtime_ns=str(s.st_mtime_ns), source=trial.get('source'))

    def runs(self):
        rows = []
        for path in self.root.glob('*/trial.json'):
            try:
                episode, trial = self.episode(path.parent.name)
                revision = self.revision(episode, trial)
                with ZipFile(episode / 'full_state.npz') as z:
                    if not {'body_q.npy', 'body_names.npy', 'cloth_q.npy'}.issubset(z.namelist()):
                        continue  # Older diagnostics lack measured robot poses.
                if not trial.get('benchmark_task_id'):
                    continue
            except (OSError, ValueError, BadZipFile):
                continue
            rows.append(dict(id=episode.name, frames=trial.get('frames'),
                             task=trial.get('benchmark_task_id'), revision=revision,
                             diagnostic=trial.get('diagnostic_only', True),
                             quality=read_json(episode / 'physical_quality.json'),
                             mtime=path.stat().st_mtime))
        return sorted(rows, key=lambda r: r['mtime'], reverse=True)

    @lru_cache(maxsize=2)
    def _state(self, name, mtime, size):
        path, _ = self.episode(name)
        with np.load(path / 'full_state.npz', allow_pickle=False) as z:
            keys = ['cloth_q', 'body_q', 'body_names', 'faces', 'cloth_id', 'time_s', 'phase']
            state = {k: z[k] for k in keys}
            if 'active_task_id' in z:
                state['active_task_id'] = z['active_task_id']
        if not np.isfinite(state['cloth_q']).all() or not np.isfinite(state['body_q']).all():
            raise ValueError('Nonfinite recorded state; inspect raw diagnostic')
        return state

    def state(self, name):
        path, trial = self.episode(name)
        rev = self.revision(path, trial)
        with self.lock:
            return self._state(name, rev['mtime_ns'], rev['size'])

    def info(self, name):
        path, trial = self.episode(name)
        s = self.state(name)
        renders = {}
        for camera in ('overview', 'head', 'left', 'right'):
            files = sorted((path / 'render' / camera).glob('*.jpg'))
            renders[camera] = [int(p.stem) for p in files if p.stem.isdigit()]
        return dict(id=name, revision=self.revision(path, trial), trial=trial,
                    quality=read_json(path / 'physical_quality.json'),
                    material_shape=read_json(path / 'material_shape.json'),
                    frames=len(s['time_s']), particles=s['cloth_q'].shape[1],
                    bodies=s['body_names'].tolist(), faces=s['faces'].tolist(),
                    cloth_id=s['cloth_id'].tolist(), time_s=s['time_s'].tolist(),
                    phase=s['phase'].tolist(), renders=renders,
                    active_task_id=s.get('active_task_id', np.full(len(s['time_s']), trial['benchmark_task_id'])).tolist(),
                    geometry_note='显式选择的 canonical 桌 / 篮及机器人几何；仅展示这些物体，完整 House 请看渲染。',
                    geometry_compatible=bool(trial.get('config', {}).get('diagnostic_canonical')))

    def chunk(self, name, start, count):
        s = self.state(name)
        if not 0 <= start < len(s['time_s']) or not 1 <= count <= 40:
            raise ValueError('Invalid frame interval')
        cloth = s['cloth_q'][start:start+count].reshape(min(count, len(s['time_s'])-start), -1)
        body = s['body_q'][start:start+count].reshape(len(cloth), -1)
        return np.concatenate([cloth, body], axis=1).astype('<f4').tobytes()

    def geometry_data(self):
        with self.lock:
            if self.static is not None:
                return self.static
            robot = read_json(self.robot)
            meshes = []
            with np.load(self.geometry, allow_pickle=False) as z:
                meta = json.loads(str(z['scene_collision_json']))
                for m in meta['meshes']:
                    k = m['key']
                    meshes.append(dict(kind=m['group'], points=z['scene_points_'+k].tolist(),
                                       triangles=z['scene_triangles_'+k].tolist()))
            for m in robot['nonhand_collision_shapes']:
                meshes.append(dict(kind='robot', link=m['link'], points=m['vertices'], triangles=m['triangles']))
            with np.load(self.robot.with_suffix('.npz'), allow_pickle=False) as z:
                for k in z.files:
                    if not k.startswith('points_'):
                        continue
                    link = k[len('points_'):]
                    # Exact deduplication only; preserve all triangles and pad surfaces.
                    points, inverse = np.unique(z[k], axis=0, return_inverse=True)
                    faces = inverse[z['triangles_'+link]]
                    meshes.append(dict(kind='gripper', link=link, points=points.tolist(), triangles=faces.tolist()))
            self.static = dict(meshes=meshes, source=dict(geometry=str(self.geometry), robot=str(self.robot)))
            return self.static

    def feedback(self):
        with self.lock:
            if not self.feedback_path.exists():
                return []
            return [json.loads(line) for line in self.feedback_path.read_text().splitlines() if line.strip()]

    def save_feedback(self, data):
        name, frame = data.get('run'), data.get('frame')
        if not isinstance(name, str) or type(frame) is not int:
            raise ValueError('Trial and integer frame required')
        path, trial = self.episode(name)
        s = self.state(name)
        if not 0 <= frame < len(s['time_s']):
            raise ValueError('Frame outside trial')
        revision = self.revision(path, trial)
        if data.get('revision') != revision:
            raise ValueError('Trial changed. Reload before saving feedback.')
        note = data.get('note', '')
        if not isinstance(note, str) or not note.strip() or len(note) > 4000:
            raise ValueError('Feedback must contain 1–4000 characters')
        issue = data.get('issue')
        if issue not in ('motion', 'grasp', 'collision', 'cloth', 'camera', 'looks_ok', 'other'):
            raise ValueError('Unknown issue type')
        item = dict(id=str(uuid4()), created_at=datetime.now(timezone.utc).isoformat(),
                    run=name, frame=frame, time_s=float(s['time_s'][frame]),
                    phase=str(s['phase'][frame]), revision=revision,
                    issue=issue, note=note.strip(), owner_approval=False)
        with self.lock:
            with self.feedback_path.open('a') as f:
                f.write(json.dumps(item, ensure_ascii=False)+'\n')
        return item


def handler(review):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, data, content_type='application/json', status=200):
            if isinstance(data, (dict, list)):
                data = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
            if isinstance(data, str):
                data = data.encode()
            compressed = len(data) > 4096 and 'gzip' in self.headers.get('Accept-Encoding', '')
            if compressed:
                data = gzip.compress(data, compresslevel=1)
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            if compressed:
                self.send_header('Content-Encoding', 'gzip')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            try:
                url = urlsplit(self.path)
                q = {k: v[0] for k, v in parse_qs(url.query).items()}
                if url.path == '/api/runs':
                    return self.respond(review.runs())
                if url.path == '/api/info':
                    return self.respond(review.info(q['run']))
                if url.path == '/api/frames':
                    return self.respond(review.chunk(q['run'], int(q['start']), int(q.get('count', 40))), 'application/octet-stream')
                if url.path == '/api/geometry':
                    return self.respond(review.geometry_data())
                if url.path == '/api/feedback':
                    return self.respond(review.feedback())
                if url.path.startswith('/render/'):
                    relative = unquote(url.path[len('/render/'):])
                    p = within(review.root, relative)
                    if p.suffix != '.jpg' or '/render/' not in '/'+relative:
                        raise ValueError('Only rendered JPEG frames are served')
                    return self.respond(p.read_bytes(), 'image/jpeg')
                if url.path == '/vendor/three.module.min.js':
                    return self.respond((review.assets / 'three.module.min.js').read_bytes(), 'text/javascript')
                paths = {'/': ('index.html', 'text/html; charset=utf-8'), '/app.js': ('app.js', 'text/javascript'), '/app.css': ('app.css', 'text/css')}
                if url.path not in paths:
                    return self.respond({'error': 'Not found'}, status=404)
                p, mime = paths[url.path]
                return self.respond((WEB / p).read_bytes(), mime)
            except (ValueError, KeyError, OSError) as e:
                self.respond({'error': str(e)}, status=400)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            try:
                # Browser writes must originate on this same local server.
                origin = self.headers.get('Origin')
                if origin and origin != 'http://'+self.headers.get('Host', ''):
                    return self.respond({'error': 'Cross-origin write rejected'}, status=403)
                if self.path != '/api/feedback':
                    return self.respond({'error': 'Not found'}, status=404)
                size = int(self.headers.get('Content-Length', 0))
                if not 0 < size <= 32768:
                    raise ValueError('Invalid request size')
                self.respond(review.save_feedback(json.loads(self.rfile.read(size))), status=201)
            except (ValueError, KeyError, OSError) as e:
                self.respond({'error': str(e)}, status=400)

    return Handler


def install_assets(path):
    path.mkdir(parents=True, exist_ok=True)
    for filename, remote in [('three.module.min.js', 'build/three.module.min.js'), ('THREE-LICENSE.txt', 'LICENSE')]:
        target = path / filename
        if not target.exists():
            with urlopen(f'https://unpkg.com/three@{THREE_VERSION}/{remote}', timeout=60) as response:
                data = response.read()
            temporary = target.with_suffix('.download')
            temporary.write_bytes(data)
            temporary.replace(target)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--robot', type=Path, required=True)
    p.add_argument('--port', type=int, default=8766)
    p.add_argument('--install-viewer-assets', action='store_true', help='Download pinned Three.js and license once into run-local viewer_assets')
    args = p.parse_args()
    assets = args.root / 'viewer_assets'
    if args.install_viewer_assets:
        install_assets(assets)
    if not (assets / 'three.module.min.js').is_file():
        p.error('First launch requires --install-viewer-assets; subsequent launches work offline')
    review = Review(args.root, args.geometry, args.robot, assets)
    review.geometry_data()  # Fail clearly before advertising a working URL.
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler(review))
    print(f'Tablecloth review: http://127.0.0.1:{args.port} — feedback: {review.feedback_path}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
