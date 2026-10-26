"""One-way measured-state observations in an owned render-only subprocess.

The physical process never calls render/app.update and never receives state
from the renderer. Rendered poses belong only to the child's visual replica;
all task facts and scoring continue reading the original native tensors.
Both processes are one private validation job under the existing GPU lease.
"""
import base64
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np


class SourceVisualClient:
    def __init__(self, *, owner, sdk_site):
        if not sdk_site or not Path(sdk_site).is_dir():
            raise ValueError('Existing configured SDK site required')
        self.root = owner.output/'visual_only'
        self.root.mkdir(exist_ok=False)
        self.count = 0
        self.closed = False
        self.shutdown_complete = False
        self.proc = None
        self.stderr = None
        source = self.root/'source.usda'
        if not owner.native['stage'].Flatten().Export(str(source)):
            raise RuntimeError('Could not export a read-only visual source copy')
        self.scene_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        for name in ('home', 'cache', 'config', 'data'):
            (self.root/name).mkdir()
        env = os.environ.copy()
        env.update(HOME=str(self.root/'home'), XDG_CACHE_HOME=str(self.root/'cache'),
                   XDG_CONFIG_HOME=str(self.root/'config'), XDG_DATA_HOME=str(self.root/'data'))
        worker = Path(__file__).with_name('visual_only_v18')/'worker.py'
        if not worker.is_file():
            raise ValueError('Task-local visual-only worker missing')
        self.stderr = (self.root/'worker_stderr.log').open('w')
        try:
            self.proc = subprocess.Popen([sys.executable, '-I', '-S', str(worker),
                '--scene', str(source), '--root', str(self.root), '--sdk-site', str(sdk_site)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr,
                text=True, env=env)
            ready = self._read()
            if ready.get('op') != 'ready' or ready.get('visual_only') is not True:
                raise RuntimeError('Render-only worker did not admit a visual scene')
            if ready.get('cpu_thread_budget') != 8:
                raise ValueError('Render-only child did not declare its approved CPU budget')
            self.paths = list(ready['required_rigid_paths'])
            if len(self.paths) != len(set(self.paths)) or not all(p.startswith('/World/') for p in self.paths):
                raise ValueError('Invalid render-only rigid identity list')
            self.view = owner.native['tensor_sim'].create_rigid_body_view(self.paths)
            if set(self.view.prim_paths) != set(self.paths) or self.view.count != len(self.paths):
                raise ValueError('Every rendered enabled rigid body must have an actual native actor')
            self.paths = list(self.view.prim_paths)
            self.ready = ready
            (self.root/'ready.json').write_text(json.dumps(ready, indent=2)+'\n')
        except BaseException as original:
            cleanup = None
            try:
                self.close()
            except BaseException as exc:
                cleanup = repr(exc)
            (self.root/'startup_failure.json').write_text(json.dumps(dict(
                observation_error=repr(original), cleanup_error=cleanup,
                render_process_returncode=self.proc.returncode if self.proc is not None else None,
                physical_state_substituted=False), indent=2)+'\n')
            raise

    def _read(self):
        line = self.proc.stdout.readline(1024*1024+1)
        if not line or len(line) > 1024*1024:
            raise RuntimeError('Render-only worker closed or exceeded the private protocol')
        value = json.loads(line)
        if value.get('op') == 'error':
            raise RuntimeError('Render-only worker reported an observation failure')
        return value

    def capture(self, owner):
        if self.closed or self.proc.poll() is not None:
            raise RuntimeError('No live owned render-only worker')
        before = owner._snapshot()
        poses = self.view.get_transforms().detach().cpu().numpy().copy()
        if poses.shape != (len(self.paths), 7) or not np.isfinite(poses).all():
            raise ValueError('Missing actual native rigid poses for observation')
        name = f'{self.count:04d}.npz'
        target = self.root/name
        if target.exists():
            raise ValueError('Never reuse a visual snapshot ID')
        arrays = dict(episode_id=owner.episode_id, physics_step=before['physics_step'],
            time_s=owner.session.clock.current_time, rigid_paths=np.asarray(self.paths),
            rigid_pose=poses, cloth_paths=np.asarray([f'/World/TableclothPhysics/Cloth_{cid}' for cid in owner.materials]))
        arrays.update({f'positions_{cid}': before['positions'][owner.native['cloth_ids'] == cid]
                       for cid in owner.materials})
        np.savez_compressed(target, **arrays)
        self.proc.stdin.write(json.dumps(dict(op='render', snapshot=name))+'\n')
        self.proc.stdin.flush()
        row = self._read()
        if (row.get('op') != 'frame' or row.get('visual_only') is not True
                or row.get('episode_id') != owner.episode_id
                or row.get('physics_step') != before['physics_step']
                or row.get('simulated_body_count') != 0
                or row.get('native_snapshot_visual_transform_guard_passed') is not True
                or set(row.get('files', {})) != {'head', 'left', 'right'}):
            raise ValueError('Cross-frame or incomplete visual-only observation')
        encoded, images = {}, {}
        for camera, size in (('head', (1280, 800)), ('left', (848, 480)), ('right', (848, 480))):
            relative = f'frames/{self.count:04d}_{camera}.png'
            record = row['files'][camera]
            if record.get('path') != relative:
                raise ValueError('Unexpected visual-only image path')
            image = self.root/relative
            if image.resolve() != image:
                raise ValueError('Aliased visual-only image')
            raw = image.read_bytes()
            if (not 33 <= len(raw) <= 8*1024*1024 or raw[:8] != b'\x89PNG\r\n\x1a\n'
                    or struct.unpack('>II', raw[16:24]) != size
                    or hashlib.sha256(raw).hexdigest() != record.get('sha256')):
                raise ValueError('Changed or incompatible native-camera observation')
            encoded[camera] = base64.b64encode(raw).decode()
            images[camera] = record['sha256']
        after = owner._snapshot()
        if before['physics_step'] != after['physics_step']:
            raise RuntimeError('Physical owner advanced while awaiting its observation')
        for key in ('positions', 'velocities', 'joints', 'joint_velocities', 'body_pose', 'body_velocity'):
            if not np.array_equal(before[key], after[key]):
                raise RuntimeError('Render-only subprocess changed physical state')
        if not np.array_equal(poses, self.view.get_transforms().detach().cpu().numpy()):
            raise RuntimeError('Native rigid scene moved while physics was frozen')
        if hashlib.sha256((self.root/'source.usda').read_bytes()).hexdigest() != self.scene_sha256:
            raise RuntimeError('Visual source copy was edited')
        with (self.root/'frame_requests.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(capture_index=self.count, episode_id=owner.episode_id,
                physics_step=before['physics_step'], control_step=owner.session.control_index,
                source_snapshot_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                images_sha256=images, physical_state_unchanged=True,
                render_reference_time=row.get('reference_time'),
                renderer_has_no_simulated_bodies=True,
                sdk_empty_bookkeeping_scenes=row.get('sdk_empty_bookkeeping_scenes'),
                parent_render_calls=0))+'\n')
        self.count += 1
        return encoded

    def close(self):
        if self.closed:
            return
        self.closed = True
        error = None
        try:
            if self.proc is not None and self.proc.poll() is None:
                self.proc.stdin.write(json.dumps(dict(op='close'))+'\n')
                self.proc.stdin.flush()
                if self._read().get('op') != 'closing':
                    raise RuntimeError('Render-only close acknowledgement missing')
                self.proc.wait(timeout=30)  # Owned-process cleanup guard, not episode scoring time.
            if self.proc is not None and self.proc.returncode != 0:
                raise RuntimeError('Render-only worker did not exit cleanly')
            self.shutdown_complete = self.proc is not None and self.proc.returncode == 0
        except BaseException as exc:
            error = exc
            if self.proc is not None and self.proc.poll() is None:
                self.proc.terminate()  # Only this exact Popen child, never another job.
                self.proc.wait(timeout=30)
        finally:
            if self.stderr is not None:
                self.stderr.close()
            if self.proc is not None:
                for stream in (self.proc.stdin, self.proc.stdout):
                    if stream is not None:
                        stream.close()
        if error is not None:
            raise error
