"""Owned RGB snapshots and bounded background recording. No simulator imports."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
from time import perf_counter

CAMERAS = ('head', 'left', 'right')
WIDTH, HEIGHT = 640, 480
RGB_BYTES = WIDTH * HEIGHT * 3


def check_rgb(images):
    if not isinstance(images, dict) or set(images) != set(CAMERAS):
        raise ValueError('All three current camera images are required')
    if any(not isinstance(images[k], bytes) or len(images[k]) != RGB_BYTES for k in CAMERAS):
        raise ValueError('Expected owned 640x480 RGB bytes, not a renderer buffer view')


def encode_png(rgb):
    from PIL import Image
    if not isinstance(rgb, bytes) or len(rgb) != RGB_BYTES:
        raise ValueError('Invalid RGB snapshot')
    output = BytesIO()
    Image.frombytes('RGB', (WIDTH, HEIGHT), rgb).save(output, format='PNG', compress_level=1)
    return output.getvalue()


def _save_png(path, rgb):
    from PIL import Image
    temporary = path.with_suffix('.png.pending')
    if path.exists():
        raise FileExistsError(path)
    with temporary.open('xb') as stream:
        Image.frombytes('RGB', (WIDTH, HEIGHT), rgb).save(stream, format='PNG', compress_level=1)
    os.replace(temporary, path)  # readers only see complete PNGs
    return hashlib.sha256(rgb).hexdigest()


class ImageRecorder:
    """Retain at most max_pending frames. Backpressure is explicit, never drop.

    Workers receive bytes only: no USD, renderer views, simulation calls, or
    borrowed NumPy buffers cross the thread boundary. Completion is drained
    before the main worker marks the run closed or destroys SimulationApp.
    """
    def __init__(self, output, *, max_pending=8, save=_save_png):
        if type(max_pending) is not int or not 1 <= max_pending <= 32:
            raise ValueError('Invalid image queue capacity')
        self.output = Path(output)
        self.capacity = max_pending
        self.save = save
        self.pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix='arena_record')
        self.pending = deque()
        self.submitted = self.completed = self.high_watermark = 0
        self.backpressure_s = 0.
        self.drain_s = 0.
        self.closed = False
        self.error = None
        self.last_time = None
        self.receipts = (self.output/'image_frames.jsonl').open('x')

    def _finish_first(self):
        index, time_s, futures = self.pending[0]
        try:
            hashes = {k: futures[k].result() for k in CAMERAS}
        except BaseException as exc:
            self.error = exc
            raise RuntimeError('Background image recording failed') from exc
        self.pending.popleft()
        self.receipts.write(json.dumps(dict(frame_index=index, time_s=time_s, rgb_sha256=hashes))+'\n')
        self.receipts.flush()
        self.completed += 1

    def check(self):
        if self.error is not None:
            raise RuntimeError('Background image recording failed') from self.error
        # Detect any failed camera promptly, even before the oldest group ends.
        for _, _, group in self.pending:
            for future in group.values():
                if future.done() and future.exception() is not None:
                    self.error = future.exception()
                    raise RuntimeError('Background image recording failed') from self.error
        while self.pending and all(f.done() for f in self.pending[0][2].values()):
            self._finish_first()

    def submit(self, index, time_s, images):
        import math
        if self.closed or type(index) is not int or index != self.submitted:
            raise ValueError('Closed recorder or missing/reordered camera frame')
        if not math.isfinite(time_s) or (self.last_time is not None and time_s <= self.last_time):
            raise ValueError('Nonmonotonic camera simulation time')
        check_rgb(images)
        self.check()
        if len(self.pending) >= self.capacity:
            started = perf_counter()
            self._finish_first()
            self.backpressure_s += perf_counter()-started
        futures = {k: self.pool.submit(self.save, self.output/f'step{index:04d}_{k}.png', images[k]) for k in CAMERAS}
        self.pending.append((index, time_s, futures))
        self.submitted += 1
        self.last_time = time_s
        self.high_watermark = max(self.high_watermark, len(self.pending))

    def report(self):
        return dict(mode='bounded_background_png', submitted_frames=self.submitted,
                    completed_frames=self.completed, dropped_frames=0,
                    max_pending_frames=self.capacity, queue_high_watermark=self.high_watermark,
                    backpressure_seconds=self.backpressure_s, drain_seconds=self.drain_s,
                    completed=self.closed and self.error is None and self.completed == self.submitted)

    def close(self):
        if self.closed:
            if self.error is not None:
                raise RuntimeError('Image recording did not complete') from self.error
            return
        started = perf_counter()
        try:
            self.check()
            while self.pending:
                self._finish_first()
        finally:
            self.pool.shutdown(wait=True)
            self.receipts.close()
            self.drain_s += perf_counter()-started
            self.closed = True
            with (self.output/'image_recording.json').open('x') as stream:
                json.dump(self.report(), stream, indent=2)
