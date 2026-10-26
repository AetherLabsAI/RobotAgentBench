"""Image layer — decode the real-robot camera streams and align them to the
proprio timeline, one mp4 per camera per episode (LeRobot-style video layout).

Streams per episode (from meta_info.json camera_list):
  head_color            H.265  (primary VLA cam; .txt gives per-frame ts)
  hand_left, hand_right JPG    (color/<ts>.jpg per frame; ts = filename)
  head_*_fisheye        H.265  (secondary)
  head_depth            zstd   (secondary; .txt gives ts + byte offset)

We resample each stream onto the proprio grid `ts` (from convert_episode) by
PRECEDING-HOLD, streaming (O(1) memory — never holds a whole video), resizing on
decode, and re-encode to mp4 at a fixed fps. Frame i of the mp4 then corresponds
to state[i]/action[i] in the episode .npz.

Requires PyAV (`av`, bundles ffmpeg) + opencv (`cv2`). No system ffmpeg needed.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import av
import cv2
import numpy as np

# GO-1's 3-cam obs; extend with fisheye/depth as needed.
PRIMARY_CAMS = ("head_color", "hand_left", "hand_right")


# -- per-stream frame iterators (ts_ns, bgr_frame) ------------------------
def _h265_frames(cam_dir: Path, cam: str, size) -> Iterator[tuple[int, np.ndarray]]:
    txt = cam_dir / f"{cam}.txt"
    ts_list = [int(l.split()[0]) for l in txt.read_text().splitlines() if l.strip()]
    with av.open(str(cam_dir / f"{cam}.h265")) as cont:
        # No B-frames in these captures (.txt shows only I/P), so decode order ==
        # capture order == ts_list order.
        for i, fr in enumerate(cont.decode(video=0)):
            if i >= len(ts_list):
                break
            img = fr.to_ndarray(format="bgr24")
            yield ts_list[i], (cv2.resize(img, size) if size else img)


def _jpg_frames(color_dir: Path, size) -> Iterator[tuple[int, np.ndarray]]:
    for p in sorted(color_dir.glob("*.jpg"), key=lambda q: int(q.stem)):
        img = cv2.imread(str(p))
        if img is None:
            continue
        yield int(p.stem), (cv2.resize(img, size) if size else img)


def camera_frames(episode_dir: Path, cam: str, size=None) -> Iterator[tuple[int, np.ndarray]]:
    """Unified (ts_ns, bgr) iterator for a camera, dispatching by stream type."""
    cam_root = episode_dir / "camera"
    if (cam_root / cam / f"{cam}.h265").exists():
        yield from _h265_frames(cam_root / cam, cam, size)
    elif (cam_root / cam / "color").is_dir():
        yield from _jpg_frames(cam_root / cam / "color", size)
    else:
        return


# -- streaming preceding-hold resampler -----------------------------------
def resample_to_grid(src: Iterator[tuple[int, np.ndarray]], grid_ts: np.ndarray) -> Iterator[np.ndarray]:
    """Emit exactly len(grid_ts) frames: the latest source frame at/-before each grid ts.
    Before the first source frame, holds the first frame. O(1) memory."""
    it = iter(src)
    cur = nxt = next(it, None)
    for gt in grid_ts:
        while nxt is not None and nxt[0] <= int(gt):
            cur = nxt
            nxt = next(it, None)
        yield (cur or nxt)[1] if (cur or nxt) is not None else None


# -- mp4 encode -----------------------------------------------------------
def write_mp4(frames: Iterator[np.ndarray], out_path: Path, fps: float, size) -> int:
    """Encode a stream of BGR frames to immediately-decodable H.264 MP4.

    Reproduce videos are commonly opened at exactly ``t=0``.  Libx264's
    default B-frame reordering gives the first packet a negative DTS, which
    makes some players display a black decoder-preroll frame before presenting
    the real frame zero.  Keep presentation and decode order identical and put
    the MP4 index before the media payload so the first image is available as
    soon as playback starts.  ``REAL2SIM_H264_ENCODER=h264_nvenc`` opts a
    production run into NVIDIA hardware encoding; software H.264 remains the
    default for portability.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size if size else (None, None)
    cont = av.open(
        str(out_path),
        mode="w",
        options={"movflags": "+faststart"},
    )
    stream = None
    n = 0
    try:
        for img in frames:
            if img is None:
                if stream is None:
                    continue
                img = np.zeros((stream.height, stream.width, 3), np.uint8)
            if stream is None:
                h, w = img.shape[:2]
                encoder = os.environ.get(
                    "REAL2SIM_H264_ENCODER", "h264").strip()
                if encoder not in {"h264", "h264_nvenc"}:
                    raise ValueError(
                        "REAL2SIM_H264_ENCODER must be h264 or h264_nvenc, "
                        f"got {encoder!r}")
                options = (
                    {
                        "preset": "p5",
                        "tune": "hq",
                        "rc": "vbr",
                        "cq": "18",
                        "profile": "high",
                        "bf": "0",
                    }
                    if encoder == "h264_nvenc" else None
                )
                stream = cont.add_stream(
                    encoder, rate=int(round(fps)) or 1, options=options)
                stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
                stream.codec_context.max_b_frames = 0
            frame = av.VideoFrame.from_ndarray(img, format="bgr24")
            for pkt in stream.encode(frame):
                cont.mux(pkt)
            n += 1
        if stream is not None:
            for pkt in stream.encode():          # flush
                cont.mux(pkt)
    finally:
        cont.close()
    return n


def write_episode_videos(episode_dir: Path, out_dir: Path, grid_ts: np.ndarray, *,
                         cams=PRIMARY_CAMS, size=(320, 256), fps: float = 30.0) -> dict[str, int]:
    """Write one aligned mp4 per camera (len == len(grid_ts)). Returns {cam: frames}."""
    written: dict[str, int] = {}
    for cam in cams:
        frames = resample_to_grid(camera_frames(episode_dir, cam, size), grid_ts)
        n = write_mp4(frames, out_dir / f"{cam}.mp4", fps, size)
        if n:
            written[cam] = n
    return written
