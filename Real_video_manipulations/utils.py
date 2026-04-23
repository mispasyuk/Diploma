"""

Covers:
  - probing video metadata (fps, resolution, frame count) via ffprobe
  - iterating frames one-by-one via OpenCV or ffmpeg pipe
  - writing frames back to disk
  - muxing original audio into augmented video via ffmpeg
"""

import cv2
import numpy as np
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


def probe_video(src: str):
    """
    Returns (fps, width, height, n_frames) via ffprobe.
    Returns None if ffprobe is unavailable or the file cannot be probed.
    """
    if shutil.which("ffprobe") is None:
        return None
    result = subprocess.run(
        ["ffprobe", "-v", "error",
         "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
         "-of", "csv=p=0", src],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        return None
    try:
        parts = result.stdout.decode().strip().split(",")
        w, h = int(parts[0]), int(parts[1])
        num, den = map(int, parts[2].split("/"))
        fps = num / den
        n_frames = int(parts[3]) if len(parts) > 3 and parts[3].strip().isdigit() else None
        return fps, w, h, n_frames
    except Exception:
        return None


def get_video_meta(src: str):
    """
    Returns (fps, width, height, backend) where backend is 'opencv' or 'ffmpeg'.
    Tries OpenCV first; falls back to ffmpeg for codecs OpenCV cannot handle
    (H.265/HEVC, AV1, ProRes, …).
    Raises FileNotFoundError or ValueError with diagnostics on failure.
    """
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"Video file not found: {src!r}\n"
            f"  Working directory: {os.getcwd()}\n"
            f"  Check that the path is correct and the file is uploaded."
        )

    # Attempt 1: OpenCV — read one test frame to confirm codec support
    cap = cv2.VideoCapture(src)
    fps_cv = cap.get(cv2.CAP_PROP_FPS) or 0.0
    w_cv   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_cv   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, _  = cap.read()
    cap.release()

    if ok and w_cv > 0 and h_cv > 0:
        return fps_cv or 25.0, w_cv, h_cv, "opencv"

    # Attempt 2: ffprobe / ffmpeg
    print(f"  [INFO] OpenCV cannot decode {src!r} — trying ffmpeg fallback.")
    info = probe_video(src)
    if info is not None:
        fps_ff, w_ff, h_ff, _ = info
        return fps_ff, w_ff, h_ff, "ffmpeg"

    # Both failed — collect diagnostics
    diag = [f"Cannot read video: {src!r}"]
    if shutil.which("ffprobe"):
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-of", "json", src],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        diag.append("ffprobe output:\n" + probe.stdout.decode(errors="replace"))
    else:
        diag.append(
            "Tip: install ffmpeg for automatic codec fallback:\n"
            "  sudo apt install ffmpeg   # Linux\n"
            "  brew install ffmpeg       # macOS\n"
            "  winget install ffmpeg     # Windows"
        )
    raise ValueError("\n".join(diag))


# FRAME ITERATORS  (memory-efficient: one frame at a time)

def iter_frames_opencv(src: str):
    """Yields (index, frame_bgr) one frame at a time via OpenCV."""
    cap = cv2.VideoCapture(src)
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield i, frame
        i += 1
    cap.release()


def iter_frames_ffmpeg(src: str, w: int, h: int):
    """
    Yields (index, frame_bgr) one frame at a time via an ffmpeg pipe.
    Only one frame worth of bytes is buffered at a time — O(1) RAM.
    Handles H.265/HEVC, AV1, ProRes and other codecs unsupported by OpenCV.
    """
    frame_bytes = w * h * 3
    proc = subprocess.Popen(
        ["ffmpeg", "-i", src, "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-loglevel", "error", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    i = 0
    while True:
        raw = proc.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()
        yield i, frame
        i += 1
    proc.stdout.close()
    proc.wait()


def read_video(src: str):
    """
    Reads ALL frames into a numpy array (N, H, W, 3).

    For large / high-resolution videos this loads gigabytes into RAM.
    Use pipeline.process_video() instead — it streams chunks and uses
    a small, fixed amount of memory regardless of video length.
    """
    fps, w, h, backend = get_video_meta(src)
    it = iter_frames_opencv(src) if backend == "opencv" else iter_frames_ffmpeg(src, w, h)
    frames = [frame for _, frame in it]
    if not frames:
        raise ValueError(f"No frames decoded from: {src!r}")
    return np.stack(frames), fps, (w, h)


# WRITING

def write_silent_video(frames_iter, dst: str, fps: float, size: tuple):
    """
    Writes frames to a video file (no audio — OpenCV limitation).
    *frames_iter* may be a numpy array or any iterable of (H,W,3) uint8 frames.
    """
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(dst, fourcc, fps, size)
    for frame in frames_iter:
        writer.write(np.clip(frame, 0, 255).astype(np.uint8))
    writer.release()


# AUDIO MUXING

def has_audio_stream(src: str) -> bool:
    """Returns True if the file contains at least one audio stream."""
    result = subprocess.run(
        ["ffmpeg", "-i", src],
        stderr=subprocess.PIPE, stdout=subprocess.DEVNULL,
    )
    return b"Audio" in result.stderr


def mux_audio(original: str, silent_video: str, dst: str) -> bool:
    """
    Copies the audio track from *original* into *silent_video* and writes
    the combined result to *dst*.

    - Video stream is copied bit-for-bit from *silent_video* (no re-encode).
    - Audio stream is copied from *original* with -c:a copy (lossless).
      Falls back to AAC transcoding if the container requires it.
    - -shortest trims to the shorter stream.

    Returns True on success.  If ffmpeg is absent or the source has no audio,
    *silent_video* is renamed to *dst* and False is returned.
    """
    if shutil.which("ffmpeg") is None:
        print(
            "  [WARN] ffmpeg not found — audio will not be preserved.\n"
            "  Install: sudo apt install ffmpeg  /  brew install ffmpeg"
        )
        if os.path.abspath(silent_video) != os.path.abspath(dst):
            os.replace(silent_video, dst)
        return False

    if not has_audio_stream(original):
        print("  [INFO] Source has no audio stream — skipping mux.")
        if os.path.abspath(silent_video) != os.path.abspath(dst):
            os.replace(silent_video, dst)
        return False

    suffix = Path(dst).suffix or ".mp4"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(tmp_fd)

    cmd = [
        "ffmpeg", "-y",
        "-i", silent_video,
        "-i", original,
        "-map", "0:v:0",
        "-map", "1:a?",
        "-c:v", "copy",
        "-c:a", "copy",
        "-shortest",
        tmp_path,
    ]
    result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL)

    if result.returncode != 0:
        print("  [INFO] Audio copy failed — retrying with AAC transcode.")
        cmd_aac = cmd.copy()
        cmd_aac[cmd_aac.index("-c:a") + 1] = "aac"
        result = subprocess.run(cmd_aac, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL)

    if result.returncode != 0:
        print(f"  [ERR] ffmpeg mux failed:\n{result.stderr.decode(errors='replace')}")
        os.remove(tmp_path)
        if os.path.abspath(silent_video) != os.path.abspath(dst):
            os.replace(silent_video, dst)
        return False

    os.replace(tmp_path, dst)
    if os.path.exists(silent_video) and os.path.abspath(silent_video) != os.path.abspath(dst):
        os.remove(silent_video)
    return True


def write_video(frames: np.ndarray, dst: str, fps: float, size: tuple,
                original: Optional[str] = None):
    """
    Writes augmented frames to *dst*.
    If *original* is provided and ffmpeg is available, the original audio
    track is muxed back into the output automatically.
    """
    if original is not None:
        suffix = Path(dst).suffix or ".mp4"
        tmp_fd, tmp_silent = tempfile.mkstemp(suffix=suffix)
        os.close(tmp_fd)
        write_silent_video(frames, tmp_silent, fps, size)
        mux_audio(original, tmp_silent, dst)
    else:
        write_silent_video(frames, dst, fps, size)


def clamp_frames(frames: np.ndarray) -> np.ndarray:
    """Clips pixel values to [0, 255] and casts to uint8."""
    return np.clip(frames, 0, 255).astype(np.uint8)
