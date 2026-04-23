"""
pipeline.py — memory-efficient video augmentation pipeline.

Frames are never loaded all at once. Instead, the video is read in small
chunks (default 64 frames), each chunk is augmented in-place, and the
result is written immediately. 
"""

import cv2
import numpy as np
import os
import random
import tempfile
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from utils import (
    get_video_meta, probe_video,
    iter_frames_opencv, iter_frames_ffmpeg,
    write_silent_video, mux_audio,
)
from augmentations import (
    AUGMENTATION_REGISTRY, PRESETS,
    augment_brightness, augment_contrast, augment_gamma, augment_flicker,
    augment_vignette, augment_color_shift, augment_exposure_burst,
    augment_camera_shake, augment_rotation_jitter, augment_zoom_pulse,
    augment_earthquake_shake, augment_compression_artifacts,
    augment_packet_loss, augment_bitrate_drop, augment_blocking_artifacts,
    augment_interlacing, augment_pixelation, augment_network_noise,
    augment_horizontal_tearing, augment_film_grain, augment_motion_blur,
    augment_lens_distortion, augment_chromatic_aberration,
)


# PARAMETER SAMPLING
# Pre-sample random values once per video so parameters stay consistent across all chunks of the same video.

def sample_aug_params(aug_names: list) -> dict:
    """
    Returns {name: fn(frames) -> frames} with fixed random parameters.
    Each augmentation draws its hyper-parameters once here, not per chunk.
    """
    p = {}
    for name in aug_names:
        if name == "brightness":
            d = random.uniform(-50, 50)
            p[name] = lambda f, _d=d: augment_brightness(f, delta=_d)
        elif name == "contrast":
            a = random.uniform(0.7, 1.5)
            p[name] = lambda f, _a=a: augment_contrast(f, alpha=_a)
        elif name == "gamma":
            g = random.uniform(0.6, 2.0)
            p[name] = lambda f, _g=g: augment_gamma(f, gamma=_g)
        elif name == "flicker":
            i = random.uniform(0.05, 0.15)
            p[name] = lambda f, _i=i: augment_flicker(f, intensity=_i)
        elif name == "vignette":
            s = random.uniform(0.3, 0.8)
            p[name] = lambda f, _s=s: augment_vignette(f, strength=_s)
        elif name == "color_shift":
            h = random.randint(-20, 20)
            p[name] = lambda f, _h=h: augment_color_shift(f, hue_delta=_h)
        elif name == "exposure_burst":
            p[name] = lambda f: augment_exposure_burst(f, prob=0.04)
        elif name == "camera_shake":
            a = random.uniform(3, 10)
            p[name] = lambda f, _a=a: augment_camera_shake(f, amplitude=_a)
        elif name == "rotation_jitter":
            m = random.uniform(0.5, 2.5)
            p[name] = lambda f, _m=m: augment_rotation_jitter(f, max_angle=_m)
        elif name == "zoom_pulse":
            p[name] = lambda f: augment_zoom_pulse(f)
        elif name == "earthquake":
            a = random.uniform(8, 20)
            p[name] = lambda f, _a=a: augment_earthquake_shake(f, amplitude=_a)
        elif name == "compression":
            q = random.randint(5, 25)
            p[name] = lambda f, _q=q: augment_compression_artifacts(f, quality=_q)
        elif name == "packet_loss":
            lp = random.uniform(0.02, 0.08)
            p[name] = lambda f, _l=lp: augment_packet_loss(f, loss_prob=_l)
        elif name == "bitrate_drop":
            p[name] = lambda f: augment_bitrate_drop(f)
        elif name == "blocking":
            p[name] = lambda f: augment_blocking_artifacts(f)
        elif name == "interlacing":
            p[name] = lambda f: augment_interlacing(f)
        elif name == "pixelation":
            s = random.uniform(0.1, 0.3)
            p[name] = lambda f, _s=s: augment_pixelation(f, scale=_s)
        elif name == "network_noise":
            n = random.uniform(10, 30)
            p[name] = lambda f, _n=n: augment_network_noise(f, noise_std=_n)
        elif name == "horizontal_tearing":
            p[name] = lambda f: augment_horizontal_tearing(f)
        elif name == "film_grain":
            g = random.uniform(5, 20)
            p[name] = lambda f, _g=g: augment_film_grain(f, grain_std=_g)
        elif name == "motion_blur":
            k = random.choice([5, 7, 9])
            p[name] = lambda f, _k=k: augment_motion_blur(f, kernel_size=_k)
        elif name == "lens_distortion":
            k1 = random.uniform(0.05, 0.2)
            p[name] = lambda f, _k=k1: augment_lens_distortion(f, k1=_k)
        elif name == "chromatic_aberration":
            s = random.randint(1, 5)
            p[name] = lambda f, _s=s: augment_chromatic_aberration(f, shift=_s)
        elif name in AUGMENTATION_REGISTRY:
            p[name] = AUGMENTATION_REGISTRY[name]
    return p


# CHUNK-STREAMING CORE

def _stream_and_augment(src: str, aug_fns: dict,
                         fps: float, w: int, h: int,
                         backend: str, chunk_size: int,
                         silent_dst: str,
                         n_frames_total: Optional[int]):
    """
    Reads *src* frame-by-frame, processes in chunks, writes to *silent_dst*.
    Never holds more than *chunk_size* decoded frames in RAM at once.
    """
    os.makedirs(os.path.dirname(silent_dst) or ".", exist_ok=True)
    writer = cv2.VideoWriter(
        silent_dst,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (w, h),
    )

    frame_iter = (iter_frames_opencv(src) if backend == "opencv"
                  else iter_frames_ffmpeg(src, w, h))

    pbar = tqdm(total=n_frames_total, desc="Augmenting", unit="frame")
    chunk: list = []
    prev_frame = None  # carry last frame across chunks for interlacing

    def _flush(chunk: list):
        nonlocal prev_frame
        if not chunk:
            return
        arr = np.stack(chunk)

        # interlacing is stateful: the first frame's odd rows come from the
        # last frame of the *previous* chunk.
        if "interlacing" in aug_fns:
            if prev_frame is not None:
                arr_ext = np.concatenate([prev_frame[np.newaxis], arr], axis=0)
                arr_ext = aug_fns["interlacing"](arr_ext)
                arr = arr_ext[1:]
            else:
                arr = aug_fns["interlacing"](arr)

        for name, fn in aug_fns.items():
            if name != "interlacing":
                arr = fn(arr)

        prev_frame = arr[-1].copy()
        for frame in arr:
            writer.write(np.clip(frame, 0, 255).astype(np.uint8))
        pbar.update(len(chunk))

    for _, frame in frame_iter:
        chunk.append(frame)
        if len(chunk) >= chunk_size:
            _flush(chunk)
            chunk = []

    _flush(chunk)  # tail frames
    pbar.close()
    writer.release()



def apply_augmentations(frames: np.ndarray,
                         augmentations: list,
                         verbose: bool = True) -> np.ndarray:
    """
    Applies augmentations to an in-memory (N, H, W, 3) uint8 array.
    Convenient for small clips or unit tests.
    For large videos use process_video() to avoid loading everything into RAM.
    """
    from tqdm import tqdm as _tqdm
    result = frames.copy()
    it = _tqdm(augmentations, desc="Augmenting") if verbose else augmentations
    for name in it:
        if name not in AUGMENTATION_REGISTRY:
            print(f"  [!] Unknown augmentation skipped: {name}")
            continue
        result = AUGMENTATION_REGISTRY[name](result)
    return result


def process_video(input_path: str,
                  output_path: str,
                  augmentations: Optional[list] = None,
                  preset: Optional[str] = None,
                  chunk_size: int = 64):
    """
    Augments a single video with low, fixed memory usage.

    Frames are decoded and written chunk by chunk (chunk_size frames at a time).
    Original audio is preserved automatically via ffmpeg muxing.

    Parameters:
    input_path   : path to the source video
    output_path  : path for the augmented output
    augmentations: list of augmentation names (see AUGMENTATION_REGISTRY keys)
    preset       : preset name (see PRESETS) — alternative to augmentations=
    chunk_size   : frames per processing chunk (lower = less RAM)
    """
    if preset:
        augs = list(PRESETS.get(preset, []))
    elif augmentations:
        augs = list(augmentations)
    else:
        raise ValueError("Provide augmentations= or preset=.")

    unknown = [a for a in augs if a not in AUGMENTATION_REGISTRY]
    if unknown:
        print(f"  [WARN] Skipping unknown augmentations: {unknown}")
        augs = [a for a in augs if a in AUGMENTATION_REGISTRY]

    fps, w, h, backend = get_video_meta(input_path)
    info = probe_video(input_path)
    n_total = info[3] if info else None

    mb = chunk_size * h * w * 3 / 1024 ** 2
    print(f"\n  Input  : {input_path}")
    print(f"  Output : {output_path}")
    print(f"  Video  : {w}×{h}  {fps:.2f} fps  backend={backend}"
          + (f"  ~{n_total} frames" if n_total else ""))
    print(f"  Augs   : {augs}")

    aug_fns = sample_aug_params(augs)

    suffix = Path(output_path).suffix or ".mp4"
    tmp_fd, tmp_silent = tempfile.mkstemp(suffix=suffix)
    os.close(tmp_fd)

    try:
        _stream_and_augment(
            src=input_path, aug_fns=aug_fns,
            fps=fps, w=w, h=h,
            backend=backend, chunk_size=chunk_size,
            silent_dst=tmp_silent,
            n_frames_total=n_total,
        )
        mux_audio(input_path, tmp_silent, output_path)
    except Exception:
        if os.path.exists(tmp_silent):
            os.remove(tmp_silent)
        raise

    print(f"  Saved  : {output_path}\n")


def batch_process(input_dir: str,
                  output_dir: str,
                  augmentations: Optional[list] = None,
                  preset: Optional[str] = None,
                  chunk_size: int = 64,
                  extensions: tuple = (".mp4", ".avi", ".mov", ".mkv")):
    """
    Runs process_video on every video in *input_dir*.
    Output files are named  <original_stem>_aug<ext>.
    Errors on individual files are caught and logged without stopping the batch.
    """
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = [f for f in sorted(input_dir.iterdir())
             if f.suffix.lower() in extensions]
    print(f"Found {len(files)} video(s) in {input_dir}")

    for vf in files:
        out = output_dir / (vf.stem + "_aug" + vf.suffix)
        try:
            process_video(str(vf), str(out),
                          augmentations=augmentations,
                          preset=preset,
                          chunk_size=chunk_size)
        except Exception as e:
            print(f"  [ERR] {vf.name}: {e}")
