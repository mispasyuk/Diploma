# Video Augmentation Pipeline

A tool for augmenting real videos. Applies realistic degradations — lighting changes, camera shake, compression artifacts, sensor noise — while preserving the original audio track.

---

## File structure

All four files must be placed in the same directory.

```
utils.py           — video I/O: frame reading, writing, audio muxing
augmentations.py   — 22 augmentation functions + registry + presets
pipeline.py        — memory-efficient chunk-streaming processor
run.py             — CLI entry point (run this file)
```

---

## Installation

**Python packages**
```bash
pip install opencv-python numpy scipy tqdm
```

**ffmpeg** — required for audio preservation and H.265 / AV1 decoding
```bash
# Linux
sudo apt install ffmpeg

# Windows
winget install ffmpeg
```

---

## Quick start

```bash
# Apply a preset to one video
python run.py input.mp4 output.mp4 --preset handheld

# Apply specific augmentations
python run.py input.mp4 output.mp4 --augs flicker camera_shake compression film_grain

# Process an entire directory
python run.py data/real/ data/augmented/ --preset bad_stream --batch

# Reduce memory usage (lower chunk = less RAM)
python run.py input.mp4 output.mp4 --preset handheld --chunk 16

# See all augmentation names and presets
python run.py --list
```

### Google Colab
```python
# Upload your video, then run in a cell:
!python run.py /content/video.mp4 /content/video_aug.mp4 --preset handheld
```

---

## CLI reference

```
python run.py INPUT OUTPUT [options]
```

| Argument | Description |
|---|---|
| `INPUT` | Source video file or directory (with `--batch`) |
| `OUTPUT` | Output file or directory (with `--batch`) |
| `--preset NAME` | Use a built-in preset (see list below) |
| `--augs AUG ...` | One or more augmentation names, applied in order |
| `--batch` | Process all videos in INPUT directory |
| `--chunk N` | Frames per processing chunk. Default: `64`. Lower = less RAM |
| `--list` | Print all available augmentations and presets, then exit |
| `--help` | Show help message |

`--preset` and `--augs` are mutually exclusive — use one or the other.

---

## Augmentations

### Lighting

| Name | Effect |
|---|---|
| `brightness` | Uniform brightness shift |
| `contrast` | Linear contrast scaling around mid-grey |
| `gamma` | Non-linear brightness curve |
| `flicker` | Smooth random brightness oscillations over time |
| `vignette` | Radial darkening of frame edges |
| `color_shift` | Hue and saturation shift |
| `exposure_burst` | Random per-frame overexposure / underexposure flashes |

### Camera shake

| Name | Effect |
|---|---|
| `camera_shake` | Smooth translational jitter — handheld filming |
| `rotation_jitter` | Micro-rotations — camera instability during panning |
| `zoom_pulse` | Slow pulsating zoom — operator breathing / autofocus |
| `earthquake` | Sharp damped shake — tripod impact or seismic event |

### Poor connection / codec artifacts

| Name | Effect |
|---|---|
| `compression` | JPEG macro-block artifacts (heavy codec compression) |
| `packet_loss` | Frame freeze — UDP packet loss in a video stream |
| `bitrate_drop` | Short bursts of very heavy compression |
| `blocking` | Random decoder blocks replaced with mean color |
| `interlacing` | Odd/even row mixing from adjacent frames (legacy TV) |
| `pixelation` | Downscale + nearest-neighbour upscale |
| `network_noise` | Gaussian noise — signal interference |
| `horizontal_tearing` | Row-level horizontal shift — sync desynchronization |

### Sensor / optics

| Name | Effect |
|---|---|
| `film_grain` | High-frequency Gaussian noise — film / sensor grain |
| `motion_blur` | Directional motion smear — long shutter |
| `lens_distortion` | Barrel / pincushion geometric warp |
| `chromatic_aberration` | Lateral RGB channel offset |

---

## Presets

| Preset | Augmentations applied |
|---|---|
| `handheld` | `camera_shake`, `rotation_jitter`, `film_grain`, `vignette` |
| `bad_stream` | `compression`, `packet_loss`, `bitrate_drop`, `network_noise` |
| `old_footage` | `flicker`, `film_grain`, `vignette`, `interlacing`, `gamma` |
| `low_light` | `brightness`, `film_grain`, `lens_distortion`, `gamma` |
| `random_mild` | 3 randomly picked augmentations |
| `random_heavy` | 6 randomly picked augmentations |

---

## Memory usage

The pipeline **never loads the full video into RAM**. Frames are decoded, augmented, and written in chunks. Peak memory:

```
chunk_size x height x width x 3 bytes
```

| Resolution | chunk=64 (default) | chunk=16 |
|---|---|---|
| 720p | ~100 MB | ~25 MB |
| 1080p | ~400 MB | ~100 MB |
| 4K | ~1.5 GB | ~380 MB |

Reduce `--chunk` if you run out of memory. The result is identical regardless of chunk size.

---

## Audio preservation

OpenCV cannot write audio. The pipeline works around this automatically:

1. Augmented frames are written to a **temporary silent file** via OpenCV.
2. `ffmpeg` copies the original audio track into the final output (`-c:a copy`, no re-encoding).
3. The temporary file is deleted.

If `ffmpeg` is not installed, a warning is printed and the output is saved without audio. If the source video has no audio stream, the mux step is skipped.

---

## Using the modules in Python

The files can also be imported directly without the CLI.

```python
from pipeline import process_video, batch_process

# Single video
process_video("input.mp4", "output.mp4", preset="handheld")

# Single video — custom augmentations and low RAM mode
process_video("input.mp4", "output.mp4",
              augmentations=["flicker", "compression", "film_grain"],
              chunk_size=16)

# Whole directory
batch_process("data/real/", "data/augmented/", preset="bad_stream")
```

