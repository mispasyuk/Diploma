"""
Categories:
1. Lighting variations  : brightness, contrast, gamma, flicker, vignette,
                          color_shift, exposure_burst
2. Video shaking        : camera_shake, rotation_jitter, zoom_pulse, earthquake
3. Poor connection      : compression, packet_loss, bitrate_drop, blocking,
                          interlacing, pixelation, network_noise, horizontal_tearing
4. Realistic noise      : film_grain, motion_blur, lens_distortion,
                          chromatic_aberration

Every augmentation function has the signature:
    fn(frames: np.ndarray, **kwargs) -> np.ndarray
where *frames* is (N, H, W, 3) uint8.
"""

import cv2
import numpy as np
import random
from scipy.signal import butter, filtfilt

from utils import clamp_frames

# 1. LIGHTING VARIATIONS

def augment_brightness(frames: np.ndarray, delta: float = 40.0) -> np.ndarray:
    """
    Uniform brightness shift for the entire video.
    delta > 0 — brighter, delta < 0 — darker.
    """
    return clamp_frames(frames.astype(np.float32) + delta)


def augment_contrast(frames: np.ndarray, alpha: float = 1.3) -> np.ndarray:
    """
    Contrast adjustment: new = alpha * (frame - 128) + 128.
    alpha > 1 — higher contrast, 0 < alpha < 1 — lower.
    """
    out = alpha * (frames.astype(np.float32) - 128.0) + 128.0
    return clamp_frames(out)


def augment_gamma(frames: np.ndarray, gamma: float = 1.5) -> np.ndarray:
    """
    Gamma correction (non-linear brightness curve).
    gamma > 1 — darker shadows, gamma < 1 — brighter shadows.
    """
    lut = ((np.arange(256) / 255.0) ** (1.0 / gamma) * 255.0).astype(np.uint8)
    return lut[frames]


def augment_flicker(frames: np.ndarray,
                    intensity: float = 0.08,
                    freq_hz: float = 0.5) -> np.ndarray:
    """
    Smooth random brightness oscillations over time (flickering light source).
    intensity : amplitude as a fraction of full brightness.
    freq_hz   : approximate oscillation frequency.
    """
    n = len(frames)
    noise = np.random.randn(n)
    b, a = butter(2, freq_hz / 12.5, btype="low")
    smooth = filtfilt(b, a, noise)
    smooth /= np.std(smooth) + 1e-8
    multipliers = 1.0 + intensity * smooth
    out = frames.astype(np.float32)
    for i, m in enumerate(multipliers):
        out[i] *= m
    return clamp_frames(out)


def augment_vignette(frames: np.ndarray, strength: float = 0.6) -> np.ndarray:
    """
    Radial darkening of frame edges (lens vignette effect).
    strength: 0 = no effect, 1 = very dark edges.
    """
    h, w = frames.shape[1:3]
    X, Y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    mask = np.clip(1.0 - strength * (X ** 2 + Y ** 2), 0, 1)[:, :, np.newaxis]
    return clamp_frames(frames.astype(np.float32) * mask[np.newaxis])


def augment_color_shift(frames: np.ndarray,
                        hue_delta: int = 15,
                        sat_scale: float = 1.2) -> np.ndarray:
    """Hue and saturation shift — simulates different lighting conditions."""
    out = []
    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.int32)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_delta) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0, 255)
        out.append(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR))
    return np.stack(out)


def augment_exposure_burst(frames: np.ndarray,
                           prob: float = 0.03,
                           magnitude: float = 60.0) -> np.ndarray:
    """
    Random per-frame brightness spikes (overexposure / underexposure flashes).
    prob : probability of a burst on any given frame.
    """
    out = frames.astype(np.float32).copy()
    for i in range(len(out)):
        if random.random() < prob:
            out[i] += random.choice([-1, 1]) * magnitude * random.uniform(0.5, 1.0)
    return clamp_frames(out)


# 2. VIDEO SHAKING

def _shake_trajectory(n: int, amplitude: float, smoothness: float = 0.3) -> np.ndarray:
    """Returns a smoothed (n, 2) array of (dx, dy) displacements in pixels."""
    raw = np.random.randn(n, 2) * amplitude
    b, a = butter(2, smoothness, btype="low")
    return np.stack([filtfilt(b, a, raw[:, i]) for i in range(2)], axis=1)


def augment_camera_shake(frames: np.ndarray,
                         amplitude: float = 6.0,
                         smoothness: float = 0.25) -> np.ndarray:
    """
    Smooth translational jitter — handheld filming.
    amplitude : max displacement in pixels.
    smoothness: lower = sharper shake.
    """
    h, w = frames.shape[1:3]
    traj = _shake_trajectory(len(frames), amplitude, smoothness)
    out = []
    for frame, (dx, dy) in zip(frames, traj):
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        out.append(cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT))
    return np.stack(out)


def augment_rotation_jitter(frames: np.ndarray,
                             max_angle: float = 1.5) -> np.ndarray:
    """Micro-rotations — camera instability during panning."""
    h, w = frames.shape[1:3]
    raw = np.random.randn(len(frames)) * max_angle
    b, a = butter(2, 0.2, btype="low")
    angles = filtfilt(b, a, raw)
    out = []
    for frame, angle in zip(frames, angles):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        out.append(cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT))
    return np.stack(out)


def augment_zoom_pulse(frames: np.ndarray,
                       scale_range: tuple = (0.97, 1.03)) -> np.ndarray:
    """Slow pulsating zoom — operator breathing / autofocus hunting."""
    h, w = frames.shape[1:3]
    scales = np.random.uniform(*scale_range, size=len(frames))
    b, a = butter(2, 0.1, btype="low")
    scales = filtfilt(b, a, scales)
    out = []
    for frame, s in zip(frames, scales):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, s)
        out.append(cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT))
    return np.stack(out)


def augment_earthquake_shake(frames: np.ndarray,
                              amplitude: float = 15.0) -> np.ndarray:
    """Sharp damped shake — tripod impact or seismic event."""
    h, w = frames.shape[1:3]
    t = np.arange(len(frames))
    decay = np.exp(-t / (len(frames) * 0.15))
    dx = amplitude * decay * np.sin(2 * np.pi * t / 4) * np.random.randn(len(frames))
    dy = amplitude * decay * np.cos(2 * np.pi * t / 3) * np.random.randn(len(frames))
    out = []
    for frame, (x, y) in zip(frames, zip(dx, dy)):
        M = np.float32([[1, 0, x], [0, 1, y]])
        out.append(cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT))
    return np.stack(out)


# 3. POOR CONNECTION / CODEC ARTIFACTS

def augment_compression_artifacts(frames: np.ndarray,
                                   quality: int = 15) -> np.ndarray:
    """
    JPEG macro-block artifacts — heavy codec compression.
    quality: 1 (worst) … 100 (lossless).
    """
    param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    out = []
    for frame in frames:
        _, buf = cv2.imencode(".jpg", frame, param)
        out.append(cv2.imdecode(buf, cv2.IMREAD_COLOR))
    return np.stack(out)


def augment_packet_loss(frames: np.ndarray,
                        loss_prob: float = 0.05,
                        freeze_duration: int = 3) -> np.ndarray:
    """
    Frame freeze — simulates UDP packet loss in a video stream.
    loss_prob      : probability of a freeze starting at each frame.
    freeze_duration: how many consecutive frames are frozen.
    """
    out = frames.copy()
    i = 0
    while i < len(frames):
        if random.random() < loss_prob:
            frozen = frames[i].copy()
            end = min(i + freeze_duration, len(frames))
            out[i:end] = frozen
            i = end
        else:
            i += 1
    return out


def augment_bitrate_drop(frames: np.ndarray,
                          low_quality: int = 5,
                          drop_prob: float = 0.08,
                          drop_duration: int = 8) -> np.ndarray:
    """Short bursts of very heavy compression — network bitrate drop."""
    out = frames.copy()
    param = [int(cv2.IMWRITE_JPEG_QUALITY), low_quality]
    i = 0
    while i < len(frames):
        if random.random() < drop_prob:
            end = min(i + drop_duration, len(frames))
            for j in range(i, end):
                _, buf = cv2.imencode(".jpg", frames[j], param)
                out[j] = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            i = end
        else:
            i += 1
    return out


def augment_blocking_artifacts(frames: np.ndarray,
                                block_size: int = 16,
                                prob: float = 0.02) -> np.ndarray:
    """Random decoder blocks replaced with their mean color."""
    out = frames.copy().astype(np.float32)
    h, w = frames.shape[1:3]
    for i in range(len(out)):
        if random.random() < prob * 10:
            for y in range(0, h, block_size):
                for x in range(0, w, block_size):
                    if random.random() < prob:
                        block = out[i, y:y+block_size, x:x+block_size]
                        out[i, y:y+block_size, x:x+block_size] = block.mean(
                            axis=(0, 1), keepdims=True)
    return clamp_frames(out)


def augment_interlacing(frames: np.ndarray) -> np.ndarray:
    """
    Interlaced scan — analog / legacy TV signal artifact.
    Odd rows come from the previous frame.
    """
    out = frames.copy()
    for i in range(1, len(frames)):
        out[i, 1::2] = frames[i - 1, 1::2]
    return out


def augment_pixelation(frames: np.ndarray, scale: float = 0.15) -> np.ndarray:
    """
    Downscale + nearest-neighbour upscale — simulates very low resolution.
    scale: smaller = larger visible pixels.
    """
    h, w = frames.shape[1:3]
    sh, sw = max(1, int(h * scale)), max(1, int(w * scale))
    out = []
    for frame in frames:
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_LINEAR)
        out.append(cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST))
    return np.stack(out)


def augment_network_noise(frames: np.ndarray, noise_std: float = 20.0) -> np.ndarray:
    """Gaussian noise over every pixel — signal interference during transmission."""
    noise = np.random.normal(0, noise_std, frames.shape).astype(np.float32)
    return clamp_frames(frames.astype(np.float32) + noise)


def augment_horizontal_tearing(frames: np.ndarray,
                                 prob: float = 0.03,
                                 max_offset: int = 20) -> np.ndarray:
    """Horizontal row shift — display sync / scan-line desynchronization."""
    out = frames.copy()
    h = frames.shape[1]
    for i in range(len(out)):
        if random.random() < prob:
            y = random.randint(0, h - 1)
            out[i, y] = np.roll(out[i, y], random.randint(-max_offset, max_offset), axis=0)
    return out

# 4. SENSOR / OPTICAL NOISE

def augment_film_grain(frames: np.ndarray, grain_std: float = 12.0) -> np.ndarray:
    """High-frequency Gaussian noise — analog film grain / sensor noise."""
    grain = np.random.normal(0, grain_std, frames.shape).astype(np.float32)
    return clamp_frames(frames.astype(np.float32) + grain)


def augment_motion_blur(frames: np.ndarray,
                         kernel_size: int = 9,
                         angle: float = 0.0) -> np.ndarray:
    """
    Directional motion blur — long shutter exposure.
    angle: blur direction in degrees (0 = horizontal).
    """
    k = kernel_size
    kernel = np.zeros((k, k))
    cx = k // 2
    rads = np.deg2rad(angle)
    for i in range(k):
        x = int(cx + (i - cx) * np.cos(rads))
        y = int(cx + (i - cx) * np.sin(rads))
        if 0 <= x < k and 0 <= y < k:
            kernel[y, x] = 1
    s = kernel.sum()
    if s > 0:
        kernel /= s
    return np.stack([cv2.filter2D(f, -1, kernel) for f in frames])


def augment_lens_distortion(frames: np.ndarray,
                             k1: float = 0.15,
                             k2: float = 0.05) -> np.ndarray:
    """
    Barrel / pincushion geometric lens distortion.
    k1 > 0 = barrel, k1 < 0 = pincushion.
    """
    h, w = frames.shape[1:3]
    fx = fy = float(max(h, w))
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array([k1, k2, 0, 0, 0], dtype=np.float64)
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K, (w, h), cv2.CV_32FC1)
    return np.stack([
        cv2.remap(f, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        for f in frames
    ])


def augment_chromatic_aberration(frames: np.ndarray, shift: int = 3) -> np.ndarray:
    """Lateral RGB channel offset — chromatic aberration of cheap optics."""
    out = []
    for frame in frames:
        b, g, r = cv2.split(frame)
        h, w = frame.shape[:2]
        r = cv2.warpAffine(r, np.float32([[1, 0,  shift], [0, 1, 0]]), (w, h),
                           borderMode=cv2.BORDER_REFLECT)
        b = cv2.warpAffine(b, np.float32([[1, 0, -shift], [0, 1, 0]]), (w, h),
                           borderMode=cv2.BORDER_REFLECT)
        out.append(cv2.merge([b, g, r]))
    return np.stack(out)


# REGISTRY

AUGMENTATION_REGISTRY: dict = {
    # Lighting
    "brightness":           lambda f: augment_brightness(f, delta=random.uniform(-50, 50)),
    "contrast":             lambda f: augment_contrast(f, alpha=random.uniform(0.7, 1.5)),
    "gamma":                lambda f: augment_gamma(f, gamma=random.uniform(0.6, 2.0)),
    "flicker":              lambda f: augment_flicker(f, intensity=random.uniform(0.05, 0.15)),
    "vignette":             lambda f: augment_vignette(f, strength=random.uniform(0.3, 0.8)),
    "color_shift":          lambda f: augment_color_shift(f, hue_delta=random.randint(-20, 20)),
    "exposure_burst":       lambda f: augment_exposure_burst(f, prob=0.04),
    # Shaking
    "camera_shake":         lambda f: augment_camera_shake(f, amplitude=random.uniform(3, 10)),
    "rotation_jitter":      lambda f: augment_rotation_jitter(f, max_angle=random.uniform(0.5, 2.5)),
    "zoom_pulse":           lambda f: augment_zoom_pulse(f),
    "earthquake":           lambda f: augment_earthquake_shake(f, amplitude=random.uniform(8, 20)),
    # Poor connection
    "compression":          lambda f: augment_compression_artifacts(f, quality=random.randint(5, 25)),
    "packet_loss":          lambda f: augment_packet_loss(f, loss_prob=random.uniform(0.02, 0.08)),
    "bitrate_drop":         lambda f: augment_bitrate_drop(f),
    "blocking":             lambda f: augment_blocking_artifacts(f),
    "interlacing":          lambda f: augment_interlacing(f),
    "pixelation":           lambda f: augment_pixelation(f, scale=random.uniform(0.1, 0.3)),
    "network_noise":        lambda f: augment_network_noise(f, noise_std=random.uniform(10, 30)),
    "horizontal_tearing":   lambda f: augment_horizontal_tearing(f),
    # Sensor / optical noise
    "film_grain":           lambda f: augment_film_grain(f, grain_std=random.uniform(5, 20)),
    "motion_blur":          lambda f: augment_motion_blur(f, kernel_size=random.choice([5, 7, 9])),
    "lens_distortion":      lambda f: augment_lens_distortion(f, k1=random.uniform(0.05, 0.2)),
    "chromatic_aberration": lambda f: augment_chromatic_aberration(f, shift=random.randint(1, 5)),
}

# PRESETS  (combinations)

PRESETS: dict = {
    "handheld":     ["camera_shake", "rotation_jitter", "film_grain", "vignette"],
    "bad_stream":   ["compression", "packet_loss", "bitrate_drop", "network_noise"],
    "old_footage":  ["flicker", "film_grain", "vignette", "interlacing", "gamma"],
    "low_light":    ["brightness", "film_grain", "lens_distortion", "gamma"],
    "random_mild":  random.sample(list(AUGMENTATION_REGISTRY.keys()), 3),
    "random_heavy": random.sample(list(AUGMENTATION_REGISTRY.keys()), 6),
}
