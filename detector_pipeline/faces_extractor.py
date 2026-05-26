import cv2
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from config import EXTRACTOR_PATH, VIDEO_FRAMES_DIR, VIDEO_DIR

VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.mpeg')


def collect_videos(input_dir):
    """Собирает список путей ко всем видеофайлам."""
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise NotADirectoryError(f"{input_dir} не является папкой")
    
    return [p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]


def find_continuous_face_window(cap, detector, fps, total_frames):
    """
    Ищет непрерывный интервал длиной 2 секунды, в котором на каждом кадре есть лицо.
    Возвращает (start_frame, end_frame) или None.
    """
    window_frames = max(1, int(round(fps * 2)))   # 2 секунды в кадрах
    if total_frames < window_frames:
        return None

    step_frames = max(1, int(round(fps * 0.5)))

    for start in range(0, total_frames - window_frames + 1, step_frames):
        all_faces = True
        for offset in range(window_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start + offset)
            ret, frame = cap.read()
            if not ret:
                all_faces = False
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            detector.setInputSize((w, h))
            _, faces = detector.detect(frame_rgb)
            if faces is None or len(faces) == 0:
                all_faces = False
                break
        if all_faces:
            return start, start + window_frames
    return None


def extract_face_from_frame(frame, detector):
    """Возвращает кроп лица (BGR) или None, если лицо не найдено."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame_rgb)
    if faces is None or len(faces) == 0:
        return None

    best_idx = np.argmax([f[2] * f[3] for f in faces])
    x, y, w_, h_ = faces[best_idx][:4].astype(np.int32)

    x1 = max(x - 50, 0)
    y1 = max(y - 50, 0)
    x2 = min(x + w_ + 50, w)
    y2 = min(y + h_ + 50, h)

    face_crop = frame_rgb[y1:y2, x1:x2]
    face_crop = cv2.resize(face_crop, (256, 256))
    return cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)


def process_video(video_path):
    """Обрабатывает одно видео: ищет подходящие 2 секунды и сохраняет 64 кадра."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Ошибка открытия видео: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or total_frames == 0:
        print(f"  Некорректное видео: fps={fps}, кадров={total_frames}")
        cap.release()
        return 0

    detector = cv2.FaceDetectorYN.create(
        EXTRACTOR_PATH, "",
        (320, 320),
        0.5, 0.4, 2,
    )

    window = find_continuous_face_window(cap, detector, fps, total_frames)
    if window is None:
        print(f"  Не найдено окно 2 секунды с лицом на всех кадрах. Видео пропущено.")
        cap.release()
        return 0

    start_frame, end_frame = window

    sample_indices = np.linspace(start_frame, end_frame - 1, 64).astype(int)

    save_dir = Path(VIDEO_FRAMES_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for i, idx in enumerate(sample_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"  Ошибка чтения кадра {idx}, пропуск.")
            continue

        face_crop = extract_face_from_frame(frame, detector)
        if face_crop is None:
            face_crop = frame
        out_path = save_dir / f"frame_{i+1:04d}.jpg"
        cv2.imwrite(str(out_path), face_crop)
        saved += 1

    cap.release()
    return saved


def main():
    videos = collect_videos(VIDEO_DIR)
    if not videos:
        print("Видеофайлы не найдены.")
        return

    total_faces = 0
    for video_path in tqdm(videos):
        processed = process_video(video_path)
        total_faces += processed


if __name__ == "__main__":
    main()