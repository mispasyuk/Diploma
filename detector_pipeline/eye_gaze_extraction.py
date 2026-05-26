import os
import cv2
import dlib
import numpy as np
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class EyeRegionExtractor:
    
    def __init__(self, predictor_path):
        """
        Args:
            predictor_path: путь к предобученной модели dlib
        """
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        
        self.left_eye_indices = [36, 37, 38, 39, 40, 41]
        self.right_eye_indices = [42, 43, 44, 45, 46, 47]
        
    def extract_eye_regions(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        face = faces[0]
        landmarks = self.predictor(gray, face)
        landmarks_points = np.array([[p.x, p.y] for p in landmarks.parts()])
        
        left_eye_points = landmarks_points[self.left_eye_indices]
        left_eye_region = self._crop_eye_region(frame, left_eye_points)
        
        right_eye_points = landmarks_points[self.right_eye_indices]
        right_eye_region = self._crop_eye_region(frame, right_eye_points)
        
        return {
            'left_eye': left_eye_region,
            'right_eye': right_eye_region,
            'left_eye_landmarks': left_eye_points.tolist(),
            'right_eye_landmarks': right_eye_points.tolist(),
            'all_landmarks': landmarks_points.tolist()
        }
    
    def _crop_eye_region(self, frame, eye_points):

        x_min, y_min = np.min(eye_points, axis=0)
        x_max, y_max = np.max(eye_points, axis=0)
        
        padding_x = int((x_max - x_min) * 0.5)
        padding_y = int((y_max - y_min) * 0.5)
        
        x_min = max(0, x_min - padding_x)
        y_min = max(0, y_min - padding_y)
        x_max = min(frame.shape[1], x_max + padding_x)
        y_max = min(frame.shape[0], y_max + padding_y)
        
        eye_region = frame[y_min:y_max, x_min:x_max]
        
        eye_region = cv2.resize(eye_region, (128, 128))
        
        return eye_region


def extract_frames_features(video_folder_path, extractor, output_dir, max_frames=64):

    video_name = Path(video_folder_path).name
    video_output_dir = Path(output_dir) / video_name
    video_output_dir.mkdir(parents=True, exist_ok=True)
    
    frame_files = sorted(list(Path(video_folder_path).glob("*.jpg")))

    frame_files = frame_files[:max_frames]
    
    features = {
        'video_name': video_name,
        'frames': [],
        'total_frames_processed': len(frame_files)
    }
    
    for frame_idx, frame_path in enumerate(tqdm(frame_files, desc=f"Processing {video_name}", leave=False)):
        frame = cv2.imread(str(frame_path))
        
        eye_data = extractor.extract_eye_regions(frame)
        
        left_eye_path = video_output_dir / f"frame_{frame_idx:04d}_left.jpg"
        right_eye_path = video_output_dir / f"frame_{frame_idx:04d}_right.jpg"
        
        cv2.imwrite(str(left_eye_path), eye_data['left_eye'])
        cv2.imwrite(str(right_eye_path), eye_data['right_eye'])
        
        frame_info = {
            'frame_index': frame_idx,
            'original_frame': frame_path.name,
            'left_eye_landmarks': eye_data['left_eye_landmarks'],
            'right_eye_landmarks': eye_data['right_eye_landmarks'],
            'left_eye_path': str(left_eye_path),
            'right_eye_path': str(right_eye_path)
        }
        
        features['frames'].append(frame_info)

    metadata_path = video_output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(features, f, indent=2)
    
    return features


def process_dataset(frames_root_dir, output_dir, predictor_path):
    """
    Обработка всего датасета с кадрами
    
    Args:
        frames_root_dir: корневая директория с папками кадров
        output_dir: директория для сохранения признаков
        predictor_path: путь к модели dlib
    """
    extractor = EyeRegionExtractor(predictor_path=predictor_path)

    video_folders = [f for f in Path(frames_root_dir).iterdir() if f.is_dir()]
    
    print(f"Found {len(video_folders)} video folders")
    
    all_features = []
    
    for video_folder in tqdm(video_folders, desc="Processing videos"):
        try:
            features = extract_frames_features(
                video_folder_path=str(video_folder),
                extractor=extractor,
                output_dir=output_dir
            )
            all_features.append(features)
                
        except Exception as e:
            print(f"Error processing {video_folder.name}: {e}")
    
    summary_path = Path(output_dir) / "all_features.json"
    with open(summary_path, 'w') as f:
        json.dump(all_features, f, indent=2)

    print(f"Processing complete!")
    print(f"Successfully processed: {len(all_features)} videos")
    print(f"Features saved to: {output_dir}")
    
    return all_features

