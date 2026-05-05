import cv2
import mediapipe as mp
import json
import os
from pathlib import Path

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,     
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
    361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
    176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109
]

def extract_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return [], []
    
    # Resize to 224x224
    image = cv2.resize(image, (224, 224))
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    all_landmarks = []
    contour_landmarks = []

    if results.multi_face_landmarks:
        for face_landmark in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmark.landmark):
                point_data = {
                    'id': idx,
                    'x': lm.x,
                    'y': lm.y,
                    'z': lm.z
                }
                all_landmarks.append(point_data)
                if idx in FACE_OVAL:
                    contour_landmarks.append(point_data)

    return all_landmarks, contour_landmarks

def save_landmarks_json(all_landmarks, contour_landmarks, json_path):
    data = {
        'all_landmarks': all_landmarks,
        'contour_landmarks': contour_landmarks
    }
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def process_folder(input_folder, output_base_dir):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    input_path = Path(input_folder)
    
    for img_file in input_path.rglob('*'):
        if img_file.suffix.lower() not in image_extensions:
            continue
        rel_path = img_file.relative_to("/kaggle/input/your_dataset")

        json_rel = rel_path.with_suffix('.json')
        
        json_full = Path(output_base_dir) / json_rel
        
        all_lm, contour_lm = extract_landmarks(str(img_file))
        save_landmarks_json(all_lm, contour_lm, str(json_full))

if __name__ == "__main__":
    input_folders = []

    output_dir = '/kaggle/working/face_landmarks'

    for folder in input_folders:
        print(f'\n Folder processing: {folder}')
        process_folder(folder, output_dir)

    print('\n All landmarks are saved to', output_dir)