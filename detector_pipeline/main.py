import config
import torch
import eye_gaze_extraction
import resnet_rgb, resnet_eyes
import argparse
import LRNet.demo.extract_landmarks, LRNet.demo.classify
import faces_extractor
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    parser = argparse.ArgumentParser(
        description='Extract landmarks sequences from input videos.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-i', '--input_path', type=str, default=config.VIDEO_DIR,
                        help="Input videos path (folder)"
                        )
    parser.add_argument('-o', '--output_path', type=str, default='./LRnet/demo/landmarks/',
                        help="Output landmarks(.txt) path (folder)"
                        )
    parser.add_argument('-v', '--visualize', action='store_true',
                        help="If visualize the extraction results."
                        )
    parser.add_argument('--visualize_path', type=str, default='./LRnet/demo/visualize/',
                        help="Visualize videos path (folder)."
                        )
    parser.add_argument('-l', '--log_file', type=str, default='landmark_logs.txt',
                        help="The log file's name (generated under the /demo by default)."
                        )
    parser.add_argument('--fd', type=str, default='blazeface',
                        choices=['blazeface', 'retinaface'],
                        help="Select the face detector. (blazeface or retinaface)"
                        )
    args = parser.parse_args()
    LRNet.demo.extract_landmarks.main(args)
    lrnet_prediction = LRNet.demo.classify.main()
    print(f"LRNet prediction {lrnet_prediction}")
    faces_extractor.main()
    model_resnet_rgb = resnet_rgb.ResNetLSTM_1(freeze_cnn=True).to(DEVICE)
    model_resnet_rgb.load_state_dict(torch.load(config.WEIGHTS_PATH_RESNET_RGB, map_location=DEVICE, weights_only=False))
    model_resnet_rgb.eval()
    confidence_resnet_rgb = resnet_rgb.predict_video(model_resnet_rgb, config.VIDEO_FRAMES_DIR, device=DEVICE)
    print(f"Full frame resnet prediction {confidence_resnet_rgb}")

    eye_gaze_extraction.process_dataset(config.FRAMES_DIR, config.EYE_OUTPUT_DIR, config.PREDICTOR_PATH)
    model_resnet_eyes = resnet_eyes.ResNetLSTM_2(freeze_cnn=True).to(DEVICE)
    model_resnet_eyes.load_state_dict(torch.load(config.WEIGHTS_PATH_RESNET_EYES, map_location=DEVICE, weights_only=False))
    model_resnet_eyes.eval()
    confidence_resnet_eyes = resnet_rgb.predict_video(model_resnet_eyes, config.EYE_DIR, device=DEVICE)
    print(f"Eyes resnet prediction {confidence_resnet_eyes}")

    mean_score = (lrnet_prediction[0] + confidence_resnet_rgb + confidence_resnet_eyes) / 3

    if mean_score >= 0.5:
        print(f"Video is classified as FAKE")
    else:
        print(f"Video is classified as REAL")
if __name__ == "__main__":
    main()