import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import models, transforms
import re
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
class ResNetLSTM_1(nn.Module):
    def __init__(self, lstm_hidden=256, lstm_layers=2, dropout=0.5, freeze_cnn=True):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.cnn_out_dim = 2048
        
        if freeze_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False
                
        self.lstm = nn.LSTM(
            input_size=self.cnn_out_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2) 
        )

    def forward(self, x):
        batch_size, frames, C, H, W = x.shape
        cnn_out = []
        for t in range(frames):
            frame = x[:, t, :, :, :]          
            feat = self.cnn(frame)            
            feat = feat.view(batch_size, -1)
            cnn_out.append(feat)
            
        cnn_seq = torch.stack(cnn_out, dim=1)

        lstm_out, _ = self.lstm(cnn_seq)      
        last_out = lstm_out[:, -1, :]        
        logits = self.classifier(last_out)
        return logits


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
NUM_FRAMES    = 16  
FRAME_SIZE    = 224 

def load_video_frames(video_dir, num_frames=NUM_FRAMES, frame_size=FRAME_SIZE):
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.jpeg', '.png', '.JPEG', '.PNG')
    frame_files = sorted([
        f for f in os.listdir(video_dir) 
        if os.path.isfile(os.path.join(video_dir, f)) and f.lower().endswith(valid_exts)
    ], key=lambda f: [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', f)])
    
    if not frame_files:
        raise ValueError(f"В директории {video_dir} не найдено кадров.")
        
    if len(frame_files) >= num_frames:
        indices = np.linspace(0, len(frame_files)-1, num_frames, dtype=int)
        frame_files = [frame_files[i] for i in indices]
    else:
        while len(frame_files) < num_frames:
            frame_files.append(frame_files[len(frame_files) % len(frame_files)])
            
    transform = transforms.Compose([
        transforms.Resize(frame_size + 32),
        transforms.CenterCrop(frame_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    frames_tensor = []
    for f in frame_files:
        img_path = os.path.join(video_dir, f)
        img = Image.open(img_path).convert('RGB')
        frames_tensor.append(transform(img))
        
    return torch.stack(frames_tensor).unsqueeze(0)


def predict_video(model, video_dir, device='cuda'):
    model.eval()
    frames_tensor = load_video_frames(video_dir).to(device)
    
    with torch.no_grad():
        logits = model(frames_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0][1].item()
        
    return confidence

