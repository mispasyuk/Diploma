import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
class ResNetLSTM_2(nn.Module):
    def __init__(self, lstm_hidden=256, lstm_layers=2, dropout=0.5, 
                 freeze_cnn=True, unfreeze_layers=0):
        super().__init__()
        
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  
        self.layer2 = resnet.layer2  
        self.layer3 = resnet.layer3  
        self.layer4 = resnet.layer4  
        
        self.avgpool = resnet.avgpool

        self.cnn_layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        
        self.cnn_out_dim = 2048

        if freeze_cnn:
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.bn1.parameters():
                param.requires_grad = False
            for layer in self.cnn_layers:
                for param in layer.parameters():
                    param.requires_grad = False

        if unfreeze_layers > 0:
            layers_to_unfreeze = self.cnn_layers[-unfreeze_layers:]
            for i, layer in enumerate(layers_to_unfreeze):
                for param in layer.parameters():
                    param.requires_grad = True
            print(f"Разморожено слоев: {unfreeze_layers}")
            print(f"   ({[f'layer{4-j}' for j in range(len(layers_to_unfreeze))][::-1]})")
        
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
            
            feat = self.conv1(frame)
            feat = self.bn1(feat)
            feat = self.relu(feat)
            feat = self.maxpool(feat)
            
            feat = self.layer1(feat)
            feat = self.layer2(feat)
            feat = self.layer3(feat)
            feat = self.layer4(feat)
            
            feat = self.avgpool(feat)
            feat = feat.view(batch_size, -1)
            cnn_out.append(feat)
            
        cnn_seq = torch.stack(cnn_out, dim=1)
        
        lstm_out, _ = self.lstm(cnn_seq)
        last_out = lstm_out[:, -1, :]
        logits = self.classifier(last_out)
        
        return logits