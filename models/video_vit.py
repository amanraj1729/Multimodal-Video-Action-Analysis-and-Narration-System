# FILE: models/video_vit.py
import torch
import torch.nn as nn
from transformers import VideoMAEModel, VideoMAEConfig

class VideoViTClassifier(nn.Module):
    def __init__(self, num_classes=50, extract_features=False):
        super(VideoViTClassifier, self).__init__()
        
        self.extract_features = extract_features
        
        # 1. Load Pre-trained VideoMAE (Base model)
        # We use the standard weights pre-trained on Kinetics-400
        print("Loading VideoMAE weights (this happens once)...")
        self.vit = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
        
        # 2. Define the Classification Head
        # VideoMAE base hidden size is usually 768
        self.hidden_size = self.vit.config.hidden_size
        self.fc = nn.Linear(self.hidden_size, num_classes)
        
        # 3. Upsampler (112x112 -> 224x224)
        # We use a simple bilinear interpolation
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

    def forward(self, x):
        # Input x shape: (Batch, Channel, Time, Height, Width) -> (B, 3, 16, 112, 112)
        
        # 1. Resize to 224x224 (Required by VideoMAE)
        # To upsample, we merge Batch and Time: (B*T, C, H, W)
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = self.upsample(x)
        
        # Reshape back to video format: (B, T, C, H, W) 
        # Note: HuggingFace VideoMAE expects (Batch, Time, Channel, Height, Width)
        x = x.reshape(B, T, C, 224, 224)
        
        # 2. Pass through Vision Transformer
        # VideoMAE output is a specific object, we need 'last_hidden_state'
        outputs = self.vit(x)
        last_hidden_state = outputs.last_hidden_state  # Shape: (B, 1568, 768)
        
        # 3. Global Average Pooling
        # Average across all tokens to get one vector per video
        video_embedding = torch.mean(last_hidden_state, dim=1) # Shape: (B, 768)
        
        if self.extract_features:
            return video_embedding
            
        # 4. Classification
        out = self.fc(video_embedding)
        return out

if __name__ == "__main__":
    # Sanity Check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoViTClassifier(num_classes=50).to(device)
    
    # Dummy Input (112x112)
    dummy = torch.randn(2, 3, 16, 112, 112).to(device)
    
    output = model(dummy)
    print(f"Video ViT Output Shape: {output.shape}")