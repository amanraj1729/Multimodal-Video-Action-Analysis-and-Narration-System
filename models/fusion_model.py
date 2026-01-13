# FILE: models/fusion_model.py
import torch
import torch.nn as nn
from models.cnn_3d import R2Plus1DClassifier
from models.video_vit import VideoViTClassifier

class VideoFusionModel(nn.Module):
    def __init__(self, num_classes=50, freeze_backbones=False):
        super(VideoFusionModel, self).__init__()
        
        # 1. Initialize Backbones
        print("Initializing Fusion Model Components...")
        self.cnn = R2Plus1DClassifier(num_classes=num_classes, extract_features=True)
        self.vit = VideoViTClassifier(num_classes=num_classes, extract_features=True)
        
        # 2. Freeze Backbones (Optional: typically we fine-tune everything)
        # If your GPU runs out of memory, set freeze_backbones=True
        if freeze_backbones:
            for param in self.cnn.parameters():
                param.requires_grad = False
            for param in self.vit.parameters():
                param.requires_grad = False
        
        # 3. Fusion Head (MLP)
        # Input = CNN Features (512) + ViT Features (768) = 1280
        fusion_dim = 512 + 768 
        
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x shape: (Batch, 3, 16, 112, 112)
        
        # 1. Get Features from both branches
        cnn_feat = self.cnn(x)  # (Batch, 512)
        vit_feat = self.vit(x)  # (Batch, 768)
        
        # 2. Concatenate
        combined = torch.cat((cnn_feat, vit_feat), dim=1) # (Batch, 1280)
        
        # 3. Final Prediction
        output = self.fusion_head(combined) # (Batch, 50)
        
        return output

if __name__ == "__main__":
    # Test Block
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoFusionModel(num_classes=50).to(device)
    dummy = torch.randn(2, 3, 16, 112, 112).to(device)
    print(f"Fusion Output: {model(dummy).shape}")