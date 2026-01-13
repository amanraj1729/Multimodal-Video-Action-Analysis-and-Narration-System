# FILE: models/cnn_3d.py
import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

class R2Plus1DClassifier(nn.Module):
    def __init__(self, num_classes=50, extract_features=False):
        super(R2Plus1DClassifier, self).__init__()
        
        self.extract_features = extract_features
        
        # 1. Load Pre-trained Model
        # We use standard weights from Kinetics-400
        weights = R2Plus1D_18_Weights.DEFAULT
        self.base_model = r2plus1d_18(weights=weights)
        
        # 2. Modify the Final Layer
        # The original model outputs 400 classes. We need 'num_classes' (50).
        in_features = self.base_model.fc.in_features  # Usually 512
        self.base_model.fc = nn.Linear(in_features, num_classes)
        
        # Store the feature dimension for Fusion later
        self.feature_dim = in_features

    def forward(self, x):
        # x shape: (Batch, 3, Frames, H, W)
        
        if self.extract_features:
            # EXTRACT EMBEDDINGS (For Fusion)
            # We need to bypass the final 'fc' layer.
            # R2Plus1D structure: stem -> layer1...layer4 -> avgpool -> fc
            
            x = self.base_model.stem(x)
            x = self.base_model.layer1(x)
            x = self.base_model.layer2(x)
            x = self.base_model.layer3(x)
            x = self.base_model.layer4(x)
            x = self.base_model.avgpool(x)
            
            # Flatten: (B, 512, 1, 1, 1) -> (B, 512)
            features = x.flatten(1)
            return features
            
        else:
            # STANDARD CLASSIFICATION (For Training Baseline)
            return self.base_model(x)

if __name__ == "__main__":
    # Test block
    model = R2Plus1DClassifier(num_classes=50)
    dummy_input = torch.randn(2, 3, 16, 112, 112)
    output = model(dummy_input)
    print(f"R(2+1)D Output Shape: {output.shape}")