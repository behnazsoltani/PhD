import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class HybridTransformer(nn.Module):
    def __init__(self, num_classes=10):
        super(HybridTransformer, self).__init__()
        
        # CNN Feature Extractor (Efficient and captures local features)
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()  # Remove final FC layer
        
        # Transformer Encoder (Global feature learning)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1),
            num_layers=2
        )
        
        # Classification Head
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        B = x.shape[0]
        
        # CNN Feature Extraction
        x = self.cnn(x)  # Shape: (B, 512)
        
        # Transformer Processing (Requires sequence input, adding sequence dim)
        x = x.unsqueeze(1)  # Shape: (B, 1, 512)
        x = self.transformer(x)  # Shape: (B, 1, 512)
        x = x.squeeze(1)  # Shape: (B, 512)
        
        # Classification Layer
        x = self.fc(x)  # Shape: (B, num_classes)
        return x


 