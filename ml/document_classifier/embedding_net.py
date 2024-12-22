import torch
import torch.nn as nn
from torchvision import models


class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim: int = 128) -> None:
        """
        Initialize the EmbeddingNet with MobileNetV2 backbone.
        
        Args:
            embedding_dim: Dimension of the output embedding
        """
        super(EmbeddingNet, self).__init__()
        self.backbone = models.mobilenet_v2(pretrained=True)
        self.backbone.classifier = nn.Sequential(
            nn.Linear(self.backbone.last_channel, embedding_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
