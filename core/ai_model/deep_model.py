import torch
import numpy as np
from utils.image_utils import preprocess_image
import random
import logging
from .base_model import BaseModel
from config.config import get_settings
import torch
import torch.nn as nn
from torchvision import models

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class DeepModel(BaseModel):
    def __init__(self, model_path: str = None, device: str = 'cpu'):
        super().__init__(model_path or get_settings().DOCUMENT_CLASSIFIER_MODEL_PATH)
    
    def load_model(self, num_classes: int):
        self.model = ClassifierNet(num_classes=num_classes)
        try:
            self.model.load_state_dict(torch.load(self.model_path,  map_location=torch.device('cpu')))
            self.model.eval()  # Set model to evaluation mode
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        except RuntimeError as e:
            raise RuntimeError(f"Error loading model: {e}")

    def predict(self, image_ndarray):
        image_tensor = preprocess_image(image_ndarray)       
        with torch.no_grad():
            outputs = self.model(image_tensor)
        return outputs

class ClassifierNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        """
        Initialize the classifier model using ResNet50.
        
        Args:
            num_classes: Number of output classes
        """
        super(ClassifierNet, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)