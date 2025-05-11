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
from contextlib import contextmanager

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@contextmanager
def temporary_seed(seed: int):
    """
    Context manager for temporarily setting a random seed and restoring
    the previous state afterward.
    
    Args:
        seed: The seed to temporarily set
        
    Example:
        with temporary_seed(42):
            # Code that needs deterministic behavior
    """
    # Store current random states
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    torch_cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    random_state = random.getstate()
    
    # Set seed
    set_seed(seed)
    
    try:
        yield
    finally:
        # Restore previous random states
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if torch_cuda_state is not None:
            torch.cuda.set_rng_state_all(torch_cuda_state)
        random.setstate(random_state)

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