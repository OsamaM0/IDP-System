# -*- coding: utf-8 -*-
from typing import List, Dict
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Base class for AI models. This class defines the interface for loading models and making predictions.
    Subclasses should implement the methods defined here.
    """
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Initialize the BaseModel with the path to the model.

        :param model_path: Path to the model file.
        """
        self.model_path = model_path
        self.device = device

    @abstractmethod
    def load_model(self):
        """
        Load the AI model from the specified path.
        """
        pass

    @abstractmethod
    def predict(self, image: str) -> List[Dict]:
        """
        Make a prediction on the given image.

        :param image: Path to the image file.
        :return: List of predictions.
        """
        pass