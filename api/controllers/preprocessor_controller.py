
from core.factories.preprocessor_factory import PreprocessorFactory
import numpy as np

class PreprocessorController:
    def __init__(self, preprocessor_type: str, image: np.ndarray):
        self.preprocessor = PreprocessorFactory.create_preprocessor(preprocessor_type)
        self.image = image

    def preprocess(self):
        return self.preprocessor.preprocess(self.image)