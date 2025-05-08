import yaml
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Dict, List

class Settings(BaseSettings):
    
    APP_NAME: str
    APP_VERSION: str
    TESSERACT_PATH: str
    TESSERACT_DIR: str
    ID_DETECTOR_MODEL_PATH: str
    MRZ_DETECTOR_MODEL_PATH: str
    ID_PARTS_DETECTOR_MODEL_PATH: str
    ID_NUMBER_DETECTOR_MODEL_PATH: str
    DOCUMENT_CLASSIFIER_MODEL_PATH: str
    TESSDATA_PREFIX: str
    
    class Config:
        env_file = ".env"
        
def get_settings():
    return Settings()

def load_coordinates():
    config_path = r"config\templates\coordinates.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)