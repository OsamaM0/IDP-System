from pydantic_settings import BaseSettings
from typing import Dict

class Settings(BaseSettings):
    APP_NAME: str
    APP_VERSION: str
    MODEL_PATH: str
    BASE_EMBEDDINGS: Dict[str, str]
    
    
    class Config:
        env_file = ".env"
        
def get_settings():
    return Settings()