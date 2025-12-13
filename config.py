import os
import logging
from pydantic_settings import BaseSettings

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("PDF-Insight")

class Settings(BaseSettings):
    API_URL: str = "http://127.0.0.1:8000"
    MODEL_NAME: str = "gemini-2.5-flash"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
logger.info(f"Loaded configuration. API_URL: {settings.API_URL}")
