"""
Relicon AI Ad Creator - Settings Configuration
Centralized configuration management with validation
"""
import os
from typing import List, Optional
from pydantic import BaseSettings, validator, Field
from pathlib import Path


class Settings(BaseSettings):
    """Application settings with validation"""
    
    # Application Info
    APP_NAME: str = "Relicon AI Ad Creator"
    APP_VERSION: str = "v0.5.4 (Relicon)"
    DEBUG: bool = False
    SECRET_KEY: str = Field(..., min_length=32)
    
    # Database
    DATABASE_URL: str = Field(..., regex=r"^postgresql://.*")
    REDIS_URL: str = Field(default="redis://localhost:6379/0")
    
    # AI Services
    OPENAI_API_KEY: str = Field(..., min_length=20)
    LUMA_API_KEY: Optional[str] = None
    ELEVENLABS_API_KEY: Optional[str] = None
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]
    
    # File Storage
    UPLOAD_DIR: str = "./uploads"
    OUTPUT_DIR: str = "./outputs"
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".mp4", ".mp3", ".wav"]
    
    # Video Generation
    DEFAULT_VIDEO_DURATION: int = 30
    MAX_VIDEO_DURATION: int = 60
    MIN_VIDEO_DURATION: int = 10
    VIDEO_RESOLUTION: str = "1920x1080"
    VIDEO_FPS: int = 30
    
    # AI Agent Configuration
    MAX_PLANNING_DEPTH: int = 5
    SCENE_MIN_DURATION: float = 2.0
    SCENE_MAX_DURATION: float = 15.0
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "{time} | {level} | {message}"
    
    # External Services
    SENTRY_DSN: Optional[str] = None
    CLOUDINARY_URL: Optional[str] = None
    
    @validator("CORS_ORIGINS", pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("UPLOAD_DIR", "OUTPUT_DIR")
    def create_directories(cls, v):
        Path(v).mkdir(parents=True, exist_ok=True)
        return v
    
    @validator("VIDEO_RESOLUTION")
    def validate_resolution(cls, v):
        if not v.count("x") == 1:
            raise ValueError("Resolution must be in format WIDTHxHEIGHT")
        width, height = v.split("x")
        try:
            int(width), int(height)
        except ValueError:
            raise ValueError("Resolution dimensions must be integers")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


# Derived settings
class DerivedSettings:
    """Settings derived from main configuration"""
    
    @property
    def video_width(self) -> int:
        return int(settings.VIDEO_RESOLUTION.split("x")[0])
    
    @property
    def video_height(self) -> int:
        return int(settings.VIDEO_RESOLUTION.split("x")[1])
    
    @property
    def is_vertical_video(self) -> bool:
        return self.video_height > self.video_width
    
    @property
    def aspect_ratio(self) -> float:
        return self.video_width / self.video_height
    
    @property
    def database_config(self) -> dict:
        """Extract database connection parameters"""
        url = settings.DATABASE_URL
        if url.startswith("postgresql://"):
            # Parse PostgreSQL URL
            import urllib.parse
            parsed = urllib.parse.urlparse(url)
            return {
                "host": parsed.hostname,
                "port": parsed.port or 5432,
                "database": parsed.path[1:],  # Remove leading slash
                "username": parsed.username,
                "password": parsed.password,
            }
        return {}


# Global derived settings
derived = DerivedSettings() 