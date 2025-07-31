"""
    Configuration Settings
    Centralized configuration management
"""

import os
from typing import Optional


class Settings:
    """Application settings"""
    
    # API Keys
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
    LUMA_API_KEY: str = os.environ.get("LUMA_API_KEY", "")
    ELEVENLABS_API_KEY: str = os.environ.get("ELEVENLABS_API_KEY", "")
    
    # Database
    DATABASE_URL: str = os.environ.get("DATABASE_URL", "sqlite:///relicon.db")
    
    # Redis
    REDIS_HOST: str = os.environ.get("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.environ.get("REDIS_PORT", "6379"))
    
    # Storage
    OUTPUT_DIR: str = os.environ.get("OUTPUT_DIR", "outputs")
    TEMP_DIR: str = os.environ.get("TEMP_DIR", "/tmp/relicon")
    
    # API Settings
    API_HOST: str = os.environ.get("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.environ.get("API_PORT", "8000"))
    
    # Video Generation
    DEFAULT_ASPECT_RATIO: str = "9:16"  # TikTok/Instagram format
    MAX_VIDEO_DURATION: int = 60  # seconds
    DEFAULT_VIDEO_QUALITY: str = "high"
    
    # Cost Limits
    MAX_COST_PER_VIDEO: float = 10.0  # dollars
    COST_WARNING_THRESHOLD: float = 5.0  # dollars
    
    # Performance
    MAX_CONCURRENT_JOBS: int = int(os.environ.get("MAX_CONCURRENT_JOBS", "3"))
    JOB_TIMEOUT: int = int(os.environ.get("JOB_TIMEOUT", "1800"))  # 30 minutes
    
    def validate(self) -> list[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        if not self.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required")
        
        if not self.LUMA_API_KEY:
            errors.append("LUMA_API_KEY is required")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.TEMP_DIR, exist_ok=True)
        
        return errors
    
    def to_dict(self) -> dict:
        """Convert settings to dictionary (excluding sensitive data)"""
        return {
            "output_dir": self.OUTPUT_DIR,
            "temp_dir": self.TEMP_DIR,
            "api_host": self.API_HOST,
            "api_port": self.API_PORT,
            "default_aspect_ratio": self.DEFAULT_ASPECT_RATIO,
            "max_video_duration": self.MAX_VIDEO_DURATION,
            "max_concurrent_jobs": self.MAX_CONCURRENT_JOBS,
            "job_timeout": self.JOB_TIMEOUT,
        }


# Global settings instance
settings = Settings()
