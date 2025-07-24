"""
Relicon AI Ad Creator - Request Models
Pydantic models for incoming API requests with comprehensive validation
"""
from datetime import datetime
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, validator

from .enums import AdPlatform, AdStyle


class AdCreationRequest(BaseModel):
    """
    Main request for AI-powered ad creation
    
    This model validates and structures all the information needed
    to create a revolutionary ad with mathematical precision.
    """
    
    # Core Business Information
    brand_name: str = Field(
        ..., 
        min_length=1, 
        max_length=100,
        description="The brand name that will be featured in the ad"
    )
    brand_description: str = Field(
        ..., 
        min_length=10, 
        max_length=1000,
        description="Detailed description of the brand, product, or service"
    )
    
    # Optional Business Details
    product_name: Optional[str] = Field(
        None, 
        max_length=100,
        description="Specific product name if different from brand"
    )
    target_audience: Optional[str] = Field(
        None, 
        max_length=500,
        description="Primary target audience demographics and characteristics"
    )
    unique_selling_point: Optional[str] = Field(
        None, 
        max_length=500,
        description="What makes this brand/product unique and compelling"
    )
    call_to_action: Optional[str] = Field(
        None, 
        max_length=100,
        description="Specific action you want viewers to take"
    )
    
    # Technical Specifications
    duration: int = Field(
        30, 
        ge=10, 
        le=60,
        description="Ad duration in seconds (10-60 seconds supported)"
    )
    platform: AdPlatform = Field(
        AdPlatform.UNIVERSAL,
        description="Target platform for optimization (affects aspect ratio and pacing)"
    )
    style: AdStyle = Field(
        AdStyle.PROFESSIONAL,
        description="Creative style that influences visual and audio direction"
    )
    
    # Asset and Customization Options
    uploaded_assets: List[str] = Field(
        default_factory=list,
        description="List of uploaded asset filenames to include in the ad"
    )
    brand_colors: Optional[List[str]] = Field(
        None, 
        max_items=5,
        description="Brand colors in hex format (e.g., ['#FF0000', '#00FF00'])"
    )
    brand_fonts: Optional[List[str]] = Field(
        None, 
        max_items=3,
        description="Preferred font families for text overlays"
    )
    
    # Audio and Visual Preferences
    include_logo: bool = Field(
        True,
        description="Whether to include brand logo in the ad"
    )
    include_music: bool = Field(
        True,
        description="Whether to include background music"
    )
    voice_preference: Optional[Literal["male", "female", "neutral"]] = Field(
        "neutral",
        description="Preferred voice type for voiceovers"
    )
    
    @validator("brand_colors", pre=True)
    def validate_colors(cls, v):
        """Validate that all colors are valid hex codes"""
        if v is None:
            return v
        
        for color in v:
            if not isinstance(color, str):
                raise ValueError("Colors must be strings")
            if not color.startswith("#") or len(color) != 7:
                raise ValueError("Colors must be valid hex codes (e.g., #FF0000)")
            # Validate hex characters
            try:
                int(color[1:], 16)
            except ValueError:
                raise ValueError(f"Invalid hex color: {color}")
        
        return v

    @validator("duration")
    def validate_duration_platform_compatibility(cls, v, values):
        """Ensure duration is compatible with selected platform"""
        platform = values.get("platform")
        
        # Platform-specific duration recommendations
        if platform == AdPlatform.TIKTOK and v > 60:
            raise ValueError("TikTok ads should be 60 seconds or less")
        if platform == AdPlatform.INSTAGRAM and v > 60:
            raise ValueError("Instagram ads should be 60 seconds or less")
        
        return v


class FileUpload(BaseModel):
    """
    Information about an uploaded file asset
    
    Tracks metadata for user-uploaded brand assets like logos,
    images, videos, or audio files to be included in ads.
    """
    filename: str = Field(..., description="Unique filename after upload")
    file_type: str = Field(..., description="MIME type of the uploaded file")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    upload_path: str = Field(..., description="Server path where file is stored")
    uploaded_at: datetime = Field(..., description="Timestamp when file was uploaded") 