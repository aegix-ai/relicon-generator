"""
Relicon AI Ad Creator - Data Models
Comprehensive Pydantic models for the entire system
"""
from datetime import datetime
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, validator
from enum import Enum


class AdPlatform(str, Enum):
    """Supported ad platforms"""
    TIKTOK = "tiktok"
    INSTAGRAM = "instagram" 
    FACEBOOK = "facebook"
    YOUTUBE_SHORTS = "youtube_shorts"
    UNIVERSAL = "universal"


class AdStyle(str, Enum):
    """Ad style preferences"""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    ENERGETIC = "energetic"
    MINIMAL = "minimal"
    CINEMATIC = "cinematic"


class JobStatus(str, Enum):
    """Job processing status"""
    PENDING = "pending"
    PLANNING = "planning"
    GENERATING_SCRIPT = "generating_script" 
    GENERATING_AUDIO = "generating_audio"
    GENERATING_VIDEO = "generating_video"
    ASSEMBLING = "assembling"
    COMPLETED = "completed"
    FAILED = "failed"


# Request Models
class AdCreationRequest(BaseModel):
    """Main request for ad creation"""
    
    # Business Information
    brand_name: str = Field(..., min_length=1, max_length=100)
    brand_description: str = Field(..., min_length=10, max_length=1000)
    product_name: Optional[str] = Field(None, max_length=100)
    target_audience: Optional[str] = Field(None, max_length=500)
    unique_selling_point: Optional[str] = Field(None, max_length=500)
    call_to_action: Optional[str] = Field(None, max_length=100)
    
    # Technical Specifications
    duration: int = Field(30, ge=10, le=60)
    platform: AdPlatform = AdPlatform.UNIVERSAL
    style: AdStyle = AdStyle.PROFESSIONAL
    
    # Asset Information
    uploaded_assets: List[str] = Field(default_factory=list)
    brand_colors: Optional[List[str]] = Field(None, max_examples=5)
    brand_fonts: Optional[List[str]] = Field(None, max_examples=3)
    
    # Additional Requirements
    include_logo: bool = True
    include_music: bool = True
    voice_preference: Optional[Literal["male", "female", "neutral"]] = "neutral"
    
    @validator("brand_colors", pre=True)
    def validate_colors(cls, v):
        if v is None:
            return v
        # Validate hex colors
        for color in v:
            if not color.startswith("#") or len(color) != 7:
                raise ValueError("Colors must be valid hex codes (e.g., #FF0000)")
        return v


class FileUpload(BaseModel):
    """File upload information"""
    filename: str
    file_type: str
    file_size: int
    upload_path: str
    uploaded_at: datetime


# AI Agent Models
class SceneComponent(BaseModel):
    """Individual scene component with ultra-detailed specifications"""
    
    # Timing
    start_time: float = Field(..., ge=0)
    duration: float = Field(..., gt=0, le=15)
    end_time: float
    
    # Visual Elements
    visual_type: Literal["video", "image", "text", "logo", "transition"]
    visual_prompt: str = Field(..., min_length=10)
    visual_style: str
    
    # Audio Elements  
    has_voiceover: bool = False
    voiceover_text: Optional[str] = None
    voice_tone: Optional[str] = None
    has_music: bool = False
    music_style: Optional[str] = None
    
    # Effects and Transitions
    entry_effect: Optional[str] = None
    exit_effect: Optional[str] = None
    overlay_effects: List[str] = Field(default_factory=list)
    
    # Generation Parameters
    luma_prompt: Optional[str] = None
    generated_asset_path: Optional[str] = None
    
    @validator("end_time", always=True)
    def calculate_end_time(cls, v, values):
        return values.get("start_time", 0) + values.get("duration", 0)


class AdScene(BaseModel):
    """Complete scene with all components"""
    
    scene_id: str = Field(..., regex=r"^scene_\d+$")
    scene_type: Literal["hook", "problem", "solution", "benefits", "cta", "transition"]
    scene_purpose: str = Field(..., min_length=10)
    
    # Timing
    start_time: float = Field(..., ge=0)
    duration: float = Field(..., gt=0)
    
    # Components
    components: List[SceneComponent] = Field(..., min_items=1)
    
    # Script
    main_script: Optional[str] = None
    script_timing: Optional[Dict[str, float]] = None
    
    # Visual Direction
    camera_direction: Optional[str] = None
    lighting_notes: Optional[str] = None
    color_palette: List[str] = Field(default_factory=list)


class MasterAdPlan(BaseModel):
    """Ultra-detailed master plan for the entire ad"""
    
    # Plan Metadata
    plan_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    total_duration: float
    
    # Strategic Elements
    ad_concept: str = Field(..., min_length=20)
    narrative_arc: str = Field(..., min_length=50) 
    emotional_journey: List[str] = Field(..., min_items=3)
    key_messages: List[str] = Field(..., min_items=1, max_items=5)
    
    # Scenes
    scenes: List[AdScene] = Field(..., min_items=1)
    scene_transitions: List[str] = Field(default_factory=list)
    
    # Audio Strategy
    overall_voice_direction: str
    music_strategy: str
    audio_pacing: Dict[str, Any] = Field(default_factory=dict)
    
    # Brand Integration
    brand_presence_timeline: Dict[str, float] = Field(default_factory=dict)
    logo_appearances: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Technical Specifications
    resolution: str = "1920x1080"
    fps: int = 30
    target_platforms: List[AdPlatform]
    
    @validator("scenes")
    def validate_scene_timing(cls, v, values):
        total_duration = values.get("total_duration", 0)
        scene_duration_sum = sum(scene.duration for scene in v)
        
        if abs(scene_duration_sum - total_duration) > 0.5:  # Allow 0.5s tolerance
            raise ValueError("Scene durations must sum to total duration")
        
        return v


# Response Models
class AdCreationResponse(BaseModel):
    """Response for ad creation request"""
    
    success: bool
    job_id: str
    status: JobStatus
    message: str
    estimated_completion: Optional[datetime] = None
    
    # Progress Information
    progress_percentage: int = Field(0, ge=0, le=100)
    current_step: Optional[str] = None
    
    # Results (when completed)
    video_url: Optional[str] = None
    master_plan: Optional[MasterAdPlan] = None
    generation_stats: Optional[Dict[str, Any]] = None


class JobStatusResponse(BaseModel):
    """Job status response"""
    
    job_id: str
    status: JobStatus
    progress_percentage: int = Field(ge=0, le=100)
    current_step: Optional[str] = None
    message: str
    
    # Timestamps
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    
    # Results
    video_url: Optional[str] = None
    error_details: Optional[str] = None
    
    # Detailed Progress
    planning_complete: bool = False
    script_complete: bool = False
    audio_complete: bool = False
    video_complete: bool = False
    assembly_complete: bool = False


class HealthCheckResponse(BaseModel):
    """System health check response"""
    
    status: Literal["healthy", "degraded", "unhealthy"]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str
    
    # Service Status
    services: Dict[str, Literal["healthy", "unhealthy"]] = Field(default_factory=dict)
    
    # System Info
    uptime_seconds: int
    active_jobs: int
    queue_size: int


# Internal Models for AI Agents
class PlanningContext(BaseModel):
    """Context for AI planning agents"""
    
    request: AdCreationRequest
    brand_analysis: Dict[str, Any] = Field(default_factory=dict)
    competitor_insights: List[str] = Field(default_factory=list)
    platform_requirements: Dict[str, Any] = Field(default_factory=dict)
    creative_constraints: Dict[str, Any] = Field(default_factory=dict)


class GenerationAssets(BaseModel):
    """Generated assets for ad creation"""
    
    # Audio Assets
    voiceover_files: List[str] = Field(default_factory=list)
    music_files: List[str] = Field(default_factory=list)
    sfx_files: List[str] = Field(default_factory=list)
    
    # Video Assets  
    scene_videos: List[str] = Field(default_factory=list)
    transition_videos: List[str] = Field(default_factory=list)
    
    # Image Assets
    generated_images: List[str] = Field(default_factory=list)
    logo_variants: List[str] = Field(default_factory=list)
    
    # Metadata
    generation_metadata: Dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Error response model"""
    
    success: bool = False
    error_code: str
    error_message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow) 