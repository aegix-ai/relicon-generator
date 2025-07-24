"""
Relicon AI Ad Creator - Planning Models
Advanced Pydantic models for AI agent planning and scene architecture
"""
from datetime import datetime
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, validator

from .enums import AdPlatform
from .requests import AdCreationRequest


class SceneComponent(BaseModel):
    """
    Individual scene component with ultra-detailed specifications
    
    Represents a single atomic element within a scene (video clip, voiceover,
    text overlay, etc.) with precise timing and generation parameters.
    """
    
    # Precise Timing Control
    start_time: float = Field(
        ..., 
        ge=0,
        description="Component start time in seconds (precise to 0.1s)"
    )
    duration: float = Field(
        ..., 
        gt=0, 
        le=15,
        description="Component duration in seconds"
    )
    end_time: float = Field(
        ...,
        description="Calculated end time (start_time + duration)"
    )
    
    # Visual Configuration
    visual_type: Literal["video", "image", "text", "logo", "transition"] = Field(
        ...,
        description="Type of visual element this component represents"
    )
    visual_prompt: str = Field(
        ..., 
        min_length=10,
        description="Detailed prompt for AI visual generation"
    )
    visual_style: str = Field(
        ...,
        description="Visual style descriptor (cinematic, clean, energetic, etc.)"
    )
    
    # Audio Configuration
    has_voiceover: bool = Field(
        False,
        description="Whether this component includes voiceover"
    )
    voiceover_text: Optional[str] = Field(
        None,
        description="Exact text to be spoken (if has_voiceover=True)"
    )
    voice_tone: Optional[str] = Field(
        None,
        description="Voice tone (confident, friendly, urgent, etc.)"
    )
    has_music: bool = Field(
        False,
        description="Whether this component has background music"
    )
    music_style: Optional[str] = Field(
        None,
        description="Music style (energetic, ambient, dramatic, etc.)"
    )
    
    # Effects and Transitions
    entry_effect: Optional[str] = Field(
        None,
        description="How this component enters (fade_in, slide, cut, etc.)"
    )
    exit_effect: Optional[str] = Field(
        None,
        description="How this component exits (fade_out, slide, cut, etc.)"
    )
    overlay_effects: List[str] = Field(
        default_factory=list,
        description="Additional overlay effects (text, particles, etc.)"
    )
    
    # AI Generation Parameters
    luma_prompt: Optional[str] = Field(
        None,
        description="Optimized prompt for Luma AI video generation"
    )
    generated_asset_path: Optional[str] = Field(
        None,
        description="Path to generated asset file (populated after generation)"
    )
    
    @validator("end_time", always=True)
    def calculate_end_time(cls, v, values):
        """Automatically calculate end time from start time and duration"""
        start_time = values.get("start_time", 0)
        duration = values.get("duration", 0)
        return start_time + duration


class AdScene(BaseModel):
    """
    Complete scene with all components and timing
    
    Represents a logically cohesive segment of the ad with multiple
    synchronized components (video, audio, effects, etc.).
    """
    
    # Scene Identification
    scene_id: str = Field(
        ..., 
        regex=r"^scene_\d+$",
        description="Unique scene identifier (scene_1, scene_2, etc.)"
    )
    scene_type: Literal["hook", "problem", "solution", "benefits", "cta", "transition"] = Field(
        ...,
        description="Scene type that determines its role in the narrative"
    )
    scene_purpose: str = Field(
        ..., 
        min_length=10,
        description="Detailed description of this scene's purpose"
    )
    
    # Timing Configuration
    start_time: float = Field(
        ..., 
        ge=0,
        description="Scene start time in the overall ad timeline"
    )
    duration: float = Field(
        ..., 
        gt=0,
        description="Total scene duration in seconds"
    )
    
    # Scene Components
    components: List[SceneComponent] = Field(
        ..., 
        min_items=1,
        description="List of all components within this scene"
    )
    
    # Script and Narrative
    main_script: Optional[str] = Field(
        None,
        description="Primary voiceover script for this scene"
    )
    script_timing: Optional[Dict[str, float]] = Field(
        None,
        description="Timing breakdown for script elements"
    )
    
    # Visual Direction
    camera_direction: Optional[str] = Field(
        None,
        description="Camera movement and positioning instructions"
    )
    lighting_notes: Optional[str] = Field(
        None,
        description="Lighting setup and mood instructions"
    )
    color_palette: List[str] = Field(
        default_factory=list,
        description="Primary colors for this scene (hex codes)"
    )


class MasterAdPlan(BaseModel):
    """
    Ultra-detailed master plan for the entire ad
    
    The complete blueprint created by the Master Planner AI agent,
    containing every detail needed for precise ad generation.
    """
    
    # Plan Metadata
    plan_id: str = Field(..., description="Unique identifier for this master plan")
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Plan creation timestamp"
    )
    total_duration: float = Field(..., description="Total ad duration in seconds")
    
    # Strategic Elements
    ad_concept: str = Field(
        ..., 
        min_length=20,
        description="Core advertising concept and strategy"
    )
    narrative_arc: str = Field(
        ..., 
        min_length=50,
        description="Overall story progression and emotional journey"
    )
    emotional_journey: List[str] = Field(
        ..., 
        min_items=3,
        description="Sequence of emotions the viewer should experience"
    )
    key_messages: List[str] = Field(
        ..., 
        min_items=1, 
        max_items=5,
        description="Core messages to communicate (in priority order)"
    )
    
    # Scene Structure
    scenes: List[AdScene] = Field(
        ..., 
        min_items=1,
        description="All scenes in chronological order"
    )
    scene_transitions: List[str] = Field(
        default_factory=list,
        description="Transition effects between scenes"
    )
    
    # Audio Strategy
    overall_voice_direction: str = Field(
        ...,
        description="Overall voice tone and style guidelines"
    )
    music_strategy: str = Field(
        ...,
        description="Background music approach and mood progression"
    )
    audio_pacing: Dict[str, Any] = Field(
        default_factory=dict,
        description="Audio pacing and rhythm specifications"
    )
    
    # Brand Integration
    brand_presence_timeline: Dict[str, float] = Field(
        default_factory=dict,
        description="When and how strongly brand elements appear"
    )
    logo_appearances: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Specific timing and positioning for logo appearances"
    )
    
    # Technical Specifications
    resolution: str = Field(
        "1920x1080",
        description="Video resolution (WIDTHxHEIGHT)"
    )
    fps: int = Field(
        30,
        description="Frames per second for video generation"
    )
    target_platforms: List[AdPlatform] = Field(
        ...,
        description="Platforms this ad is optimized for"
    )
    
    @validator("scenes")
    def validate_scene_timing(cls, v, values):
        """
        Validate that scene timing is mathematically precise
        
        Ensures all scenes fit within the total duration with proper timing.
        """
        total_duration = values.get("total_duration", 0)
        
        if not v:  # Empty scenes list
            return v
            
        # Calculate total scene duration
        scene_duration_sum = sum(scene.duration for scene in v)
        
        # Allow 0.5 second tolerance for rounding
        tolerance = 0.5
        if abs(scene_duration_sum - total_duration) > tolerance:
            raise ValueError(
                f"Scene durations ({scene_duration_sum}s) must sum to total duration ({total_duration}s)"
            )
        
        # Validate scene order and timing
        expected_start = 0.0
        for i, scene in enumerate(v):
            if abs(scene.start_time - expected_start) > 0.1:  # 0.1s precision
                raise ValueError(
                    f"Scene {scene.scene_id} starts at {scene.start_time}s, "
                    f"expected {expected_start}s"
                )
            expected_start += scene.duration
        
        return v


class PlanningContext(BaseModel):
    """
    Context information for AI planning agents
    
    Provides all the necessary context and constraints for the
    AI agents to create optimal plans.
    """
    
    # Core Request
    request: AdCreationRequest = Field(
        ...,
        description="Original ad creation request"
    )
    
    # Analysis Results
    brand_analysis: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed brand analysis results"
    )
    competitor_insights: List[str] = Field(
        default_factory=list,
        description="Competitive landscape insights"
    )
    
    # Platform Requirements
    platform_requirements: Dict[str, Any] = Field(
        default_factory=dict,
        description="Platform-specific requirements and constraints"
    )
    
    # Creative Constraints
    creative_constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description="Creative guidelines and limitations"
    ) 