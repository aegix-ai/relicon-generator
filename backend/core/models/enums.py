"""
Relicon AI Ad Creator - Enums
System-wide enumeration types for consistent data validation
"""
from enum import Enum


class AdPlatform(str, Enum):
    """
    Supported advertising platforms with specific optimization requirements
    
    Each platform has unique characteristics:
    - TIKTOK: Vertical 9:16, fast-paced, trend-focused content
    - INSTAGRAM: Square 1:1, aesthetic-focused, engagement-driven
    - FACEBOOK: Landscape 16:9, longer-form, conversion-optimized
    - YOUTUBE_SHORTS: Vertical 9:16, retention-focused
    - UNIVERSAL: Adaptable format for multiple platforms
    """
    TIKTOK = "tiktok"
    INSTAGRAM = "instagram" 
    FACEBOOK = "facebook"
    YOUTUBE_SHORTS = "youtube_shorts"
    UNIVERSAL = "universal"


class AdStyle(str, Enum):
    """
    Creative style preferences that influence visual and audio direction
    
    Style impacts:
    - PROFESSIONAL: Clean, corporate, trust-building
    - CASUAL: Approachable, friendly, conversational
    - ENERGETIC: Dynamic, exciting, attention-grabbing
    - MINIMAL: Simple, elegant, focus-driven
    - CINEMATIC: Dramatic, film-quality, artistic
    """
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    ENERGETIC = "energetic"
    MINIMAL = "minimal"
    CINEMATIC = "cinematic"


class JobStatus(str, Enum):
    """
    Ad creation workflow status tracking
    
    Process flow:
    1. PENDING: Job received and queued
    2. PLANNING: AI agents creating master plan
    3. GENERATING_SCRIPT: Finalizing voiceover scripts
    4. GENERATING_AUDIO: ElevenLabs voice synthesis
    5. GENERATING_VIDEO: Luma AI video generation
    6. ASSEMBLING: FFmpeg final video assembly
    7. COMPLETED: Ready for download
    8. FAILED: Error occurred, check error_details
    """
    PENDING = "pending"
    PLANNING = "planning"
    GENERATING_SCRIPT = "generating_script" 
    GENERATING_AUDIO = "generating_audio"
    GENERATING_VIDEO = "generating_video"
    ASSEMBLING = "assembling"
    COMPLETED = "completed"
    FAILED = "failed" 