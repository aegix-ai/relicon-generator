"""
Relicon AI Ad Creator - Asset Models
Models for tracking generated and uploaded assets throughout the ad creation process
"""
from typing import List, Dict, Any
from pydantic import BaseModel, Field


class GenerationAssets(BaseModel):
    """
    Collection of all generated assets for ad creation
    
    Tracks all AI-generated audio, video, and image assets produced
    during the ad creation workflow, with metadata for assembly.
    """
    
    # Audio Assets
    voiceover_files: List[str] = Field(
        default_factory=list,
        description="Paths to generated voiceover audio files (ElevenLabs)"
    )
    music_files: List[str] = Field(
        default_factory=list,
        description="Paths to background music files"
    )
    sfx_files: List[str] = Field(
        default_factory=list,
        description="Paths to sound effect files"
    )
    
    # Video Assets  
    scene_videos: List[str] = Field(
        default_factory=list,
        description="Paths to generated scene video files (Luma AI)"
    )
    transition_videos: List[str] = Field(
        default_factory=list,
        description="Paths to transition effect videos"
    )
    
    # Image Assets
    generated_images: List[str] = Field(
        default_factory=list,
        description="Paths to AI-generated image assets"
    )
    logo_variants: List[str] = Field(
        default_factory=list,
        description="Paths to processed logo variations"
    )
    
    # Asset Metadata
    generation_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about asset generation (timing, quality, etc.)"
    )
    
    def get_all_asset_paths(self) -> List[str]:
        """
        Get all asset file paths in a single list
        
        Returns:
            List of all asset file paths across all categories
        """
        all_assets = []
        all_assets.extend(self.voiceover_files)
        all_assets.extend(self.music_files)
        all_assets.extend(self.sfx_files)
        all_assets.extend(self.scene_videos)
        all_assets.extend(self.transition_videos)
        all_assets.extend(self.generated_images)
        all_assets.extend(self.logo_variants)
        return all_assets
    
    def get_assets_by_type(self, asset_type: str) -> List[str]:
        """
        Get asset paths filtered by type
        
        Args:
            asset_type: Type of assets to retrieve
                       ('audio', 'video', 'image', 'voiceover', etc.)
        
        Returns:
            List of asset paths matching the specified type
        """
        type_mapping = {
            'audio': self.voiceover_files + self.music_files + self.sfx_files,
            'video': self.scene_videos + self.transition_videos,
            'image': self.generated_images + self.logo_variants,
            'voiceover': self.voiceover_files,
            'music': self.music_files,
            'sfx': self.sfx_files,
            'scene_video': self.scene_videos,
            'transition': self.transition_videos,
            'generated_image': self.generated_images,
            'logo': self.logo_variants
        }
        
        return type_mapping.get(asset_type, [])
    
    def add_asset(self, asset_path: str, asset_type: str) -> None:
        """
        Add a new asset to the appropriate collection
        
        Args:
            asset_path: Path to the asset file
            asset_type: Type of asset ('voiceover', 'scene_video', etc.)
        """
        if asset_type == 'voiceover':
            self.voiceover_files.append(asset_path)
        elif asset_type == 'music':
            self.music_files.append(asset_path)
        elif asset_type == 'sfx':
            self.sfx_files.append(asset_path)
        elif asset_type == 'scene_video':
            self.scene_videos.append(asset_path)
        elif asset_type == 'transition':
            self.transition_videos.append(asset_path)
        elif asset_type == 'generated_image':
            self.generated_images.append(asset_path)
        elif asset_type == 'logo':
            self.logo_variants.append(asset_path)
        else:
            raise ValueError(f"Unknown asset type: {asset_type}")
    
    def get_asset_count(self) -> Dict[str, int]:
        """
        Get count of assets by type
        
        Returns:
            Dictionary with asset type counts
        """
        return {
            'voiceover_files': len(self.voiceover_files),
            'music_files': len(self.music_files),
            'sfx_files': len(self.sfx_files),
            'scene_videos': len(self.scene_videos),
            'transition_videos': len(self.transition_videos),
            'generated_images': len(self.generated_images),
            'logo_variants': len(self.logo_variants),
            'total': len(self.get_all_asset_paths())
        }
    
    def is_complete_for_assembly(self) -> bool:
        """
        Check if we have sufficient assets for final video assembly
        
        Returns:
            True if we have the minimum required assets for assembly
        """
        # Minimum requirements for assembly
        has_video_content = len(self.scene_videos) > 0 or len(self.generated_images) > 0
        has_audio_content = len(self.voiceover_files) > 0 or len(self.music_files) > 0
        
        return has_video_content and has_audio_content 