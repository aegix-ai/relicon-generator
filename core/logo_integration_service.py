"""
Dynamic Logo Integration Service
Intelligently integrates logos into video scenes based on analysis and scene context.
"""

import os
import json
import tempfile
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from core.logo_analysis_service import LogoAnalysisService, LogoAnalysisResult
from core.logger import get_logger

logger = get_logger(__name__)

@dataclass
class LogoPlacement:
    """Represents logo placement configuration for a scene."""
    scene_number: int
    position: str  # corner, center, floating, watermark
    size_percentage: float  # 0.0-1.0 of video width
    opacity: float  # 0.0-1.0
    animation: str  # fade_in, slide_in, scale_in, static
    timing: Tuple[float, float]  # (start_time, end_time) in seconds
    style_adaptations: Dict[str, Any]

class LogoIntegrationService:
    """Dynamic logo integration with scene-aware placement and styling."""
    
    def __init__(self):
        self.logo_analyzer = LogoAnalysisService()
        
    def create_logo_integration_plan(
        self, 
        logo_path: str, 
        architecture: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create comprehensive logo integration plan based on logo analysis and video architecture.
        
        Args:
            logo_path: Path to logo image file
            architecture: Video architecture from planning service
            
        Returns:
            Complete logo integration plan
        """
        try:
            logger.info("Creating logo integration plan", action="logo.integration.start")
            
            # Analyze logo characteristics
            logo_analysis = self.logo_analyzer.analyze_logo(logo_path)
            if not logo_analysis:
                logger.error("Logo analysis failed")
                return self._create_fallback_plan(architecture)
            
            # Extract video structure
            scenes = architecture.get('scene_architecture', {}).get('scenes', [])
            total_duration = architecture.get('scene_architecture', {}).get('total_duration', 18)
            
            # Create scene-specific logo placements
            logo_placements = self._create_scene_placements(logo_analysis, scenes)
            
            # Generate logo variations for different scenes
            logo_variations = self._generate_logo_variations(logo_path, logo_analysis, scenes)
            
            # Create integration timeline
            integration_timeline = self._create_integration_timeline(logo_placements, total_duration)
            
            # Generate style-adapted prompts for video generation
            enhanced_prompts = self._enhance_scene_prompts_with_logo(
                scenes, logo_analysis, logo_placements
            )
            
            integration_plan = {
                'logo_analysis': logo_analysis,
                'logo_placements': [asdict(placement) for placement in logo_placements],
                'logo_variations': logo_variations,
                'integration_timeline': integration_timeline,
                'enhanced_scene_prompts': enhanced_prompts,
                'ffmpeg_filters': self._generate_ffmpeg_filters(logo_placements),
                'brand_color_palette': {
                    'primary': logo_analysis.color_palette.primary_color,
                    'secondary': logo_analysis.color_palette.secondary_color,
                    'accents': logo_analysis.color_palette.accent_colors
                }
            }
            
            logger.info(
                "Logo integration plan created",
                action="logo.integration.complete",
                placements_count=len(logo_placements)
            )
            
            return integration_plan
            
        except Exception as e:
            logger.error(f"Logo integration planning failed: {e}", exc_info=True)
            return self._create_fallback_plan(architecture)
    
    def _create_scene_placements(
        self, 
        logo_analysis: LogoAnalysisResult, 
        scenes: List[Dict[str, Any]]
    ) -> List[LogoPlacement]:
        """Create intelligent logo placements for each scene."""
        placements = []
        suggestions = logo_analysis.video_integration_suggestions
        
        current_time = 0.0
        
        for i, scene in enumerate(scenes):
            scene_duration = scene.get('duration', 6.0)
            scene_purpose = scene.get('purpose', 'unknown')
            
            # Determine placement strategy based on scene purpose and logo style
            placement_config = self._determine_placement_for_scene(
                scene_purpose, logo_analysis, i + 1, len(scenes)
            )
            
            placement = LogoPlacement(
                scene_number=i + 1,
                position=placement_config['position'],
                size_percentage=placement_config['size'],
                opacity=placement_config['opacity'],
                animation=placement_config['animation'],
                timing=(current_time, current_time + scene_duration),
                style_adaptations=placement_config['adaptations']
            )
            
            placements.append(placement)
            current_time += scene_duration
        
        return placements
    
    def _determine_placement_for_scene(
        self, 
        scene_purpose: str, 
        logo_analysis: LogoAnalysisResult, 
        scene_number: int, 
        total_scenes: int
    ) -> Dict[str, Any]:
        """Determine optimal logo placement for specific scene."""
        style_category = logo_analysis.style_profile.style_category
        brand_personality = logo_analysis.brand_personality
        
        # Scene 1 (Hook): Subtle introduction
        if scene_number == 1:
            if style_category == 'minimalist':
                return {
                    'position': 'top_right_corner',
                    'size': 0.12,
                    'opacity': 0.7,
                    'animation': 'fade_in',
                    'adaptations': {'blur_background': False}
                }
            else:
                return {
                    'position': 'bottom_right_corner',
                    'size': 0.15,
                    'opacity': 0.8,
                    'animation': 'slide_in',
                    'adaptations': {'shadow': True}
                }
        
        # Scene 2 (Problem/Solution): Maintain presence
        elif scene_number == 2:
            return {
                'position': 'watermark',
                'size': 0.08,
                'opacity': 0.4,
                'animation': 'static',
                'adaptations': {'blend_mode': 'overlay'}
            }
        
        # Scene 3 (CTA): Strong presence
        else:
            if brand_personality.get('professional', 0) > 0.7:
                return {
                    'position': 'center_bottom',
                    'size': 0.20,
                    'opacity': 0.9,
                    'animation': 'scale_in',
                    'adaptations': {'background_blur': True, 'glow': True}
                }
            else:
                return {
                    'position': 'bottom_right_corner',
                    'size': 0.18,
                    'opacity': 0.85,
                    'animation': 'bounce_in',
                    'adaptations': {'shadow': True, 'border': True}
                }
    
    def _generate_logo_variations(
        self, 
        logo_path: str, 
        logo_analysis: LogoAnalysisResult, 
        scenes: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Generate logo variations adapted for different scene contexts."""
        try:
            logo_image = Image.open(logo_path)
            variations = {}
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Original logo
                original_path = os.path.join(temp_dir, "logo_original.png")
                logo_image.save(original_path, "PNG")
                variations['original'] = original_path
                
                # Watermark version (low opacity, optimized size)
                watermark = logo_image.copy()
                if watermark.mode != 'RGBA':
                    watermark = watermark.convert('RGBA')
                
                # Reduce opacity
                watermark = self._adjust_image_opacity(watermark, 0.3)
                watermark_path = os.path.join(temp_dir, "logo_watermark.png")
                watermark.save(watermark_path, "PNG")
                variations['watermark'] = watermark_path
                
                # High contrast version for overlays
                high_contrast = logo_image.copy()
                enhancer = ImageEnhance.Contrast(high_contrast)
                high_contrast = enhancer.enhance(1.5)
                
                contrast_path = os.path.join(temp_dir, "logo_high_contrast.png")
                high_contrast.save(contrast_path, "PNG")
                variations['high_contrast'] = contrast_path
                
                # Monochrome version
                monochrome = logo_image.convert('L').convert('RGBA')
                mono_path = os.path.join(temp_dir, "logo_monochrome.png")
                monochrome.save(mono_path, "PNG")
                variations['monochrome'] = mono_path
                
                # Copy variations to permanent location
                permanent_variations = {}
                for variant_name, temp_path in variations.items():
                    permanent_path = f"outputs/logo_{variant_name}_{int(time.time())}.png"
                    import shutil
                    shutil.copy2(temp_path, permanent_path)
                    permanent_variations[variant_name] = permanent_path
                
                return permanent_variations
                
        except Exception as e:
            logger.error(f"Failed to generate logo variations: {e}")
            return {'original': logo_path}
    
    def _adjust_image_opacity(self, image: Image.Image, opacity: float) -> Image.Image:
        """Adjust image opacity while preserving transparency."""
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Create new image with adjusted alpha
        data = list(image.getdata())
        new_data = []
        
        for pixel in data:
            new_alpha = int(pixel[3] * opacity)
            new_data.append((pixel[0], pixel[1], pixel[2], new_alpha))
        
        new_image = Image.new('RGBA', image.size)
        new_image.putdata(new_data)
        return new_image
    
    def _create_integration_timeline(
        self, 
        placements: List[LogoPlacement], 
        total_duration: float
    ) -> Dict[str, Any]:
        """Create detailed timeline for logo appearances."""
        timeline = {
            'total_duration': total_duration,
            'logo_events': [],
            'style_transitions': [],
            'animation_cues': []
        }
        
        for placement in placements:
            event = {
                'scene': placement.scene_number,
                'start_time': placement.timing[0],
                'end_time': placement.timing[1],
                'position': placement.position,
                'animation_in': placement.animation,
                'animation_out': 'fade_out',
                'opacity_curve': self._generate_opacity_curve(placement)
            }
            timeline['logo_events'].append(event)
        
        return timeline
    
    def _generate_opacity_curve(self, placement: LogoPlacement) -> List[Tuple[float, float]]:
        """Generate opacity curve for smooth logo transitions."""
        duration = placement.timing[1] - placement.timing[0]
        max_opacity = placement.opacity
        
        if placement.animation == 'fade_in':
            return [
                (0.0, 0.0),
                (0.5, max_opacity * 0.8),
                (1.0, max_opacity)
            ]
        elif placement.animation == 'slide_in':
            return [
                (0.0, 0.0),
                (0.3, max_opacity),
                (1.0, max_opacity)
            ]
        else:  # static
            return [
                (0.0, max_opacity),
                (1.0, max_opacity)
            ]
    
    def _enhance_scene_prompts_with_logo(
        self,
        scenes: List[Dict[str, Any]],
        logo_analysis: LogoAnalysisResult,
        placements: List[LogoPlacement]
    ) -> Dict[str, str]:
        """Enhance video generation prompts with professional logo-aware styling."""
        enhanced_prompts = {}
        
        # Extract comprehensive brand intelligence
        primary_color = logo_analysis.color_palette.primary_color
        secondary_color = logo_analysis.color_palette.secondary_color
        accent_colors = logo_analysis.color_palette.accent_colors
        style_category = logo_analysis.style_profile.style_category
        brand_personality = logo_analysis.brand_personality
        color_harmony = logo_analysis.color_palette.color_harmony
        brightness = logo_analysis.color_palette.brightness
        saturation = logo_analysis.color_palette.saturation
        
        # Determine overall brand mood
        if brand_personality.get('energetic', 0) > 0.7:
            brand_mood = "dynamic and energetic"
            lighting_style = "vibrant, high-energy lighting"
        elif brand_personality.get('elegant', 0) > 0.7:
            brand_mood = "sophisticated and elegant"
            lighting_style = "soft, premium lighting with subtle shadows"
        elif brand_personality.get('professional', 0) > 0.7:
            brand_mood = "professional and trustworthy"
            lighting_style = "clean, corporate lighting"
        else:
            brand_mood = "modern and approachable"
            lighting_style = "natural, balanced lighting"
        
        for i, (scene, placement) in enumerate(zip(scenes, placements)):
            original_prompt = scene.get('luma_prompt', scene.get('visual_concept', ''))
            scene_purpose = scene.get('purpose', 'unknown')
            
            # Professional brand color integration
            color_integration = self._create_color_integration_prompt(
                primary_color, secondary_color, accent_colors, color_harmony, 
                brightness, saturation, scene_purpose
            )
            
            # Visual style consistency
            style_consistency = self._create_style_consistency_prompt(
                style_category, brand_personality, scene_purpose
            )
            
            # Composition and spacing for logo
            composition_guide = self._create_composition_prompt(
                placement, scene_purpose, logo_analysis.aspect_ratio
            )
            
            # Lighting and mood alignment
            lighting_mood = f"with {lighting_style}, {brand_mood} atmosphere, "
            
            # Professional quality standards
            quality_standards = (
                "shot with professional commercial cinematography, "
                "perfect color grading, high production value, "
                "brand-consistent visual identity, "
                "suitable for premium advertising"
            )
            
            # Construct enhanced prompt
            enhanced_prompt = (
                f"{original_prompt.rstrip('.')}. "
                f"{color_integration} {style_consistency} "
                f"{composition_guide} {lighting_mood} "
                f"{quality_standards}"
            )
            
            enhanced_prompts[f'scene_{i+1}'] = enhanced_prompt
            
            # Debug logging
            logger.info(
                f"Enhanced scene {i+1} prompt with logo integration",
                scene=i+1,
                brand_mood=brand_mood,
                primary_color=primary_color,
                style=style_category
            )
        
        return enhanced_prompts
    
    def _create_color_integration_prompt(
        self, primary: str, secondary: str, accents: List[str], 
        harmony: str, brightness: float, saturation: float, scene_purpose: str
    ) -> str:
        """Create professional color integration prompt."""
        
        # Adjust color emphasis based on scene purpose
        if scene_purpose == 'hook':
            color_emphasis = "subtle brand color accents"
        elif scene_purpose == 'problem':
            color_emphasis = "strategic brand color highlights"
        else:  # solution/cta
            color_emphasis = "prominent brand color integration"
        
        # Color temperature and mood
        if brightness > 0.6:
            color_mood = "bright, optimistic color palette"
        else:
            color_mood = "sophisticated, premium color palette"
        
        if saturation > 0.6:
            saturation_desc = "vibrant, engaging colors"
        else:
            saturation_desc = "refined, professional tones"
        
        return (
            f"Incorporating {color_emphasis} featuring {primary} as primary brand color, "
            f"{secondary} as supporting tone, maintaining {color_mood} with {saturation_desc}, "
            f"following {harmony} color harmony principles."
        )
    
    def _create_style_consistency_prompt(
        self, style_category: str, personality: Dict[str, float], scene_purpose: str
    ) -> str:
        """Create style consistency prompt based on brand personality."""
        
        style_descriptors = {
            'minimalist': 'clean, uncluttered compositions with purposeful negative space',
            'bold': 'strong, confident visual elements with impactful presence',
            'modern': 'contemporary, sleek design with innovative touches',
            'classic': 'timeless, refined aesthetics with traditional elegance',
            'playful': 'creative, engaging visuals with dynamic energy'
        }
        
        base_style = style_descriptors.get(style_category, 'professional, polished visual design')
        
        # Add personality-driven enhancements
        personality_enhancements = []
        if personality.get('innovative', 0) > 0.6:
            personality_enhancements.append("forward-thinking visual approach")
        if personality.get('trustworthy', 0) > 0.6:
            personality_enhancements.append("reliable, dependable visual language")
        if personality.get('energetic', 0) > 0.6:
            personality_enhancements.append("dynamic, motivating visual energy")
        
        enhancement_text = ", ".join(personality_enhancements) if personality_enhancements else "brand-aligned visual identity"
        
        return f"Maintaining {base_style}, conveying {enhancement_text}."
    
    def _create_composition_prompt(
        self, placement: LogoPlacement, scene_purpose: str, logo_aspect: float
    ) -> str:
        """Create composition guidance for logo placement."""
        
        position_guides = {
            'top_right_corner': 'preserving clean top-right corner space for branding with balanced composition',
            'bottom_right_corner': 'maintaining unobstructed bottom-right area for logo integration',
            'center_bottom': 'keeping lower third clear for central branding placement',
            'watermark': 'designing with subtle overlay space consideration',
            'center': 'balanced composition allowing for centered brand element integration'
        }
        
        position_guide = position_guides.get(placement.position, 'thoughtful composition for brand integration')
        
        # Adjust for scene purpose
        if scene_purpose == 'hook':
            composition_focus = "drawing attention without overwhelming the brand space"
        elif scene_purpose == 'problem':
            composition_focus = "maintaining visual hierarchy with brand visibility"
        else:  # solution/cta
            composition_focus = "emphasizing call-to-action while showcasing brand prominence"
        
        return f"Composed with {position_guide}, {composition_focus}."
    
    def _generate_ffmpeg_filters(self, placements: List[LogoPlacement]) -> List[str]:
        """Generate FFmpeg filter commands for logo overlay."""
        filters = []
        
        for placement in placements:
            # Calculate position coordinates
            position_coords = self._calculate_ffmpeg_position(placement.position)
            
            # Create overlay filter with timing
            filter_cmd = (
                f"overlay={position_coords['x']}:{position_coords['y']}:"
                f"enable='between(t,{placement.timing[0]:.2f},{placement.timing[1]:.2f})':"
                f"alpha={placement.opacity}"
            )
            
            filters.append(filter_cmd)
        
        return filters
    
    def _calculate_ffmpeg_position(self, position: str) -> Dict[str, str]:
        """Calculate FFmpeg overlay position coordinates."""
        position_map = {
            'top_right_corner': {'x': 'W-w-20', 'y': '20'},
            'bottom_right_corner': {'x': 'W-w-20', 'y': 'H-h-20'},
            'center_bottom': {'x': '(W-w)/2', 'y': 'H-h-40'},
            'watermark': {'x': 'W-w-10', 'y': 'H-h-10'},
            'center': {'x': '(W-w)/2', 'y': '(H-h)/2'}
        }
        
        return position_map.get(position, position_map['bottom_right_corner'])
    
    def _create_fallback_plan(self, architecture: Dict[str, Any]) -> Dict[str, Any]:
        """Create basic fallback logo integration plan."""
        scenes = architecture.get('scene_architecture', {}).get('scenes', [])
        
        return {
            'logo_analysis': None,
            'logo_placements': [],
            'logo_variations': {},
            'integration_timeline': {'total_duration': 18.0, 'logo_events': []},
            'enhanced_scene_prompts': {},
            'ffmpeg_filters': [],
            'brand_color_palette': {
                'primary': '#333333',
                'secondary': '#666666',
                'accents': ['#999999']
            },
            'fallback': True
        }


# Global logo integration service instance  
logo_integration_service = LogoIntegrationService()

# Import time for timestamp generation
import time
from dataclasses import asdict