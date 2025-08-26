"""
Advanced Logo Analysis Service
Extracts colors, style, typography information from logos for dynamic video integration.
"""

import os
import json
import colorsys
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from sklearn.cluster import KMeans
from core.logger import get_logger

logger = get_logger(__name__)

@dataclass
class ColorPalette:
    """Represents extracted color palette from logo."""
    primary_color: str  # Hex color
    secondary_color: str
    accent_colors: List[str]
    dominant_colors: List[str]
    color_harmony: str  # monochromatic, complementary, triadic, etc.
    brightness: float  # 0-1 scale
    saturation: float  # 0-1 scale

@dataclass
class StyleProfile:
    """Represents logo style characteristics."""
    style_category: str  # modern, classic, minimalist, bold, etc.
    geometric_complexity: float  # 0-1 scale
    text_to_image_ratio: float  # 0-1 scale
    edge_density: float  # 0-1 scale
    symmetry_score: float  # 0-1 scale
    recommended_fonts: List[str]
    style_keywords: List[str]

@dataclass
class LogoAnalysisResult:
    """Complete logo analysis result."""
    logo_path: str
    color_palette: ColorPalette
    style_profile: StyleProfile
    dimensions: Tuple[int, int]
    aspect_ratio: float
    dominant_shapes: List[str]
    typography_analysis: Dict[str, Any]
    brand_personality: Dict[str, float]
    video_integration_suggestions: Dict[str, Any]

class LogoAnalysisService:
    """Advanced logo analysis with computer vision and style extraction."""
    
    def __init__(self):
        self.supported_formats = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']
        
    def analyze_logo(self, logo_path: str) -> Optional[LogoAnalysisResult]:
        """
        Perform comprehensive logo analysis.
        
        Args:
            logo_path: Path to logo image file
            
        Returns:
            Complete logo analysis result
        """
        try:
            logger.info("Starting logo analysis", action="logo.analysis.start", logo_path=logo_path)
            
            if not self._validate_logo_file(logo_path):
                logger.error("Logo file validation failed", logo_path=logo_path)
                return None
            
            # Load and preprocess image
            image = self._load_and_preprocess_image(logo_path)
            if image is None:
                return None
            
            # Extract color palette
            color_palette = self._extract_color_palette(image)
            
            # Analyze style characteristics
            style_profile = self._analyze_style_profile(image)
            
            # Analyze typography (if text is present)
            typography_analysis = self._analyze_typography(image)
            
            # Determine brand personality
            brand_personality = self._analyze_brand_personality(color_palette, style_profile)
            
            # Generate video integration suggestions
            video_suggestions = self._generate_video_integration_suggestions(
                color_palette, style_profile, brand_personality
            )
            
            # Create analysis result
            result = LogoAnalysisResult(
                logo_path=logo_path,
                color_palette=color_palette,
                style_profile=style_profile,
                dimensions=image.size,
                aspect_ratio=image.size[0] / image.size[1],
                dominant_shapes=self._detect_dominant_shapes(image),
                typography_analysis=typography_analysis,
                brand_personality=brand_personality,
                video_integration_suggestions=video_suggestions
            )
            
            logger.info(
                "Logo analysis completed",
                action="logo.analysis.complete",
                primary_color=color_palette.primary_color,
                style=style_profile.style_category
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Logo analysis failed: {e}", exc_info=True)
            return None
    
    def _validate_logo_file(self, logo_path: str) -> bool:
        """Validate logo file exists and has supported format."""
        if not os.path.exists(logo_path):
            return False
        
        file_ext = Path(logo_path).suffix.lower()
        return file_ext in self.supported_formats
    
    def _load_and_preprocess_image(self, logo_path: str) -> Optional[Image.Image]:
        """Load and preprocess logo image."""
        try:
            image = Image.open(logo_path)
            
            # Convert to RGBA for transparency support
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            
            # Resize if too large (for performance)
            max_size = 1024
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            return image
            
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return None
    
    def _extract_color_palette(self, image: Image.Image) -> ColorPalette:
        """Extract comprehensive color palette from logo."""
        try:
            # Convert to RGB array, removing transparent pixels
            img_array = np.array(image)
            
            # Filter out transparent pixels
            alpha_channel = img_array[:, :, 3]
            opaque_mask = alpha_channel > 128
            
            rgb_pixels = img_array[opaque_mask][:, :3]
            
            if len(rgb_pixels) == 0:
                # Fallback to default colors if image is mostly transparent
                return self._create_default_color_palette()
            
            # Use KMeans clustering to find dominant colors
            n_colors = min(8, len(rgb_pixels) // 100 + 1)
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(rgb_pixels)
            
            # Get dominant colors
            dominant_colors = []
            for color in kmeans.cluster_centers_:
                hex_color = self._rgb_to_hex(tuple(color.astype(int)))
                dominant_colors.append(hex_color)
            
            # Sort by frequency
            labels = kmeans.labels_
            color_counts = np.bincount(labels)
            sorted_indices = np.argsort(color_counts)[::-1]
            
            dominant_colors = [dominant_colors[i] for i in sorted_indices]
            
            # Determine primary and secondary colors
            primary_color = dominant_colors[0] if dominant_colors else "#000000"
            secondary_color = dominant_colors[1] if len(dominant_colors) > 1 else primary_color
            accent_colors = dominant_colors[2:5] if len(dominant_colors) > 2 else [primary_color]
            
            # Analyze color properties
            primary_rgb = self._hex_to_rgb(primary_color)
            primary_hsv = colorsys.rgb_to_hsv(
                primary_rgb[0]/255, primary_rgb[1]/255, primary_rgb[2]/255
            )
            
            brightness = primary_hsv[2]
            saturation = primary_hsv[1]
            
            # Determine color harmony
            color_harmony = self._analyze_color_harmony(dominant_colors[:4])
            
            return ColorPalette(
                primary_color=primary_color,
                secondary_color=secondary_color,
                accent_colors=accent_colors,
                dominant_colors=dominant_colors,
                color_harmony=color_harmony,
                brightness=brightness,
                saturation=saturation
            )
            
        except Exception as e:
            logger.error(f"Color palette extraction failed: {e}")
            return self._create_default_color_palette()
    
    def _analyze_style_profile(self, image: Image.Image) -> StyleProfile:
        """Analyze logo style characteristics."""
        try:
            # Convert to grayscale for analysis
            gray_image = image.convert('L')
            img_array = np.array(gray_image)
            
            # Calculate edge density using Sobel operator
            from scipy import ndimage
            sobel_x = ndimage.sobel(img_array, axis=0)
            sobel_y = ndimage.sobel(img_array, axis=1)
            edge_magnitude = np.hypot(sobel_x, sobel_y)
            edge_density = np.mean(edge_magnitude) / 255.0
            
            # Calculate geometric complexity
            non_zero_pixels = np.count_nonzero(img_array)
            total_pixels = img_array.size
            geometric_complexity = min(edge_density * 2, 1.0)
            
            # Analyze symmetry
            symmetry_score = self._calculate_symmetry(img_array)
            
            # Determine text ratio (simplified)
            text_to_image_ratio = self._estimate_text_ratio(img_array)
            
            # Determine style category
            style_category = self._classify_style(
                edge_density, geometric_complexity, symmetry_score, text_to_image_ratio
            )
            
            # Recommend fonts based on style
            recommended_fonts = self._recommend_fonts(style_category, edge_density)
            
            # Generate style keywords
            style_keywords = self._generate_style_keywords(
                style_category, edge_density, geometric_complexity
            )
            
            return StyleProfile(
                style_category=style_category,
                geometric_complexity=geometric_complexity,
                text_to_image_ratio=text_to_image_ratio,
                edge_density=edge_density,
                symmetry_score=symmetry_score,
                recommended_fonts=recommended_fonts,
                style_keywords=style_keywords
            )
            
        except Exception as e:
            logger.error(f"Style profile analysis failed: {e}")
            return self._create_default_style_profile()
    
    def _analyze_typography(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze typography characteristics in the logo."""
        # This is a simplified implementation
        # In production, you might use OCR libraries like pytesseract
        return {
            'has_text': True,  # Assume logos may have text
            'text_style': 'modern',
            'font_weight': 'bold',
            'text_position': 'integrated'
        }
    
    def _analyze_brand_personality(self, color_palette: ColorPalette, style_profile: StyleProfile) -> Dict[str, float]:
        """Analyze brand personality traits based on visual elements."""
        personality = {
            'professional': 0.5,
            'modern': 0.5,
            'trustworthy': 0.5,
            'innovative': 0.5,
            'energetic': 0.5,
            'elegant': 0.5,
            'playful': 0.5
        }
        
        # Adjust based on color palette
        if color_palette.brightness > 0.7:
            personality['energetic'] += 0.2
            personality['playful'] += 0.1
        else:
            personality['professional'] += 0.2
            personality['trustworthy'] += 0.1
        
        if color_palette.saturation > 0.6:
            personality['energetic'] += 0.2
            personality['innovative'] += 0.1
        else:
            personality['elegant'] += 0.2
            personality['professional'] += 0.1
        
        # Adjust based on style
        if style_profile.style_category == 'minimalist':
            personality['modern'] += 0.3
            personality['elegant'] += 0.2
        elif style_profile.style_category == 'bold':
            personality['energetic'] += 0.3
            personality['innovative'] += 0.2
        
        # Normalize values
        for key in personality:
            personality[key] = min(personality[key], 1.0)
        
        return personality
    
    def _generate_video_integration_suggestions(
        self, color_palette: ColorPalette, style_profile: StyleProfile, brand_personality: Dict[str, float]
    ) -> Dict[str, Any]:
        """Generate suggestions for integrating logo into video."""
        suggestions = {
            'recommended_placement': [],
            'animation_style': 'fade',
            'timing_suggestions': {},
            'color_integration': {},
            'style_adaptations': []
        }
        
        # Placement suggestions based on style
        if style_profile.style_category == 'minimalist':
            suggestions['recommended_placement'] = ['corner_overlay', 'end_card']
        elif style_profile.style_category == 'bold':
            suggestions['recommended_placement'] = ['center_intro', 'corner_overlay']
        else:
            suggestions['recommended_placement'] = ['corner_overlay', 'end_card', 'watermark']
        
        # Animation style based on brand personality
        if brand_personality.get('energetic', 0) > 0.7:
            suggestions['animation_style'] = 'slide_in'
        elif brand_personality.get('elegant', 0) > 0.7:
            suggestions['animation_style'] = 'fade'
        else:
            suggestions['animation_style'] = 'scale_in'
        
        # Color integration
        suggestions['color_integration'] = {
            'use_brand_colors_in_text': True,
            'background_tint': color_palette.primary_color,
            'accent_elements': color_palette.accent_colors[:2]
        }
        
        # Timing suggestions
        suggestions['timing_suggestions'] = {
            'intro_duration': 1.0,
            'outro_duration': 2.0,
            'watermark_opacity': 0.3 if style_profile.geometric_complexity < 0.5 else 0.5
        }
        
        return suggestions
    
    def _create_default_color_palette(self) -> ColorPalette:
        """Create default color palette as fallback."""
        return ColorPalette(
            primary_color="#333333",
            secondary_color="#666666",
            accent_colors=["#999999"],
            dominant_colors=["#333333", "#666666", "#999999"],
            color_harmony="monochromatic",
            brightness=0.4,
            saturation=0.2
        )
    
    def _create_default_style_profile(self) -> StyleProfile:
        """Create default style profile as fallback."""
        return StyleProfile(
            style_category="modern",
            geometric_complexity=0.5,
            text_to_image_ratio=0.3,
            edge_density=0.4,
            symmetry_score=0.5,
            recommended_fonts=["Arial", "Helvetica", "Open Sans"],
            style_keywords=["modern", "clean", "professional"]
        )
    
    # Helper methods
    def _rgb_to_hex(self, rgb: Tuple[int, int, int]) -> str:
        """Convert RGB tuple to hex color."""
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def _analyze_color_harmony(self, colors: List[str]) -> str:
        """Analyze color harmony type."""
        if len(colors) < 2:
            return "monochromatic"
        
        # Simplified color harmony analysis
        # In production, this would be more sophisticated
        return "complementary"
    
    def _calculate_symmetry(self, img_array: np.ndarray) -> float:
        """Calculate symmetry score of the image."""
        try:
            height, width = img_array.shape
            
            # Horizontal symmetry
            left_half = img_array[:, :width//2]
            right_half = np.fliplr(img_array[:, width//2:])
            
            # Resize to same dimensions
            min_width = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_width]
            right_half = right_half[:, :min_width]
            
            # Calculate similarity
            diff = np.abs(left_half.astype(float) - right_half.astype(float))
            symmetry = 1.0 - (np.mean(diff) / 255.0)
            
            return max(0.0, min(1.0, symmetry))
            
        except Exception:
            return 0.5
    
    def _estimate_text_ratio(self, img_array: np.ndarray) -> float:
        """Estimate the ratio of text to image content."""
        # Simplified implementation
        # Could be improved with OCR or text detection
        return 0.3
    
    def _classify_style(self, edge_density: float, geometric_complexity: float, 
                       symmetry_score: float, text_ratio: float) -> str:
        """Classify logo style based on calculated metrics."""
        if edge_density < 0.2 and geometric_complexity < 0.3:
            return "minimalist"
        elif edge_density > 0.6 or geometric_complexity > 0.7:
            return "bold"
        elif symmetry_score > 0.7:
            return "classic"
        elif text_ratio > 0.6:
            return "typography-focused"
        else:
            return "modern"
    
    def _recommend_fonts(self, style_category: str, edge_density: float) -> List[str]:
        """Recommend fonts based on logo style."""
        font_recommendations = {
            "minimalist": ["Helvetica Neue", "Avenir", "Futura"],
            "bold": ["Impact", "Bebas Neue", "Oswald"],
            "classic": ["Times New Roman", "Georgia", "Trajan"],
            "modern": ["Open Sans", "Lato", "Montserrat"],
            "typography-focused": ["Custom", "Brand Font", "Helvetica"]
        }
        
        return font_recommendations.get(style_category, ["Arial", "Helvetica", "sans-serif"])
    
    def _generate_style_keywords(self, style_category: str, edge_density: float, 
                                geometric_complexity: float) -> List[str]:
        """Generate descriptive keywords for the logo style."""
        keywords = [style_category]
        
        if edge_density < 0.3:
            keywords.extend(["clean", "simple"])
        elif edge_density > 0.6:
            keywords.extend(["detailed", "complex"])
        
        if geometric_complexity < 0.3:
            keywords.append("geometric")
        elif geometric_complexity > 0.6:
            keywords.extend(["organic", "intricate"])
        
        return keywords
    
    def _detect_dominant_shapes(self, image: Image.Image) -> List[str]:
        """Detect dominant geometric shapes in the logo."""
        # Simplified implementation
        # Could be enhanced with computer vision libraries
        return ["circular", "angular"]


# Global logo analysis service instance
logo_analysis_service = LogoAnalysisService()