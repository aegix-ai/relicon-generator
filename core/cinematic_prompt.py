"""
Cinematic Prompt Module for Relicon
This module contains prompts and methods for generating hyperrealistic, cinematic ad scenes.
"""

from typing import Dict, Any

def generate_cinematic_ad_scene_prompt(brand_info: Dict[str, Any], niche: str, target_audience: str, duration: int, 
                                     business_type: str = None, brand_colors: list = None, scene_purpose: str = None) -> str:
    """
    Generate a professional, production-ready prompt for creating cinematic ad scenes with technical specifications.
    """
    brand_name = brand_info.get('name', 'the brand')
    brand_personality = brand_info.get('personality', 'innovative and professional')
    brand_values = brand_info.get('values', 'excellence and customer satisfaction')
    
    # Enhanced brand color integration
    color_guidance = ""
    if brand_colors:
        color_hex = ", ".join(brand_colors[:3])  # Use top 3 brand colors
        color_guidance = f"\n- **Brand Color Integration**: Incorporate brand colors ({color_hex}) naturally in lighting, backgrounds, or environmental elements"
    
    # Business type specific requirements
    business_focus = {
        'product': "product showcases with detailed close-ups, unboxing sequences, feature demonstrations",
        'service': "professional environments, expertise demonstrations, client interaction scenarios",
        'platform': "user interface interactions, connectivity visualizations, ecosystem demonstrations",
        'hybrid': "integrated solution showcases combining product features with service excellence"
    }
    
    focus_requirements = business_focus.get(business_type, "professional brand representation")
    
    # Scene purpose specific guidance
    purpose_guidance = {
        'hook': "attention-grabbing opening with strong visual impact and immediate brand recognition",
        'problem': "relatable problem scenarios with emotional resonance and clear pain points",
        'solution': "clear solution demonstration with benefit visualization and professional execution",
        'cta': "compelling call-to-action with brand prominence and clear next steps"
    }
    
    scene_specific = purpose_guidance.get(scene_purpose, "engaging brand storytelling")

    prompt = f"""
You are a world-class creative director and cinematographer creating a PRODUCTION-READY commercial advertisement scene for {brand_name}. This scene will be part of a professional video campaign targeting {target_audience} in the {niche} industry.

**TECHNICAL SPECIFICATIONS:**
- Duration: {duration} seconds (exact timing critical)
- Resolution: 720p minimum quality for professional broadcast
- Aspect Ratio: 9:16 vertical format (mobile-optimized)
- Frame Rate: Smooth 24fps cinematic quality
- Lighting: Professional three-point lighting setup with soft shadows
- Color Grading: Cinema-quality color correction with consistent brand palette{color_guidance}

**CREATIVE REQUIREMENTS:**
- Scene Purpose: {scene_specific}
- Business Focus: {focus_requirements}
- Brand Personality: {brand_personality}
- Core Values: {brand_values}
- Visual Style: Hyperrealistic commercial photography quality

**MANDATORY QUALITY STANDARDS:**
1. **Visual Excellence**: 
   - Sharp focus with professional depth of field
   - Consistent lighting without harsh shadows or overexposure
   - Smooth camera movements with stabilized footage
   - No visual artifacts, grain, or technical imperfections

2. **Composition Standards**:
   - Rule of thirds for subject placement
   - Leading lines to guide viewer attention
   - Balanced frame composition with breathing room
   - Brand elements integrated naturally without obstruction

3. **Technical Requirements**:
   - No text overlays, watermarks, or burned-in captions
   - Clean audio recording environment (if applicable)
   - Consistent color temperature throughout scene
   - Professional-grade motion blur and natural transitions

4. **Brand Integration**:
   - Subtle but memorable brand presence
   - Colors that complement brand palette
   - Professional environment appropriate for brand positioning
   - Clear brand story progression

**STRICT AVOIDANCE CRITERIA:**
- Amateur handheld camera shake or poor stabilization
- Harsh lighting creating unflattering shadows
- Cluttered backgrounds or distracting elements  
- Generic stock footage aesthetics
- Inconsistent visual quality between shots
- Technical artifacts or rendering issues

**OUTPUT FORMAT:**
Provide a detailed scene specification in JSON format:
{{
    "scene_title": "Professional scene identifier",
    "technical_specs": {{
        "camera_setup": "Specific camera angle, lens, and movement instructions",
        "lighting_design": "Professional lighting setup with key, fill, and rim lights",
        "composition_notes": "Frame composition and subject positioning"
    }},
    "visual_narrative": {{
        "setting": "Detailed environment description with professional elements",
        "subject_action": "Clear subject movements and interactions",
        "brand_integration": "How brand elements appear naturally in scene"
    }},
    "quality_assurance": {{
        "visual_checklist": ["Key quality checkpoints for this scene"],
        "brand_consistency": "How scene maintains brand identity",
        "production_notes": "Technical considerations for professional execution"
    }}
}}

**FINAL REQUIREMENT:** This scene must meet broadcast television commercial standards. Every visual element should contribute to a cohesive, premium brand experience that drives viewer action while maintaining absolute technical and creative excellence.
"""
    return prompt
