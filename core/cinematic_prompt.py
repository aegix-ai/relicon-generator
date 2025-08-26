"""
Cinematic Prompt Module for Relicon
This module contains prompts and methods for generating hyperrealistic, cinematic ad scenes.
"""

from typing import Dict, Any

def generate_cinematic_ad_scene_prompt(brand_info: Dict[str, Any], niche: str, target_audience: str, duration: int) -> str:
    """
    Generate a prompt for GPT-4o to create a hyperrealistic, cinematic ad scene tailored to any business niche.
    """
    brand_name = brand_info.get('name', 'the brand')
    brand_personality = brand_info.get('personality', 'innovative and professional')
    brand_values = brand_info.get('values', 'excellence and customer satisfaction')

    prompt = f"""
You are a world-class creative director tasked with generating a cinematic, hyperrealistic advertisement scene for {brand_name}, a {niche} business. The ad must be visually stunning, emotionally engaging, and professionally crafted to resonate with {target_audience}. The scene should last approximately {duration} seconds.

**Brand Information:**
- Personality: {brand_personality}
- Core Values: {brand_values}

**Objective:**
Create a single, dynamic ad scene that showcases the essence of {brand_name} through a hyperrealistic and cinematic lens. The scene should:
1. Highlight the unique aspects of a {niche} business, focusing on services or solutions rather than just products.
2. Use vivid, detailed imagery to create a hyperrealistic experience (e.g., intricate lighting, lifelike textures, dramatic camera angles).
3. Incorporate storytelling elements that evoke emotion and connect with the audience.
4. Suggest background music or sound design that complements the cinematic feel.

**Output Format:**
Provide the scene description in a structured JSON format with the following fields:
- 'scene_title': A short, catchy title for the scene.
- 'visual_description': A detailed paragraph describing the visuals, including setting, characters, and actions.
- 'dialogue_or_voiceover': Any spoken content or voiceover script for the scene.
- 'sound_design': Description of background music or sound effects.
- 'camera_movement': Specific instructions for camera angles or movements to enhance the cinematic effect.
- 'emotional_impact': The intended emotional response from the audience.
- 'niche_relevance': How the scene connects specifically to the {niche} industry.

Ensure the scene feels premium, professional, and tailored to {brand_name}'s identity. Avoid generic or overly product-focused content; instead, emphasize the brand's value and impact in its niche.
"""
    return prompt
