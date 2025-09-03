"""
Professional Niche-Specific Prompt Engineering Templates
Generates high-accuracy video prompts adapted to business niches and brand requirements.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from core.brand_intelligence import BusinessNiche, BrandElements, BusinessType
from core.universal_brand_intelligence import UniversalBrandIntelligenceGenerator
import openai
import json
import re

def safe_parse_json_response(response_content: str, context: str) -> Dict[str, Any]:
    """
    Safely parse JSON response with robust error handling and extraction.
    
    Args:
        response_content: Raw response content from AI
        context: Context for error messages
        
    Returns:
        Parsed JSON dictionary
        
    Raises:
        ValueError: If JSON parsing fails
    """
    try:
        if not response_content or not response_content.strip():
            raise ValueError(f"Empty {context}")
        
        response_content = response_content.strip()
        
        # Handle markdown-wrapped JSON
        if response_content.startswith("```"):
            # Extract JSON from markdown code block
            json_match = re.search(r"```(?:json)?\s*({.*?})\s*```", response_content, re.DOTALL)
            if json_match:
                response_content = json_match.group(1)
            else:
                raise ValueError(f"No valid JSON found in {context}")
        
        # Try to extract JSON object if response contains other text
        if not response_content.startswith('{'):
            json_match = re.search(r"\{.*\}", response_content, re.DOTALL)
            if json_match:
                response_content = json_match.group()
            else:
                raise ValueError(f"No valid JSON found in {context}")
        
        return json.loads(response_content)
        
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parsing error in {context}: {e}")
    except Exception as e:
        raise ValueError(f"Failed to parse {context} JSON: {e}")


@dataclass
class SceneTemplate:
    """Template for a single video scene."""
    scene_type: str  # 'hook', 'problem', 'solution'
    duration: int    # seconds
    visual_focus: str
    narrative_structure: str
    technical_specs: Dict[str, str]

class NichePromptTemplateEngine:
    """Advanced prompt engineering engine with niche-specific templates."""
    
    def __init__(self):
        self.niche_templates = self._initialize_niche_templates()
        self.scene_structures = self._initialize_scene_structures()
        self.technical_specifications = self._initialize_technical_specs()
        self.universal_brand_generator = UniversalBrandIntelligenceGenerator()
    
    def _initialize_niche_templates(self) -> Dict[BusinessNiche, Dict[str, Any]]:
        """Initialize comprehensive niche-specific prompt templates."""
        return {
            BusinessNiche.TECHNOLOGY: {
                'visual_style': 'sleek, modern, high-tech, digital interfaces, clean environments',
                'color_palette': ['blue', 'white', 'silver', 'neon accents'],
                'scene_settings': [
                    'server room with rows of humming machines and blue LED indicators',
                    'holographic interface projections in darkened control center', 
                    'macro shots of circuit boards with electrical currents flowing',
                    'data center with cables and fiber optic light streams',
                    'robotic assembly line with precision mechanical movements',
                    'cloud computing visualization with abstract data particle flows',
                    'cybersecurity command center with real-time threat mapping displays',
                    'modern office with multiple monitors showing live data analytics'
                ],
                'product_showcase': 'screen recordings, interface demonstrations, data visualizations',
                'lifestyle_context': 'professional productivity, digital transformation, innovation',
                'brand_elements': 'logos on screens, branded interfaces, digital branding',
                'call_to_action_style': 'modern button overlays, website URLs, QR codes'
            },
            
            BusinessNiche.HEALTHCARE: {
                'visual_style': 'clean, professional, trustworthy, medical environments, calming tones',
                'color_palette': ['white', 'blue', 'green', 'soft pastels'],
                'scene_settings': [
                    'modern medical facility with natural lighting',
                    'clean examination room or consultation space',
                    'peaceful wellness environment'
                ],
                'product_showcase': 'medical devices in use, health monitoring, professional consultations',
                'lifestyle_context': 'health improvement, wellness journey, professional care',
                'brand_elements': 'medical logos, certification badges, professional branding',
                'call_to_action_style': 'professional contact information, appointment scheduling'
            },
            
            BusinessNiche.FOOD_BEVERAGE: {
                'visual_style': 'appetizing, fresh, vibrant, natural lighting, food photography',
                'color_palette': ['warm tones', 'natural colors', 'vibrant greens', 'golden browns'],
                'scene_settings': [
                    'modern kitchen with natural lighting',
                    'fresh ingredient displays',
                    'dining environment with happy customers'
                ],
                'product_showcase': 'food preparation, ingredient close-ups, final dish presentation',
                'lifestyle_context': 'family dining, healthy eating, culinary enjoyment',
                'brand_elements': 'packaging display, restaurant branding, logo on uniforms',
                'call_to_action_style': 'menu highlights, delivery apps, location information'
            },
            
            BusinessNiche.FASHION_BEAUTY: {
                'visual_style': 'stylish, elegant, trendy, fashion photography, beauty lighting',
                'color_palette': ['sophisticated neutrals', 'fashion colors', 'luxury tones'],
                'scene_settings': [
                    'stylish boutique or showroom',
                    'elegant beauty salon or spa',
                    'fashionable lifestyle environments'
                ],
                'product_showcase': 'product styling, before/after transformations, lifestyle modeling',
                'lifestyle_context': 'personal style, beauty transformation, fashion confidence',
                'brand_elements': 'designer labels, luxury packaging, brand accessories',
                'call_to_action_style': 'shopping interfaces, style consultations, brand websites'
            },
            
            BusinessNiche.FITNESS_WELLNESS: {
                'visual_style': 'energetic, motivational, dynamic movement, natural lighting',
                'color_palette': ['vibrant colors', 'energy tones', 'natural greens', 'motivational oranges'],
                'scene_settings': [
                    'modern fitness facility with equipment',
                    'outdoor workout environments',
                    'wellness studio with natural elements'
                ],
                'product_showcase': 'exercise demonstrations, fitness equipment in use, wellness practices',
                'lifestyle_context': 'fitness journey, health transformation, active lifestyle',
                'brand_elements': 'fitness logos, workout apparel branding, equipment branding',
                'call_to_action_style': 'membership offers, fitness app downloads, class scheduling'
            },
            
            BusinessNiche.FINANCE: {
                'visual_style': 'professional, trustworthy, sophisticated, corporate environments',
                'color_palette': ['navy blue', 'white', 'gold accents', 'professional grays'],
                'scene_settings': [
                    'executive office with city views',
                    'modern banking environment',
                    'professional consultation space'
                ],
                'product_showcase': 'financial dashboards, consultation meetings, success metrics',
                'lifestyle_context': 'financial security, wealth building, professional success',
                'brand_elements': 'corporate logos, financial certifications, professional branding',
                'call_to_action_style': 'consultation bookings, financial planning, contact information'
            },
            
            BusinessNiche.REAL_ESTATE: {
                'visual_style': 'luxurious, spacious, architectural beauty, natural lighting',
                'color_palette': ['warm neutrals', 'luxury tones', 'architectural whites', 'natural colors'],
                'scene_settings': [
                    'stunning property interiors with natural light',
                    'architectural exterior shots',
                    'luxury amenities and features'
                ],
                'product_showcase': 'property tours, architectural features, luxury amenities',
                'lifestyle_context': 'dream home, luxury living, investment opportunities',
                'brand_elements': 'real estate signage, agency branding, professional presentations',
                'call_to_action_style': 'property listings, agent contact, virtual tours'
            },
            
            BusinessNiche.E_COMMERCE: {
                'visual_style': 'product-focused, lifestyle integration, shopping experience',
                'color_palette': ['brand-specific colors', 'shopping environment tones', 'product highlights'],
                'scene_settings': [
                    'lifestyle environments with products in use',
                    'clean product display backgrounds',
                    'shopping and unboxing experiences'
                ],
                'product_showcase': 'product demonstrations, lifestyle usage, unboxing experiences',
                'lifestyle_context': 'convenience shopping, product benefits, customer satisfaction',
                'brand_elements': 'product packaging, brand logos, shopping interfaces',
                'call_to_action_style': 'shopping buttons, discount codes, website links'
            },
            
            BusinessNiche.PROFESSIONAL_SERVICES: {
                'visual_style': 'professional, trustworthy, business-focused, corporate environments',
                'color_palette': ['navy blue', 'white', 'professional grays', 'brand accents'],
                'scene_settings': [
                    'modern office environment with professional lighting',
                    'consultation or meeting space with clean design',
                    'business setting with corporate branding'
                ],
                'product_showcase': 'service demonstrations, professional consultations, expertise display',
                'lifestyle_context': 'business success, professional growth, reliable partnerships',
                'brand_elements': 'professional branding, certifications, corporate identity',
                'call_to_action_style': 'consultation booking, professional contact, service inquiry'
            },
            
            BusinessNiche.HOME_LIFESTYLE: {
                'visual_style': 'comfortable, stylish, home-focused, lifestyle photography',
                'color_palette': ['warm neutrals', 'home colors', 'lifestyle tones', 'cozy accents'],
                'scene_settings': [
                    'beautiful home interior with natural lighting',
                    'lifestyle setting with home products',
                    'comfortable living space with style elements'
                ],
                'product_showcase': 'home products in use, lifestyle integration, comfort demonstration',
                'lifestyle_context': 'home comfort, lifestyle improvement, family life',
                'brand_elements': 'home branding, lifestyle logos, product integration',
                'call_to_action_style': 'home shopping, lifestyle catalogs, comfort solutions'
            },
            
            BusinessNiche.ENTERTAINMENT: {
                'visual_style': 'dynamic, engaging, entertainment-focused, media production',
                'color_palette': ['vibrant colors', 'entertainment tones', 'dynamic accents', 'media colors'],
                'scene_settings': [
                    'entertainment venue or media environment',
                    'dynamic performance or content space',
                    'engaging audience or community setting'
                ],
                'product_showcase': 'entertainment content, performance demonstrations, audience engagement',
                'lifestyle_context': 'entertainment enjoyment, content consumption, community participation',
                'brand_elements': 'entertainment branding, media logos, content integration',
                'call_to_action_style': 'content subscription, event tickets, entertainment access'
            },
            
            BusinessNiche.SUSTAINABILITY: {
                'visual_style': 'natural, eco-friendly, green environments, sustainability focus',
                'color_palette': ['natural greens', 'earth tones', 'sustainable colors', 'eco accents'],
                'scene_settings': [
                    'natural environment with eco-friendly elements',
                    'sustainable facility or green space',
                    'environmental setting with conservation focus'
                ],
                'product_showcase': 'eco-friendly products, sustainability practices, environmental benefits',
                'lifestyle_context': 'environmental responsibility, sustainable living, green choices',
                'brand_elements': 'eco branding, sustainability certifications, green identity',
                'call_to_action_style': 'sustainable choices, environmental action, green solutions'
            }
        }
    
    def _initialize_scene_structures(self) -> Dict[str, SceneTemplate]:
        """Initialize 3-scene video structure templates optimized for 30-second format."""
        return {
            'hook_scene': SceneTemplate(
                scene_type='hook',
                duration=10,
                visual_focus='attention_grabbing_product_or_lifestyle_moment',
                narrative_structure='visual_intrigue_with_brand_introduction',
                technical_specs={
                    'camera_movement': 'dynamic establishing shots, product reveals',
                    'lighting': 'professional, brand-appropriate lighting',
                    'audio': 'engaging music with brand introduction',
                    'pacing': 'quick cuts to maintain attention'
                }
            ),
            'problem_solution_scene': SceneTemplate(
                scene_type='problem_solution',
                duration=10,
                visual_focus='problem_demonstration_followed_by_solution',
                narrative_structure='problem_identification_and_immediate_solution_demonstration',
                technical_specs={
                    'camera_movement': 'problem close-ups, solution demonstrations',
                    'lighting': 'contrast between problem/solution moments',
                    'audio': 'transition from problem to solution tonality',
                    'pacing': 'build tension, then release with solution'
                }
            ),
            'call_to_action_scene': SceneTemplate(
                scene_type='call_to_action',
                duration=10,
                visual_focus='strong_brand_presence_with_clear_action_prompt',
                narrative_structure='benefit_reinforcement_with_compelling_action_request',
                technical_specs={
                    'camera_movement': 'confident product shots, brand display',
                    'lighting': 'bright, positive, professional',
                    'audio': 'uplifting music with clear call-to-action',
                    'pacing': 'strong finish with memorable brand moment'
                }
            )
        }
    
    def _initialize_technical_specs(self) -> Dict[str, Dict[str, str]]:
        """Initialize technical specifications for hyperrealistic video generation."""
        return {
            'luma_optimized': {
                'prompt_length': 'detailed_descriptive_300_chars_max',
                'visual_style': 'hyperrealistic_cinematic_commercial_quality',
                'camera_work': 'dynamic_professional_movements_with_depth',
                'duration_per_scene': '5_seconds_precise',
                'lighting': 'natural_volumetric_lighting_with_soft_shadows',
                'rendering': '4K_photorealistic_with_motion_blur',
                'human_subjects': 'authentic_diverse_characters_with_emotions',
                'environment': 'detailed_atmospheric_settings_with_depth'
            },
            'hailuo_optimized': {
                'prompt_length': 'focused_storytelling_200_chars_max',
                'visual_style': 'hyperrealistic_story_driven_commercial',
                'camera_work': 'cinematic_movements_with_storytelling_focus',
                'duration_per_scene': '10_seconds_flexible',
                'lighting': 'dramatic_natural_lighting_for_emotion',
                'rendering': 'photorealistic_commercial_quality',
                'human_subjects': 'relatable_characters_with_clear_motivations',
                'environment': 'immersive_branded_environments'
            },
            'hyperrealistic_storytelling': {
                'narrative_structure': 'emotional_arc_with_tension_and_resolution',
                'character_development': 'relatable_personas_with_clear_motivations',
                'visual_continuity': 'seamless_scene_transitions_with_story_flow',
                'emotional_triggers': 'authentic_human_moments_and_reactions',
                'brand_integration': 'natural_product_placement_in_story_context',
                'call_to_action': 'emotionally_compelling_story_conclusion'
            }
        }
    
    def generate_niche_specific_scenes(
        self, 
        brand_elements: BrandElements,
        target_duration: int = 18,
        service_type: str = "luma",
        creative_style: str = "professional",
        storytelling_approach: str = "problem-solution",
        logo_info: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate 3 professional scenes with business-type-aware creative storytelling and logo integration.
        
        Args:
            brand_elements: Extracted brand intelligence with business_type classification
            target_duration: Total video duration in seconds
            service_type: Video generation service (luma/hailuo)
            creative_style: Creative style (professional, modern, elegant, dynamic)
            storytelling_approach: Story approach (problem-solution, transformation, testimonial, showcase)
            logo_info: Logo information for integration
            
        Returns:
            List of 3 optimized scene configurations with story structure
        """
        # CRITICAL: Get business type for creative direction
        business_type = getattr(brand_elements, 'business_type', BusinessType.SERVICE)

        # Optional overrides via settings (session-level control)
        try:
            from config.settings import settings
            if getattr(settings, 'BUSINESS_TYPE_OVERRIDE', None):
                forced = str(settings.BUSINESS_TYPE_OVERRIDE).strip().lower()
                bt_map = {
                    'product': BusinessType.PRODUCT,
                    'service': BusinessType.SERVICE,
                    'platform': BusinessType.PLATFORM,
                    'hybrid': BusinessType.HYBRID
                }
                business_type = bt_map.get(forced, business_type)
        except Exception:
            pass
        
        # Get creative direction based on business type to prevent mismatches
        from core.brand_intelligence import brand_intelligence_service
        creative_direction = brand_intelligence_service.get_creative_direction_for_business_type(
            business_type, brand_elements.niche
        )

        # Apply optional focus/CTA overrides from settings
        try:
            from config.settings import settings
            if getattr(settings, 'FOCUS_OVERRIDE', None):
                creative_direction['focus'] = str(settings.FOCUS_OVERRIDE).strip()
            if getattr(settings, 'CTA_STYLE_OVERRIDE', None):
                creative_direction['cta_style'] = str(settings.CTA_STYLE_OVERRIDE).strip()
        except Exception:
            pass
        
        # Validate alignment and log warnings if needed
        brand_intelligence_service._validate_business_type_niche_alignment(
            business_type, brand_elements.niche, 
            f"{brand_elements.brand_name}: {' '.join(brand_elements.key_benefits[:3])}"
        )
        
        print(f"🎯 BUSINESS TYPE ANALYSIS: {business_type.value.upper()} | Focus: {creative_direction.get('focus', 'generic')}")
        print(f"   Visual Priority: {', '.join(creative_direction.get('visual_priority', [])[:3])}")
        print(f"   CTA Style: {creative_direction.get('cta_style', 'generic')}")
        
        niche_template = self.niche_templates.get(
            brand_elements.niche,
            self.niche_templates[BusinessNiche.PROFESSIONAL_SERVICES]
        )
        
        # Calculate scene durations for 18-second format
        scene_duration = target_duration // 3  # 6 seconds per scene for 18s total
        
        # Apply business-type-aware creative style enhancements
        enhanced_template = self._enhance_template_with_business_type(
            niche_template, creative_style, business_type, creative_direction
        )
        
        # Generate scenes based on storytelling approach
        scenes = self._generate_scenes_by_storytelling_approach(
            brand_elements, enhanced_template, scene_duration, service_type, storytelling_approach, logo_info
        )
        
        return scenes
    
    def _enhance_template_with_business_type(
        self, 
        niche_template: Dict[str, Any], 
        creative_style: str, 
        business_type: BusinessType,
        creative_direction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhance niche template with business-type-aware creative modifications."""
        enhanced_template = niche_template.copy()
        
        # Apply creative style first
        enhanced_template = self._enhance_template_with_creative_style(enhanced_template, creative_style)
        
        # Apply business-type-specific enhancements
        if business_type == BusinessType.PRODUCT:
            enhanced_template['business_type_focus'] = {
                'visual_emphasis': 'product_showcase_and_features',
                'scene_settings': (enhanced_template.get('scene_settings', ['professional environment']) + [
                    'product display environments',
                    'unboxing and reveal scenarios',
                    'product in real-world usage contexts'
                ]),
                'storytelling_elements': 'product benefits, feature demonstrations, value propositions',
                'avoid_elements': creative_direction.get('avoid', []),
                'cta_integration': 'purchase-focused call-to-action overlays'
            }
            
        elif business_type == BusinessType.SERVICE:
            enhanced_template['business_type_focus'] = {
                'visual_emphasis': 'expertise_demonstration_and_results',
                'scene_settings': (enhanced_template.get('scene_settings', ['professional environment']) + [
                    'professional consultation environments',
                    'client success story settings',
                    'expert team collaboration spaces'
                ]),
                'storytelling_elements': 'problem-solving expertise, client outcomes, professional credibility',
                'avoid_elements': creative_direction.get('avoid', []),
                'cta_integration': 'consultation-focused engagement invitations'
            }
            
        elif business_type == BusinessType.PLATFORM:
            enhanced_template['business_type_focus'] = {
                'visual_emphasis': 'connectivity_and_ecosystem_benefits',
                'scene_settings': (enhanced_template.get('scene_settings', ['professional environment']) + [
                    'connected workflow environments',
                    'user interaction demonstrations',
                    'network effect visualizations'
                ]),
                'storytelling_elements': 'connectivity solutions, scalability benefits, ecosystem value',
                'avoid_elements': creative_direction.get('avoid', []),
                'cta_integration': 'platform-joining and signup motivations'
            }
            
        elif business_type == BusinessType.HYBRID:
            enhanced_template['business_type_focus'] = {
                'visual_emphasis': 'integrated_solution_ecosystem',
                'scene_settings': (enhanced_template.get('scene_settings', ['professional environment']) + [
                    'comprehensive solution environments',
                    'end-to-end journey demonstrations',
                    'integrated value showcases'
                ]),
                'storytelling_elements': 'comprehensive solutions, integrated benefits, one-stop value',
                'avoid_elements': creative_direction.get('avoid', []),
                'cta_integration': 'complete-solution-oriented invitations'
            }
        
        # Add business type metadata for prompt generation
        enhanced_template['business_metadata'] = {
            'type': business_type.value,
            'focus': creative_direction.get('focus', 'value_proposition'),
            'visual_priority': creative_direction.get('visual_priority', []),
            'messaging_priority': creative_direction.get('messaging_priority', []),
            'scene_structure': creative_direction.get('scene_structure', {})
        }
        
        return enhanced_template
    
    def _enhance_template_with_creative_style(self, niche_template: Dict[str, Any], creative_style: str) -> Dict[str, Any]:
        """Enhance niche template with creative style modifications."""
        enhanced_template = niche_template.copy()
        
        creative_enhancements = {
            'professional': {
                'visual_enhancement': 'clean, corporate, trustworthy',
                'lighting_enhancement': 'professional studio lighting',
                'atmosphere_enhancement': 'confident, authoritative'
            },
            'modern': {
                'visual_enhancement': 'sleek, contemporary, minimalist',
                'lighting_enhancement': 'modern architectural lighting',
                'atmosphere_enhancement': 'innovative, forward-thinking'
            },
            'elegant': {
                'visual_enhancement': 'sophisticated, refined, luxurious',
                'lighting_enhancement': 'soft, premium lighting with warmth',
                'atmosphere_enhancement': 'upscale, distinguished'
            },
            'dynamic': {
                'visual_enhancement': 'energetic, bold, engaging',
                'lighting_enhancement': 'dramatic, high-contrast lighting',
                'atmosphere_enhancement': 'exciting, motivational'
            }
        }
        
        style_config = creative_enhancements.get(creative_style, creative_enhancements['professional'])
        enhanced_template['creative_style'] = style_config
        
        return enhanced_template
    
    def _generate_scenes_by_storytelling_approach(
        self, 
        brand_elements: BrandElements, 
        template: Dict[str, Any], 
        scene_duration: int, 
        service_type: str, 
        storytelling_approach: str, 
        logo_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate scenes based on specific storytelling approach."""
        
        storytelling_frameworks = {
            'problem-solution': self._generate_problem_solution_story,
            'transformation': self._generate_transformation_story,
            'testimonial': self._generate_testimonial_story,
            'showcase': self._generate_showcase_story
        }
        
        story_generator = storytelling_frameworks.get(
            storytelling_approach, 
            storytelling_frameworks['problem-solution']
        )
        
        return story_generator(brand_elements, template, scene_duration, service_type, logo_info)
    
    def _generate_problem_solution_story(
        self, 
        brand_elements: BrandElements, 
        template: Dict[str, Any], 
        scene_duration: int, 
        service_type: str, 
        logo_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate problem-solution storytelling approach scenes."""
        scenes = []
        
        # Scene 1: Problem Introduction with Hook
        problem_scene = self._create_creative_scene(
            scene_id=1,
            scene_type='problem_hook',
            brand_elements=brand_elements,
            template=template,
            duration=scene_duration,
            service_type=service_type,
            logo_info=logo_info,
            focus='problem_identification_with_emotional_impact'
        )
        scenes.append(problem_scene)
        
        # Scene 2: Solution Revelation
        solution_scene = self._create_creative_scene(
            scene_id=2,
            scene_type='solution_reveal',
            brand_elements=brand_elements,
            template=template,
            duration=scene_duration,
            service_type=service_type,
            logo_info=logo_info,
            focus='brand_solution_demonstration_with_transformation'
        )
        scenes.append(solution_scene)
        
        # Scene 3: Results and Call to Action
        results_scene = self._create_creative_scene(
            scene_id=3,
            scene_type='results_cta',
            brand_elements=brand_elements,
            template=template,
            duration=scene_duration,
            service_type=service_type,
            logo_info=logo_info,
            focus='successful_outcome_with_compelling_action'
        )
        scenes.append(results_scene)
        
        return scenes
    
    def _generate_transformation_story(
        self, 
        brand_elements: BrandElements, 
        template: Dict[str, Any], 
        scene_duration: int, 
        service_type: str, 
        logo_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate transformation storytelling approach scenes."""
        scenes = []
        
        # Scene 1: Before State
        before_scene = self._create_creative_scene(
            scene_id=1,
            scene_type='transformation_before',
            brand_elements=brand_elements,
            template=template,
            duration=scene_duration,
            service_type=service_type,
            logo_info=logo_info,
            focus='current_state_with_transformation_potential'
        )
        scenes.append(before_scene)
        
        # Scene 2: Transformation Process
        process_scene = self._create_creative_scene(
            scene_id=2,
            scene_type='transformation_process',
            brand_elements=brand_elements,
            template=template,
            duration=scene_duration,
            service_type=service_type,
            logo_info=logo_info,
            focus='brand_driven_transformation_in_action'
        )
        scenes.append(process_scene)
        
        # Scene 3: After State and Results
        after_scene = self._create_creative_scene(
            scene_id=3,
            scene_type='transformation_after',
            brand_elements=brand_elements,
            template=template,
            duration=scene_duration,
            service_type=service_type,
            logo_info=logo_info,
            focus='transformed_state_with_success_invitation'
        )
        scenes.append(after_scene)
        
        return scenes
    
    def _generate_testimonial_story(
        self, 
        brand_elements: BrandElements, 
        template: Dict[str, Any], 
        scene_duration: int, 
        service_type: str, 
        logo_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate testimonial storytelling approach scenes."""
        scenes = []
        
        # Scene 1: Customer Introduction
        intro_scene = self._create_creative_scene(
            scene_id=1,
            scene_type='testimonial_intro',
            brand_elements=brand_elements,
            template=template,
            duration=scene_duration,
            service_type=service_type,
            logo_info=logo_info,
            focus='authentic_customer_introduction_and_challenge'
        )
        scenes.append(intro_scene)
        
        # Scene 2: Experience with Brand
        experience_scene = self._create_creative_scene(
            scene_id=2,
            scene_type='testimonial_experience',
            brand_elements=brand_elements,
            template=template,
            duration=scene_duration,
            service_type=service_type,
            logo_info=logo_info,
            focus='positive_brand_experience_and_benefits'
        )
        scenes.append(experience_scene)
        
        # Scene 3: Recommendation and Results
        recommendation_scene = self._create_creative_scene(
            scene_id=3,
            scene_type='testimonial_recommendation',
            brand_elements=brand_elements,
            template=template,
            duration=scene_duration,
            service_type=service_type,
            logo_info=logo_info,
            focus='strong_recommendation_and_call_to_action'
        )
        scenes.append(recommendation_scene)
        
        return scenes
    
    def _generate_showcase_story(
        self, 
        brand_elements: BrandElements, 
        template: Dict[str, Any], 
        scene_duration: int, 
        service_type: str, 
        logo_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate showcase storytelling approach scenes."""
        scenes = []
        
        # Scene 1: Problem Identification & Character Introduction
        problem_scene = self._create_creative_scene(
            scene_id=1,
            scene_type='problem_identification',
            brand_elements=brand_elements,
            template=template,
            duration=scene_duration,
            service_type=service_type,
            logo_info=logo_info,
            focus='authentic_character_facing_relatable_challenge_that_audience_experiences'
        )
        scenes.append(problem_scene)
        
        # Scene 2: Solution Discovery & Transformation
        transformation_scene = self._create_creative_scene(
            scene_id=2,
            scene_type='solution_discovery',
            brand_elements=brand_elements,
            template=template,
            duration=scene_duration,
            service_type=service_type,
            logo_info=logo_info,
            focus='character_discovers_brand_solution_and_experiences_meaningful_transformation'
        )
        scenes.append(transformation_scene)
        
        # Scene 3: Success State & Aspirational Future
        success_scene = self._create_creative_scene(
            scene_id=3,
            scene_type='success_showcase',
            brand_elements=brand_elements,
            template=template,
            duration=scene_duration,
            service_type=service_type,
            logo_info=logo_info,
            focus='character_thriving_in_improved_life_state_with_compelling_call_to_action'
        )
        scenes.append(success_scene)
        
        return scenes
    
    def _create_creative_scene(
        self,
        scene_id: int,
        scene_type: str,
        brand_elements: BrandElements,
        template: Dict[str, Any],
        duration: int,
        service_type: str,
        logo_info: Dict[str, Any],
        focus: str
    ) -> Dict[str, Any]:
        return self._generate_scene_with_gpt4o(brand_elements, scene_type, focus, duration, service_type, logo_info)

    def _generate_scene_with_gpt4o(
        self,
        brand_elements: BrandElements,
        scene_type: str,
        focus: str,
        duration: int,
        service_type: str,
        logo_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a creative scene with GPT-4o using universal brand intelligence."""
        
        # Try to get universal brand intelligence for deeper creative insights
        universal_intelligence = None
        brand_name = brand_elements.brand_name
        
        # Extract company description from brand elements
        company_description = None
        if hasattr(brand_elements, 'company_description'):
            company_description = brand_elements.company_description
        elif hasattr(brand_elements, 'key_benefits') and brand_elements.key_benefits:
            company_description = f"A business specializing in {', '.join(brand_elements.key_benefits[:3])}"
        elif hasattr(brand_elements, 'industry'):
            company_description = f"A {brand_elements.industry} company"
        else:
            company_description = f"A {brand_elements.niche.value} business"
        
        # Get universal brand intelligence for deeper creativity
        try:
            additional_context = {}
            if hasattr(brand_elements, 'target_demographics'):
                additional_context['target_audience'] = brand_elements.target_demographics
            if hasattr(brand_elements, 'competitive_advantages'):
                additional_context['competitive_advantages'] = brand_elements.competitive_advantages
            if hasattr(brand_elements, 'brand_personality'):
                additional_context['brand_personality'] = brand_elements.brand_personality
                
            universal_intelligence = self.universal_brand_generator.analyze_brand_deeply(
                brand_name, company_description, additional_context
            )
        except Exception as e:
            print(f"Universal brand intelligence unavailable for {brand_name}, using standard approach: {e}")
        
        # Build enhanced strategic brand intelligence section
        if universal_intelligence:
            brand_intelligence_section = f"""
        **DEEP UNIVERSAL BRAND INTELLIGENCE:**
        • Brand Name: {universal_intelligence.brand_name}
        • Business Type: {universal_intelligence.business_type}
        • Industry Category: {universal_intelligence.industry_category}
        • Authentic Brand Personality: {', '.join(universal_intelligence.brand_essence.get('authentic_personality', ['professional']))}
        • Core Values Demonstrated: {', '.join(universal_intelligence.brand_essence.get('core_values_demonstrated', ['quality']))}
        • Brand Voice Characteristics: {universal_intelligence.brand_essence.get('brand_voice_characteristics', 'professional and trustworthy')}
        • Primary Customer Profile: {universal_intelligence.customer_profile.get('primary_customer_profile', 'professionals')}
        • Customer Motivations: {universal_intelligence.customer_profile.get('customer_motivations', 'quality solutions')}
        • Unique Value Delivery: {universal_intelligence.value_architecture.get('unique_value_delivery', 'trusted solutions')}
        • Emotional Territory: {universal_intelligence.brand_essence.get('emotional_territory', 'trust and confidence')}
        • Competitive Differentiation: {universal_intelligence.value_architecture.get('competitive_differentiation', 'market leadership')}
        • Authenticity Markers: {', '.join(universal_intelligence.authenticity_markers[:3])}
        • Creative Opportunities: {', '.join(universal_intelligence.creative_opportunities[:3])}
        
        **AVOID THESE CREATIVE MISMATCHES:**
        - Tech innovation focus if brand is not technology-focused
        - Holographic/digital elements unless brand is actually tech-focused
        - Generic corporate settings - use industry-appropriate environments
        - One-size-fits-all approaches - customize to actual brand essence"""
        else:
            brand_intelligence_section = f"""
        **STRATEGIC BRAND INTELLIGENCE:**
        • Primary Brand: {brand_elements.brand_name}
        • Core Business: {brand_elements.industry if hasattr(brand_elements, 'industry') else brand_elements.niche.value}
        • Key Differentiators: {brand_elements.key_benefits if hasattr(brand_elements, 'key_benefits') else 'Premium quality and innovation'}
        • Competitive Edge: {brand_elements.competitive_advantages if hasattr(brand_elements, 'competitive_advantages') else 'Market leadership and excellence'}
        • Brand DNA: {brand_elements.brand_personality if hasattr(brand_elements, 'brand_personality') else 'Professional, trustworthy, innovative'}
        • Emotional Drivers: {brand_elements.emotional_triggers if hasattr(brand_elements, 'emotional_triggers') else 'Aspiration, success, transformation'}
        • Target Psychographics: {brand_elements.target_demographics if hasattr(brand_elements, 'target_demographics') else 'Achievement-oriented professionals'}"""
        
        prompt = f"""
        You are an award-winning Executive Creative Director at a top-tier global advertising agency (think Wieden+Kennedy, BBDO, Ogilvy level). You've created iconic campaigns for BMW, Coca-Cola, Apple, and Nike. Your expertise spans consumer psychology, cinematic storytelling, and cutting-edge video production.

        **CREATIVE BRIEF:**
        Client: {brand_elements.brand_name}
        Industry: {brand_elements.niche.value}
        Objective: Create a {scene_type} scene that {focus}
        Duration: {duration} seconds of pure cinematic excellence
        Production Budget: $500K+ equivalent quality

        {brand_intelligence_section}

        **ADVANCED CREATIVE STRATEGY:**

        1. **CONSUMER PSYCHOLOGY APPLICATION:**
        Apply Cialdini's persuasion principles (Authority, Social Proof, Scarcity) and create cognitive hooks that bypass rational defenses. Use emotional resonance over logical arguments.

        2. **PROFESSIONAL STORYTELLING FRAMEWORK:**
        
        **Scene-Specific Narrative Strategy:**
        
        → **PROBLEM_IDENTIFICATION:** Show authentic character in relatable struggle. Think Dove's self-esteem campaign - real emotion, genuine pain points. Visual: tension in environment/body language. Avoid: fake corporate problems, obvious setup.
        
        → **SOLUTION_DISCOVERY:** The "lightbulb moment" - character encounters brand naturally. Think Apple's iPhone reveal ads - seamless integration. Visual: transformation in lighting/environment. Avoid: forced product placement.
        
        → **SUCCESS_SHOWCASE:** Aspirational future state achieved. Think Nike's "Just Do It" - emotional payoff, inspirational conclusion. Visual: elevated environment, confident character. Avoid: cheesy happiness, unrealistic outcomes.
        
        **Universal Story Elements:**
        - Emotional Arc: Establish → Escalate → Transform (within {duration} seconds)
        - Visual Metaphors: Use symbolic imagery that reinforces brand values subconsciously
        - Character Journey: Relatable → Challenged → Empowered

        3. **TECHNICAL EXCELLENCE STANDARDS:**
        - Cinematography: Roger Deakins/Emmanuel Lubezki level visual poetry
        - Lighting: Chivo Lubezki's naturalistic-cinematic hybrid approach
        - Camera Language: Intentional movements that serve narrative (no arbitrary motion)
        - Color Science: Deliberate palette that reinforces brand emotional positioning

        4. **ANTI-CLICHÉ INNOVATION SYSTEM:**
        Avoid these overused elements: stock footage aesthetics, corporate handshakes, fake smiles, generic office settings, predictable product shots, obvious symbolism.

        **PRODUCTION SPECIFICATIONS:**

        **Visual Concept Requirements:**
        Create a mini-masterpiece that could win Cannes Lions. Think Denis Villeneuve directing a luxury brand commercial. Every frame should justify the production budget through visual sophistication, authentic human truth, and brand-story integration.

        **SCENE DEPTH & VARIETY MANDATE:**
        - **Environmental Storytelling**: Use locations as active narrative elements, not just backdrops
        - **Multi-Layer Compositions**: Foreground action, midground context, background atmosphere 
        - **Product/Service Integration**: Show functionality in authentic contexts beyond person demonstrations
        - **Macro/Micro Details**: Close-ups of textures, materials, processes that reveal quality/craftsmanship
        - **Symbolic Visual Elements**: Abstract concepts made tangible through objects, lighting, movement
        - **Process Visualization**: Show behind-the-scenes, creation, transformation, or methodology
        - **Atmospheric Depth**: Weather, time of day, seasonal elements that enhance narrative mood
        - **Technical Demonstrations**: Equipment, tools, systems working in sophisticated ways
        - **Spatial Relationships**: Architecture, geometry, scale that reinforces brand positioning
        - **Temporal Layering**: Before/after states, progression sequences, time-lapse elements

        **AVOID OVERDEPENDENCE ON:**
        - Generic person-talking-to-camera shots
        - Basic product-in-hand demonstrations  
        - Standard office/meeting room settings
        - Predictable lifestyle poses
        - Surface-level human interactions

        **Luma AI Optimization (280 chars max):**
        Luma excels with: detailed environmental descriptions, specific lighting conditions, precise camera movements, emotional character states, realistic physics, atmospheric elements. Use technical cinematography language.

        **Hailuo Optimization (120 chars max):**  
        Hailuo thrives on: dynamic action verbs, clear subject-object relationships, specific visual elements, emotional descriptors, concise scene composition. Focus on what's happening, not how it looks.

        **Script Strategy:**
        Channel David Droga's copywriting genius. Every word must earn its place. Create cognitive dissonance, then resolve it with brand truth. Use power words that trigger emotional responses.

        **Character Development:**
        Cast from real life, not stock photo catalogs. Think Dove Real Beauty campaign authenticity. Characters should feel like someone's actual neighbor/colleague/friend, with genuine micro-expressions and natural mannerisms.

        **Audio Architecture:**
        Music: Hans Zimmer meets Apple commercial sophistication. Build emotional crescendo that aligns with visual narrative peak.
        Sound Design: Every sound serves story purpose. Create acoustic environment that enhances psychological immersion.

        **BRAND INTEGRATION MANDATE:**
        The brand should feel inevitable, not forced. Like seeing a BMW emblem - it doesn't interrupt the story, it completes it. Integrate through context, environment, character choices, not product placement.

        **OUTPUT STRUCTURE (STRICT JSON):**
        {{
            "visual_concept": "[Write a 200-word cinematic treatment that would make Ridley Scott nod in approval. Include specific camera techniques, lighting setups, character blocking, environmental details, and emotional subtext. Example: 'Golden hour streams through floor-to-ceiling windows of a minimalist workspace where our protagonist - a driven architect in her early 30s - studies building schematics with intense focus. The camera begins with an extreme wide establishing shot, then executes a slow dolly-in combined with a subtle crane descent, transitioning to an intimate over-shoulder medium shot. Her workspace reflects creative chaos: scattered blueprints, a half-empty coffee cup, a vintage desk lamp casting warm practical light that battles the cooler window light. She pauses, looks up through the window at the city skyline, and we see both determination and vulnerability in her eyes. The moment captures the solitary intensity of creation, while ambient city sounds and soft paper rustling create an authentic workspace atmosphere.']",
            "luma_prompt": "[280 chars max] Professional architect, 30s, reviewing blueprints in minimalist office. Golden hour window light vs warm desk lamp. Slow dolly-in + crane descent to over-shoulder medium. City skyline backdrop. Determined yet vulnerable expression. Creative chaos workspace. Cinematic depth of field.",
            "hailuo_prompt": "[120 chars max] Focused architect reviewing blueprints. Golden hour office. Dolly camera movement. City view. Professional intensity.",
            "script_line": "[Channel David Droga's genius: maximum impact, minimum words, psychological hook + brand truth resolution]",
            "character_description": "[Real person casting brief: specific age, authentic styling, genuine emotional range, relatable imperfections, natural mannerisms]",
            "music_cues": "[Hans Zimmer meets Apple commercial sophistication: specific instruments, emotional arc, build/release timing]",
            "audio_cues": "[Acoustic environment serving story: every sound justified, psychological immersion enhanced]"
        }}

        Create advertising that transcends product promotion to become cultural conversation starters. Make something that people want to share, not skip.
        """

        try:
            from openai import OpenAI
            client = OpenAI()
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a professional video content creator. Respond ONLY with valid JSON. No markdown, no explanations, no extra text. The JSON must be parseable and valid. Start with { and end with }. Do not wrap in ```json```. Only return the JSON object."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024,
                temperature=0.7,
            )

            scene_data = safe_parse_json_response(response.choices[0].message.content, "GPT-4o scene generation")

            return {
                'scene_id': 0, # This will be set later
                'scene_type': scene_type,
                'duration': duration,
                'visual_concept': scene_data['visual_concept'],
                'luma_prompt': scene_data['luma_prompt'],
                'hailuo_prompt': scene_data['hailuo_prompt'],
                'script_line': scene_data['script_line'],
                'character_description': scene_data['character_description'],
                'creative_focus': focus,
                'logo_integration': "",
                'music_cues': scene_data['music_cues'],
                'audio_cues': scene_data['audio_cues'],
                'brand_alignment': f"Enhanced {brand_elements.brand_name} brand integration with professional positioning",
                'niche_optimization': brand_elements.niche.value,
                'hyperrealistic_elements': f"authentic_expressions_with_cinematic_cinematography"
            }
        except Exception as e:
            print(f"Error generating scene with GPT-4o: {e}")
            # Fallback to the old method
            return self._create_creative_scene_fallback(
                0, scene_type, brand_elements, {}, duration, service_type, logo_info, focus
            )
    def _create_creative_scene_fallback(
        self,
        scene_id: int,
        scene_type: str,
        brand_elements: BrandElements,
        template: Dict[str, Any],
        duration: int,
        service_type: str,
        logo_info: Dict[str, Any],
        focus: str
    ) -> Dict[str, Any]:
        """Create a creative scene with enhanced storytelling and logo integration."""
        
        # Get character for consistency
        character_description = self._generate_character_for_niche(brand_elements.niche)
        
        # Apply creative style enhancements
        creative_style = template.get('creative_style', {})
        visual_enhancement = creative_style.get('visual_enhancement', 'professional')
        lighting_enhancement = creative_style.get('lighting_enhancement', 'professional lighting')
        atmosphere_enhancement = creative_style.get('atmosphere_enhancement', 'confident')
        
        # Logo functionality removed
        logo_enhancement = {
            'color_environment': 'professional branded environment',
            'brand_style': 'consistent brand styling',
            'lighting_style': 'professional commercial lighting',
            'font_style': 'clean professional typography',
            'visual_consistency': 'brand-aligned visual elements'
        }
        
        # Logo integration removed
        logo_integration = ""
        
        # Build LUXURY CINEMATIC COMMERCIAL visual concept (BMW/Coca-Cola level)
        # Advanced cinematography elements for premium brand commercials
        camera_techniques = [
            "smooth dolly tracking shot revealing product elegance",
            "crane movement establishing premium lifestyle context", 
            "steadicam follow shot showcasing professional confidence",
            "macro lens highlighting premium craftsmanship details",
            "wide establishing shot with luxury environmental context",
            "handheld shot for a more personal and authentic feel",
            "drone shot for a breathtaking aerial view",
            "first-person point-of-view shot to immerse the viewer"
        ]
        
        lighting_styles = [
            "golden hour natural lighting with soft rim lighting",
            "three-point lighting with professional key light setup", 
            "dramatic chiaroscuro lighting emphasizing luxury",
            "soft diffused lighting creating premium atmosphere",
            "directional lighting highlighting product premium features",
            "neon and futuristic lighting for a modern and edgy look",
            "warm and inviting lighting for a comfortable and cozy feel",
            "bright and clean lighting for a fresh and sterile look"
        ]
        
        import random
        selected_camera = random.choice(camera_techniques)
        selected_lighting = random.choice(lighting_styles)
        
        # Safely get scene settings with fallback
        scene_settings = template.get('scene_settings', ['professional business environment', 'modern workspace', 'brand showcase setting'])
        first_setting = scene_settings[0] if scene_settings else 'professional business environment'
        
        visual_concept = (
            f"A {creative_style.get('visual_enhancement', 'professional')} and {atmosphere_enhancement} cinematic ad for {brand_elements.brand_name}, a {brand_elements.niche.value} company. "
            f"The scene features a {character_description} in a {first_setting}. "
            f"The visual style is {template['visual_style']} with {selected_lighting} and {selected_camera}. "
            f"The ad should convey a sense of {brand_elements.brand_personality.get('primary', 'trust')} and {brand_elements.emotional_triggers[0] if brand_elements.emotional_triggers else 'excitement'}."
            f"{logo_integration}"
        )
        
        # Generate optimized prompts
        luma_prompt = self._optimize_for_luma_hyperrealistic(visual_concept, brand_elements, character_description)
        hailuo_prompt = self._optimize_for_hailuo_hyperrealistic(visual_concept, brand_elements, character_description)
        
        # Generate scene-specific script
        script_line = self._generate_scene_script(scene_type, brand_elements, focus)
        
        # Generate music and audio cues
        music_cues = self._generate_music_cues(scene_type, brand_elements, atmosphere_enhancement)
        audio_cues = self._generate_audio_cues(scene_type, template, atmosphere_enhancement)
        
        return {
            'scene_id': scene_id,
            'scene_type': scene_type,
            'duration': duration,
            'visual_concept': visual_concept,
            'luma_prompt': luma_prompt,
            'hailuo_prompt': hailuo_prompt,
            'script_line': script_line,
            'character_description': character_description,
            'creative_focus': focus,
            'logo_integration': logo_integration,
            'music_cues': music_cues,
            'audio_cues': audio_cues,
            'brand_alignment': f"Enhanced {brand_elements.brand_name} brand integration with {atmosphere_enhancement} positioning",
            'niche_optimization': brand_elements.niche.value,
            'hyperrealistic_elements': f"authentic_{atmosphere_enhancement}_expressions_with_{visual_enhancement}_cinematography"
        }
    
    
    def _generate_scene_script(self, scene_type: str, brand_elements: BrandElements, focus: str) -> str:
        """Generate luxury commercial narratives matching BMW/Coca-Cola quality using GPT-4o."""
        brand_name = brand_elements.brand_name
        primary_benefit = brand_elements.key_benefits[0] if brand_elements.key_benefits else "excellence"

        script_prompt = f"""
        As a master storyteller for luxury brands, craft a concise, impactful script line for a {scene_type} scene.
        The scene is for {brand_name}, a {brand_elements.niche.value} company.
        The core focus of this scene is: {focus}.
        Integrate the brand's key benefit: {primary_benefit}.
        The script should be professional, emotionally resonant, and suitable for a hyper-realistic cinematic ad.
        Keep it under 25 words.
        """

        try:
            from openai import OpenAI
            client = OpenAI()
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": script_prompt}],
                max_tokens=50,
                temperature=0.8,
            )
            script_line = response.choices[0].message.content.strip()
            return script_line
        except Exception as e:
            print(f"Error generating script with GPT-4o: {e}")
            # Fallback to predefined templates
            script_templates = {
                'problem_hook': f"Excellence demands precision. Innovation requires vision. {brand_name} delivers both.",
                'solution_reveal': f"Where others see limits, {brand_name} sees possibilities. Where others compromise, we perfect.",  
                'results_cta': f"This is more than {primary_benefit}. This is {brand_name}. This is mastery.",
                'transformation_before': f"Before {brand_name}, good enough was enough. Now, excellence is the only standard.",
                'transformation_process': f"Every detail refined. Every moment perfected. {brand_name} elevates {primary_benefit}.",
                'transformation_after': f"The difference isn't just visible. It's transformational. It's {brand_name}.",
                'testimonial_intro': f"When expertise meets excellence, {brand_name} becomes the obvious choice.",
                'testimonial_experience': f"From the first moment, you know this isn't ordinary. This is {brand_name}.",
                'testimonial_recommendation': f"Some things speak for themselves. {brand_name} is one of them.",
                'showcase_intro': f"Crafted for those who refuse to compromise. Designed for those who demand excellence.",
                'showcase_demo': f"Every element purposeful. Every innovation meaningful. This is {brand_name}.",
                'showcase_impact': f"The choice of professionals. The standard of excellence. The future is {brand_name}."
            }
            return script_templates.get(scene_type, f"Excellence redefined. Innovation perfected. Welcome to {brand_name}.")
    
    def _generate_music_cues(self, scene_type: str, brand_elements: BrandElements, atmosphere_enhancement: str) -> str:
        """Generate music cues for scenes based on type and brand personality."""
        music_templates = {
            'problem_hook': f"Gentle emotional build with {atmosphere_enhancement} energy, creating anticipation",
            'solution_reveal': f"Emotional crescendo with {atmosphere_enhancement} breakthrough moment",
            'results_cta': f"Inspiring crescendo to memorable finish with {atmosphere_enhancement} motivation",
            'transformation_before': f"Subtle {atmosphere_enhancement} undertone building tension",
            'transformation_process': f"Dynamic {atmosphere_enhancement} transformation theme with rising energy",
            'transformation_after': f"Triumphant {atmosphere_enhancement} resolution with satisfaction",
            'testimonial_intro': f"Warm {atmosphere_enhancement} introduction with human connection",
            'testimonial_experience': f"Uplifting {atmosphere_enhancement} experience with positive energy",
            'testimonial_recommendation': f"Confident {atmosphere_enhancement} endorsement with trust",
            'showcase_intro': f"Professional {atmosphere_enhancement} presentation with authority",
            'showcase_demo': f"Engaging {atmosphere_enhancement} demonstration with expertise",
            'showcase_impact': f"Powerful {atmosphere_enhancement} finale with lasting impression"
        }
        
        primary_personality = list(brand_elements.brand_personality.keys())[0] if brand_elements.brand_personality else 'professional'
        base_cue = music_templates.get(scene_type, f"{atmosphere_enhancement} energy with brand alignment")
        return f"{base_cue}, {primary_personality} brand personality"
    
    def _generate_audio_cues(self, scene_type: str, template: Dict[str, Any], atmosphere_enhancement: str) -> str:
        """Generate audio cues for scenes based on niche and atmosphere."""
        niche_audio = {
            'problem_hook': f"Subtle ambient sounds building {atmosphere_enhancement} connection",
            'solution_reveal': f"Dramatic transition to relief with {atmosphere_enhancement} resolution",
            'results_cta': f"Confident success ambiance with {atmosphere_enhancement} invitation",
            'transformation_before': f"Tension-building ambient with {atmosphere_enhancement} potential",
            'transformation_process': f"Dynamic transformation sounds with {atmosphere_enhancement} progress",
            'transformation_after': f"Triumphant success ambient with {atmosphere_enhancement} achievement",
            'testimonial_intro': f"Warm conversational ambient with {atmosphere_enhancement} authenticity",
            'testimonial_experience': f"Positive experience sounds with {atmosphere_enhancement} satisfaction",
            'testimonial_recommendation': f"Confident recommendation tone with {atmosphere_enhancement} trust",
            'showcase_intro': f"Professional presentation ambient with {atmosphere_enhancement} authority",
            'showcase_demo': f"Engaging demonstration sounds with {atmosphere_enhancement} expertise",
            'showcase_impact': f"Powerful conclusion ambient with {atmosphere_enhancement} impact"
        }
        
        visual_style = template.get('visual_style', 'professional')
        base_cue = niche_audio.get(scene_type, f"{atmosphere_enhancement} environmental audio")
        return f"{base_cue}, {visual_style} environmental audio"
    
    def _generate_hook_scene(
        self,
        brand_elements: BrandElements,
        niche_template: Dict[str, Any],
        duration: int,
        service_type: str
    ) -> Dict[str, Any]:
        """Generate hyperrealistic hook scene with logo-consistent brand elements."""
        
        # Extract brand-specific visual elements
        brand_name = brand_elements.brand_name
        visual_style = niche_template['visual_style']
        scene_settings = niche_template.get('scene_settings', ['professional environment', 'modern workspace', 'brand showcase'])
        settings = scene_settings[0] if scene_settings else 'professional environment'  # First setting for hook
        
        # Create hyperrealistic character-driven opening
        character_description = self._generate_character_for_niche(brand_elements.niche)
        
        # Logo functionality removed - using default enhancement
        logo_enhancement = {
            'color_environment': 'professional branded environment',
            'brand_style': 'consistent brand styling',
            'lighting_style': 'professional commercial lighting',
            'font_style': 'clean professional typography',
            'visual_consistency': 'brand-aligned visual elements'
        }
        
        # Build hyperrealistic visual concept with logo consistency
        visual_concept = (
            f"Hyperrealistic commercial opening: {character_description} in {settings}, "
            f"{logo_enhancement['color_environment']}, "
            f"natural volumetric lighting with soft shadows, 4K photorealistic quality, "
            f"{visual_style} cinematography with emotional depth, authentic human moment showing anticipation, "
            f"subtle {brand_name} brand integration in environment with {logo_enhancement['brand_style']}, "
            f"dynamic camera movement revealing story context, "
            f"atmospheric details: {logo_enhancement['lighting_style']}, shallow depth of field, authentic expressions"
        )
        
        # Generate service-optimized prompt
        if service_type.lower() == "hailuo":
            prompt = self._optimize_for_hailuo_hyperrealistic(visual_concept, brand_elements, character_description)
        else:
            prompt = self._optimize_for_luma_hyperrealistic(visual_concept, brand_elements, character_description)
        
        # Create emotional narrative script
        emotional_hook = self._generate_emotional_hook(brand_elements)
        script_line = f"{emotional_hook} Meet {brand_name} - where {brand_elements.key_benefits[0] if brand_elements.key_benefits else 'transformation'} begins."
        
        return {
            'scene_id': 1,
            'duration': duration,
            'visual_concept': visual_concept,
            'luma_prompt': prompt,
            'hailuo_prompt': self._optimize_for_hailuo_hyperrealistic(visual_concept, brand_elements, character_description),
            'script_line': script_line,
            'character_description': character_description,
            'emotional_arc': 'curiosity_and_anticipation',
            'visual_storytelling': 'establishing_character_and_world',
            'audio_cues': f"Subtle ambient sounds building emotional connection, {niche_template['visual_style']} environmental audio",
            'music_cues': f"Gentle emotional build, {brand_elements.brand_personality.get('energetic', 'medium')} energy with human warmth",
            'brand_alignment': f"Natural {brand_name} brand presence integrated into authentic human story",
            'niche_optimization': brand_elements.niche.value,
            'hyperrealistic_elements': 'authentic_character_emotions_natural_lighting_environmental_depth'
        }
    
    def _generate_problem_solution_scene(
        self,
        brand_elements: BrandElements,
        niche_template: Dict[str, Any],
        duration: int,
        service_type: str
    ) -> Dict[str, Any]:
        """Generate hyperrealistic problem identification and emotional solution transformation scene."""
        
        brand_name = brand_elements.brand_name
        visual_style = niche_template['visual_style']
        scene_settings = niche_template.get('scene_settings', ['professional environment', 'modern workspace', 'brand showcase'])
        settings = scene_settings[1] if len(scene_settings) > 1 else scene_settings[0] if scene_settings else 'modern workspace'  # Second setting for problem/solution
        
        # Create same character continuity from hook scene
        character_description = self._generate_character_for_niche(brand_elements.niche)
        
        # Extract key benefit for solution
        primary_benefit = brand_elements.key_benefits[0] if brand_elements.key_benefits else "enhanced efficiency"
        
        # Logo functionality removed - using default enhancement
        logo_enhancement = {
            'color_environment': 'professional branded environment',
            'brand_style': 'consistent brand styling',
            'lighting_style': 'professional commercial lighting',
            'font_style': 'clean professional typography',
            'visual_consistency': 'brand-aligned visual elements'
        }
        
        # Build hyperrealistic transformation scene with logo consistency
        visual_concept = (
            f"Hyperrealistic transformation scene: {character_description} experiencing problem frustration then relief, "
            f"in {settings}, {logo_enhancement['lighting_style']} transition from shadow to warm light, "
            f"4K photorealistic quality showing authentic human struggle and breakthrough, "
            f"{visual_style} cinematography with emotional close-ups, "
            f"{brand_name} product naturally solving challenge with {logo_enhancement['brand_style']}, "
            f"genuine relief and satisfaction expressions, "
            f"atmospheric details: {logo_enhancement['color_environment']}, lighting shift represents emotional transformation, "
            f"realistic before/after moment showing {primary_benefit}, {logo_enhancement['visual_consistency']}"
        )
        
        # Generate service-optimized prompt
        if service_type.lower() == "hailuo":
            prompt = self._optimize_for_hailuo_hyperrealistic(visual_concept, brand_elements, character_description)
        else:
            prompt = self._optimize_for_luma_hyperrealistic(visual_concept, brand_elements, character_description)
        
        # Emotional problem-solution narrative
        pain_point = brand_elements.target_demographics.get('pain_points', ['everyday challenges'])[0] if brand_elements.target_demographics.get('pain_points') else 'everyday challenges'
        emotional_transition = self._generate_emotional_transition(brand_elements)
        script_line = f"When {pain_point} feel overwhelming... {emotional_transition} {brand_name} transforms everything. Experience {primary_benefit}."
        
        return {
            'scene_id': 2,
            'duration': duration,
            'visual_concept': visual_concept,
            'luma_prompt': prompt,
            'hailuo_prompt': self._optimize_for_hailuo_hyperrealistic(visual_concept, brand_elements, character_description),
            'script_line': script_line,
            'character_description': character_description,
            'emotional_arc': 'struggle_to_breakthrough_and_relief',
            'visual_storytelling': 'problem_tension_to_solution_transformation',
            'transformation_moment': f"authentic_relief_discovering_{primary_benefit}",
            'audio_cues': f"Dramatic transition from tension to relief, emotional resolution sounds, {niche_template['visual_style']} environmental audio",
            'music_cues': "Emotional crescendo from struggle to triumph, authentic human moment resolution",
            'brand_alignment': f"{brand_name} positioned as life-changing solution in authentic story context",
            'niche_optimization': brand_elements.niche.value,
            'hyperrealistic_elements': 'authentic_emotional_transformation_natural_lighting_transition_human_expressions'
        }
    
    def _generate_cta_scene(
        self,
        brand_elements: BrandElements,
        niche_template: Dict[str, Any],
        duration: int,
        service_type: str
    ) -> Dict[str, Any]:
        """Generate hyperrealistic call-to-action scene with emotional brand conclusion and compelling action."""
        
        brand_name = brand_elements.brand_name
        visual_style = niche_template['visual_style']
        scene_settings = niche_template.get('scene_settings', ['professional environment', 'modern workspace', 'brand showcase'])
        settings = scene_settings[2] if len(scene_settings) > 2 else scene_settings[-1] if scene_settings else 'brand showcase'  # Third setting for CTA
        
        # Maintain character continuity
        character_description = self._generate_character_for_niche(brand_elements.niche)
        
        # Extract competitive advantage
        advantage = brand_elements.competitive_advantages[0] if brand_elements.competitive_advantages else "proven results"
        
        # Logo functionality removed - using default enhancement
        logo_enhancement = {
            'color_environment': 'professional branded environment',
            'brand_style': 'consistent brand styling',
            'lighting_style': 'professional commercial lighting',
            'font_style': 'clean professional typography',
            'visual_consistency': 'brand-aligned visual elements'
        }
        
        # Build hyperrealistic conclusion with logo-consistent satisfaction
        visual_concept = (
            f"Hyperrealistic commercial conclusion: {character_description} now confident and satisfied, "
            f"in {settings}, {logo_enhancement['lighting_style']}, 4K photorealistic quality, "
            f"authentic joy and confidence expressions, {brand_name} brand prominently featured with {logo_enhancement['brand_style']}, "
            f"{niche_template['call_to_action_style']} seamlessly integrated, "
            f"{visual_style} cinematography with inspiring final moments, "
            f"atmospheric details: {logo_enhancement['color_environment']} suggesting success and satisfaction, "
            f"clear branding showcase with {advantage}, {logo_enhancement['visual_consistency']}, compelling visual invitation to action"
        )
        
        # Generate service-optimized prompt
        if service_type.lower() == "hailuo":
            prompt = self._optimize_for_hailuo_hyperrealistic(visual_concept, brand_elements, character_description)
        else:
            prompt = self._optimize_for_luma_hyperrealistic(visual_concept, brand_elements, character_description)
        
        # Emotionally compelling call-to-action
        emotional_close = self._generate_emotional_close(brand_elements)
        cta_text = f"{emotional_close} Join thousands who chose {brand_name}. Experience {advantage} today."
        
        return {
            'scene_id': 3,
            'duration': duration,
            'visual_concept': visual_concept,
            'luma_prompt': prompt,
            'hailuo_prompt': self._optimize_for_hailuo_hyperrealistic(visual_concept, brand_elements, character_description),
            'script_line': cta_text,
            'character_description': character_description,
            'emotional_arc': 'satisfaction_and_confident_invitation',
            'visual_storytelling': 'triumphant_conclusion_with_clear_action_path',
            'success_moment': f"authentic_satisfaction_with_{advantage}",
            'audio_cues': f"Confident triumphant resolution, inspiring call-to-action tone, {niche_template['visual_style']} success ambiance",
            'music_cues': "Inspiring crescendo to memorable brand finish, emotional satisfaction and motivation",
            'brand_alignment': f"Maximum {brand_name} brand impact with authentic story conclusion and compelling action invitation",
            'niche_optimization': brand_elements.niche.value,
            'hyperrealistic_elements': 'authentic_satisfaction_expressions_triumphant_lighting_natural_brand_integration',
            'action_motivation': f"emotionally_compelling_invitation_to_experience_{advantage}"
        }
    
    def _optimize_for_luma(self, visual_concept: str, brand_elements: BrandElements) -> str:
        """Optimize prompt for Luma Dream Machine with detailed descriptions."""
        
        # Luma works well with detailed cinematic descriptions
        enhanced_prompt = (
            f"Professional commercial cinematography: {visual_concept}, "
            f"branded environment featuring {brand_elements.brand_name}, "
            f"high-quality product integration, {brand_elements.niche.value} industry styling, "
            f"commercial lighting and composition, brand-appropriate color palette, "
            f"professional video production quality"
        )
        
        # Limit to optimal length for Luma
        return enhanced_prompt[:280]
    
    def _optimize_for_hailuo(self, visual_concept: str, brand_elements: BrandElements) -> str:
        """Optimize prompt for Hailuo with concise, focused direction."""
        
        # Extract key elements for concise prompt
        key_words = [
            'commercial',
            brand_elements.brand_name.lower(),
            brand_elements.niche.value,
            'professional',
            'product demonstration',
            'branded environment'
        ]
        
        # Build concise but effective prompt
        concise_prompt = f"Commercial scene: {brand_elements.brand_name} {brand_elements.niche.value} product, professional environment, branded presentation"
        
        # Limit to optimal length for Hailuo
        return concise_prompt[:120]
    
    def _generate_character_for_niche(self, niche: BusinessNiche) -> str:
        """Generate cinema-quality character descriptions for luxury brand commercials."""
        # Premium character archetypes for BMW/Coca-Cola level commercials
        character_templates = {
            BusinessNiche.TECHNOLOGY: "sophisticated executive, 30s, sharp modern attire, commanding presence, genuine confidence with cutting-edge tech, single focused individual",
            BusinessNiche.HEALTHCARE: "distinguished healthcare leader, professional white coat, warm trustworthy demeanor, single person showcasing medical innovation, authentic authority",
            BusinessNiche.FOOD_BEVERAGE: "master chef, pristine culinary whites, artistic precision, single craftsperson creating culinary excellence, passionate expertise",
            BusinessNiche.FASHION_BEAUTY: "elegant fashion model, timeless style, confident poise, single person embodying luxury lifestyle, sophisticated beauty",
            BusinessNiche.FITNESS_WELLNESS: "elite athlete, peak physical condition, determined focus, single person achieving excellence, inspiring dedication",
            BusinessNiche.FINANCE: "senior financial advisor, tailored business suit, authoritative presence, single professional managing success, trusted expertise",
            BusinessNiche.REAL_ESTATE: "luxury property specialist, elegant professional attire, confident sophistication, single person presenting premium properties, exclusive expertise",
            BusinessNiche.E_COMMERCE: "discerning customer, premium lifestyle, sophisticated taste, single person experiencing luxury products, refined satisfaction",
            BusinessNiche.PROFESSIONAL_SERVICES: "industry expert, executive presence, genuine authority, single professional delivering excellence, trusted leadership",
            BusinessNiche.HOME_LIFESTYLE: "sophisticated homeowner, refined living space, elegant casual attire, single person enjoying luxury lifestyle, tasteful success",
            BusinessNiche.ENTERTAINMENT: "creative professional, artistic expression, confident performance, single talent showcasing excellence, authentic artistry",
            BusinessNiche.SUSTAINABILITY: "environmental leader, modern eco-conscious style, genuine passion, single person championing sustainability, authentic commitment"
        }
        return character_templates.get(niche, "distinguished professional, executive presence, single person with authentic authority and sophisticated demeanor")
    
    def _generate_emotional_hook(self, brand_elements: BrandElements) -> str:
        """Generate emotional hooks based on brand personality and benefits."""
        personality_hooks = {
            'energetic': "Feel the energy.",
            'professional': "Excellence awaits.",
            'friendly': "You belong here.",
            'luxury': "Indulge yourself.",
            'innovative': "Imagine the possibilities.",
            'trustworthy': "Trust your instincts.",
            'casual': "Life's better when..."
        }
        
        primary_personality = list(brand_elements.brand_personality.keys())[0] if brand_elements.brand_personality else 'professional'
        return personality_hooks.get(primary_personality, "Transform your world.")
    
    def _generate_emotional_transition(self, brand_elements: BrandElements) -> str:
        """Generate emotional transition phrases for problem-solution moments."""
        transition_phrases = {
            'energetic': "Feel the shift.",
            'professional': "See the change.",
            'friendly': "Watch as",
            'luxury': "Witness the transformation.",
            'innovative': "Experience the breakthrough.",
            'trustworthy': "Discover how",
            'casual': "That's when"
        }
        
        primary_personality = list(brand_elements.brand_personality.keys())[0] if brand_elements.brand_personality else 'professional'
        return transition_phrases.get(primary_personality, "That's when")
    
    def _generate_emotional_close(self, brand_elements: BrandElements) -> str:
        """Generate emotional closing phrases for compelling calls-to-action."""
        closing_phrases = {
            'energetic': "Ready to energize your life?",
            'professional': "Join the professionals who choose excellence.",
            'friendly': "Your journey starts here.",
            'luxury': "You deserve the finest.",
            'innovative': "Step into the future.",
            'trustworthy': "Trust the choice thousands have made.",
            'casual': "Why wait? Your moment is now."
        }
        
        primary_personality = list(brand_elements.brand_personality.keys())[0] if brand_elements.brand_personality else 'professional'
        return closing_phrases.get(primary_personality, "Your transformation awaits.")
    
    def _optimize_for_luma_hyperrealistic(self, visual_concept: str, brand_elements: BrandElements, character_description: str) -> str:
        """Cinematic commercial prompts for Luma Dream Machine - BMW/Coca-Cola level quality."""
        
        # Extract key elements for luxury brand commercial construction
        core_character = character_description.split(',')[0].strip()
        brand_name = brand_elements.brand_name
        niche = brand_elements.niche.value
        
        # Hyperrealistic cinematography elements focused on natural quality
        cinematic_elements = [
            "hyperrealistic cinematography with smooth natural camera movements and perfect motion blur",
            "photorealistic lighting with accurate shadows and natural reflections", 
            "authentic human expressions with natural facial micro-movements and realistic body language",
            "seamless temporal consistency with no morphing or distortion artifacts",
            "fluid character movements with accurate physics and natural walking patterns",
            "realistic environmental interactions with proper lighting response"
        ]
        
        # Visual quality enhancement elements for hyperrealistic results
        luxury_visuals = [
            "ultra-high definition detail with perfect edge definition and skin texture accuracy",
            "natural color grading with accurate skin tones and realistic material properties",
            "smooth motion interpolation with 60fps quality and no frame stuttering",
            "photorealistic human features with natural pore detail and authentic expressions",
            "consistent character appearance throughout with no face or body morphing",
            "cinema-grade visual fidelity with professional color accuracy and contrast"
        ]
        
        # Select random elements for uniqueness (like force_unique in Luma provider)
        import random
        selected_cinematic = random.choice(cinematic_elements)
        selected_luxury = random.choice(luxury_visuals)
        
        # Build hyperrealistic quality-focused prompt 
        enhanced_prompt = (
            f"HYPERREALISTIC: {core_character} featuring {brand_name} {niche}, "
            f"{selected_cinematic}, {selected_luxury}, "
            f"natural human movement with realistic physics, no artificial motion artifacts, "
            f"8K hyperrealistic quality with perfect temporal consistency"
        )
        
        # Optimal length for Luma Dream Machine (250 chars for best quality/cost ratio)
        return enhanced_prompt[:250]
    
    def _optimize_for_hailuo_hyperrealistic(self, visual_concept: str, brand_elements: BrandElements, character_description: str) -> str:
        """Cost-optimized prompt for Hailuo with maximum efficiency and no text overlays."""
        
        # Ultra-concise prompt for maximum cost savings while maintaining quality
        essential_elements = [
            character_description.split(',')[0],  # Core character only
            brand_elements.brand_name,  # Brand name
            brand_elements.niche.value,  # Business niche
            "professional commercial",  # Quality indicator
            "no text overlays"  # Prevent text rendering
        ]
        
        # Build ultra-efficient prompt
        story_prompt = f"{' '.join(essential_elements[:4])}, natural lighting, {essential_elements[4]}"
        
        # Limit to cost-optimized length (extended slightly to include no-text instruction)
        return story_prompt[:130]
    
    
    
    def generate_audio_architecture(
        self,
        brand_elements: BrandElements,
        scenes: List[Dict[str, Any]],
        total_duration: int = 18
    ) -> Dict[str, Any]:
        """Generate comprehensive audio architecture for the video."""
        
        # Determine voice characteristics based on brand personality
        voice_gender = "female" if "friendly" in brand_elements.brand_personality else "male"
        
        energy_mapping = {
            "energetic": "high",
            "professional": "medium", 
            "luxury": "low",
            "friendly": "medium"
        }
        
        # Get primary personality trait
        primary_trait = list(brand_elements.brand_personality.keys())[0] if brand_elements.brand_personality else "professional"
        energy_level = energy_mapping.get(primary_trait, "medium")
        
        # Niche-specific music styles
        music_styles = {
            BusinessNiche.TECHNOLOGY: "electronic",
            BusinessNiche.HEALTHCARE: "ambient",
            BusinessNiche.FINANCE: "orchestral",
            BusinessNiche.FITNESS_WELLNESS: "upbeat",
            BusinessNiche.FASHION_BEAUTY: "sophisticated"
        }
        
        music_style = music_styles.get(brand_elements.niche, "cinematic")
        
        return {
            'voice_gender': voice_gender,
            'voice_tone': primary_trait,
            'energy_level': energy_level,
            'music_style': music_style,
            'tempo_bpm': 120 if energy_level == "high" else 90,
            'total_duration': total_duration,
            'scene_audio_sync': [
                {
                    'scene_id': scene['scene_id'],
                    'duration': scene['duration'],
                    'script': scene['script_line'],
                    'music_cues': scene['music_cues'],
                    'audio_cues': scene['audio_cues']
                }
                for scene in scenes
            ]
        }
    
    def get_fallback_template(self, niche: BusinessNiche, service_type: str = 'luma') -> List[Dict[str, Any]]:
        """Get proven high-performing fallback template for quality gating failures."""
        
        # High-performing universal fallback template optimized for all niches
        fallback_scenes = [
            {
                'scene_id': 1,
                'scene_type': 'hook',
                'duration': 10,
                'visual_concept': 'Professional individual confidently using product/service in clean, modern environment',
                'luma_prompt': 'Professional person confidently demonstrating product in modern office setting, natural lighting, 4K cinematic quality, smooth camera movement',
                'hailuo_prompt': 'Professional demonstration, clean modern office, confident person using product',
                'script_line': 'Discover the solution that professionals trust worldwide.',
                'music_cues': ['upbeat_intro'],
                'audio_cues': ['ambient_office'],
                'character_description': 'Professional, confident individual',
                'emotional_arc': 'introduction_confidence',
                'niche_optimization': 'universal_professional',
                'quality_optimized': True,
                'fallback_template': True
            },
            {
                'scene_id': 2,
                'scene_type': 'transformation',
                'duration': 10,
                'visual_concept': 'Clear before/after transformation showing improved workflow/results',
                'luma_prompt': 'Split screen showing workflow transformation, from challenge to smooth solution, professional cinematography with natural transitions',
                'hailuo_prompt': 'Before and after comparison, smooth workflow transformation, professional setting',
                'script_line': 'See the immediate difference it makes in your daily operations.',
                'music_cues': ['building_momentum'],
                'audio_cues': ['success_tone'],
                'character_description': 'Same professional, now showing satisfaction',
                'emotional_arc': 'problem_to_solution_relief',
                'niche_optimization': 'universal_results',
                'quality_optimized': True,
                'fallback_template': True
            },
            {
                'scene_id': 3,
                'scene_type': 'call_to_action',
                'duration': 10,
                'visual_concept': 'Compelling call-to-action with clear next steps and contact information',
                'luma_prompt': 'Professional looking directly at camera with confident smile, clean branded background, clear call-to-action text overlay',
                'hailuo_prompt': 'Direct camera address, professional smile, branded background with clear CTA',
                'script_line': 'Join thousands of satisfied professionals. Get started today.',
                'music_cues': ['triumphant_close'],
                'audio_cues': ['call_to_action_chime'],
                'character_description': 'Professional, directly engaging viewer',
                'emotional_arc': 'call_to_action_motivation',
                'niche_optimization': 'universal_cta',
                'quality_optimized': True,
                'fallback_template': True
            }
        ]
        
        # Customize prompts based on service type
        for scene in fallback_scenes:
            if service_type.lower() == 'luma':
                # Luma prefers detailed cinematic descriptions
                scene['primary_prompt'] = scene['luma_prompt']
            else:
                # Hailuo prefers concise action-focused prompts  
                scene['primary_prompt'] = scene['hailuo_prompt']
        
        return fallback_scenes
    
    def get_template_performance_history(self) -> Dict[str, Any]:
        """Get performance history for continuous improvement (placeholder for database integration)."""
        
        # In a full implementation, this would query the database for:
        # - Template usage statistics
        # - Success rates by niche
        # - Quality scores over time
        # - A/B test results
        
        return {
            'high_performing_templates': [
                {
                    'template_id': 'professional_transformation_v2',
                    'success_rate': 0.94,
                    'average_quality_score': 0.87,
                    'recommended_niches': ['technology', 'professional_services', 'healthcare']
                },
                {
                    'template_id': 'product_demonstration_v1',
                    'success_rate': 0.91,
                    'average_quality_score': 0.83,
                    'recommended_niches': ['retail', 'technology', 'manufacturing']
                }
            ],
            'underperforming_templates': [
                {
                    'template_id': 'abstract_concept_v1',
                    'success_rate': 0.67,
                    'average_quality_score': 0.62,
                    'needs_optimization': True
                }
            ],
            'optimization_recommendations': [
                'Increase specificity in visual descriptions',
                'Add more emotional engagement elements',
                'Optimize prompt length for service type'
            ]
        }
