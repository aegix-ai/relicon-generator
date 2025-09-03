"""
Brand Scene Service - Integration layer for brand-aware scene generation
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from core.brand_aware_scene_generator import (
    BrandAwareSceneGenerator, BrandIntelligence, SceneObjective, SceneType, IndustryVertical,
    create_brand_intelligence_for_company
)
from core.universal_brand_intelligence import UniversalBrandIntelligenceGenerator
from core.logger import get_logger

logger = get_logger(__name__)

class BrandSceneService:
    """Service for generating brand-aware scenes integrated with existing architecture."""
    
    def __init__(self):
        self.scene_generator = BrandAwareSceneGenerator()
        self.universal_brand_generator = UniversalBrandIntelligenceGenerator()
        self.industry_mapping = self._initialize_industry_mapping()
        
    def generate_scenes_for_architecture(self, 
                                       architecture: Dict[str, Any],
                                       brand_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate brand-aware scenes for video architecture.
        
        Args:
            architecture: Existing video architecture from enhanced planning service
            brand_context: Additional brand context if available
            
        Returns:
            Enhanced architecture with brand-aware scenes
        """
        try:
            logger.info("Generating brand-aware scenes for architecture",
                       action="brand_scene.generation.start")
            
            # Extract brand intelligence from architecture
            brand_intelligence = self._extract_brand_intelligence(architecture, brand_context)
            
            # Generate scenes for each act in the architecture
            enhanced_scenes = []
            
            scenes = architecture.get('scene_architecture', {}).get('scenes', [])
            
            for i, scene in enumerate(scenes):
                # Determine scene type based on position and purpose
                scene_type = self._determine_scene_type(i, len(scenes), scene)
                
                # Create scene objective
                scene_objective = self._create_scene_objective(scene, scene_type)
                
                # Generate brand-aware scene
                brand_scene = self.scene_generator.generate_brand_aligned_scene(
                    brand_intelligence=brand_intelligence,
                    scene_objective=scene_objective,
                    creative_constraints=self._extract_creative_constraints(architecture)
                )
                
                # Merge with existing scene data
                enhanced_scene = self._merge_with_existing_scene(scene, brand_scene)
                enhanced_scenes.append(enhanced_scene)
            
            # Update architecture with enhanced scenes
            enhanced_architecture = architecture.copy()
            enhanced_architecture['scene_architecture']['scenes'] = enhanced_scenes
            enhanced_architecture['brand_aware_generation'] = True
            enhanced_architecture['brand_intelligence_used'] = {
                'brand_name': brand_intelligence.brand_name,
                'industry': brand_intelligence.industry_vertical.value,
                'alignment_approach': 'professional_brand_aware'
            }
            
            logger.info(f"Generated {len(enhanced_scenes)} brand-aware scenes",
                       action="brand_scene.generation.complete")
            
            return enhanced_architecture
            
        except Exception as e:
            logger.error(f"Brand-aware scene generation failed: {e}", exc_info=True)
            # Return original architecture on failure
            return architecture
    
    def _extract_brand_intelligence(self, 
                                  architecture: Dict[str, Any], 
                                  brand_context: Optional[Dict[str, Any]]) -> BrandIntelligence:
        """Extract or create brand intelligence using universal GPT-4o analysis."""
        
        # Try to get brand info from architecture
        brand_elements = architecture.get('brand_elements', {})
        brand_name = brand_elements.get('brand_name', 'Brand')
        
        # Get company description from various sources
        company_description = (
            brand_elements.get('company_description') or
            brand_elements.get('business_description') or
            brand_elements.get('description') or
            f"A business offering {', '.join(brand_elements.get('key_benefits', ['quality solutions']))}"
        )
        
        # Use universal brand intelligence if we have enough information
        if brand_name and company_description:
            try:
                logger.info(f"Using universal brand intelligence for {brand_name}", 
                           action="brand_intelligence.universal.start")
                
                # Combine brand context with architecture elements for comprehensive analysis
                additional_context = {}
                if brand_context:
                    additional_context.update(brand_context)
                
                # Add relevant architecture context
                if brand_elements.get('target_demographics'):
                    additional_context['target_audience'] = brand_elements['target_demographics']
                if brand_elements.get('key_benefits'):
                    additional_context['key_benefits'] = brand_elements['key_benefits']
                if brand_elements.get('competitive_advantages'):
                    additional_context['competitive_advantages'] = brand_elements['competitive_advantages']
                
                # Get universal brand intelligence
                universal_intelligence = self.universal_brand_generator.analyze_brand_deeply(
                    brand_name, company_description, additional_context
                )
                
                # Convert to BrandIntelligence format
                return self._convert_universal_to_brand_intelligence(universal_intelligence)
                
            except Exception as e:
                logger.warning(f"Universal brand intelligence failed: {e}, falling back to standard extraction")
        
        # Fallback to existing methods
        niche = brand_elements.get('niche')
        industry = self._map_niche_to_industry(niche)
        
        if brand_context:
            return self._create_brand_intelligence_from_context(brand_context, brand_name, industry)
        
        return self._create_brand_intelligence_from_elements(brand_elements, industry)
    
    def _convert_universal_to_brand_intelligence(self, universal_intelligence) -> BrandIntelligence:
        """Convert UniversalBrandIntelligence to BrandIntelligence format."""
        
        # Map industry category to IndustryVertical
        industry_mapping = {
            'technology': IndustryVertical.TECHNOLOGY,
            'healthcare': IndustryVertical.HEALTH_WELLNESS,
            'health': IndustryVertical.HEALTH_WELLNESS,
            'wellness': IndustryVertical.HEALTH_WELLNESS,
            'beauty': IndustryVertical.BEAUTY_COSMETICS,
            'cosmetics': IndustryVertical.BEAUTY_COSMETICS,
            'food': IndustryVertical.FOOD_BEVERAGE,
            'beverage': IndustryVertical.FOOD_BEVERAGE,
            'fashion': IndustryVertical.FASHION_LIFESTYLE,
            'lifestyle': IndustryVertical.FASHION_LIFESTYLE,
            'fitness': IndustryVertical.FITNESS_SPORTS,
            'sports': IndustryVertical.FITNESS_SPORTS,
            'finance': IndustryVertical.FINANCE,
            'financial': IndustryVertical.FINANCE,
            'real estate': IndustryVertical.REAL_ESTATE,
            'property': IndustryVertical.REAL_ESTATE,
            'retail': IndustryVertical.RETAIL_COMMERCE,
            'commerce': IndustryVertical.RETAIL_COMMERCE,
            'ecommerce': IndustryVertical.RETAIL_COMMERCE,
            'e-commerce': IndustryVertical.RETAIL_COMMERCE,
            'professional services': IndustryVertical.PROFESSIONAL_SERVICES,
            'consulting': IndustryVertical.PROFESSIONAL_SERVICES,
            'home': IndustryVertical.HOME_LIVING,
            'automotive': IndustryVertical.AUTOMOTIVE,
            'travel': IndustryVertical.TRAVEL_HOSPITALITY,
            'hospitality': IndustryVertical.TRAVEL_HOSPITALITY,
            'education': IndustryVertical.EDUCATION,
            'entertainment': IndustryVertical.ENTERTAINMENT
        }
        
        # Find matching industry vertical
        industry_vertical = IndustryVertical.PROFESSIONAL_SERVICES  # Default
        industry_lower = universal_intelligence.industry_category.lower()
        
        for key, vertical in industry_mapping.items():
            if key in industry_lower:
                industry_vertical = vertical
                break
        
        # Extract brand personality from brand essence
        brand_personality = []
        if universal_intelligence.brand_essence.get('authentic_personality'):
            personality_data = universal_intelligence.brand_essence['authentic_personality']
            if isinstance(personality_data, list):
                brand_personality = personality_data
            elif isinstance(personality_data, str):
                brand_personality = [trait.strip() for trait in personality_data.split(',')]
        
        # Extract core values
        core_values = []
        if universal_intelligence.brand_essence.get('core_values_demonstrated'):
            values_data = universal_intelligence.brand_essence['core_values_demonstrated']
            if isinstance(values_data, list):
                core_values = values_data
            elif isinstance(values_data, str):
                core_values = [value.strip() for value in values_data.split(',')]
        
        # Extract target audience information
        target_audience = {}
        if universal_intelligence.customer_profile:
            target_audience = {
                'demographics': universal_intelligence.customer_profile.get('primary_customer_profile', 'General audience'),
                'psychographics': universal_intelligence.customer_profile.get('customer_motivations', 'Value-conscious consumers'),
                'behaviors': universal_intelligence.customer_profile.get('behavioral_patterns', {}).get('discovery_behavior', 'Active researchers'),
                'aspirations': universal_intelligence.customer_profile.get('transformation_seeking', 'Positive outcomes and satisfaction')
            }
        
        # Extract unique value proposition
        unique_value_proposition = (
            universal_intelligence.value_architecture.get('unique_value_delivery', '') or
            universal_intelligence.value_architecture.get('core_problem_solved', '') or
            f"Trusted {universal_intelligence.brand_name} solutions"
        )
        
        # Extract emotional drivers
        emotional_drivers = []
        emotional_landscape = universal_intelligence.emotional_landscape or {}
        if emotional_landscape.get('customer_emotional_drivers'):
            emotional_drivers = emotional_landscape['customer_emotional_drivers']
        elif emotional_landscape.get('brand_emotional_territory'):
            emotional_drivers = [emotional_landscape['brand_emotional_territory']]
        
        if not emotional_drivers:
            emotional_drivers = ['trust', 'satisfaction', 'confidence']
        
        # Extract brand voice
        brand_voice = (
            universal_intelligence.brand_essence.get('brand_voice_characteristics', '') or
            'Professional and trustworthy'
        )
        
        # Extract visual identity
        visual_identity = {}
        if universal_intelligence.visual_identity_cues:
            visual_cues = universal_intelligence.visual_identity_cues
            visual_identity = {
                'style': visual_cues.get('aesthetic_direction', ['modern', 'professional'])[0] if visual_cues.get('aesthetic_direction') else 'professional',
                'aesthetic': ', '.join(visual_cues.get('aesthetic_direction', ['clean', 'trustworthy']))
            }
        else:
            visual_identity = {
                'style': 'professional',
                'aesthetic': 'clean and trustworthy'
            }
        
        # Extract competitive positioning
        competitive_positioning = (
            universal_intelligence.competitive_context.get('market_position', '') or
            universal_intelligence.competitive_context.get('competitive_landscape', '') or
            f"{universal_intelligence.brand_name} - committed to excellence and customer value"
        )
        
        return BrandIntelligence(
            brand_name=universal_intelligence.brand_name,
            industry_vertical=industry_vertical,
            brand_personality=brand_personality or ['professional', 'trustworthy'],
            core_values=core_values or ['quality', 'service', 'innovation'],
            target_audience=target_audience,
            unique_value_proposition=unique_value_proposition,
            emotional_drivers=emotional_drivers,
            brand_voice=brand_voice,
            visual_identity=visual_identity,
            competitive_positioning=competitive_positioning
        )
    
    def _map_niche_to_industry(self, niche) -> IndustryVertical:
        """Map business niche to industry vertical."""
        if not niche:
            return IndustryVertical.RETAIL_COMMERCE
        
        niche_str = str(niche).lower()
        
        mapping = {
            'beauty': IndustryVertical.BEAUTY_COSMETICS,
            'cosmetics': IndustryVertical.BEAUTY_COSMETICS,
            'health': IndustryVertical.HEALTH_WELLNESS,
            'wellness': IndustryVertical.HEALTH_WELLNESS,
            'fitness': IndustryVertical.FITNESS_SPORTS,
            'fashion': IndustryVertical.FASHION_LIFESTYLE,
            'food': IndustryVertical.FOOD_BEVERAGE,
            'beverage': IndustryVertical.FOOD_BEVERAGE,
            'home': IndustryVertical.HOME_LIVING,
            'automotive': IndustryVertical.AUTOMOTIVE,
            'travel': IndustryVertical.TRAVEL_HOSPITALITY,
            'education': IndustryVertical.EDUCATION,
            'finance': IndustryVertical.FINANCE,
            'professional': IndustryVertical.PROFESSIONAL_SERVICES
        }
        
        for key, industry in mapping.items():
            if key in niche_str:
                return industry
        
        return IndustryVertical.RETAIL_COMMERCE  # Default
    
    def _create_brand_intelligence_from_context(self, 
                                              brand_context: Dict[str, Any], 
                                              brand_name: str,
                                              industry: IndustryVertical) -> BrandIntelligence:
        """Create brand intelligence from provided context."""
        
        return BrandIntelligence(
            brand_name=brand_context.get('brand_name', brand_name),
            industry_vertical=industry,
            brand_personality=brand_context.get('brand_personality', ['professional', 'trustworthy']),
            core_values=brand_context.get('core_values', ['quality', 'service', 'innovation']),
            target_audience=brand_context.get('target_audience', {
                'demographics': 'General consumer audience',
                'psychographics': 'Value-conscious, quality-seeking',
                'behaviors': 'Research before purchase, brand loyal',
                'aspirations': 'Quality solutions that improve their life'
            }),
            unique_value_proposition=brand_context.get('unique_value_proposition', 
                                                     'Trusted provider of quality solutions'),
            emotional_drivers=brand_context.get('emotional_drivers', 
                                              ['trust', 'confidence', 'satisfaction']),
            brand_voice=brand_context.get('brand_voice', 'Professional and reliable'),
            visual_identity=brand_context.get('visual_identity', {
                'style': 'clean and professional',
                'aesthetic': 'modern and approachable'
            }),
            competitive_positioning=brand_context.get('competitive_positioning', 
                                                    'Industry leader focused on customer value')
        )
    
    def _create_brand_intelligence_from_elements(self, 
                                               brand_elements: Dict[str, Any],
                                               industry: IndustryVertical) -> BrandIntelligence:
        """Create brand intelligence from existing brand elements."""
        
        brand_name = brand_elements.get('brand_name', 'Brand')
        
        # Extract personality from brand elements
        brand_personality = []
        if brand_elements.get('brand_personality'):
            if isinstance(brand_elements['brand_personality'], dict):
                brand_personality = list(brand_elements['brand_personality'].values())
            elif isinstance(brand_elements['brand_personality'], list):
                brand_personality = brand_elements['brand_personality']
        
        if not brand_personality:
            brand_personality = ['professional', 'trustworthy']
        
        # Extract values
        core_values = brand_elements.get('core_values', ['quality', 'service'])
        if isinstance(core_values, str):
            core_values = [core_values]
        
        return BrandIntelligence(
            brand_name=brand_name,
            industry_vertical=industry,
            brand_personality=brand_personality,
            core_values=core_values,
            target_audience={
                'demographics': brand_elements.get('target_demographics', 'General audience'),
                'psychographics': 'Value-conscious consumers',
                'behaviors': 'Active researchers and decision makers',
                'aspirations': 'Quality solutions and positive outcomes'
            },
            unique_value_proposition=brand_elements.get('value_proposition', 
                                                       f'Trusted {brand_name} experience'),
            emotional_drivers=brand_elements.get('emotional_triggers', 
                                               ['trust', 'confidence', 'satisfaction']),
            brand_voice='Professional yet approachable',
            visual_identity={
                'style': 'modern professional',
                'aesthetic': 'clean and trustworthy'
            },
            competitive_positioning=f'{brand_name} - committed to customer value and excellence'
        )
    
    def _determine_scene_type(self, scene_index: int, total_scenes: int, scene_data: Dict[str, Any]) -> SceneType:
        """Determine the scene type based on position and content."""
        
        # Check if scene has explicit type information
        if 'scene_type' in scene_data:
            scene_type_str = scene_data['scene_type'].lower()
            type_mapping = {
                'hook': SceneType.HOOK,
                'problem': SceneType.PROBLEM,
                'solution': SceneType.SOLUTION,
                'transformation': SceneType.TRANSFORMATION,
                'social_proof': SceneType.SOCIAL_PROOF,
                'call_to_action': SceneType.CALL_TO_ACTION,
                'cta': SceneType.CALL_TO_ACTION
            }
            
            for key, scene_type in type_mapping.items():
                if key in scene_type_str:
                    return scene_type
        
        # Determine by position in sequence
        if scene_index == 0:
            return SceneType.HOOK
        elif scene_index == total_scenes - 1:
            return SceneType.CALL_TO_ACTION
        elif scene_index == 1 and total_scenes > 3:
            return SceneType.PROBLEM
        elif scene_index == total_scenes - 2:
            return SceneType.TRANSFORMATION
        else:
            return SceneType.SOLUTION
    
    def _create_scene_objective(self, scene_data: Dict[str, Any], scene_type: SceneType) -> SceneObjective:
        """Create scene objective from scene data and type."""
        
        # Get duration from scene data
        duration = int(scene_data.get('duration', 10))
        
        # Create objectives based on scene type
        objective_mapping = {
            SceneType.HOOK: {
                'primary_goal': 'capture attention and establish brand relevance',
                'emotional_outcome': 'viewer feels curious and connected to brand story',
                'behavioral_trigger': 'motivates continued viewing and engagement',
                'visual_priority': ['attention-grabbing moment', 'brand introduction', 'audience connection']
            },
            SceneType.PROBLEM: {
                'primary_goal': 'establish relatable challenge or unmet need',
                'emotional_outcome': 'viewer recognizes their own pain point or aspiration',
                'behavioral_trigger': 'creates desire for solution',
                'visual_priority': ['authentic problem demonstration', 'emotional relatability', 'tension building']
            },
            SceneType.SOLUTION: {
                'primary_goal': 'introduce brand as natural solution to established need',
                'emotional_outcome': 'viewer feels hopeful and sees possibility',
                'behavioral_trigger': 'builds interest in brand offering',
                'visual_priority': ['organic brand introduction', 'solution demonstration', 'value clarity']
            },
            SceneType.TRANSFORMATION: {
                'primary_goal': 'demonstrate positive outcome and brand impact',
                'emotional_outcome': 'viewer feels inspired and confident about potential change',
                'behavioral_trigger': 'motivates trial or purchase consideration',
                'visual_priority': ['authentic transformation', 'positive outcomes', 'emotional payoff']
            },
            SceneType.CALL_TO_ACTION: {
                'primary_goal': 'motivate specific next step with brand',
                'emotional_outcome': 'viewer feels motivated and empowered to act',
                'behavioral_trigger': 'drives immediate action or engagement',
                'visual_priority': ['clear action demonstration', 'urgency creation', 'barrier removal']
            }
        }
        
        objective_data = objective_mapping.get(scene_type, objective_mapping[SceneType.SOLUTION])
        
        return SceneObjective(
            primary_goal=objective_data['primary_goal'],
            emotional_outcome=objective_data['emotional_outcome'],
            behavioral_trigger=objective_data['behavioral_trigger'],
            narrative_function=scene_type,
            duration_seconds=duration,
            visual_priority=objective_data['visual_priority']
        )
    
    def _extract_creative_constraints(self, architecture: Dict[str, Any]) -> Dict[str, Any]:
        """Extract creative constraints from architecture."""
        
        constraints = {}
        
        # Budget/production constraints
        if 'production_budget' in architecture:
            constraints['budget_level'] = architecture['production_budget']
        
        # Visual style constraints
        if 'visual_style' in architecture:
            constraints['visual_style_requirements'] = architecture['visual_style']
        
        # Technical constraints
        technical_specs = architecture.get('technical_specifications', {})
        if technical_specs:
            constraints['technical_requirements'] = technical_specs
        
        return constraints
    
    def _merge_with_existing_scene(self, 
                                 original_scene: Dict[str, Any], 
                                 brand_scene: Dict[str, Any]) -> Dict[str, Any]:
        """Merge brand-aware scene specification with existing scene data."""
        
        # Start with original scene
        enhanced_scene = original_scene.copy()
        
        # Enhance with brand-aware elements
        enhanced_scene['brand_aware_spec'] = brand_scene
        
        # Update key fields with brand-aware content
        if brand_scene.get('scene_concept'):
            concept = brand_scene['scene_concept']
            enhanced_scene['enhanced_title'] = concept.get('title')
            enhanced_scene['core_narrative'] = concept.get('core_narrative')
            enhanced_scene['emotional_arc'] = concept.get('emotional_arc')
        
        if brand_scene.get('visual_specification'):
            visual_spec = brand_scene['visual_specification']
            enhanced_scene['brand_aligned_visual'] = visual_spec
            
            # Update main prompt with brand-aware content
            if 'prompt' in enhanced_scene:
                # Enhance existing prompt with brand insights
                original_prompt = enhanced_scene['prompt']
                brand_setting = visual_spec.get('setting', '')
                brand_mood = visual_spec.get('mood_and_atmosphere', '')
                
                if brand_setting and brand_mood:
                    enhanced_scene['enhanced_prompt'] = f"{original_prompt}. {brand_setting}. {brand_mood}. Professional brand storytelling."
        
        if brand_scene.get('production_notes'):
            enhanced_scene['brand_production_notes'] = brand_scene['production_notes']
        
        # Add brand alignment info
        if brand_scene.get('brand_alignment'):
            enhanced_scene['brand_alignment'] = brand_scene['brand_alignment']
        
        return enhanced_scene
    
    def _initialize_industry_mapping(self) -> Dict[str, IndustryVertical]:
        """Initialize mapping of business niches to industry verticals."""
        return {
            'TECHNOLOGY': IndustryVertical.PROFESSIONAL_SERVICES,
            'HEALTHCARE': IndustryVertical.HEALTH_WELLNESS,
            'FOOD_BEVERAGE': IndustryVertical.FOOD_BEVERAGE,
            'BEAUTY_COSMETICS': IndustryVertical.BEAUTY_COSMETICS,
            'FITNESS_WELLNESS': IndustryVertical.FITNESS_SPORTS,
            'REAL_ESTATE': IndustryVertical.PROFESSIONAL_SERVICES,
            'FINANCE_CONSULTING': IndustryVertical.FINANCE,
            'ECOMMERCE': IndustryVertical.RETAIL_COMMERCE,
            'PROFESSIONAL_SERVICES': IndustryVertical.PROFESSIONAL_SERVICES,
            'HOME_LIFESTYLE': IndustryVertical.HOME_LIVING,
            'ENTERTAINMENT': IndustryVertical.RETAIL_COMMERCE,
            'SUSTAINABILITY': IndustryVertical.RETAIL_COMMERCE
        }

    def enhance_existing_prompt(self, 
                              original_prompt: str, 
                              brand_scene: Dict[str, Any],
                              brand_intelligence: BrandIntelligence) -> str:
        """Enhance existing Luma prompt with brand-aware elements."""
        
        try:
            # Extract brand-aware enhancements
            visual_spec = brand_scene.get('visual_specification', {})
            character_direction = brand_scene.get('character_direction', {})
            production_notes = brand_scene.get('production_notes', {})
            
            enhancements = []
            
            # Add authentic setting
            setting = visual_spec.get('setting')
            if setting and 'tech' not in setting.lower():  # Avoid tech bias
                enhancements.append(setting)
            
            # Add mood and atmosphere
            mood = production_notes.get('mood_and_atmosphere')
            if mood:
                enhancements.append(mood)
            
            # Add character authenticity
            character = character_direction.get('primary_character')
            if character and 'innovator' not in character.lower():  # Avoid tech bias
                enhancements.append(character)
            
            # Add brand values as visual elements
            for value in brand_intelligence.core_values[:2]:  # Top 2 values
                if value == 'inclusivity':
                    enhancements.append('diverse and inclusive representation')
                elif value == 'quality':
                    enhancements.append('premium quality details and craftsmanship')
                elif value == 'community':
                    enhancements.append('warm community connection')
                elif value == 'innovation':
                    enhancements.append('creative and innovative approach')
            
            # Combine original with enhancements
            if enhancements:
                enhanced_prompt = f"{original_prompt}. {'. '.join(enhancements)}. Authentic brand storytelling."
                
                # Remove tech-focused elements
                tech_terms = ['holographic', 'tech innovator', 'digital', 'screens', 'tech integration']
                for term in tech_terms:
                    enhanced_prompt = enhanced_prompt.replace(term, '')
                
                # Clean up double spaces and punctuation
                enhanced_prompt = ' '.join(enhanced_prompt.split())
                enhanced_prompt = enhanced_prompt.replace('..', '.').replace('.,', ',')
                
                return enhanced_prompt
            else:
                return original_prompt
                
        except Exception as e:
            logger.error(f"Prompt enhancement failed: {e}")
            return original_prompt

# Convenience function for easy integration
def enhance_architecture_with_brand_awareness(architecture: Dict[str, Any], 
                                            brand_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convenience function to enhance architecture with brand awareness."""
    
    service = BrandSceneService()
    return service.generate_scenes_for_architecture(architecture, brand_context)

def analyze_any_business_for_scenes(brand_name: str, 
                                  company_description: str,
                                  additional_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Analyze any business using universal brand intelligence and generate brand-aware scenes.
    
    Args:
        brand_name: Name of the brand/company
        company_description: Description of what the business does
        additional_context: Any additional context (optional)
        
    Returns:
        Enhanced architecture with universal brand-aware scenes
    """
    from core.universal_brand_intelligence import analyze_any_brand
    
    try:
        # Get universal brand intelligence
        universal_intelligence = analyze_any_brand(brand_name, company_description, additional_context)
        
        # Create mock architecture for scene generation
        mock_architecture = {
            'brand_elements': {
                'brand_name': brand_name,
                'company_description': company_description,
                'business_description': company_description,
                'key_benefits': [universal_intelligence.value_architecture.get('unique_value_delivery', 'quality solutions')],
                'target_demographics': universal_intelligence.customer_profile.get('primary_customer_profile', 'professionals'),
                'competitive_advantages': [universal_intelligence.value_architecture.get('competitive_differentiation', 'market leadership')]
            },
            'scene_architecture': {
                'scenes': [
                    {'duration': 10, 'scene_number': 1},
                    {'duration': 10, 'scene_number': 2}, 
                    {'duration': 10, 'scene_number': 3}
                ]
            }
        }
        
        # Enhance with universal brand awareness
        service = BrandSceneService()
        enhanced_architecture = service.generate_scenes_for_architecture(mock_architecture, additional_context)
        
        # Add universal intelligence metadata
        enhanced_architecture['universal_brand_intelligence'] = {
            'brand_name': universal_intelligence.brand_name,
            'business_type': universal_intelligence.business_type,
            'industry_category': universal_intelligence.industry_category,
            'authenticity_markers': universal_intelligence.authenticity_markers,
            'creative_opportunities': universal_intelligence.creative_opportunities,
            'analysis_source': 'universal_gpt4o_deep_creativity'
        }
        
        return enhanced_architecture
        
    except Exception as e:
        print(f"Universal business analysis failed: {e}")
        return {
            'error': f"Analysis failed: {e}",
            'fallback': 'Standard brand analysis recommended'
        }

def create_ulta_brand_context() -> Dict[str, Any]:
    """Create Ulta Beauty brand context for testing."""
    return {
        'brand_name': 'Ulta Beauty',
        'brand_personality': ['inclusive', 'accessible', 'diverse', 'empowering'],
        'core_values': ['inclusivity', 'beauty for all', 'innovation', 'community'],
        'target_audience': {
            'demographics': 'Beauty enthusiasts of all ages, income levels, and backgrounds',
            'psychographics': 'Self-expression focused, community-oriented, value-conscious',
            'behaviors': 'Product discovery, loyalty program engagement, omnichannel shopping',
            'pain_points': 'Finding right products for individual needs, navigating vast selection',
            'aspirations': 'Personal beauty expression, confidence, belonging to beauty community'
        },
        'unique_value_proposition': 'One-stop beauty destination offering 25,000+ products from mass to prestige with expert services',
        'emotional_drivers': ['confidence', 'self-expression', 'belonging', 'discovery'],
        'brand_voice': 'Friendly, inclusive, empowering, and knowledgeable',
        'visual_identity': {
            'primary_colors': ['warm colors', 'inclusive tones'],
            'style': 'accessible luxury', 
            'aesthetic': 'diverse beauty celebration'
        },
        'competitive_positioning': 'Largest specialty beauty retailer democratizing beauty for all'
    }