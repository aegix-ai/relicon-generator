"""
Professional Brand-Aware Scene Generator
Generates authentic, brand-aligned scenes using GPT-4o without tech bias.
"""

import os
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from core.logger import get_logger
from providers.openai import OpenAIProvider

logger = get_logger(__name__)

class SceneType(Enum):
    """Professional scene types for brand storytelling."""
    HOOK = "hook"
    PROBLEM = "problem" 
    SOLUTION = "solution"
    TRANSFORMATION = "transformation"
    SOCIAL_PROOF = "social_proof"
    CALL_TO_ACTION = "call_to_action"

class IndustryVertical(Enum):
    """Industry verticals for brand-specific optimization."""
    BEAUTY_COSMETICS = "beauty_cosmetics"
    HEALTH_WELLNESS = "health_wellness"
    FASHION_LIFESTYLE = "fashion_lifestyle"
    FOOD_BEVERAGE = "food_beverage"
    HOME_LIVING = "home_living"
    PROFESSIONAL_SERVICES = "professional_services"
    RETAIL_COMMERCE = "retail_commerce"
    AUTOMOTIVE = "automotive"
    TRAVEL_HOSPITALITY = "travel_hospitality"
    FITNESS_SPORTS = "fitness_sports"
    EDUCATION = "education"
    FINANCE = "finance"

@dataclass
class BrandIntelligence:
    """Comprehensive brand intelligence for scene generation."""
    brand_name: str
    industry_vertical: IndustryVertical
    brand_personality: List[str]  # e.g., ["premium", "accessible", "innovative"]
    core_values: List[str]  # e.g., ["inclusivity", "quality", "community"]
    target_audience: Dict[str, Any]  # demographics, psychographics, behaviors
    unique_value_proposition: str
    emotional_drivers: List[str]  # e.g., ["confidence", "belonging", "transformation"]
    brand_voice: str  # e.g., "friendly and empowering", "sophisticated yet approachable"
    visual_identity: Dict[str, Any]  # colors, style, aesthetic preferences
    competitive_positioning: str
    cultural_context: Optional[Dict[str, Any]] = None

@dataclass
class SceneObjective:
    """Specific objective for the scene within the brand narrative."""
    primary_goal: str  # e.g., "demonstrate product efficacy"
    emotional_outcome: str  # e.g., "viewer feels inspired and confident"
    behavioral_trigger: str  # e.g., "motivates trial or purchase"
    narrative_function: SceneType
    duration_seconds: int
    visual_priority: List[str]  # ordered list of what should be visually prominent

class BrandAwareSceneGenerator:
    """Professional scene generator focused on authentic brand storytelling."""
    
    def __init__(self):
        self.openai_provider = OpenAIProvider()
        self.industry_expertise = self._initialize_industry_expertise()
        self.narrative_frameworks = self._initialize_narrative_frameworks()
        
    def generate_brand_aligned_scene(self, 
                                   brand_intelligence: BrandIntelligence,
                                   scene_objective: SceneObjective,
                                   creative_constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a professional, brand-aligned scene using GPT-4o.
        
        Args:
            brand_intelligence: Comprehensive brand information
            scene_objective: Specific scene goals and requirements
            creative_constraints: Production limitations, preferences, etc.
            
        Returns:
            Complete scene specification with visual prompts, narrative elements, etc.
        """
        try:
            logger.info(f"Generating brand-aligned scene for {brand_intelligence.brand_name}",
                       action="scene.generation.start")
            
            # Build comprehensive creative brief
            creative_brief = self._build_creative_brief(
                brand_intelligence, scene_objective, creative_constraints
            )
            
            # Generate scene using GPT-4o
            scene_specification = self._generate_scene_with_gpt4o(creative_brief)
            
            # Validate and enhance scene
            validated_scene = self._validate_brand_alignment(
                scene_specification, brand_intelligence, scene_objective
            )
            
            logger.info("Brand-aligned scene generated successfully",
                       action="scene.generation.complete")
            
            return validated_scene
            
        except Exception as e:
            logger.error(f"Brand-aware scene generation failed: {e}", exc_info=True)
            return self._generate_fallback_scene(brand_intelligence, scene_objective)
    
    def _build_creative_brief(self, 
                            brand_intelligence: BrandIntelligence,
                            scene_objective: SceneObjective,
                            creative_constraints: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Build comprehensive creative brief for GPT-4o."""
        
        # Get industry-specific insights
        industry_context = self.industry_expertise.get(
            brand_intelligence.industry_vertical, {}
        )
        
        # Get narrative framework for scene type
        narrative_framework = self.narrative_frameworks.get(
            scene_objective.narrative_function, {}
        )
        
        creative_brief = {
            "brand_context": {
                "brand_name": brand_intelligence.brand_name,
                "industry": brand_intelligence.industry_vertical.value,
                "brand_essence": {
                    "personality": brand_intelligence.brand_personality,
                    "values": brand_intelligence.core_values,
                    "voice": brand_intelligence.brand_voice,
                    "positioning": brand_intelligence.competitive_positioning
                },
                "unique_value_proposition": brand_intelligence.unique_value_proposition,
                "emotional_drivers": brand_intelligence.emotional_drivers
            },
            
            "audience_intelligence": brand_intelligence.target_audience,
            
            "scene_requirements": {
                "objective": scene_objective.primary_goal,
                "emotional_outcome": scene_objective.emotional_outcome,
                "behavioral_trigger": scene_objective.behavioral_trigger,
                "scene_type": scene_objective.narrative_function.value,
                "duration": scene_objective.duration_seconds,
                "visual_priorities": scene_objective.visual_priority
            },
            
            "industry_context": industry_context,
            "narrative_framework": narrative_framework,
            "visual_identity": brand_intelligence.visual_identity,
            "creative_constraints": creative_constraints or {},
            
            "anti_patterns": self._get_anti_patterns(brand_intelligence.industry_vertical),
            "authenticity_markers": self._get_authenticity_markers(brand_intelligence)
        }
        
        return creative_brief
    
    def _generate_scene_with_gpt4o(self, creative_brief: Dict[str, Any]) -> Dict[str, Any]:
        """Generate scene using GPT-4o with professional creative direction."""
        
        prompt = self._build_professional_prompt(creative_brief)
        
        try:
            response = self.openai_provider.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a world-class Creative Director with 20+ years experience creating award-winning brand campaigns for Fortune 500 companies. You specialize in authentic brand storytelling that drives emotional connection and business results."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.8,
                max_tokens=2000
            )
            
            # Parse response as JSON
            scene_content = response.choices[0].message.content
            
            # Try to extract JSON from response
            try:
                if "```json" in scene_content:
                    json_start = scene_content.find("```json") + 7
                    json_end = scene_content.find("```", json_start)
                    json_content = scene_content[json_start:json_end].strip()
                else:
                    json_content = scene_content
                
                scene_data = json.loads(json_content)
                return scene_data
                
            except json.JSONDecodeError:
                # If JSON parsing fails, create structured response from text
                return self._parse_text_response(scene_content, creative_brief)
                
        except Exception as e:
            logger.error(f"GPT-4o scene generation failed: {e}")
            raise
    
    def _build_professional_prompt(self, creative_brief: Dict[str, Any]) -> str:
        """Build professional creative direction prompt for GPT-4o."""
        
        brand_context = creative_brief["brand_context"]
        scene_requirements = creative_brief["scene_requirements"]
        audience = creative_brief["audience_intelligence"]
        
        prompt = f"""
**CREATIVE BRIEF: BRAND-ALIGNED SCENE GENERATION**

**CLIENT:** {brand_context['brand_name']}
**INDUSTRY:** {brand_context['industry'].replace('_', ' ').title()}
**SCENE OBJECTIVE:** {scene_requirements['objective']}

**BRAND INTELLIGENCE:**
• **Brand Personality:** {', '.join(brand_context['brand_essence']['personality'])}
• **Core Values:** {', '.join(brand_context['brand_essence']['values'])}
• **Brand Voice:** {brand_context['brand_essence']['voice']}
• **Unique Value Prop:** {brand_context['unique_value_proposition']}
• **Emotional Drivers:** {', '.join(brand_context['emotional_drivers'])}

**SCENE REQUIREMENTS:**
• **Type:** {scene_requirements['scene_type'].title()} Scene
• **Duration:** {scene_requirements['duration']} seconds
• **Emotional Outcome:** {scene_requirements['emotional_outcome']}
• **Visual Priorities:** {', '.join(scene_requirements['visual_priorities'])}
• **Behavioral Goal:** {scene_requirements['behavioral_trigger']}

**TARGET AUDIENCE INSIGHTS:**
{self._format_audience_insights(audience)}

**CREATIVE DIRECTION MANDATE:**

**AUTHENTICITY FIRST:**
Create scenes that feel genuine and true to real human experience. Avoid:
• Generic stock-photo aesthetics
• Overproduced, artificial scenarios  
• Tech-centric solutions unless genuinely relevant
• Clichéd industry tropes
• Performative diversity without substance

**BRAND ALIGNMENT:**
Every element should reinforce the brand's authentic positioning:
• Visual style must reflect brand personality
• Narrative should embody core values
• Emotional tone aligns with brand voice
• Scenarios relevant to actual customer experience

**HUMAN-CENTERED STORYTELLING:**
Focus on real human truths and authentic moments:
• Genuine emotions and reactions
• Relatable situations and contexts
• Natural interactions and behaviors
• Real-world environments and settings
• Diverse representation that feels organic

**INDUSTRY AUTHENTICITY:**
{self._get_industry_specific_direction(creative_brief)}

**ANTI-PATTERNS TO AVOID:**
{self._format_anti_patterns(creative_brief.get('anti_patterns', []))}

**DELIVERABLE:**
Provide a detailed scene specification in JSON format with:

```json
{{
    "scene_concept": {{
        "title": "Brief scene title",
        "core_narrative": "One sentence describing the human story",
        "emotional_arc": "How the viewer should feel by scene end"
    }},
    "visual_specification": {{
        "setting": "Specific, authentic environment description", 
        "lighting": "Natural, mood-appropriate lighting description",
        "composition": "Camera angles and framing that serve the story",
        "color_palette": "Colors that reinforce brand and emotion",
        "visual_style": "Overall aesthetic approach"
    }},
    "character_direction": {{
        "primary_character": "Authentic character description with motivation",
        "character_arc": "Brief transformation or realization within scene",
        "performance_notes": "Natural behaviors and genuine reactions"
    }},
    "narrative_elements": {{
        "story_beats": ["Ordered list of key story moments"],
        "dialogue_or_voiceover": "Natural, brand-voice aligned copy",
        "product_integration": "How brand/product appears organically"
    }},
    "production_notes": {{
        "key_visuals": ["Specific shots that must be captured"],
        "props_and_details": ["Important environmental elements"],
        "mood_and_atmosphere": "Overall feeling and energy"
    }},
    "brand_alignment": {{
        "brand_values_reinforced": ["Which brand values this scene demonstrates"],
        "emotional_connection": "How this creates authentic brand affinity",
        "differentiation": "What makes this uniquely this brand's story"
    }}
}}
```

Create a scene that would make the CMO proud to represent their brand and the audience genuinely connect with the message.
        """
        
        return prompt.strip()
    
    def _format_audience_insights(self, audience: Dict[str, Any]) -> str:
        """Format audience intelligence for creative direction."""
        insights = []
        
        if audience.get('demographics'):
            demo = audience['demographics']
            insights.append(f"**Demographics:** {demo}")
        
        if audience.get('psychographics'):
            psycho = audience['psychographics'] 
            insights.append(f"**Psychographics:** {psycho}")
        
        if audience.get('behaviors'):
            behaviors = audience['behaviors']
            insights.append(f"**Behaviors:** {behaviors}")
        
        if audience.get('pain_points'):
            pains = audience['pain_points']
            insights.append(f"**Pain Points:** {pains}")
        
        if audience.get('aspirations'):
            aspirations = audience['aspirations']
            insights.append(f"**Aspirations:** {aspirations}")
        
        return '\n'.join(insights) if insights else "General consumer audience"
    
    def _get_industry_specific_direction(self, creative_brief: Dict[str, Any]) -> str:
        """Get industry-specific creative direction."""
        industry = creative_brief["brand_context"]["industry"]
        
        industry_directions = {
            "beauty_cosmetics": """
**BEAUTY INDUSTRY AUTHENTICITY:**
• Show real transformations, not just before/after
• Include diverse skin tones, ages, and beauty expressions
• Focus on the emotional journey, not just the physical change
• Authentic application techniques and realistic results
• Real environments (bathrooms, bedrooms, getting ready spaces)
• Genuine moments of self-discovery and confidence building
            """,
            
            "health_wellness": """
**HEALTH & WELLNESS AUTHENTICITY:**
• Real wellness journeys, not quick fixes
• Sustainable lifestyle changes, not extreme transformations  
• Inclusive health that respects different bodies and abilities
• Evidence-based approaches over trendy solutions
• Community support and professional guidance
• Mental and physical wellness integration
            """,
            
            "food_beverage": """
**FOOD & BEVERAGE AUTHENTICITY:**
• Real cooking and preparation processes
• Authentic cultural contexts and traditions
• Genuine family/social moments around food
• Natural ingredient stories and sourcing
• Real kitchens and dining environments
• Diverse food traditions and dietary approaches
            """,
            
            "fashion_lifestyle": """
**FASHION & LIFESTYLE AUTHENTICITY:**
• Real personal style expression, not just trends
• Diverse body types and style preferences
• Authentic lifestyle contexts and occasions
• Sustainable and conscious fashion choices
• Individual creativity over conformity
• Real wardrobe integration and styling
            """,
            
            "professional_services": """
**PROFESSIONAL SERVICES AUTHENTICITY:**
• Real expertise and professional knowledge
• Genuine client relationships and success stories
• Authentic work environments and processes
• Professional development and growth
• Ethical business practices and values
• Community impact and contribution
            """
        }
        
        return industry_directions.get(industry, "Focus on authentic industry representation and genuine customer experience.")
    
    def _format_anti_patterns(self, anti_patterns: List[str]) -> str:
        """Format anti-patterns for creative direction."""
        if not anti_patterns:
            return "• Generic stock photography aesthetics\n• Overly corporate or sterile environments\n• Inauthentic diversity casting"
        
        return '\n'.join([f"• {pattern}" for pattern in anti_patterns])
    
    def _get_anti_patterns(self, industry: IndustryVertical) -> List[str]:
        """Get industry-specific anti-patterns to avoid."""
        patterns = {
            IndustryVertical.BEAUTY_COSMETICS: [
                "Perfect, unrealistic beauty standards",
                "Dramatic before/after comparisons that seem fake",
                "Laboratory or sterile beauty environments", 
                "Tech-focused beauty solutions over human experience",
                "One-size-fits-all beauty ideals"
            ],
            
            IndustryVertical.HEALTH_WELLNESS: [
                "Extreme fitness transformations",
                "Medical equipment without context",
                "Supplements as miracle solutions",
                "Yoga/fitness poses without proper form",
                "One-dimensional wellness approaches"
            ],
            
            IndustryVertical.PROFESSIONAL_SERVICES: [
                "Generic corporate handshakes",
                "Sterile office environments",
                "Overly formal business interactions",
                "Charts and graphs without context",
                "Professional services that feel impersonal"
            ]
        }
        
        return patterns.get(industry, [])
    
    def _get_authenticity_markers(self, brand_intelligence: BrandIntelligence) -> List[str]:
        """Get authenticity markers specific to the brand."""
        markers = []
        
        # Based on brand values
        if "inclusivity" in brand_intelligence.core_values:
            markers.append("Diverse representation that feels natural and unforced")
        
        if "quality" in brand_intelligence.core_values:
            markers.append("Attention to craftsmanship and detail")
        
        if "community" in brand_intelligence.core_values:
            markers.append("Genuine social connections and shared experiences")
        
        if "innovation" in brand_intelligence.core_values:
            markers.append("Creative solutions that improve real human problems")
        
        # Based on brand personality
        if "premium" in brand_intelligence.brand_personality:
            markers.append("Sophisticated environments and refined aesthetics")
        
        if "accessible" in brand_intelligence.brand_personality:
            markers.append("Approachable settings and relatable scenarios")
        
        if "playful" in brand_intelligence.brand_personality:
            markers.append("Joyful moments and lighthearted interactions")
        
        return markers
    
    def _initialize_industry_expertise(self) -> Dict[IndustryVertical, Dict[str, Any]]:
        """Initialize industry-specific expertise and insights."""
        return {
            IndustryVertical.BEAUTY_COSMETICS: {
                "key_moments": ["application", "transformation", "confidence_boost", "self_expression"],
                "authentic_settings": ["bathroom", "bedroom", "salon", "getting_ready_space"],
                "emotional_journey": ["self_doubt", "experimentation", "discovery", "confidence"],
                "visual_priorities": ["transformation", "texture", "color", "expression"],
                "common_mistakes": ["unrealistic_results", "sterile_environments", "one_beauty_standard"]
            },
            
            IndustryVertical.HEALTH_WELLNESS: {
                "key_moments": ["commitment", "progress", "setback", "breakthrough", "lifestyle_integration"],
                "authentic_settings": ["home", "gym", "outdoors", "kitchen", "community_space"],
                "emotional_journey": ["awareness", "motivation", "challenge", "persistence", "achievement"],
                "visual_priorities": ["real_progress", "sustainable_habits", "community_support"],
                "common_mistakes": ["quick_fixes", "extreme_transformations", "isolated_wellness"]
            }
            
            # Add more industries as needed
        }
    
    def _initialize_narrative_frameworks(self) -> Dict[SceneType, Dict[str, Any]]:
        """Initialize narrative frameworks for different scene types."""
        return {
            SceneType.HOOK: {
                "primary_function": "capture_attention_and_establish_relevance",
                "narrative_structure": "problem_awareness_or_aspiration_activation",
                "emotional_goal": "curiosity_and_connection",
                "visual_approach": "relatable_moment_or_intriguing_scenario",
                "duration_optimization": "first_3_seconds_critical"
            },
            
            SceneType.PROBLEM: {
                "primary_function": "establish_relatable_challenge_or_need",
                "narrative_structure": "authentic_problem_demonstration",
                "emotional_goal": "recognition_and_empathy", 
                "visual_approach": "genuine_frustration_or_limitation",
                "duration_optimization": "build_tension_without_dwelling"
            },
            
            SceneType.SOLUTION: {
                "primary_function": "introduce_brand_as_natural_solution",
                "narrative_structure": "discovery_and_trial_moment",
                "emotional_goal": "hope_and_possibility",
                "visual_approach": "organic_product_integration",
                "duration_optimization": "show_solution_in_action"
            },
            
            SceneType.TRANSFORMATION: {
                "primary_function": "demonstrate_positive_change_and_outcome",
                "narrative_structure": "before_during_after_journey",
                "emotional_goal": "inspiration_and_aspiration",
                "visual_approach": "authentic_improvement_and_confidence",
                "duration_optimization": "meaningful_change_over_time"
            },
            
            SceneType.SOCIAL_PROOF: {
                "primary_function": "build_credibility_through_others_experience",
                "narrative_structure": "community_validation_and_shared_success",
                "emotional_goal": "trust_and_belonging",
                "visual_approach": "genuine_testimonials_and_community_moments",
                "duration_optimization": "authentic_voices_and_experiences"
            },
            
            SceneType.CALL_TO_ACTION: {
                "primary_function": "motivate_specific_next_step",
                "narrative_structure": "clear_path_forward_with_urgency",
                "emotional_goal": "motivation_and_confidence_to_act",
                "visual_approach": "simple_clear_action_demonstration",
                "duration_optimization": "remove_barriers_to_action"
            }
        }
    
    def _validate_brand_alignment(self, 
                                 scene_spec: Dict[str, Any], 
                                 brand_intelligence: BrandIntelligence,
                                 scene_objective: SceneObjective) -> Dict[str, Any]:
        """Validate and enhance scene for brand alignment."""
        
        # Add validation scoring
        scene_spec["brand_alignment_score"] = self._calculate_alignment_score(
            scene_spec, brand_intelligence
        )
        
        # Add production readiness notes
        scene_spec["production_readiness"] = {
            "feasibility_score": self._assess_production_feasibility(scene_spec),
            "technical_requirements": self._extract_technical_requirements(scene_spec),
            "budget_implications": self._assess_budget_impact(scene_spec)
        }
        
        return scene_spec
    
    def _calculate_alignment_score(self, 
                                 scene_spec: Dict[str, Any], 
                                 brand_intelligence: BrandIntelligence) -> Dict[str, Any]:
        """Calculate brand alignment score for the scene."""
        
        alignment_factors = {
            "brand_values_integration": 0.0,
            "target_audience_relevance": 0.0, 
            "emotional_authenticity": 0.0,
            "visual_brand_consistency": 0.0,
            "narrative_voice_alignment": 0.0
        }
        
        # Simple scoring based on presence of key elements
        brand_alignment = scene_spec.get("brand_alignment", {})
        
        if brand_alignment.get("brand_values_reinforced"):
            alignment_factors["brand_values_integration"] = 0.9
        
        if brand_alignment.get("emotional_connection"):
            alignment_factors["emotional_authenticity"] = 0.85
        
        if scene_spec.get("visual_specification", {}).get("color_palette"):
            alignment_factors["visual_brand_consistency"] = 0.8
        
        overall_score = sum(alignment_factors.values()) / len(alignment_factors)
        
        return {
            "overall_score": overall_score,
            "factor_scores": alignment_factors,
            "recommendations": self._generate_alignment_recommendations(alignment_factors)
        }
    
    def _generate_alignment_recommendations(self, factor_scores: Dict[str, float]) -> List[str]:
        """Generate recommendations for improving brand alignment."""
        recommendations = []
        
        for factor, score in factor_scores.items():
            if score < 0.7:
                if factor == "brand_values_integration":
                    recommendations.append("Strengthen connection to core brand values in narrative")
                elif factor == "emotional_authenticity":
                    recommendations.append("Enhance emotional authenticity and human truth")
                elif factor == "visual_brand_consistency":
                    recommendations.append("Align visual elements more closely with brand identity")
        
        return recommendations
    
    def _assess_production_feasibility(self, scene_spec: Dict[str, Any]) -> float:
        """Assess production feasibility of the scene."""
        # Simple feasibility scoring - can be enhanced
        complexity_factors = []
        
        setting = scene_spec.get("visual_specification", {}).get("setting", "")
        if "studio" in setting.lower() or "controlled" in setting.lower():
            complexity_factors.append(0.9)  # Easy
        elif "outdoor" in setting.lower():
            complexity_factors.append(0.7)  # Medium
        elif "multiple locations" in setting.lower():
            complexity_factors.append(0.4)  # Hard
        else:
            complexity_factors.append(0.8)  # Default medium-easy
        
        return sum(complexity_factors) / len(complexity_factors) if complexity_factors else 0.8
    
    def _extract_technical_requirements(self, scene_spec: Dict[str, Any]) -> List[str]:
        """Extract technical production requirements from scene spec."""
        requirements = []
        
        visual_spec = scene_spec.get("visual_specification", {})
        
        if "macro" in str(visual_spec).lower():
            requirements.append("Macro lens capability")
        
        if "slow motion" in str(scene_spec).lower():
            requirements.append("High frame rate recording")
        
        if "golden hour" in str(visual_spec).lower():
            requirements.append("Natural lighting timing constraints")
        
        production_notes = scene_spec.get("production_notes", {})
        props = production_notes.get("props_and_details", [])
        if len(props) > 5:
            requirements.append("Extensive prop sourcing and styling")
        
        return requirements
    
    def _assess_budget_impact(self, scene_spec: Dict[str, Any]) -> str:
        """Assess budget impact of scene requirements."""
        # Simplified budget assessment
        setting = scene_spec.get("visual_specification", {}).get("setting", "")
        
        if any(term in setting.lower() for term in ["studio", "simple", "minimal"]):
            return "Low - controlled environment"
        elif any(term in setting.lower() for term in ["location", "outdoor", "public"]):
            return "Medium - location requirements"
        elif any(term in setting.lower() for term in ["multiple", "complex", "elaborate"]):
            return "High - complex production needs"
        else:
            return "Medium - standard production"
    
    def _parse_text_response(self, text_content: str, creative_brief: Dict[str, Any]) -> Dict[str, Any]:
        """Parse non-JSON GPT-4o response into structured format."""
        
        # Fallback structure if JSON parsing fails
        scene_spec = {
            "scene_concept": {
                "title": f"Brand Scene for {creative_brief['brand_context']['brand_name']}",
                "core_narrative": "Authentic brand moment showcasing genuine customer experience",
                "emotional_arc": creative_brief["scene_requirements"]["emotional_outcome"]
            },
            "visual_specification": {
                "setting": "Authentic environment relevant to brand and audience",
                "lighting": "Natural, mood-appropriate lighting",
                "composition": "Human-centered framing and perspective", 
                "color_palette": "Brand-aligned color story",
                "visual_style": "Professional yet authentic aesthetic"
            },
            "character_direction": {
                "primary_character": "Relatable brand audience member",
                "character_arc": "Positive transformation or realization",
                "performance_notes": "Natural, genuine behaviors and reactions"
            },
            "narrative_elements": {
                "story_beats": ["Opening moment", "Challenge or opportunity", "Brand interaction", "Positive outcome"],
                "dialogue_or_voiceover": f"Brand-voice aligned messaging for {creative_brief['brand_context']['brand_name']}",
                "product_integration": "Organic brand presence within authentic scenario"
            },
            "raw_gpt4o_response": text_content  # Preserve original response
        }
        
        return scene_spec
    
    def _generate_fallback_scene(self, 
                               brand_intelligence: BrandIntelligence,
                               scene_objective: SceneObjective) -> Dict[str, Any]:
        """Generate fallback scene when GPT-4o fails."""
        
        logger.warning("Using fallback scene generation", 
                      action="scene.fallback")
        
        return {
            "scene_concept": {
                "title": f"Authentic {brand_intelligence.brand_name} Moment",
                "core_narrative": f"Real customer experiencing {brand_intelligence.brand_name} value",
                "emotional_arc": scene_objective.emotional_outcome
            },
            "visual_specification": {
                "setting": "Natural environment relevant to brand usage",
                "lighting": "Soft, natural lighting that feels welcoming",
                "composition": "Medium shots focusing on human experience",
                "color_palette": "Warm, approachable tones with brand accent colors",
                "visual_style": "Authentic, documentary-style cinematography"
            },
            "character_direction": {
                "primary_character": "Genuine brand audience representative", 
                "character_arc": "Discovery and positive experience with brand",
                "performance_notes": "Natural reactions and authentic engagement"
            },
            "production_notes": {
                "approach": "Focus on real human experience over production complexity",
                "key_message": "Authentic brand value in everyday context"
            },
            "brand_alignment": {
                "brand_values_reinforced": brand_intelligence.core_values[:2],
                "emotional_connection": scene_objective.emotional_outcome,
                "differentiation": "Genuine human-centered brand experience"
            },
            "fallback_generated": True
        }

def create_brand_intelligence_for_company(company_description: str, 
                                        brand_name: str,
                                        industry: IndustryVertical) -> BrandIntelligence:
    """Helper function to create BrandIntelligence from company description."""
    
    # This could be enhanced to use GPT-4o to extract intelligence from description
    # For now, providing a manual mapping approach
    
    intelligence_mapping = {
        "ulta beauty": BrandIntelligence(
            brand_name="Ulta Beauty",
            industry_vertical=IndustryVertical.BEAUTY_COSMETICS,
            brand_personality=["inclusive", "accessible", "diverse", "empowering"],
            core_values=["inclusivity", "beauty for all", "innovation", "community"],
            target_audience={
                "demographics": "Beauty enthusiasts of all ages, income levels, and backgrounds",
                "psychographics": "Self-expression focused, community-oriented, value-conscious",
                "behaviors": "Product discovery, loyalty program engagement, omnichannel shopping",
                "pain_points": "Finding right products for individual needs, navigating vast selection",
                "aspirations": "Personal beauty expression, confidence, belonging to beauty community"
            },
            unique_value_proposition="One-stop beauty destination offering 25,000+ products from mass to prestige with expert services",
            emotional_drivers=["confidence", "self-expression", "belonging", "discovery"],
            brand_voice="Friendly, inclusive, empowering, and knowledgeable",
            visual_identity={
                "primary_colors": ["warm colors", "inclusive tones"],
                "style": "accessible luxury", 
                "aesthetic": "diverse beauty celebration"
            },
            competitive_positioning="Largest specialty beauty retailer democratizing beauty for all",
            cultural_context={
                "inclusivity_focus": "Beauty without barriers",
                "community_aspect": "Shared beauty experiences and discovery"
            }
        )
    }
    
    return intelligence_mapping.get(brand_name.lower(), 
        BrandIntelligence(
            brand_name=brand_name,
            industry_vertical=industry,
            brand_personality=["professional", "trustworthy"],
            core_values=["quality", "service"],
            target_audience={"demographics": "General consumer audience"},
            unique_value_proposition="Quality products and services",
            emotional_drivers=["trust", "satisfaction"],
            brand_voice="Professional and reliable",
            visual_identity={"style": "clean and professional"},
            competitive_positioning="Trusted industry leader"
        )
    )

# Example usage and testing
if __name__ == "__main__":
    
    # Example: Generate scene for Ulta Beauty
    generator = BrandAwareSceneGenerator()
    
    # Create brand intelligence for Ulta
    ulta_intelligence = create_brand_intelligence_for_company(
        company_description="""Ulta Beauty, Inc. is the largest specialty beauty retailer in the United States, 
        offering over 25,000 products from more than 600 brands across cosmetics, skincare, haircare, fragrance, 
        and wellness categories. Ulta emphasizes inclusivity, innovation, and community engagement.""",
        brand_name="Ulta Beauty",
        industry=IndustryVertical.BEAUTY_COSMETICS
    )
    
    # Define scene objective  
    scene_objective = SceneObjective(
        primary_goal="showcase product discovery and personal beauty expression",
        emotional_outcome="viewer feels inspired and confident about their beauty journey",
        behavioral_trigger="motivates store visit or app exploration",
        narrative_function=SceneType.TRANSFORMATION,
        duration_seconds=10,
        visual_priority=["authentic transformation", "product interaction", "confidence boost"]
    )
    
    # Generate brand-aligned scene
    try:
        scene_result = generator.generate_brand_aligned_scene(
            brand_intelligence=ulta_intelligence,
            scene_objective=scene_objective
        )
        
        print("✅ Brand-Aware Scene Generated Successfully!")
        print(json.dumps(scene_result, indent=2))
        
    except Exception as e:
        print(f"❌ Scene generation failed: {e}")