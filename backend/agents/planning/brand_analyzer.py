"""
Relicon AI Ad Creator - Brand Analyzer
Advanced brand analysis and strategic positioning for AI ad planning
"""
import json
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage

from core.models import AdCreationRequest, AdPlatform, AdStyle
from core.settings import settings


class BrandAnalyzer:
    """
    Brand Analysis Engine
    
    Performs deep strategic analysis of brands, competitive positioning,
    and target audience psychology to inform AI ad creation.
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=settings.OPENAI_API_KEY,
            temperature=0.2,  # Low temperature for consistent analysis
            max_tokens=3000
        )
    
    async def analyze_brand_strategy(self, request: AdCreationRequest) -> Dict[str, Any]:
        """
        Perform comprehensive brand strategy analysis
        
        Args:
            request: Ad creation request with brand information
            
        Returns:
            Dictionary with detailed brand analysis
        """
        print(f"🔍 Brand Analyzer: Analyzing {request.brand_name}")
        
        analysis_prompt = self._build_analysis_prompt(request)
        response = self.llm.invoke([SystemMessage(content=analysis_prompt)])
        
        try:
            analysis = json.loads(response.content)
            print(f"✅ Brand analysis complete for {request.brand_name}")
            return analysis
        except json.JSONDecodeError:
            print(f"❌ Failed to parse brand analysis for {request.brand_name}")
            return self._get_fallback_analysis(request)
    
    def _build_analysis_prompt(self, request: AdCreationRequest) -> str:
        """Build the brand analysis prompt"""
        return f"""
        You are a world-class brand strategist analyzing this brand for AI ad creation:
        
        BRAND INFORMATION:
        - Brand Name: {request.brand_name}
        - Description: {request.brand_description}
        - Product: {request.product_name or "Not specified"}
        - Target Audience: {request.target_audience or "General"}
        - USP: {request.unique_selling_point or "Not specified"}
        - Platform: {request.platform}
        - Style: {request.style}
        - Duration: {request.duration} seconds
        
        Provide ultra-detailed strategic analysis in JSON format:
        {{
            "brand_personality": {{
                "core_values": ["value1", "value2", "value3"],
                "emotional_attributes": ["attr1", "attr2", "attr3"],
                "competitive_positioning": "clear positioning statement",
                "brand_archetype": "archetype name and description",
                "voice_characteristics": ["characteristic1", "characteristic2"]
            }},
            "target_psychology": {{
                "primary_motivations": ["motivation1", "motivation2"],
                "pain_points": ["pain1", "pain2"],
                "emotional_triggers": ["trigger1", "trigger2"],
                "attention_span": "estimated seconds for platform",
                "platform_behavior": "how they behave on {request.platform}",
                "decision_factors": ["factor1", "factor2"]
            }},
            "ad_strategy": {{
                "primary_objective": "main goal of the ad",
                "secondary_objectives": ["obj1", "obj2"],
                "success_metrics": ["metric1", "metric2"],
                "messaging_hierarchy": ["message1", "message2", "message3"],
                "call_to_action_strategy": "CTA approach"
            }},
            "creative_direction": {{
                "visual_style": "detailed visual approach",
                "tone_of_voice": "specific tone description",
                "pacing_strategy": "fast/medium/slow with reasoning",
                "energy_curve": "how energy should progress",
                "color_psychology": ["color1 meaning", "color2 meaning"],
                "music_mood": "background music approach"
            }},
            "competitive_analysis": {{
                "category_trends": ["trend1", "trend2"],
                "differentiation_opportunities": ["opp1", "opp2"],
                "messaging_gaps": ["gap1", "gap2"],
                "visual_differentiation": "how to stand out visually"
            }}
        }}
        
        Be extremely detailed and specific. This analysis will drive every creative decision.
        """
    
    def _get_fallback_analysis(self, request: AdCreationRequest) -> Dict[str, Any]:
        """Provide fallback analysis if AI analysis fails"""
        print("⚠️ Using fallback brand analysis")
        
        return {
            "brand_personality": {
                "core_values": ["quality", "reliability", "innovation"],
                "emotional_attributes": ["trustworthy", "professional"],
                "competitive_positioning": f"{request.brand_name} offers unique value",
                "brand_archetype": "The Expert",
                "voice_characteristics": ["confident", "knowledgeable"]
            },
            "target_psychology": {
                "primary_motivations": ["solve problems", "save time"],
                "pain_points": ["inefficiency", "complexity"],
                "emotional_triggers": ["convenience", "results"],
                "attention_span": f"{min(request.duration, 5)} seconds",
                "platform_behavior": f"Active on {request.platform}",
                "decision_factors": ["value", "trust"]
            },
            "ad_strategy": {
                "primary_objective": "Generate awareness and interest",
                "secondary_objectives": ["build trust", "drive action"],
                "success_metrics": ["engagement", "conversions"],
                "messaging_hierarchy": [
                    request.unique_selling_point or "Key benefit",
                    "How it works",
                    request.call_to_action or "Take action"
                ],
                "call_to_action_strategy": "Clear and direct"
            },
            "creative_direction": {
                "visual_style": f"{request.style} approach with clean aesthetics",
                "tone_of_voice": "Professional and approachable",
                "pacing_strategy": "medium" if request.duration >= 30 else "fast",
                "energy_curve": "Build excitement throughout",
                "color_psychology": ["blue for trust", "green for growth"],
                "music_mood": "Uplifting and energetic"
            },
            "competitive_analysis": {
                "category_trends": ["digital transformation", "user experience"],
                "differentiation_opportunities": ["unique approach", "better results"],
                "messaging_gaps": ["emotional connection", "clear benefits"],
                "visual_differentiation": "Modern and distinctive design"
            }
        }
    
    def analyze_platform_requirements(self, platform: AdPlatform) -> Dict[str, Any]:
        """
        Analyze platform-specific requirements and characteristics
        
        Args:
            platform: Target advertising platform
            
        Returns:
            Dictionary with platform requirements and constraints
        """
        platform_specs = {
            AdPlatform.TIKTOK: {
                "aspect_ratio": "9:16",
                "optimal_duration": "15-30 seconds",
                "hook_timing": "first 3 seconds critical",
                "style_preferences": ["trendy", "authentic", "fast-paced"],
                "content_approach": "entertainment-first",
                "text_overlay": "minimal, large fonts",
                "music_importance": "very high",
                "audience_behavior": "scroll-heavy, low attention span",
                "success_factors": ["viral potential", "trend alignment", "authenticity"]
            },
            AdPlatform.INSTAGRAM: {
                "aspect_ratio": "1:1 or 9:16",
                "optimal_duration": "15-60 seconds", 
                "hook_timing": "first 2 seconds",
                "style_preferences": ["aesthetic", "polished", "aspirational"],
                "content_approach": "visual-first",
                "text_overlay": "moderate, stylish fonts",
                "music_importance": "high",
                "audience_behavior": "browse-focused, visual-driven",
                "success_factors": ["visual appeal", "brand aesthetic", "engagement"]
            },
            AdPlatform.FACEBOOK: {
                "aspect_ratio": "16:9 or 1:1",
                "optimal_duration": "15-120 seconds",
                "hook_timing": "first 3 seconds",
                "style_preferences": ["informative", "trustworthy", "relatable"],
                "content_approach": "story-first",
                "text_overlay": "heavy, clear messaging",
                "music_importance": "medium",
                "audience_behavior": "content-focused, longer attention",
                "success_factors": ["clear value prop", "social proof", "accessibility"]
            },
            AdPlatform.YOUTUBE_SHORTS: {
                "aspect_ratio": "9:16",
                "optimal_duration": "15-60 seconds",
                "hook_timing": "first 5 seconds",
                "style_preferences": ["educational", "entertaining", "high-value"],
                "content_approach": "content-first",
                "text_overlay": "moderate, readable",
                "music_importance": "medium-high",
                "audience_behavior": "content-hungry, retention-focused",
                "success_factors": ["watch time", "educational value", "entertainment"]
            },
            AdPlatform.UNIVERSAL: {
                "aspect_ratio": "16:9",
                "optimal_duration": "30-60 seconds",
                "hook_timing": "first 3 seconds",
                "style_preferences": ["professional", "clear", "versatile"],
                "content_approach": "balanced",
                "text_overlay": "moderate, professional",
                "music_importance": "medium",
                "audience_behavior": "varied, platform-dependent",
                "success_factors": ["clear messaging", "broad appeal", "adaptability"]
            }
        }
        
        return platform_specs.get(platform, platform_specs[AdPlatform.UNIVERSAL])
    
    def get_style_requirements(self, style: AdStyle) -> Dict[str, Any]:
        """
        Get style-specific creative requirements
        
        Args:
            style: Creative style preference
            
        Returns:
            Dictionary with style requirements
        """
        style_specs = {
            AdStyle.PROFESSIONAL: {
                "color_scheme": "corporate colors, blues, greys",
                "pacing": "measured and deliberate",
                "visual_complexity": "clean and uncluttered",
                "font_style": "sans-serif, professional",
                "music_style": "corporate, uplifting",
                "tone_keywords": ["authoritative", "trustworthy", "expert"],
                "visual_elements": ["clean lines", "professional imagery", "subtle animations"]
            },
            AdStyle.ENERGETIC: {
                "color_scheme": "vibrant, high-contrast",
                "pacing": "fast and dynamic",
                "visual_complexity": "high energy, multiple elements",
                "font_style": "bold, impactful",
                "music_style": "upbeat, motivating",
                "tone_keywords": ["exciting", "dynamic", "powerful"],
                "visual_elements": ["quick cuts", "dynamic movements", "bright colors"]
            },
            AdStyle.MINIMAL: {
                "color_scheme": "monochrome, subtle accents",
                "pacing": "slow and contemplative",
                "visual_complexity": "ultra-simple, focus-driven",
                "font_style": "thin, elegant",
                "music_style": "ambient, subtle",
                "tone_keywords": ["elegant", "sophisticated", "refined"],
                "visual_elements": ["white space", "single focus", "gentle transitions"]
            },
            AdStyle.CASUAL: {
                "color_scheme": "warm, approachable",
                "pacing": "relaxed, conversational",
                "visual_complexity": "moderate, friendly",
                "font_style": "rounded, friendly",
                "music_style": "light, approachable",
                "tone_keywords": ["friendly", "relatable", "approachable"],
                "visual_elements": ["natural imagery", "soft edges", "warm lighting"]
            },
            AdStyle.CINEMATIC: {
                "color_scheme": "dramatic, film-like",
                "pacing": "varied, story-driven",
                "visual_complexity": "high production value",
                "font_style": "dramatic, serif accents",
                "music_style": "orchestral, dramatic",
                "tone_keywords": ["dramatic", "artistic", "epic"],
                "visual_elements": ["film techniques", "dramatic lighting", "story arcs"]
            }
        }
        
        return style_specs.get(style, style_specs[AdStyle.PROFESSIONAL]) 