"""
Relicon AI Ad Creator - Narrative Designer
Advanced narrative structure and emotional journey design for compelling ads
"""
import json
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage

from core.models import AdCreationRequest
from core.settings import settings


class NarrativeDesigner:
    """
    Narrative Design Engine
    
    Creates compelling story structures and emotional journeys that 
    capture attention and drive action with mathematical precision.
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=settings.OPENAI_API_KEY,
            temperature=0.3,  # Balanced creativity and consistency
            max_tokens=2500
        )
    
    async def design_narrative_arc(
        self, 
        request: AdCreationRequest, 
        brand_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Design a mathematically precise narrative arc
        
        Args:
            request: Ad creation request
            brand_analysis: Strategic brand analysis
            
        Returns:
            Dictionary with narrative structure and emotional journey
        """
        print(f"📖 Narrative Designer: Creating story arc for {request.brand_name}")
        
        narrative_prompt = self._build_narrative_prompt(request, brand_analysis)
        response = self.llm.invoke([SystemMessage(content=narrative_prompt)])
        
        try:
            narrative = json.loads(response.content)
            print(f"✅ Narrative arc complete: {narrative.get('narrative_concept', 'Unknown')}")
            return narrative
        except json.JSONDecodeError:
            print(f"❌ Failed to parse narrative design")
            return self._get_fallback_narrative(request, brand_analysis)
    
    def _build_narrative_prompt(self, request: AdCreationRequest, brand_analysis: Dict[str, Any]) -> str:
        """Build the narrative design prompt"""
        return f"""
        You are a master storyteller and advertising narrative expert. Create a mathematically precise 
        narrative arc for this {request.duration}-second ad:
        
        BRAND CONTEXT:
        {json.dumps(brand_analysis, indent=2)}
        
        AD SPECIFICATIONS:
        - Duration: {request.duration} seconds
        - Platform: {request.platform}
        - Style: {request.style}
        - Target: {request.target_audience or "General audience"}
        
        Create a narrative structure with exact mathematical precision in JSON format:
        {{
            "narrative_concept": "One powerful sentence describing the core story",
            "story_framework": "Choose: Problem-Solution, Before-After, Journey, Transformation, or Discovery",
            "story_arc": {{
                "hook_percentage": 20,
                "development_percentage": 35,
                "climax_percentage": 25,
                "resolution_percentage": 20
            }},
            "emotional_journey": [
                {{
                    "emotion": "curiosity", 
                    "intensity": 8, 
                    "timing_start": 0.0, 
                    "timing_end": 3.5,
                    "trigger": "what creates this emotion"
                }},
                {{
                    "emotion": "desire", 
                    "intensity": 9, 
                    "timing_start": 3.5, 
                    "timing_end": 12.0,
                    "trigger": "what creates this emotion"
                }},
                {{
                    "emotion": "urgency", 
                    "intensity": 10, 
                    "timing_start": 12.0, 
                    "timing_end": {request.duration},
                    "trigger": "what creates this emotion"
                }}
            ],
            "attention_curve": [
                {{"second": 0, "attention_level": 10, "technique": "technique used"}},
                {{"second": {request.duration // 4}, "attention_level": 7, "technique": "technique used"}},
                {{"second": {request.duration // 2}, "attention_level": 9, "technique": "technique used"}},
                {{"second": {request.duration * 3 // 4}, "attention_level": 10, "technique": "technique used"}},
                {{"second": {request.duration}, "attention_level": 10, "technique": "technique used"}}
            ],
            "key_moments": [
                {{
                    "timing": 1.5, 
                    "moment_type": "hook_peak", 
                    "description": "Maximum attention grab moment",
                    "required_elements": ["specific element needed"]
                }},
                {{
                    "timing": {request.duration * 0.6}, 
                    "moment_type": "value_reveal", 
                    "description": "Core value proposition moment",
                    "required_elements": ["specific element needed"]
                }},
                {{
                    "timing": {request.duration * 0.9}, 
                    "moment_type": "action_trigger", 
                    "description": "Call to action moment",
                    "required_elements": ["specific element needed"]
                }}
            ],
            "narrative_tension": {{
                "conflict_introduction": "When and how conflict/problem is introduced",
                "tension_buildup": "How tension increases throughout",
                "resolution_method": "How the solution is revealed",
                "satisfaction_delivery": "How resolution satisfies viewer"
            }},
            "pacing_strategy": {{
                "opening_pace": "fast/medium/slow",
                "middle_pace": "fast/medium/slow", 
                "closing_pace": "fast/medium/slow",
                "rhythm_changes": ["description of major pace shifts"],
                "timing_rationale": "Why this pacing works for the audience"
            }}
        }}
        
        CRITICAL REQUIREMENTS:
        1. All percentages must add up to 100%
        2. All timing must be precise to 0.1 seconds
        3. Emotional journey must have 3-5 distinct phases
        4. Attention curve must account for platform behavior
        5. Moments must have exact timing and purpose
        
        Make every second count with mathematical precision!
        """
    
    def _get_fallback_narrative(
        self, 
        request: AdCreationRequest, 
        brand_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Provide fallback narrative if AI fails"""
        print("⚠️ Using fallback narrative design")
        
        duration = request.duration
        
        return {
            "narrative_concept": f"Transform your experience with {request.brand_name}",
            "story_framework": "Problem-Solution",
            "story_arc": {
                "hook_percentage": 20,
                "development_percentage": 40, 
                "climax_percentage": 25,
                "resolution_percentage": 15
            },
            "emotional_journey": [
                {
                    "emotion": "curiosity",
                    "intensity": 8,
                    "timing_start": 0.0,
                    "timing_end": duration * 0.3,
                    "trigger": "intriguing opening"
                },
                {
                    "emotion": "interest", 
                    "intensity": 7,
                    "timing_start": duration * 0.3,
                    "timing_end": duration * 0.7,
                    "trigger": "problem identification"
                },
                {
                    "emotion": "desire",
                    "intensity": 9,
                    "timing_start": duration * 0.7,
                    "timing_end": duration * 0.9,
                    "trigger": "solution revelation"
                },
                {
                    "emotion": "urgency",
                    "intensity": 10,
                    "timing_start": duration * 0.9,
                    "timing_end": duration,
                    "trigger": "call to action"
                }
            ],
            "attention_curve": [
                {"second": 0, "attention_level": 10, "technique": "strong hook"},
                {"second": duration * 0.25, "attention_level": 7, "technique": "story development"},
                {"second": duration * 0.5, "attention_level": 9, "technique": "problem revelation"},
                {"second": duration * 0.75, "attention_level": 10, "technique": "solution showcase"},
                {"second": duration, "attention_level": 10, "technique": "strong CTA"}
            ],
            "key_moments": [
                {
                    "timing": 1.5,
                    "moment_type": "hook_peak",
                    "description": "Grab attention immediately",
                    "required_elements": ["visual hook", "audio impact"]
                },
                {
                    "timing": duration * 0.6,
                    "moment_type": "value_reveal", 
                    "description": "Show the transformation",
                    "required_elements": ["clear benefit", "visual proof"]
                },
                {
                    "timing": duration * 0.9,
                    "moment_type": "action_trigger",
                    "description": "Drive immediate action",
                    "required_elements": ["clear CTA", "urgency"]
                }
            ],
            "narrative_tension": {
                "conflict_introduction": "Early problem identification",
                "tension_buildup": "Escalating problem consequences",
                "resolution_method": "Clear solution presentation",
                "satisfaction_delivery": "Transformation showcase"
            },
            "pacing_strategy": {
                "opening_pace": "fast",
                "middle_pace": "medium",
                "closing_pace": "fast",
                "rhythm_changes": ["accelerate at solution reveal"],
                "timing_rationale": "Match platform attention patterns"
            }
        }
    
    def calculate_scene_timing(self, narrative: Dict[str, Any], total_duration: float) -> Dict[str, Any]:
        """
        Calculate precise scene timing based on narrative structure
        
        Args:
            narrative: Narrative structure from design_narrative_arc
            total_duration: Total ad duration in seconds
            
        Returns:
            Dictionary with scene timing breakdown
        """
        story_arc = narrative["story_arc"]
        
        # Calculate scene durations based on percentages
        hook_duration = total_duration * (story_arc["hook_percentage"] / 100)
        development_duration = total_duration * (story_arc["development_percentage"] / 100)
        climax_duration = total_duration * (story_arc["climax_percentage"] / 100)
        resolution_duration = total_duration * (story_arc["resolution_percentage"] / 100)
        
        # Calculate start times
        hook_start = 0.0
        development_start = hook_duration
        climax_start = development_start + development_duration
        resolution_start = climax_start + climax_duration
        
        return {
            "scene_timing": {
                "hook": {
                    "start_time": hook_start,
                    "duration": hook_duration,
                    "end_time": hook_start + hook_duration,
                    "purpose": "Capture attention and establish context"
                },
                "development": {
                    "start_time": development_start,
                    "duration": development_duration, 
                    "end_time": development_start + development_duration,
                    "purpose": "Build story and emotional connection"
                },
                "climax": {
                    "start_time": climax_start,
                    "duration": climax_duration,
                    "end_time": climax_start + climax_duration,
                    "purpose": "Reveal solution and peak emotion"
                },
                "resolution": {
                    "start_time": resolution_start,
                    "duration": resolution_duration,
                    "end_time": total_duration,
                    "purpose": "Call to action and satisfaction"
                }
            },
            "validation": {
                "total_calculated": hook_duration + development_duration + climax_duration + resolution_duration,
                "total_target": total_duration,
                "timing_error": abs((hook_duration + development_duration + climax_duration + resolution_duration) - total_duration),
                "is_valid": abs((hook_duration + development_duration + climax_duration + resolution_duration) - total_duration) < 0.1
            }
        }
    
    def get_narrative_templates(self) -> Dict[str, Dict[str, Any]]:
        """
        Get pre-built narrative templates for different ad types
        
        Returns:
            Dictionary of narrative templates
        """
        return {
            "problem_solution": {
                "description": "Classic problem-solution structure",
                "story_arc": {"hook_percentage": 15, "development_percentage": 40, "climax_percentage": 30, "resolution_percentage": 15},
                "emotional_flow": ["concern", "frustration", "relief", "satisfaction"],
                "best_for": ["B2B", "practical solutions", "clear value props"]
            },
            "transformation": {
                "description": "Before and after transformation",
                "story_arc": {"hook_percentage": 20, "development_percentage": 30, "climax_percentage": 35, "resolution_percentage": 15},
                "emotional_flow": ["curiosity", "aspiration", "amazement", "desire"],
                "best_for": ["lifestyle", "beauty", "fitness", "personal development"]
            },
            "discovery": {
                "description": "Journey of discovery and revelation",
                "story_arc": {"hook_percentage": 25, "development_percentage": 35, "climax_percentage": 25, "resolution_percentage": 15},
                "emotional_flow": ["intrigue", "curiosity", "surprise", "enlightenment"],
                "best_for": ["education", "innovation", "technology", "research"]
            },
            "testimonial": {
                "description": "Personal story and social proof",
                "story_arc": {"hook_percentage": 20, "development_percentage": 45, "climax_percentage": 20, "resolution_percentage": 15},
                "emotional_flow": ["relatability", "empathy", "trust", "confidence"],
                "best_for": ["services", "healthcare", "finance", "personal stories"]
            },
            "urgency": {
                "description": "Time-sensitive opportunity",
                "story_arc": {"hook_percentage": 30, "development_percentage": 25, "climax_percentage": 30, "resolution_percentage": 15},
                "emotional_flow": ["attention", "interest", "urgency", "action"],
                "best_for": ["sales", "limited offers", "events", "seasonal"]
            }
        } 