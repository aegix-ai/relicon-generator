"""
Relicon AI Ad Creator - Master Planner Agent
Revolutionary AI agent for ultra-detailed ad planning with mathematical precision
"""
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from core.models import (
    AdCreationRequest, MasterAdPlan, AdScene, SceneComponent, 
    PlanningContext, AdPlatform, AdStyle
)
from core.settings import settings


class MasterPlannerAgent:
    """
    Revolutionary Master Planner AI Agent
    
    Creates ultra-detailed, mathematically precise ad plans that break down
    every second of the ad into atomic components. This is the brain of the system.
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=settings.OPENAI_API_KEY,
            temperature=0.3,  # Lower temperature for precision
            max_tokens=4000
        )
        self.planning_graph = self._create_planning_graph()
        
    def _create_planning_graph(self) -> StateGraph:
        """Create the LangGraph state machine for planning"""
        
        # Define the planning state as a TypedDict
        from typing import TypedDict
        
        class PlanningState(TypedDict):
            request: AdCreationRequest
            context: PlanningContext
            strategic_analysis: Dict[str, Any]
            narrative_structure: Dict[str, Any]
            scene_breakdown: List[Dict[str, Any]]
            timing_analysis: Dict[str, Any]
            master_plan: Optional[MasterAdPlan]
            
        # Create the graph
        workflow = StateGraph(PlanningState)
        
        # Add nodes for each planning stage
        workflow.add_node("analyze_brand", self._analyze_brand_strategy)
        workflow.add_node("design_narrative", self._design_narrative_arc)
        workflow.add_node("architect_scenes", self._architect_scenes)
        workflow.add_node("calculate_timing", self._calculate_precise_timing)
        workflow.add_node("integrate_brand", self._integrate_brand_elements)
        workflow.add_node("finalize_plan", self._finalize_master_plan)
        
        # Define the flow
        workflow.set_entry_point("analyze_brand")
        workflow.add_edge("analyze_brand", "design_narrative")
        workflow.add_edge("design_narrative", "architect_scenes")
        workflow.add_edge("architect_scenes", "calculate_timing")
        workflow.add_edge("calculate_timing", "integrate_brand")
        workflow.add_edge("integrate_brand", "finalize_plan")
        workflow.add_edge("finalize_plan", END)
        
        return workflow.compile()
    
    async def create_master_plan(self, request: AdCreationRequest) -> MasterAdPlan:
        """
        Create an ultra-detailed master plan for the ad
        This is the main entry point that orchestrates the entire planning process
        """
        print(f"🧠 Master Planner: Starting ultra-detailed planning for {request.brand_name}")
        
        # Initialize planning context
        context = PlanningContext(
            request=request,
            brand_analysis={},
            platform_requirements=self._get_platform_requirements(request.platform),
            creative_constraints=self._get_creative_constraints(request)
        )
        
        # Execute the planning graph
        initial_state = {
            "request": request,
            "context": context,
            "strategic_analysis": {},
            "narrative_structure": {},
            "scene_breakdown": [],
            "timing_analysis": {},
            "master_plan": None
        }
        
        final_state = await self.planning_graph.ainvoke(initial_state)
        
        print(f"✅ Master Planner: Ultra-detailed plan completed with {len(final_state['scene_breakdown'])} scenes")
        return final_state["master_plan"]
    
    def _analyze_brand_strategy(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze brand strategy and positioning with mathematical precision"""
        request = state["request"]
        
        analysis_prompt = f"""
        You are a world-class brand strategist and advertising genius. Analyze this brand with mathematical precision:
        
        Brand: {request.brand_name}
        Description: {request.brand_description}
        Target Audience: {request.target_audience or "General"}
        USP: {request.unique_selling_point or "Not specified"}
        Platform: {request.platform}
        Duration: {request.duration} seconds
        Style: {request.style}
        
        Provide ultra-detailed strategic analysis in JSON format:
        {{
            "brand_personality": {{
                "core_values": ["value1", "value2", "value3"],
                "emotional_attributes": ["attr1", "attr2"],
                "competitive_positioning": "positioning statement",
                "brand_archetype": "archetype name"
            }},
            "target_psychology": {{
                "primary_motivations": ["motivation1", "motivation2"],
                "pain_points": ["pain1", "pain2"],
                "emotional_triggers": ["trigger1", "trigger2"],
                "attention_span": "X seconds",
                "platform_behavior": "behavior description"
            }},
            "ad_strategy": {{
                "primary_objective": "objective",
                "secondary_objectives": ["obj1", "obj2"],
                "success_metrics": ["metric1", "metric2"],
                "messaging_hierarchy": ["message1", "message2", "message3"]
            }},
            "creative_direction": {{
                "visual_style": "style description",
                "tone_of_voice": "tone description",
                "pacing_strategy": "fast/medium/slow",
                "energy_curve": "description of energy progression"
            }}
        }}
        
        Be extremely detailed and precise. This analysis will drive every creative decision.
        """
        
        response = self.llm.invoke([SystemMessage(content=analysis_prompt)])
        strategic_analysis = json.loads(response.content)
        
        state["strategic_analysis"] = strategic_analysis
        return state
    
    def _design_narrative_arc(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Design the narrative arc with dramatic precision"""
        request = state["request"]
        strategic_analysis = state["strategic_analysis"]
        
        narrative_prompt = f"""
        Based on this strategic analysis, design a mathematically precise narrative arc for a {request.duration}-second ad:
        
        Strategic Analysis: {json.dumps(strategic_analysis, indent=2)}
        
        Create a narrative structure in JSON format:
        {{
            "narrative_concept": "One sentence concept",
            "story_arc": {{
                "setup_percentage": 15,
                "conflict_percentage": 25,
                "resolution_percentage": 35,
                "cta_percentage": 25
            }},
            "emotional_journey": [
                {{"emotion": "curiosity", "intensity": 7, "timing_start": 0, "timing_end": 3}},
                {{"emotion": "desire", "intensity": 9, "timing_start": 3, "timing_end": 8}},
                {{"emotion": "urgency", "intensity": 10, "timing_start": 8, "timing_end": 12}}
            ],
            "attention_curve": [
                {{"second": 0, "attention_level": 10}},
                {{"second": 3, "attention_level": 8}},
                {{"second": 6, "attention_level": 9}},
                {{"second": 9, "attention_level": 10}}
            ],
            "key_moments": [
                {{"timing": 1.5, "moment": "hook_peak", "description": "Maximum attention grab"}},
                {{"timing": 8.0, "moment": "transformation_reveal", "description": "Show the solution"}},
                {{"timing": 12.0, "moment": "urgency_peak", "description": "Call to action"}}
            ]
        }}
        
        Calculate exact percentages that add up to 100%. Be mathematically precise.
        """
        
        response = self.llm.invoke([SystemMessage(content=narrative_prompt)])
        narrative_structure = json.loads(response.content)
        
        state["narrative_structure"] = narrative_structure
        return state
    
    def _architect_scenes(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Architect individual scenes with atomic precision"""
        request = state["request"]
        narrative = state["narrative_structure"]
        strategic = state["strategic_analysis"]
        
        scenes_prompt = f"""
        Now architect the individual scenes with ATOMIC precision. Break down every second:
        
        Duration: {request.duration} seconds
        Narrative Arc: {json.dumps(narrative, indent=2)}
        Brand Strategy: {json.dumps(strategic, indent=2)}
        
        Create ultra-detailed scene breakdown in JSON format:
        {{
            "scenes": [
                {{
                    "scene_id": "scene_1",
                    "scene_type": "hook",
                    "start_time": 0.0,
                    "duration": 4.0,
                    "scene_purpose": "Grab attention and create curiosity",
                    "components": [
                        {{
                            "component_id": "hook_visual",
                            "start_time": 0.0,
                            "duration": 4.0,
                            "visual_type": "video",
                            "visual_prompt": "Ultra-detailed prompt for Luma AI",
                            "visual_style": "cinematic close-up",
                            "luma_prompt": "Specific technical prompt",
                            "has_voiceover": true,
                            "voiceover_text": "Exact script text",
                            "voice_tone": "intriguing",
                            "entry_effect": "fade_in",
                            "exit_effect": "quick_cut"
                        }}
                    ],
                    "main_script": "Complete voiceover script",
                    "camera_direction": "Extreme close-up, shallow depth of field",
                    "lighting_notes": "Dramatic side lighting",
                    "color_palette": ["#FF6B35", "#F7931E", "#FFD23F"]
                }}
            ]
        }}
        
        Requirements:
        1. Each scene must have EXACT timing (start_time + duration)
        2. Every component needs ultra-detailed prompts for generation
        3. All timing must add up to exactly {request.duration} seconds
        4. Include detailed visual direction for every element
        5. Create 3-5 scenes that flow perfectly together
        6. Make each prompt extremely specific and actionable for Luma AI
        
        This is the foundation of the entire ad - be EXTREMELY detailed and precise.
        """
        
        response = self.llm.invoke([SystemMessage(content=scenes_prompt)])
        scene_breakdown = json.loads(response.content)
        
        state["scene_breakdown"] = scene_breakdown["scenes"]
        return state
    
    def _calculate_precise_timing(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate mathematically precise timing for all elements"""
        scenes = state["scene_breakdown"]
        request = state["request"]
        
        timing_analysis = {
            "total_duration": request.duration,
            "scene_count": len(scenes),
            "timing_validation": [],
            "pacing_analysis": {},
            "sync_points": []
        }
        
        current_time = 0.0
        for i, scene in enumerate(scenes):
            # Validate timing
            scene_duration = scene["duration"]
            timing_validation = {
                "scene_id": scene["scene_id"],
                "expected_start": current_time,
                "actual_start": scene["start_time"],
                "duration": scene_duration,
                "timing_accurate": abs(scene["start_time"] - current_time) < 0.1
            }
            timing_analysis["timing_validation"].append(timing_validation)
            
            # Calculate sync points for audio-visual alignment
            for component in scene["components"]:
                if component.get("has_voiceover"):
                    sync_point = {
                        "time": component["start_time"],
                        "type": "voiceover_start",
                        "text": component["voiceover_text"][:50] + "...",
                        "duration": component["duration"]
                    }
                    timing_analysis["sync_points"].append(sync_point)
            
            current_time += scene_duration
        
        # Verify total timing
        timing_analysis["total_calculated"] = current_time
        timing_analysis["timing_error"] = abs(current_time - request.duration)
        timing_analysis["timing_valid"] = timing_analysis["timing_error"] < 0.5
        
        state["timing_analysis"] = timing_analysis
        return state
    
    def _integrate_brand_elements(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate brand elements throughout the ad"""
        request = state["request"]
        scenes = state["scene_breakdown"]
        
        # Calculate brand presence timeline
        brand_timeline = {}
        logo_appearances = []
        
        for scene in scenes:
            scene_id = scene["scene_id"]
            start_time = scene["start_time"]
            duration = scene["duration"]
            
            # Determine brand presence intensity for this scene
            if scene["scene_type"] in ["hook", "solution"]:
                brand_intensity = 0.8
            elif scene["scene_type"] in ["problem", "benefits"]:
                brand_intensity = 0.6
            else:  # cta, transition
                brand_intensity = 1.0
            
            brand_timeline[scene_id] = {
                "start_time": start_time,
                "duration": duration,
                "brand_intensity": brand_intensity
            }
            
            # Add logo appearances
            if scene["scene_type"] == "cta" or (scene_id == "scene_1"):
                logo_appearances.append({
                    "time": start_time + (duration * 0.8),  # Near end of scene
                    "duration": 1.5,
                    "size": "medium" if scene["scene_type"] == "hook" else "large",
                    "position": "bottom_right" if scene["scene_type"] == "hook" else "center"
                })
        
        # Update state with brand integration
        state["brand_integration"] = {
            "brand_presence_timeline": brand_timeline,
            "logo_appearances": logo_appearances,
            "brand_color_usage": request.brand_colors or ["#2C3E50", "#E74C3C"],
            "brand_consistency_score": 0.95  # High consistency
        }
        
        return state
    
    def _finalize_master_plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Create the final ultra-detailed master plan"""
        request = state["request"]
        strategic = state["strategic_analysis"]
        narrative = state["narrative_structure"]
        scenes_data = state["scene_breakdown"]
        timing = state["timing_analysis"]
        brand_integration = state["brand_integration"]
        
        # Convert scene data to AdScene objects
        scenes = []
        for scene_data in scenes_data:
            components = []
            for comp_data in scene_data["components"]:
                component = SceneComponent(
                    start_time=comp_data["start_time"],
                    duration=comp_data["duration"],
                    end_time=comp_data["start_time"] + comp_data["duration"],
                    visual_type=comp_data["visual_type"],
                    visual_prompt=comp_data["visual_prompt"],
                    visual_style=comp_data["visual_style"],
                    has_voiceover=comp_data.get("has_voiceover", False),
                    voiceover_text=comp_data.get("voiceover_text"),
                    voice_tone=comp_data.get("voice_tone"),
                    has_music=comp_data.get("has_music", False),
                    music_style=comp_data.get("music_style"),
                    entry_effect=comp_data.get("entry_effect"),
                    exit_effect=comp_data.get("exit_effect"),
                    luma_prompt=comp_data.get("luma_prompt")
                )
                components.append(component)
            
            scene = AdScene(
                scene_id=scene_data["scene_id"],
                scene_type=scene_data["scene_type"],
                scene_purpose=scene_data["scene_purpose"],
                start_time=scene_data["start_time"],
                duration=scene_data["duration"],
                components=components,
                main_script=scene_data.get("main_script"),
                camera_direction=scene_data.get("camera_direction"),
                lighting_notes=scene_data.get("lighting_notes"),
                color_palette=scene_data.get("color_palette", [])
            )
            scenes.append(scene)
        
        # Create the master plan
        master_plan = MasterAdPlan(
            plan_id=f"plan_{int(time.time())}_{request.brand_name.lower().replace(' ', '_')}",
            total_duration=float(request.duration),
            ad_concept=strategic["ad_strategy"]["primary_objective"],
            narrative_arc=narrative["narrative_concept"],
            emotional_journey=[ej["emotion"] for ej in narrative["emotional_journey"]],
            key_messages=strategic["ad_strategy"]["messaging_hierarchy"],
            scenes=scenes,
            scene_transitions=[],  # Will be calculated by scene architect
            overall_voice_direction=strategic["creative_direction"]["tone_of_voice"],
            music_strategy=strategic["creative_direction"]["pacing_strategy"],
            brand_presence_timeline=brand_integration["brand_presence_timeline"],
            logo_appearances=brand_integration["logo_appearances"],
            target_platforms=[request.platform]
        )
        
        state["master_plan"] = master_plan
        print(f"🎯 Master Plan completed: {len(scenes)} scenes, {timing['sync_points'].__len__()} sync points")
        
        return state
    
    def _get_platform_requirements(self, platform: AdPlatform) -> Dict[str, Any]:
        """Get platform-specific requirements"""
        requirements = {
            AdPlatform.TIKTOK: {
                "aspect_ratio": "9:16",
                "max_duration": 60,
                "optimal_hook_duration": 3,
                "text_overlay": "minimal",
                "pacing": "fast"
            },
            AdPlatform.INSTAGRAM: {
                "aspect_ratio": "1:1",
                "max_duration": 60,
                "optimal_hook_duration": 2,
                "text_overlay": "moderate",
                "pacing": "medium"
            },
            AdPlatform.FACEBOOK: {
                "aspect_ratio": "16:9",
                "max_duration": 240,
                "optimal_hook_duration": 3,
                "text_overlay": "heavy",
                "pacing": "medium"
            },
            AdPlatform.UNIVERSAL: {
                "aspect_ratio": "16:9",
                "max_duration": 60,
                "optimal_hook_duration": 3,
                "text_overlay": "moderate",
                "pacing": "medium"
            }
        }
        return requirements.get(platform, requirements[AdPlatform.UNIVERSAL])
    
    def _get_creative_constraints(self, request: AdCreationRequest) -> Dict[str, Any]:
        """Get creative constraints based on request"""
        return {
            "duration_constraint": {"min": 10, "max": request.duration, "optimal": request.duration},
            "style_requirements": self._get_style_requirements(request.style),
            "brand_constraints": {
                "colors": request.brand_colors or [],
                "include_logo": request.include_logo,
                "voice_preference": request.voice_preference
            }
        }
    
    def _get_style_requirements(self, style: AdStyle) -> Dict[str, Any]:
        """Get style-specific requirements"""
        styles = {
            AdStyle.PROFESSIONAL: {
                "color_scheme": "corporate",
                "pacing": "measured",
                "visual_complexity": "clean",
                "font_style": "sans-serif"
            },
            AdStyle.ENERGETIC: {
                "color_scheme": "vibrant",
                "pacing": "fast",
                "visual_complexity": "dynamic",
                "font_style": "bold"
            },
            AdStyle.MINIMAL: {
                "color_scheme": "monochrome",
                "pacing": "slow",
                "visual_complexity": "simple",
                "font_style": "thin"
            }
        }
        return styles.get(style, styles[AdStyle.PROFESSIONAL])


# Global master planner instance
master_planner = MasterPlannerAgent() 