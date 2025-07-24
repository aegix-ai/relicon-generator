"""
Relicon AI Ad Creator - Master Planner Agent (Refactored)
Revolutionary AI agent using modular planning components for ultra-detailed ad planning
"""
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from langchain_openai import ChatOpenAI
from langgraph import StateGraph, END

from core.models import AdCreationRequest, MasterAdPlan, AdScene, SceneComponent, PlanningContext
from core.settings import settings
from .planning import BrandAnalyzer, NarrativeDesigner


class MasterPlannerAgent:
    """
    Revolutionary Master Planner AI Agent (Refactored)
    
    Now uses modular planning components for better maintainability,
    testing, and extensibility. Each planning phase is handled by
    a specialized module with focused responsibility.
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=settings.OPENAI_API_KEY,
            temperature=0.3,
            max_tokens=4000
        )
        
        # Initialize modular planning components
        self.brand_analyzer = BrandAnalyzer()
        self.narrative_designer = NarrativeDesigner()
        
        # Create the planning state machine
        self.planning_graph = self._create_planning_graph()
        
    def _create_planning_graph(self) -> StateGraph:
        """Create the LangGraph state machine for modular planning"""
        
        # Define the planning state
        class PlanningState:
            request: AdCreationRequest
            context: PlanningContext
            strategic_analysis: Dict[str, Any]
            narrative_structure: Dict[str, Any]
            scene_breakdown: List[Dict[str, Any]]
            timing_analysis: Dict[str, Any]
            master_plan: Optional[MasterAdPlan]
            
        # Create the graph with modular nodes
        workflow = StateGraph(PlanningState)
        
        # Add modular planning nodes
        workflow.add_node("analyze_brand", self._analyze_brand_strategy)
        workflow.add_node("design_narrative", self._design_narrative_arc)
        workflow.add_node("architect_scenes", self._architect_scenes)
        workflow.add_node("calculate_timing", self._calculate_precise_timing)
        workflow.add_node("integrate_brand", self._integrate_brand_elements)
        workflow.add_node("finalize_plan", self._finalize_master_plan)
        
        # Define the refined workflow
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
        Create an ultra-detailed master plan using modular components
        
        Args:
            request: Ad creation request with brand information
            
        Returns:
            MasterAdPlan: Complete detailed plan for ad creation
        """
        print(f"🧠 Master Planner: Starting modular planning for {request.brand_name}")
        
        # Initialize planning context with platform and style analysis
        context = PlanningContext(
            request=request,
            brand_analysis={},
            platform_requirements=self.brand_analyzer.analyze_platform_requirements(request.platform),
            creative_constraints=self._get_creative_constraints(request)
        )
        
        # Execute the modular planning graph
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
        
        print(f"✅ Master Planner: Modular plan completed with {len(final_state['scene_breakdown'])} scenes")
        return final_state["master_plan"]
    
    async def _analyze_brand_strategy(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze brand strategy using the modular Brand Analyzer"""
        print("🔍 Phase 1: Brand Analysis (Modular)")
        
        request = state["request"]
        
        # Use the modular brand analyzer
        strategic_analysis = await self.brand_analyzer.analyze_brand_strategy(request)
        
        state["strategic_analysis"] = strategic_analysis
        state["context"].brand_analysis = strategic_analysis
        
        print("✅ Brand analysis complete")
        return state
    
    async def _design_narrative_arc(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Design narrative arc using the modular Narrative Designer"""
        print("📖 Phase 2: Narrative Design (Modular)")
        
        request = state["request"]
        strategic_analysis = state["strategic_analysis"]
        
        # Use the modular narrative designer
        narrative_structure = await self.narrative_designer.design_narrative_arc(
            request, strategic_analysis
        )
        
        state["narrative_structure"] = narrative_structure
        
        print("✅ Narrative design complete")
        return state
    
    async def _architect_scenes(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Architect individual scenes with ultra precision"""
        print("🏗️ Phase 3: Scene Architecture")
        
        request = state["request"]
        narrative = state["narrative_structure"]
        strategic = state["strategic_analysis"]
        
        # Calculate scene timing using narrative designer
        scene_timing = self.narrative_designer.calculate_scene_timing(
            narrative, float(request.duration)
        )
        
        # Create detailed scene breakdown
        scenes = await self._create_detailed_scenes(
            request, narrative, strategic, scene_timing
        )
        
        state["scene_breakdown"] = scenes
        
        print(f"✅ Scene architecture complete: {len(scenes)} scenes")
        return state
    
    async def _create_detailed_scenes(
        self, 
        request: AdCreationRequest, 
        narrative: Dict[str, Any], 
        strategic: Dict[str, Any],
        scene_timing: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create detailed scene breakdown using AI"""
        
        scenes_prompt = f"""
        Create ultra-detailed scene breakdown based on this analysis:
        
        Duration: {request.duration} seconds
        Narrative: {json.dumps(narrative.get('narrative_concept'), indent=2)}
        Scene Timing: {json.dumps(scene_timing['scene_timing'], indent=2)}
        Brand Strategy: {json.dumps(strategic.get('ad_strategy', {}), indent=2)}
        
        Create detailed scenes in JSON format:
        {{
            "scenes": [
                {{
                    "scene_id": "scene_1",
                    "scene_type": "hook",
                    "start_time": 0.0,
                    "duration": 4.0,
                    "scene_purpose": "Capture attention with intriguing opening",
                    "components": [
                        {{
                            "component_id": "hook_visual",
                            "start_time": 0.0,
                            "duration": 4.0,
                            "visual_type": "video",
                            "visual_prompt": "Ultra-detailed prompt for Luma AI",
                            "visual_style": "cinematic style",
                            "has_voiceover": true,
                            "voiceover_text": "Exact script text",
                            "voice_tone": "intriguing",
                            "luma_prompt": "Specific Luma AI prompt"
                        }}
                    ],
                    "main_script": "Complete voiceover script",
                    "camera_direction": "Detailed camera instructions",
                    "lighting_notes": "Lighting setup notes",
                    "color_palette": ["#FF6B35", "#F7931E"]
                }}
            ]
        }}
        
        Requirements:
        1. Exact timing that adds up to {request.duration} seconds
        2. Ultra-detailed prompts for each component
        3. 3-5 scenes with smooth transitions
        4. Mathematical precision in all timing
        """
        
        response = self.llm.invoke([{"role": "system", "content": scenes_prompt}])
        
        try:
            scene_data = json.loads(response.content)
            return scene_data["scenes"]
        except (json.JSONDecodeError, KeyError):
            print("⚠️ Using fallback scene creation")
            return self._create_fallback_scenes(request, narrative, scene_timing)
    
    def _create_fallback_scenes(
        self, 
        request: AdCreationRequest, 
        narrative: Dict[str, Any],
        scene_timing: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create fallback scenes if AI generation fails"""
        
        timing = scene_timing["scene_timing"]
        
        return [
            {
                "scene_id": "scene_1",
                "scene_type": "hook",
                "start_time": timing["hook"]["start_time"],
                "duration": timing["hook"]["duration"],
                "scene_purpose": "Capture attention immediately",
                "components": [
                    {
                        "component_id": "hook_visual",
                        "start_time": timing["hook"]["start_time"],
                        "duration": timing["hook"]["duration"],
                        "visual_type": "video",
                        "visual_prompt": f"Attention-grabbing opening for {request.brand_name}",
                        "visual_style": f"{request.style} style",
                        "has_voiceover": True,
                        "voiceover_text": f"Discover how {request.brand_name} changes everything",
                        "voice_tone": "intriguing",
                        "luma_prompt": f"Dynamic opening scene for {request.brand_name}"
                    }
                ],
                "main_script": f"Discover how {request.brand_name} changes everything",
                "camera_direction": "Dynamic opening shot",
                "lighting_notes": "Bright, engaging lighting",
                "color_palette": ["#2C3E50", "#E74C3C"]
            }
        ]
    
    async def _calculate_precise_timing(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate and validate precise timing"""
        print("⏱️ Phase 4: Timing Calculation")
        
        scenes = state["scene_breakdown"]
        request = state["request"]
        
        # Validate timing precision
        total_calculated = sum(scene["duration"] for scene in scenes)
        timing_error = abs(total_calculated - request.duration)
        
        timing_analysis = {
            "total_duration": request.duration,
            "total_calculated": total_calculated,
            "timing_error": timing_error,
            "timing_valid": timing_error < 0.5,
            "scene_count": len(scenes),
            "precision_level": "high" if timing_error < 0.1 else "medium"
        }
        
        state["timing_analysis"] = timing_analysis
        
        print(f"✅ Timing validation: {timing_analysis['precision_level']} precision")
        return state
    
    async def _integrate_brand_elements(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate brand elements throughout the ad"""
        print("🎨 Phase 5: Brand Integration")
        
        request = state["request"]
        scenes = state["scene_breakdown"]
        
        # Calculate brand presence and logo timing
        brand_integration = {
            "brand_presence_timeline": {},
            "logo_appearances": [],
            "brand_color_usage": request.brand_colors or ["#2C3E50", "#E74C3C"],
            "brand_consistency_score": 0.95
        }
        
        # Add logo appearances
        for i, scene in enumerate(scenes):
            scene_id = scene["scene_id"]
            if i == 0 or scene["scene_type"] == "cta":  # First scene and CTA
                brand_integration["logo_appearances"].append({
                    "time": scene["start_time"] + (scene["duration"] * 0.8),
                    "duration": 1.5,
                    "size": "medium" if i == 0 else "large",
                    "position": "bottom_right" if i == 0 else "center"
                })
        
        state["brand_integration"] = brand_integration
        
        print("✅ Brand integration complete")
        return state
    
    async def _finalize_master_plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Create the final master plan using modular components"""
        print("🎯 Phase 6: Plan Finalization")
        
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
        
        # Create the comprehensive master plan
        master_plan = MasterAdPlan(
            plan_id=f"plan_{int(time.time())}_{request.brand_name.lower().replace(' ', '_')}",
            total_duration=float(request.duration),
            ad_concept=strategic.get("ad_strategy", {}).get("primary_objective", "Brand awareness"),
            narrative_arc=narrative.get("narrative_concept", "Compelling brand story"),
            emotional_journey=[ej.get("emotion", "positive") for ej in narrative.get("emotional_journey", [])],
            key_messages=strategic.get("ad_strategy", {}).get("messaging_hierarchy", ["Key message"]),
            scenes=scenes,
            scene_transitions=[],
            overall_voice_direction=strategic.get("creative_direction", {}).get("tone_of_voice", "Professional"),
            music_strategy=strategic.get("creative_direction", {}).get("music_mood", "Uplifting"),
            brand_presence_timeline=brand_integration["brand_presence_timeline"],
            logo_appearances=brand_integration["logo_appearances"],
            target_platforms=[request.platform]
        )
        
        state["master_plan"] = master_plan
        
        print(f"✅ Master Plan finalized: {len(scenes)} scenes, modular architecture")
        return state
    
    def _get_creative_constraints(self, request: AdCreationRequest) -> Dict[str, Any]:
        """Get creative constraints using modular style analyzer"""
        style_requirements = self.brand_analyzer.get_style_requirements(request.style)
        
        return {
            "duration_constraint": {
                "min": 10, 
                "max": request.duration, 
                "optimal": request.duration
            },
            "style_requirements": style_requirements,
            "brand_constraints": {
                "colors": request.brand_colors or [],
                "include_logo": request.include_logo,
                "voice_preference": request.voice_preference
            }
        }


# Global master planner instance (now using modular architecture)
master_planner = MasterPlannerAgent() 