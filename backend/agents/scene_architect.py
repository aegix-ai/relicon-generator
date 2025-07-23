"""
Relicon AI Ad Creator - Scene Architect Agent
Ultra-precise scene-by-scene breakdown and component architecture
"""
import json
import math
from typing import Dict, Any, List, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage
from core.models import AdScene, SceneComponent, MasterAdPlan
from core.settings import settings


class SceneArchitectAgent:
    """
    Scene Architect Agent - The precision engineer of ad scenes
    
    Takes the master plan and architects each scene with atomic precision,
    ensuring every component is perfectly timed and orchestrated.
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=settings.OPENAI_API_KEY,
            temperature=0.1,  # Very low temperature for precision
            max_tokens=3000
        )
        
        # Scene component templates
        self.component_templates = {
            "hook": self._get_hook_templates(),
            "problem": self._get_problem_templates(),
            "solution": self._get_solution_templates(),
            "benefits": self._get_benefits_templates(),
            "cta": self._get_cta_templates()
        }
    
    async def architect_scene(self, scene: AdScene, context: Dict[str, Any]) -> AdScene:
        """
        Architect a single scene with atomic precision
        
        Args:
            scene: The scene to architect
            context: Brand and planning context
            
        Returns:
            Fully architected scene with detailed components
        """
        print(f"🏗️ Scene Architect: Architecting {scene.scene_id} ({scene.scene_type})")
        
        # Get scene template
        template = self.component_templates.get(scene.scene_type, {})
        
        # Calculate component breakdown
        components = await self._calculate_scene_components(scene, context, template)
        
        # Optimize timing
        optimized_components = self._optimize_component_timing(components, scene.duration)
        
        # Generate ultra-detailed prompts
        detailed_components = await self._generate_detailed_prompts(
            optimized_components, scene, context
        )
        
        # Create final scene
        architected_scene = AdScene(
            scene_id=scene.scene_id,
            scene_type=scene.scene_type,
            scene_purpose=scene.scene_purpose,
            start_time=scene.start_time,
            duration=scene.duration,
            components=detailed_components,
            main_script=await self._generate_scene_script(detailed_components, context),
            camera_direction=await self._generate_camera_direction(scene, context),
            lighting_notes=await self._generate_lighting_notes(scene, context),
            color_palette=self._calculate_color_palette(scene, context)
        )
        
        print(f"✅ Scene Architect: {scene.scene_id} architected with {len(detailed_components)} components")
        return architected_scene
    
    async def _calculate_scene_components(
        self, 
        scene: AdScene, 
        context: Dict[str, Any], 
        template: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Calculate the optimal component breakdown for a scene"""
        
        components_prompt = f"""
        You are a master scene architect. Break down this scene into atomic components:
        
        Scene: {scene.scene_id}
        Type: {scene.scene_type}
        Duration: {scene.duration} seconds
        Purpose: {scene.scene_purpose}
        Brand: {context.get('brand_name', 'Unknown')}
        
        Template Guidelines: {json.dumps(template, indent=2)}
        
        Create component breakdown in JSON format:
        {{
            "components": [
                {{
                    "component_id": "unique_id",
                    "start_time": 0.0,
                    "duration": 2.5,
                    "component_type": "primary_visual|secondary_visual|text_overlay|transition",
                    "priority": "high|medium|low",
                    "visual_requirements": {{
                        "shot_type": "close_up|medium|wide|extreme_close",
                        "camera_movement": "static|slow_pan|quick_cut|zoom",
                        "visual_complexity": "simple|moderate|complex",
                        "foreground_elements": ["element1", "element2"],
                        "background_elements": ["element1", "element2"]
                    }},
                    "audio_requirements": {{
                        "has_voiceover": true,
                        "voiceover_priority": "primary|secondary",
                        "music_layer": "background|foreground|none",
                        "sfx_needed": ["sfx1", "sfx2"]
                    }},
                    "timing_constraints": {{
                        "can_overlap": true,
                        "must_sync_with": "audio|beat|previous_component",
                        "transition_buffer": 0.2
                    }}
                }}
            ]
        }}
        
        Requirements:
        1. All components must fit within {scene.duration} seconds
        2. Components can overlap but must be precisely timed
        3. Primary components need detailed visual requirements
        4. Consider pacing and energy flow
        5. Ensure smooth transitions between components
        """
        
        response = self.llm.invoke([SystemMessage(content=components_prompt)])
        components_data = json.loads(response.content)
        
        return components_data["components"]
    
    def _optimize_component_timing(
        self, 
        components: List[Dict[str, Any]], 
        total_duration: float
    ) -> List[Dict[str, Any]]:
        """Optimize component timing with mathematical precision"""
        
        # Sort components by start time
        components.sort(key=lambda x: x["start_time"])
        
        # Calculate optimal timing based on priority and duration
        optimized = []
        current_time = 0.0
        
        for i, component in enumerate(components):
            # Adjust start time if needed
            if component["start_time"] < current_time:
                component["start_time"] = current_time
            
            # Ensure component doesn't exceed scene duration
            max_end_time = total_duration
            if i < len(components) - 1:
                next_component = components[i + 1]
                buffer = component["timing_constraints"].get("transition_buffer", 0.1)
                max_end_time = min(max_end_time, next_component["start_time"] - buffer)
            
            # Adjust duration if necessary
            max_duration = max_end_time - component["start_time"]
            if component["duration"] > max_duration:
                component["duration"] = max(max_duration, 0.5)  # Minimum 0.5 seconds
            
            # Calculate precise end time
            component["end_time"] = component["start_time"] + component["duration"]
            
            # Update current time for overlap calculations
            if not component["timing_constraints"].get("can_overlap", False):
                current_time = component["end_time"]
            
            optimized.append(component)
        
        return optimized
    
    async def _generate_detailed_prompts(
        self, 
        components: List[Dict[str, Any]], 
        scene: AdScene, 
        context: Dict[str, Any]
    ) -> List[SceneComponent]:
        """Generate ultra-detailed prompts for each component"""
        
        detailed_components = []
        
        for comp_data in components:
            # Generate Luma AI prompt
            luma_prompt = await self._generate_luma_prompt(comp_data, scene, context)
            
            # Generate voiceover text
            voiceover_text = None
            if comp_data["audio_requirements"]["has_voiceover"]:
                voiceover_text = await self._generate_voiceover_text(comp_data, context)
            
            # Create scene component
            component = SceneComponent(
                start_time=comp_data["start_time"],
                duration=comp_data["duration"],
                end_time=comp_data["end_time"],
                visual_type=self._map_component_type_to_visual_type(comp_data["component_type"]),
                visual_prompt=self._create_visual_prompt(comp_data, context),
                visual_style=self._determine_visual_style(comp_data, context),
                has_voiceover=comp_data["audio_requirements"]["has_voiceover"],
                voiceover_text=voiceover_text,
                voice_tone=self._determine_voice_tone(comp_data, context),
                has_music=comp_data["audio_requirements"]["music_layer"] != "none",
                music_style=self._determine_music_style(comp_data, context),
                entry_effect=self._calculate_entry_effect(comp_data),
                exit_effect=self._calculate_exit_effect(comp_data),
                luma_prompt=luma_prompt
            )
            
            detailed_components.append(component)
        
        return detailed_components
    
    async def _generate_luma_prompt(
        self, 
        component: Dict[str, Any], 
        scene: AdScene, 
        context: Dict[str, Any]
    ) -> str:
        """Generate ultra-specific Luma AI prompt"""
        
        visual_req = component["visual_requirements"]
        brand_name = context.get("brand_name", "product")
        
        luma_prompt = f"""
        You are creating a Luma AI prompt for a {scene.scene_type} scene. Generate an ultra-specific prompt:
        
        Component Type: {component["component_type"]}
        Duration: {component["duration"]} seconds
        Shot Type: {visual_req["shot_type"]}
        Camera Movement: {visual_req["camera_movement"]}
        Brand Context: {brand_name}
        Scene Purpose: {scene.scene_purpose}
        
        Visual Requirements:
        - Foreground: {visual_req["foreground_elements"]}
        - Background: {visual_req["background_elements"]}
        - Complexity: {visual_req["visual_complexity"]}
        
        Create a Luma AI prompt that is:
        1. Extremely specific and actionable
        2. Includes camera angles, lighting, and movement
        3. Specifies colors, textures, and mood
        4. Optimized for {component["duration"]} seconds
        5. Professional advertising quality
        
        Format: Single paragraph, maximum 200 characters for Luma AI efficiency.
        """
        
        response = self.llm.invoke([SystemMessage(content=luma_prompt)])
        return response.content.strip()
    
    async def _generate_voiceover_text(
        self, 
        component: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> str:
        """Generate natural voiceover text"""
        
        voiceover_prompt = f"""
        Generate natural voiceover text for this component:
        
        Duration: {component["duration"]} seconds
        Component Type: {component["component_type"]}
        Priority: {component["priority"]}
        Brand: {context.get("brand_name")}
        Brand Description: {context.get("brand_description", "")}
        
        Requirements:
        1. Must be speakable in exactly {component["duration"]} seconds
        2. Natural, conversational tone
        3. Engaging and compelling
        4. Fits the component's purpose
        5. No more than {int(component["duration"] * 2.5)} words
        
        Generate ONLY the voiceover text, nothing else.
        """
        
        response = self.llm.invoke([SystemMessage(content=voiceover_prompt)])
        return response.content.strip()
    
    async def _generate_scene_script(
        self, 
        components: List[SceneComponent], 
        context: Dict[str, Any]
    ) -> str:
        """Generate the complete scene script"""
        
        voiceover_parts = []
        for component in components:
            if component.has_voiceover and component.voiceover_text:
                voiceover_parts.append(component.voiceover_text)
        
        return " ".join(voiceover_parts)
    
    async def _generate_camera_direction(
        self, 
        scene: AdScene, 
        context: Dict[str, Any]
    ) -> str:
        """Generate detailed camera direction"""
        
        direction_prompt = f"""
        Generate professional camera direction for this scene:
        
        Scene Type: {scene.scene_type}
        Duration: {scene.duration} seconds
        Purpose: {scene.scene_purpose}
        
        Include:
        1. Primary camera angle and position
        2. Camera movement throughout the scene
        3. Lens choice and depth of field
        4. Any special techniques
        
        Keep it concise but specific for a cinematographer.
        """
        
        response = self.llm.invoke([SystemMessage(content=direction_prompt)])
        return response.content.strip()
    
    async def _generate_lighting_notes(
        self, 
        scene: AdScene, 
        context: Dict[str, Any]
    ) -> str:
        """Generate lighting direction"""
        
        lighting_prompt = f"""
        Generate lighting notes for this {scene.scene_type} scene:
        
        Scene Purpose: {scene.scene_purpose}
        Duration: {scene.duration} seconds
        Brand Style: {context.get("style", "professional")}
        
        Include:
        1. Key light position and intensity
        2. Fill light requirements
        3. Background lighting
        4. Mood and atmosphere
        
        Keep concise but specific.
        """
        
        response = self.llm.invoke([SystemMessage(content=lighting_prompt)])
        return response.content.strip()
    
    def _calculate_color_palette(self, scene: AdScene, context: Dict[str, Any]) -> List[str]:
        """Calculate optimal color palette for the scene"""
        
        # Base colors from brand
        brand_colors = context.get("brand_colors", ["#2C3E50", "#E74C3C", "#F39C12"])
        
        # Scene-specific color adjustments
        scene_color_map = {
            "hook": {"intensity": "high", "contrast": "high"},
            "problem": {"intensity": "medium", "contrast": "medium"},
            "solution": {"intensity": "high", "contrast": "high"},
            "benefits": {"intensity": "medium", "contrast": "low"},
            "cta": {"intensity": "very_high", "contrast": "very_high"}
        }
        
        scene_style = scene_color_map.get(scene.scene_type, {"intensity": "medium", "contrast": "medium"})
        
        # Return optimized palette
        return brand_colors[:3]  # Use top 3 brand colors
    
    def _map_component_type_to_visual_type(self, component_type: str) -> str:
        """Map component type to visual type"""
        mapping = {
            "primary_visual": "video",
            "secondary_visual": "image",
            "text_overlay": "text",
            "transition": "transition"
        }
        return mapping.get(component_type, "video")
    
    def _create_visual_prompt(self, component: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Create detailed visual prompt"""
        visual_req = component["visual_requirements"]
        return f"{visual_req['shot_type']} shot with {visual_req['camera_movement']} movement, featuring {', '.join(visual_req['foreground_elements'])}"
    
    def _determine_visual_style(self, component: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Determine visual style for component"""
        style_map = {
            "professional": "clean, corporate, polished",
            "energetic": "vibrant, dynamic, bold",
            "minimal": "simple, elegant, understated",
            "cinematic": "dramatic, film-like, artistic"
        }
        return style_map.get(context.get("style", "professional"), "professional")
    
    def _determine_voice_tone(self, component: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Determine voice tone for component"""
        if component["priority"] == "high":
            return "confident"
        elif component["component_type"] == "primary_visual":
            return "engaging"
        else:
            return "supportive"
    
    def _determine_music_style(self, component: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Determine music style for component"""
        music_layer = component["audio_requirements"]["music_layer"]
        if music_layer == "foreground":
            return "energetic"
        elif music_layer == "background":
            return "ambient"
        else:
            return "none"
    
    def _calculate_entry_effect(self, component: Dict[str, Any]) -> str:
        """Calculate optimal entry effect"""
        if component["priority"] == "high":
            return "fade_in"
        else:
            return "cut"
    
    def _calculate_exit_effect(self, component: Dict[str, Any]) -> str:
        """Calculate optimal exit effect"""
        if component["timing_constraints"].get("can_overlap", False):
            return "fade_out"
        else:
            return "cut"
    
    # Template methods for different scene types
    def _get_hook_templates(self) -> Dict[str, Any]:
        """Get templates for hook scenes"""
        return {
            "optimal_duration": 3.0,
            "component_count": 2,
            "primary_focus": "attention_grab",
            "pacing": "fast",
            "visual_complexity": "high"
        }
    
    def _get_problem_templates(self) -> Dict[str, Any]:
        """Get templates for problem scenes"""
        return {
            "optimal_duration": 4.0,
            "component_count": 3,
            "primary_focus": "problem_identification",
            "pacing": "medium",
            "visual_complexity": "medium"
        }
    
    def _get_solution_templates(self) -> Dict[str, Any]:
        """Get templates for solution scenes"""
        return {
            "optimal_duration": 5.0,
            "component_count": 3,
            "primary_focus": "solution_presentation",
            "pacing": "medium",
            "visual_complexity": "high"
        }
    
    def _get_benefits_templates(self) -> Dict[str, Any]:
        """Get templates for benefits scenes"""
        return {
            "optimal_duration": 4.0,
            "component_count": 2,
            "primary_focus": "benefit_showcase",
            "pacing": "medium",
            "visual_complexity": "medium"
        }
    
    def _get_cta_templates(self) -> Dict[str, Any]:
        """Get templates for CTA scenes"""
        return {
            "optimal_duration": 3.0,
            "component_count": 2,
            "primary_focus": "action_motivation",
            "pacing": "fast",
            "visual_complexity": "low"
        }


# Global scene architect instance
scene_architect = SceneArchitectAgent() 