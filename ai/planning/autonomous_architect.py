"""
Autonomous AI Video Architect
Revolutionary AI system that designs everything from first principles
Full control over timing, visuals, audio, colors, voice, and every creative decision
"""
import os
import json
from typing import Dict, List, Any
from openai import OpenAI

class AutonomousVideoArchitect:
    def __init__(self):
        # Initialize OpenAI client with compatibility fix
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        try:
            # Use basic client initialization to avoid httpx conflicts
            # Compatibility fix for Python 3.13 and GitHub Actions
            self.client = OpenAI(api_key=api_key)
        except TypeError as e:
            if "proxies" in str(e):
                # Fallback for version conflicts
                try:
                    self.client = OpenAI(api_key=api_key)
                except:
                    import openai
                    openai.api_key = api_key
                    self.client = openai
            else:
                raise e
        except Exception as e:
            print(f"⚠️ OpenAI client initialization failed: {e}")
            raise ValueError(f"Failed to initialize OpenAI client: {e}")
    
    def architect_complete_video(self, brand_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Revolutionary autonomous architecting - AI decides EVERYTHING
        From macro strategy to micro execution details
        """
        target_duration = brand_info.get('duration', 15)
        
        architecture_prompt = f"""
        You are the world's most advanced autonomous video architect. You have FULL CREATIVE CONTROL.
        Design a revolutionary {target_duration}-second video advertisement from first principles.
        
        Brand Context:
        - Brand: {brand_info.get('brand_name', 'Unknown')}
        - Description: {brand_info.get('brand_description', 'A brand')}
        - Target: {brand_info.get('target_audience', 'general')}
        - CTA: {brand_info.get('call_to_action', 'Learn more')}
        
        YOUR AUTONOMOUS DECISIONS TO MAKE:
        1. Video Structure & Timing (flexible, dynamic)
        2. Visual Style & Cinematography 
        3. Color Palette & Mood
        4. Audio Strategy (voice gender, tone, energy)
        5. Scene Concepts & Transitions
        6. Pacing & Rhythm
        
        CREATIVE CONSTRAINTS:
        - Total duration: EXACTLY {target_duration} seconds
        - ULTRA-REALISTIC advertisement quality (NO sci-fi, fantasy, abstract)
        - Real people in professional commercial settings
        - Cost-optimized (2-3 substantial scenes minimum 5s each)
        - Advertisement-ready, commercial-grade content
        - Each scene must be visually distinct and realistic
        
        AUTONOMOUS ARCHITECTURE REQUIREMENTS:
        Design flexible timing (not rigid timestamps) - let creativity flow
        Choose optimal scene count and durations based on narrative needs
        Select voice characteristics that match the brand personality
        Design each scene with unique REALISTIC visual identity
        CRITICAL: All scenes must be ultra-realistic advertisement content
        - Real people in professional environments
        - Commercial product demonstrations
        - Modern office, home, or studio settings
        - NO fantasy, sci-fi, or abstract visuals
        
        Respond in JSON format with your autonomous creative decisions:
        {{
            "creative_vision": {{
                "overall_concept": "Your unique creative concept",
                "visual_style": "Your chosen aesthetic (cinematic/modern/artistic/etc)",
                "color_palette": ["color1", "color2", "color3"],
                "mood": "Your selected emotional tone",
                "cinematography": "Your camera/visual approach"
            }},
            "audio_architecture": {{
                "voice_gender": "male/female/neutral",
                "voice_tone": "energetic/calm/authoritative/friendly/etc",
                "script_style": "conversational/dramatic/technical/story/etc",
                "energy_level": "high/medium/low"
            }},
            "scene_architecture": {{
                "scene_count": 2-3,
                "total_duration": {target_duration},
                "scenes": [
                    {{
                        "scene_id": 1,
                        "duration": "your_choice_5_to_10_seconds",
                        "purpose": "your_creative_purpose",
                        "visual_concept": "your_unique_visual_idea",
                        "colors": ["scene_specific_colors"],
                        "mood": "scene_specific_mood",
                        "camera_style": "your_camera_choice",
                        "lighting": "your_lighting_choice"
                    }}
                ]
            }},
            "unified_script": "Write ONE continuous energetic script for the entire {target_duration} seconds",
            "cost_optimization": {{
                "estimated_scenes": "scene_count",
                "rationale": "why this structure is cost-effective"
            }}
        }}
        
        BE REVOLUTIONARY. BE CREATIVE. MAKE AUTONOMOUS DECISIONS.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": architecture_prompt}],
            response_format={"type": "json_object"},
            temperature=0.8  # High creativity
        )
        
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty architecture response")
        
        architecture = json.loads(content)
        
        # Validate timing
        scenes = architecture.get('scene_architecture', {}).get('scenes', [])
        total_duration = sum(scene.get('duration', 0) for scene in scenes)
        
        if total_duration != target_duration:
            # AI should self-correct
            architecture = self._auto_correct_timing(architecture, target_duration)
        
        return architecture
    
    def _auto_correct_timing(self, architecture: Dict[str, Any], target_duration: int) -> Dict[str, Any]:
        """Auto-correct timing if AI made calculation errors"""
        scenes = architecture.get('scene_architecture', {}).get('scenes', [])
        scene_count = len(scenes)
        
        if scene_count == 2:
            scenes[0]['duration'] = 7
            scenes[1]['duration'] = 8
        elif scene_count == 3:
            scenes[0]['duration'] = 5
            scenes[1]['duration'] = 5
            scenes[2]['duration'] = 5
        
        architecture['scene_architecture']['scenes'] = scenes
        architecture['scene_architecture']['total_duration'] = target_duration
        
        return architecture
    
    def create_cutting_edge_prompts(self, architecture: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate cutting-edge Luma prompts for each scene
        Revolutionary quality with 2025 standards
        """
        scenes = architecture.get('scene_architecture', {}).get('scenes', [])
        creative_vision = architecture.get('creative_vision', {})
        
        enhanced_scenes = []
        
        for i, scene in enumerate(scenes):
            prompt_creation = f"""
            Create an ULTRA-REALISTIC ADVERTISEMENT video prompt for this scene.
            This MUST be commercial-grade, professional advertisement content - NOT sci-fi, fantasy, or abstract.
            
            Scene Context:
            - Scene {i+1} of {len(scenes)}
            - Duration: {scene.get('duration')}s
            - Purpose: {scene.get('purpose')}
            - Visual Concept: {scene.get('visual_concept')}
            
            Overall Video Vision:
            - Style: {creative_vision.get('visual_style')}
            - Colors: {creative_vision.get('color_palette')}
            - Mood: {creative_vision.get('mood')}
            - Cinematography: {creative_vision.get('cinematography')}
            
            Scene Specifics:
            - Colors: {scene.get('colors')}
            - Mood: {scene.get('mood')}
            - Camera: {scene.get('camera_style')}
            - Lighting: {scene.get('lighting')}
            
            CRITICAL REQUIREMENTS FOR ULTRA-REALISTIC ADVERTISEMENT:
            - Real people in professional settings (office, home, studio, etc.)
            - Commercial product demonstration or lifestyle scenes
            - Professional lighting and cinematography like TV commercials
            - Modern, contemporary environments and clothing
            - Clear focus on realistic human interactions
            - Advertisement-ready visual quality
            - NO sci-fi, fantasy, abstract, or unrealistic elements
            - Professional makeup, styling, and wardrobe
            - High-end commercial production value
            
            Example good advertisement scenes:
            - Professional businesswoman using laptop in modern office with natural lighting
            - Family enjoying product at dining table with warm home lighting
            - Young professional demonstrating app on smartphone in contemporary workspace
            - Product close-up with professional studio lighting and clean background
            
            Write a detailed, specific prompt for ULTRA-REALISTIC commercial advertisement content.
            Focus on real people, real environments, and professional commercial aesthetics.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt_creation}],
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            cutting_edge_prompt = content.strip() if content else "Default prompt"
            
            enhanced_scene = {
                **scene,
                'luma_prompt': cutting_edge_prompt,
                'scene_number': i + 1
            }
            enhanced_scenes.append(enhanced_scene)
        
        return enhanced_scenes
