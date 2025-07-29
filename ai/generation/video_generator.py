"""
    Complete AI Video Generation System
    Orchestrates the entire video creation pipeline from planning to final output
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Dict, Any

# Add the relicon directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from ai.planning.autonomous_architect import AutonomousVideoArchitect
from services.luma.video_service import LumaVideoService
from services.audio.tts_service import TTSService
from services.video.assembly_service import VideoAssemblyService

class CompleteVideoGenerator:
    def __init__(self):
        self.architect = AutonomousVideoArchitect()
        self.luma_service = LumaVideoService()
        self.tts_service = TTSService()
        self.assembly_service = VideoAssemblyService()
        self.temp_dir = "/tmp/relicon_generation"
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def progress_update(self, progress: int, message: str):
        """Send progress update"""
        print(f"PROGRESS:{progress}:{message}")
        sys.stdout.flush()
    
    def generate_complete_video(self, brand_info: Dict[str, Any], output_path: str) -> Dict[str, Any]:
        """
        Generate complete video from brand information
        Returns generation results with video path and metadata
        """
        try:
            self.progress_update(5, "🚀 Autonomous AI architect designing revolutionary video...")
            
            # Step 1: Autonomous architectural design
            architecture = self.architect.architect_complete_video(brand_info)
            
            print("🎨 AUTONOMOUS ARCHITECTURE CREATED:")
            print(f"  Vision: {architecture.get('creative_vision', {}).get('overall_concept', 'N/A')[:50]}...")
            print(f"  Style: {architecture.get('creative_vision', {}).get('visual_style', 'N/A')}")
            print(f"  Voice: {architecture.get('audio_architecture', {}).get('voice_gender', 'N/A')} {architecture.get('audio_architecture', {}).get('voice_tone', 'N/A')}")
            print(f"  Scenes: {architecture.get('scene_architecture', {}).get('scene_count', 'N/A')}")
            
            # Step 2: Generate cutting-edge scene prompts
            enhanced_scenes = self.architect.create_cutting_edge_prompts(architecture)
            
            self.progress_update(15, f"🎬 AI architected {len(enhanced_scenes)} revolutionary scenes")
            
            # Step 3: Generate autonomous audio with AI-selected voice
            target_duration = brand_info.get('duration', 15)
            self.progress_update(20, f"🎙️ Creating AI-designed {target_duration}s voiceover")
            
            unified_audio_file = os.path.join(self.temp_dir, "unified_audio.mp3")
            unified_script = architecture.get('unified_script', 'Default script')
            audio_config = architecture.get('audio_architecture', {})
            
            if not self.tts_service.generate_autonomous_audio(unified_script, unified_audio_file, audio_config):
                raise Exception("Failed to generate autonomous audio")
            
            # Step 4: Generate cutting-edge videos
            video_files = []
            
            for i, scene in enumerate(enhanced_scenes):
                scene_num = scene.get('scene_number', i + 1)
                progress = 30 + (i * 35 // len(enhanced_scenes))
                
                self.progress_update(progress, f"🎬 Creating revolutionary scene {int(scene_num)}: {scene.get('purpose', 'scene')}")
                
                # Use ultra-realistic advertisement prompt with uniqueness
                ultra_realistic_prompt = scene.get('luma_prompt', 'Default prompt')
                video_url = self.luma_service.generate_video(ultra_realistic_prompt, aspect_ratio="9:16", force_unique=True)
                
                # Download video
                video_file = os.path.join(self.temp_dir, f"scene_{int(scene_num)}.mp4")
                if not self.luma_service.download_video(video_url, video_file):
                    raise Exception(f"Failed to download video for scene {int(scene_num)}")
                
                video_files.append(video_file)
                print(f"✅ Revolutionary scene {int(scene_num)} completed")
            
            self.progress_update(70, "🎬 Assembling revolutionary video with AI-designed audio")
            
            # Step 5: Assemble final video with unified audio
            temp_final = os.path.join(self.temp_dir, "assembled_video.mp4")
            target_duration = architecture.get('scene_architecture', {}).get('total_duration', 15)
            
            if not self.assembly_service.combine_videos_with_unified_audio(video_files, unified_audio_file, temp_final, target_duration):
                raise Exception("Failed to assemble final video with unified audio")
            
            self.progress_update(85, "✨ Finalizing revolutionary video")
            
            # Step 6: Optimize for cutting-edge quality
            if not self.assembly_service.optimize_for_social_media(temp_final, output_path, "9:16"):
                # If optimization fails, use the assembled video
                import shutil
                shutil.copy2(temp_final, output_path)
            
            self.progress_update(95, "🎯 AI video architecture complete")
            
            # Step 7: Final output
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            scene_count = len(enhanced_scenes)
            
            self.progress_update(100, f"🚀 Revolutionary AI video created!")
            
            return {
                'success': True,
                'output_path': output_path,
                'file_size_mb': round(file_size, 1),
                'duration': target_duration,
                'scenes': scene_count,
                'architecture': architecture,
                'cost_breakdown': self.estimate_autonomous_cost(scene_count),
                'aspect_ratio': '9:16'
            }
            
        except Exception as e:
            error_msg = f"Video generation failed: {str(e)}"
            self.progress_update(0, f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg
            }
    
    def create_luma_prompt(self, scene: Dict[str, Any], master_plan: Dict[str, Any]) -> str:
        """
        Create sophisticated Luma prompt from scene details
        """
        visual_desc = scene.get('visual_description', '')
        camera_movement = scene.get('camera_movement', 'dynamic')
        lighting = scene.get('lighting', 'cinematic')
        mood = scene.get('mood', 'professional')
        visual_style = master_plan.get('visual_style', 'modern')
        
        # Create comprehensive prompt
        prompt = f"{visual_desc}"
        
        # Add visual style
        if visual_style:
            prompt += f", {visual_style} aesthetic"
        
        # Add camera movement
        if camera_movement != 'static':
            prompt += f", {camera_movement} camera movement"
        
        # Add lighting and mood
        prompt += f", {lighting} lighting, {mood} atmosphere"
        
        # Add technical specs for quality
        prompt += ", cinematic quality, professional video production, 9:16 vertical format"
        
        return prompt.strip()
    
    def estimate_autonomous_cost(self, scene_count: int) -> Dict[str, float]:
        """
        Estimate generation costs for autonomous system
        """
        # Cost estimates (optimized)
        ai_architecture_cost = 0.02  # Enhanced AI planning
        unified_tts_cost = 0.015  # Single TTS generation
        luma_video_cost = scene_count * 1.50  # Luma per video
        
        total_cost = ai_architecture_cost + unified_tts_cost + luma_video_cost
        
        return {
            'ai_architecture': ai_architecture_cost,
            'unified_audio': unified_tts_cost,
            'video_generation': luma_video_cost,
            'total_estimated': round(total_cost, 2)
        }

def main():
    """
    Command line interface for video generation
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate AI video advertisement')
    parser.add_argument('--brand-name', required=True, help='Brand name')
    parser.add_argument('--brand-description', required=True, help='Brand description')
    parser.add_argument('--target-audience', default='general audience', help='Target audience')
    parser.add_argument('--tone', default='professional', help='Tone of voice')
    parser.add_argument('--duration', type=int, default=15, help='Duration in seconds')
    parser.add_argument('--call-to-action', default='Learn more', help='Call to action')
    parser.add_argument('--output', required=True, help='Output video file path')
    
    args = parser.parse_args()
    
    brand_info = {
        'brand_name': args.brand_name,
        'brand_description': args.brand_description,
        'target_audience': args.target_audience,
        'tone': args.tone,
        'duration': args.duration,
        'call_to_action': args.call_to_action
    }
    
    generator = CompleteVideoGenerator()
    result = generator.generate_complete_video(brand_info, args.output)
    
    if result['success']:
        print(f"\n🎉 Video generation completed successfully!")
        print(f"📁 Output: {result['output_path']}")
        print(f"📊 File size: {result['file_size_mb']}MB")
        print(f"⏱️ Duration: {result['duration']}s")
        print(f"🎬 Scenes: {result['scenes']}")
        print(f"💰 Estimated cost: ${result['cost_breakdown']['total_estimated']}")
    else:
        print(f"\n❌ Generation failed: {result['error']}")
        sys.exit(1)

if __name__ == "__main__":
    main()
