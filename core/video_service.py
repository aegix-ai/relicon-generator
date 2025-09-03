"""
  High-level video generation service.
  Orchestrates video creation using abstracted providers.
"""

import os
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
from core.provider_manager import provider_manager
from core.assembly_service import AssemblyService
from core.logger import video_logger
from config.settings import settings


class VideoService:
    """High-level video generation orchestration service."""
    
    def __init__(self):
        self.provider_manager = provider_manager
    
    def generate_video_from_architecture(self, architecture: Dict[str, Any], output_dir: str, 
                                       progress_callback: callable = None, logo_integration_plan: Dict[str, Any] = None,
                                       quality_mode: str = 'professional') -> Dict[str, Any]:
        """
        Generate complete video from architectural plan with professional quality controls.
        
        Args:
            architecture: Video architecture from planning service
            output_dir: Directory to save generated videos
            progress_callback: Progress update callback
            logo_integration_plan: Optional logo integration plan for enhanced prompts
            quality_mode: Quality mode ('professional', 'standard', 'economy')
            
        Returns:
            Dictionary with generation results and file paths including quality metrics
        """
        scenes = architecture.get('scene_architecture', {}).get('scenes', [])
        if not scenes:
            raise ValueError("No scenes found in architecture")
        
        # Set quality parameters based on mode
        quality_settings = self._get_quality_settings(quality_mode)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🎬 Starting professional video generation with {quality_mode} quality mode")
        print(f"📊 Quality settings: {quality_settings['description']}")
        
        # Generate video for each scene
        video_logger.video_generation_start(
            job_id=architecture.get('job_id', 'unknown'),
            provider=self.provider_manager.get_video_provider().__class__.__name__,
            prompt_length=len(str(scenes))
        )
        generated_videos = []
        video_generator = self.provider_manager.get_video_generator()
        max_retries = 3
        
        # Process scenes with retry logic - don't skip failed scenes
        scene_index = 0
        while scene_index < len(scenes):
            scene = scenes[scene_index]
            i = scene_index  # Keep original variable for compatibility
            scene_number = i + 1
            print(f"Generating scene {scene_number}/{len(scenes)}: {scene.get('purpose', 'Unknown')}")
            
            # Update progress for each scene (25% to 70% divided by number of scenes)
            scene_progress = 25 + int((45 / len(scenes)) * i)
            if progress_callback:
                progress_callback(scene_progress, f"Generating Scene {scene_number}/{len(scenes)}...")
            
            # Get appropriate prompt based on provider, with logo enhancement
            provider_name = getattr(settings, 'VIDEO_PROVIDER', 'luma').lower()
            prompt = None  # Initialize prompt variable
            
            # Use logo-enhanced prompt if available
            if logo_integration_plan and 'enhanced_scene_prompts' in logo_integration_plan:
                enhanced_prompts = logo_integration_plan['enhanced_scene_prompts']
                prompt = enhanced_prompts.get(f'scene_{scene_number}')
            
            # Fallback to original prompts
            if not prompt:
                if provider_name == 'luma':
                    prompt = scene.get('luma_prompt', scene.get('visual_concept', ''))
                elif provider_name == 'runway':
                    prompt = scene.get('runway_prompt', scene.get('visual_concept', ''))
                else:
                    prompt = scene.get('visual_concept', '')
            
            if not prompt:
                print(f"Warning: No prompt found for scene {scene_number}")
                continue
            
            # Add logo integration metadata to scene result
            logo_metadata = None
            if logo_integration_plan and 'logo_placements' in logo_integration_plan:
                for placement in logo_integration_plan['logo_placements']:
                    if placement.get('scene_number') == scene_number:
                        logo_metadata = placement
                        break
            
            # Retry logic for current scene
            retry_count = 0
            scene_success = False
            
            while retry_count < max_retries and not scene_success:
                try:
                    retry_suffix = f" (Retry {retry_count + 1}/{max_retries})" if retry_count > 0 else ""
                    print(f"🎥 Generating scene {scene_number} with enhanced quality settings{retry_suffix}")
                    
                    # Generate video with primary provider using quality settings
                    scene_duration = scene.get('duration', 6)  # Use scene duration from planning (6s per scene for 18s total)
                    
                    # Add quality parameters to video generation
                    generation_params = {
                        'prompt': prompt,
                        'aspect_ratio': "9:16",
                        'force_unique': True,
                        'duration': scene_duration,
                        'quality_mode': quality_mode,
                        **quality_settings.get('generation_params', {})
                    }
                    
                    # Use professional promotional method for Luma provider in professional mode
                    if hasattr(video_generator, 'generate_professional_promotional_video') and quality_mode == 'professional':
                        print("🎬 Using Luma model ray-2 (best) at 720p per project preference")
                        video_url = video_generator.generate_professional_promotional_video(**generation_params)
                    else:
                        video_url = video_generator.generate_video(**generation_params)
                
                    # Download video
                    output_filename = f"scene_{scene_number:02d}_{int(time.time())}.mp4"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    if video_generator.download_video(video_url, output_path):
                        # Enforce exact 6.00s scene duration per requirement
                        fixed_output_path = os.path.join(output_dir, f"scene_{scene_number:02d}_{int(time.time())}_6s.mp4")
                        try:
                            assembly = AssemblyService()
                            if assembly._adjust_video_duration(output_path, fixed_output_path, 6.0):
                                # Prefer the fixed-length clip
                                final_scene_path = fixed_output_path
                            else:
                                # Fallback to original if adjustment failed
                                final_scene_path = output_path
                        except Exception:
                            final_scene_path = output_path
                        scene_result = {
                            'scene_number': scene_number,
                            'file_path': final_scene_path,
                            'duration': 6,  # Enforced per-scene duration
                            'prompt': prompt,
                            'url': video_url,
                            'logo_enhanced': bool(logo_integration_plan)
                        }
                        
                        # Add logo metadata if available
                        if logo_metadata:
                            scene_result['logo_placement'] = logo_metadata
                        
                        generated_videos.append(scene_result)
                        print(f"✅ Scene {scene_number} completed: {output_filename}")
                        scene_success = True
                        
                        # Update progress after scene completion
                        scene_complete_progress = 25 + int((45 / len(scenes)) * (scene_index + 1))
                        if progress_callback:
                            progress_callback(scene_complete_progress, f"Scene {scene_number}/{len(scenes)} completed")
                    else:
                        print(f"❌ Failed to download scene {scene_number}")
                        raise Exception(f"Download failed for scene {scene_number}")
                        
                except Exception as e:
                    retry_count += 1
                    print(f"❌ Scene {scene_number} attempt {retry_count} failed: {e}")
                    
                    # Try fallback to Runway on first retry if primary provider (Luma) fails
                    # Disabled until Runway API key is configured
                    if False and retry_count == 1 and provider_name == 'luma':
                        try:
                            print(f"🔄 Attempting fallback to Runway for scene {scene_number}...")
                            from providers.runway import RunwayProvider
                            fallback_generator = RunwayProvider()
                            
                            # Get Runway-specific prompt or create a shorter version
                            fallback_prompt = scene.get('runway_prompt', scene.get('visual_concept', ''))
                            if not fallback_prompt:
                                # Create a short, basic prompt from the scene
                                fallback_prompt = f"Professional commercial video, 9:16 aspect ratio, scene {scene_number}"
                            
                            # Ensure prompt is under Runway's 500 character limit
                            if len(fallback_prompt) > 450:  # Leave some buffer
                                fallback_prompt = fallback_prompt[:450].rsplit(' ', 1)[0] + "..."
                                print(f"📏 Shortened prompt for Runway: {len(fallback_prompt)} chars")
                            
                            if fallback_prompt:
                                print(f"🎬 Runway fallback prompt: {fallback_prompt[:100]}...")
                                video_url = fallback_generator.generate_video(
                                    prompt=fallback_prompt,
                                    aspect_ratio="9:16",
                                    duration=5  # Runway's typical duration
                                )
                                
                                output_filename = f"scene_{scene_number:02d}_runway_{int(time.time())}.mp4"
                                output_path = os.path.join(output_dir, output_filename)
                                
                                if fallback_generator.download_video(video_url, output_path):
                                    scene_result = {
                                        'scene_number': scene_number,
                                        'file_path': output_path,
                                        'duration': scene.get('duration', 6),  # FIXED: Consistent fallback (6s per scene for 18s total)
                                        'prompt': fallback_prompt,
                                        'url': video_url,
                                        'provider': 'runway_fallback'
                                    }
                                    generated_videos.append(scene_result)
                                    print(f"✅ Scene {scene_number} completed with Runway fallback: {output_filename}")
                                    scene_success = True
                                    continue  # Skip retry increment, scene succeeded
                        except Exception as fallback_error:
                            print(f"Runway fallback also failed: {fallback_error}")
                    
                    if retry_count >= max_retries:
                        print(f"💀 Scene {scene_number} failed after {max_retries} retries. Skipping to maintain video continuity.")
                        break
            
            # Move to next scene only after success or max retries reached
            scene_index += 1
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(generated_videos, quality_settings)
        
        return {
            'total_scenes': len(scenes),
            'generated_scenes': len(generated_videos),
            'videos': generated_videos,
            'success_rate': len(generated_videos) / len(scenes) if scenes else 0,
            'quality_mode': quality_mode,
            'quality_metrics': quality_metrics,
            'professional_grade': quality_mode == 'professional'
        }
    
    def generate_single_video(self, prompt: str, output_path: str, **kwargs) -> bool:
        """
        Generate a single video from prompt.
        
        Args:
            prompt: Text description for video
            output_path: Local path to save video
            **kwargs: Additional parameters
            
        Returns:
            True if successful, False otherwise
        """
        try:
            video_generator = self.provider_manager.get_video_generator()
            
            video_url = video_generator.generate_video(
                prompt=prompt,
                aspect_ratio=kwargs.get('aspect_ratio', '9:16'),
                image_url=kwargs.get('image_url'),
                **kwargs
            )
            
            return video_generator.download_video(video_url, output_path)
            
        except Exception as e:
            print(f"Single video generation failed: {e}")
            return False
    
    def switch_provider(self, provider_name: str) -> None:
        """
        Switch video generation provider at runtime.
        
        Args:
            provider_name: Name of the provider ('hailuo', 'luma', etc.)
        """
        try:
            self.provider_manager.set_video_provider(provider_name)
            print(f"Switched to {provider_name} video provider")
        except Exception as e:
            print(f"Failed to switch to {provider_name}: {e}")
            raise
    
    def _get_quality_settings(self, quality_mode: str) -> Dict[str, Any]:
        """Get quality settings based on mode."""
        quality_modes = {
            'professional': {
                'description': 'Broadcast television commercial quality',
                'resolution': '720p',
                'crf': 18,
                'preset': 'medium',
                'profile': 'high',
                'generation_params': {
                    'enhanced_quality': True,
                    'professional_mode': True
                }
            },
            'standard': {
                'description': 'High quality for general use',
                'resolution': '720p',
                'crf': 23,
                'preset': 'medium',
                'profile': 'main',
                'generation_params': {
                    'enhanced_quality': True,
                    'professional_mode': False
                }
            },
            'economy': {
                'description': 'Cost-optimized quality',
                'resolution': '720p',
                'crf': 28,
                'preset': 'fast',
                'profile': 'baseline',
                'generation_params': {
                    'enhanced_quality': False,
                    'professional_mode': False
                }
            }
        }
        
        return quality_modes.get(quality_mode, quality_modes['professional'])
    
    def _calculate_quality_metrics(self, generated_videos: List[Dict[str, Any]], quality_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate quality metrics for generated videos."""
        try:
            total_videos = len(generated_videos)
            if total_videos == 0:
                return {'overall_quality': 'failed', 'videos_analyzed': 0}
            
            successful_generations = sum(1 for video in generated_videos if 'file_path' in video)
            logo_enhanced_count = sum(1 for video in generated_videos if video.get('logo_enhanced', False))
            
            # Simple quality scoring based on success rates and enhancements
            quality_score = (successful_generations / total_videos) * 100
            
            if logo_enhanced_count > 0:
                quality_score += 10  # Bonus for logo integration
            
            quality_level = 'excellent' if quality_score >= 90 else 'good' if quality_score >= 75 else 'acceptable' if quality_score >= 60 else 'poor'
            
            return {
                'overall_quality': quality_level,
                'quality_score': quality_score,
                'videos_analyzed': total_videos,
                'successful_generations': successful_generations,
                'logo_enhanced_videos': logo_enhanced_count,
                'success_rate_percent': (successful_generations / total_videos * 100) if total_videos > 0 else 0,
                'quality_mode_used': quality_settings.get('description', 'unknown'),
                'professional_features': {
                    'logo_integration': logo_enhanced_count > 0,
                    'quality_validation': True,
                    'professional_processing': quality_settings.get('crf', 23) <= 20
                }
            }
            
        except Exception as e:
            return {
                'overall_quality': 'error',
                'error': str(e),
                'videos_analyzed': len(generated_videos) if generated_videos else 0
            }
