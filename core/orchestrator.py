"""
  Main orchestration service for complete video generation pipeline.
  Enterprise-grade coordination of all services and providers.
"""

import os
import time
import tempfile
import uuid
from typing import Dict, Any, List, Optional
from pathlib import Path

from core.enhanced_planning_service import EnhancedPlanningService
from core.video_service import VideoService
from core.audio_service import AudioService
from core.assembly_service import AssemblyService
from core.cost_tracker import cost_tracker
from core.logger import orchestrator_logger, set_trace_context
from core.preflight_validator import PreflightValidator
from config.settings import settings


class VideoOrchestrator:
    """Enterprise-grade video generation orchestrator."""
    
    def __init__(self):
        self.planning_service = EnhancedPlanningService()
        self.video_service = VideoService()
        self.audio_service = AudioService()
        self.assembly_service = AssemblyService()
        self.preflight_validator = PreflightValidator()
    
    def create_complete_video(self, brand_info: Dict[str, Any], output_path: str, 
                           progress_callback: callable = None, video_provider: Optional[str] = None,
                           quality_mode: str = 'professional') -> Dict[str, Any]:
        """
        Create a complete video from brand information with professional quality controls.
        
        Args:
            brand_info: Dictionary containing brand information
            output_path: Path to save the final video
            progress_callback: Optional progress callback function
            video_provider: Optional video provider override
            quality_mode: Quality mode ('professional', 'standard', 'economy')
            
        Returns:
            Dictionary with generation results and quality metrics
        """
        start_time = time.time()
        generation_id = f"gen_{int(start_time)}"
        
        # Initialize trace context
        set_trace_context(f"video_gen_{generation_id}", generation_id)
        
        try:
            orchestrator_logger.info("Video generation started", "orchestration.start",
                                   **{"orchestration.generation_id": generation_id, 
                                      "orchestration.brand": brand_info.get('brand_name', 'unknown'),
                                      "orchestration.duration": brand_info.get('duration', 18)})
            
            # Validate inputs
            if not brand_info:
                orchestrator_logger.error("Brand information missing", "orchestration.validation.failed",
                                        **{"orchestration.missing_field": "brand_info"})
                raise ValueError("Brand information is required")
                
            # Cost estimation and validation
            cost_estimate = cost_tracker.estimate_video_cost(
                scene_count=3, 
                resolution="720p", 
                duration=brand_info.get('duration', 18)
            )
            budget_check = cost_tracker.validate_budget(cost_estimate.total_estimated_cost)
            
            print(f"Estimated cost: ${cost_estimate.total_estimated_cost:.2f} "
                  f"(3 scenes × ${cost_estimate.video_cost_per_scene} + audio ${cost_estimate.audio_cost} + planning ${cost_estimate.planning_cost})")
            
            if not budget_check["within_budget"]:
                print(f"Budget warning: {budget_check.get('warning', 'Cost exceeds limits')}")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            
            # CRITICAL: Run preflight validation to prevent token waste
            if progress_callback:
                progress_callback(2, "Running preflight validation checks...")
            print("🛡️ Running preflight validation to prevent API token waste...")
            
            validation_result = self.preflight_validator.validate_complete_pipeline(
                brand_info, quality_mode
            )
            
            if not validation_result['is_valid']:
                error_msg = f"Preflight validation failed: {validation_result['error_message']}"
                print(f"❌ {error_msg}")
                for issue in validation_result.get('validation_issues', []):
                    print(f"   • {issue}")
                raise Exception(error_msg)
            
            print("✅ Preflight validation passed - proceeding with video generation")
            
            # Proceed directly to video generation
            
            # Step 2: Create enterprise video blueprint (10-20%)
            if progress_callback:
                progress_callback(15, "Creating video blueprint...")
            print("Creating enterprise video blueprint...")
            architecture = self.planning_service.create_enterprise_blueprint(
                brand_info, 
                video_provider=video_provider,
                creative_brief_mode="professional"  # Use maximum professional GPT-4o creativity
            )
            
            target_duration = architecture.get('scene_architecture', {}).get('total_duration', 18)
            print(f"DEBUG: Orchestrator target_duration = {target_duration}s from architecture")
            
            if progress_callback:
                progress_callback(20, "Blueprint created, starting audio-driven timing...")
            
            # STEP 3: Generate audio FIRST to determine actual duration (20-40%)
            print("🎵 Generating audio track to establish master timing...")
            
            # Initialize variables that will be used outside temp_dir context
            actual_audio_duration = target_duration  # fallback
            adjusted_architecture = architecture  # fallback
            audio_success = False
            video_results = {'generated_scenes': 0, 'videos': []}
            assembly_success = False
            
            with tempfile.TemporaryDirectory() as temp_dir:
                if progress_callback:
                    progress_callback(25, "Generating master audio track...")
                
                audio_path = os.path.join(temp_dir, "audio_track.mp3")
                audio_success = self.audio_service.generate_audio_from_architecture(
                    architecture, audio_path
                )
                
                if not audio_success:
                    raise Exception("Audio generation failed - cannot establish master timing")
                
                # STEP 4: Detect actual audio duration and adjust architecture (40-45%)
                if progress_callback:
                    progress_callback(40, "Detecting audio duration and adjusting scenes...")
                
                actual_audio_duration = self._detect_audio_duration(audio_path)
                print(f"🎵 Master timing established: Audio duration = {actual_audio_duration:.2f}s")
                
                # Dynamically adjust scene architecture to match audio
                adjusted_architecture = self._adjust_architecture_to_audio_duration(
                    architecture, actual_audio_duration
                )
                
                if progress_callback:
                    progress_callback(45, f"Scenes adjusted to match {actual_audio_duration:.1f}s audio")
                
                # STEP 5: Generate video scenes with audio-matched durations (45-80%)
                print("🎬 Generating video scenes with audio-synchronized timing...")
                if progress_callback:
                    progress_callback(50, "Generating Scene 1/3 (audio-synchronized)...")
                
                video_results = self.video_service.generate_video_from_architecture(
                    adjusted_architecture, temp_dir, progress_callback, None, quality_mode
                )
                
                if video_results['generated_scenes'] == 0:
                    raise Exception("No video scenes were generated successfully")
                
                if progress_callback:
                    progress_callback(80, "All video scenes generated with perfect audio sync")
                
                # STEP 6: Generate synchronized subtitles with audio-matched timing (80-85%)
                if progress_callback:
                    progress_callback(82, "Generating subtitles with audio-synchronized timing...")
                print("📝 Generating subtitles synchronized to audio duration...")
                
                from core.advanced_subtitle_service import subtitle_service
                # Generate subtitles using adjusted architecture with actual audio duration
                subtitle_segments = subtitle_service.generate_subtitles_from_script(adjusted_architecture)
                
                # If that fails, we'll generate them from the audio later in the assembly step
                if not subtitle_segments:
                    print("Will generate subtitles from audio in post-processing step")
                
                if progress_callback:
                    progress_callback(85, "Subtitles and audio generated")
                
                # Step 6: Assemble final video with overlays (85-95%)
                if progress_callback:
                    progress_callback(88, "Assembling final video with subtitles...")
                print("Assembling final video with overlays...")
                video_files = [video['file_path'] for video in video_results['videos']]
                
                # Get quality settings for assembly
                from core.video_service import VideoService
                quality_settings = VideoService()._get_quality_settings(quality_mode)
                
                # Assemble with subtitle overlays using audio-matched timing
                assembly_success = self.assembly_service.assemble_final_video_with_script(
                    video_files, audio_path, output_path, actual_audio_duration,  # Use actual audio duration
                    subtitle_segments, None, quality_settings, adjusted_architecture  # Use adjusted architecture
                )
                
                if not assembly_success:
                    raise Exception("Video assembly failed")
                
                if progress_callback:
                    progress_callback(95, "Video assembly completed")
            
            # Step 7: Validate final output (95-100%)
            if progress_callback:
                progress_callback(98, "Validating final video...")
            print("Validating final video against audio duration...")
            validation = self.assembly_service.validate_video_output(
                output_path, actual_audio_duration  # Validate against actual audio duration
            )
            
            if progress_callback:
                progress_callback(100, "Video generation completed successfully")
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Log actual costs for tracking
            cost_tracker.log_generation_cost(
                job_id=generation_id,
                actual_cost=cost_estimate.total_estimated_cost,
                scenes_generated=video_results['generated_scenes'],
                resolution="720p"
            )
            
            # Compile results
            results = {
                'success': True,
                'generation_id': generation_id,
                'output_path': output_path,
                'duration': total_time,
                'target_duration': actual_audio_duration,  # Use actual audio duration
                'original_planned_duration': target_duration,  # Keep original for reference
                'architecture': adjusted_architecture,  # Use adjusted architecture
                'video_results': video_results,
                'audio_success': audio_success,
                'assembly_success': assembly_success,
                'validation': validation,
                'cost_breakdown': {
                    'estimated_cost': cost_estimate.total_estimated_cost,
                    'video_cost': cost_estimate.total_video_cost,
                    'audio_cost': cost_estimate.audio_cost,
                    'planning_cost': cost_estimate.planning_cost,
                    'resolution': cost_estimate.resolution,
                    'scenes_generated': video_results['generated_scenes']
                },
                'metadata': {
                    'created_at': time.time(),
                    'video_provider': settings.VIDEO_PROVIDER,
                    'audio_provider': settings.AUDIO_PROVIDER,
                    'text_provider': settings.TEXT_PROVIDER,
                    'cost_optimized': True,
                    'version': '2.0'
                }
            }
            
            print(f"Video generation completed in {total_time:.1f}s: {output_path}")
            return results
            
        except Exception as e:
            end_time = time.time()
            total_time = end_time - start_time
            
            error_results = {
                'success': False,
                'generation_id': generation_id,
                'error': str(e),
                'duration': total_time,
                'output_path': output_path,
                'metadata': {
                    'created_at': time.time(),
                    'error_occurred': True
                }
            }
            
            print(f"Video generation failed after {total_time:.1f}s: {e}")
            return error_results
    
    def _validate_timing_consistency(self, architecture: Dict[str, Any]) -> None:
        """Validate that all services use consistent scene durations."""
        scenes = architecture.get('scene_architecture', {}).get('scenes', [])
        total_duration = architecture.get('scene_architecture', {}).get('total_duration', 30)
        
        if scenes:
            # Check each scene has proper duration
            scene_durations = []
            for i, scene in enumerate(scenes):
                duration = scene.get('duration', 6)  # Use consistent default (6s per scene for 18s total)
                scene_durations.append(duration)
                
                # Force consistent duration if missing
                if 'duration' not in scene:
                    scene['duration'] = 6  # Default scene duration (6s per scene for 18s total)
                    print(f"⚠️  Scene {i+1} missing duration, set to 6s")
            
            # Validate total matches
            calculated_total = sum(scene_durations)
            if abs(calculated_total - total_duration) > 0.1:  # 100ms tolerance
                print(f"⚠️  TIMING MISMATCH: Scenes total {calculated_total:.1f}s but architecture expects {total_duration:.1f}s")
                
                # Auto-fix by adjusting proportionally
                if calculated_total != total_duration:
                    adjustment_factor = total_duration / calculated_total
                    for scene in scenes:
                        scene['duration'] = scene['duration'] * adjustment_factor
                    print(f"✅ Auto-fixed scene durations with {adjustment_factor:.3f}x adjustment")
            
            print(f"🎯 Timing validation: {len(scenes)} scenes, {total_duration:.1f}s total, consistent durations")
    
    def _detect_audio_duration(self, audio_path: str) -> float:
        """Detect the actual duration of generated audio file."""
        import subprocess
        import json
        
        try:
            # Use ffprobe to get precise audio duration
            probe_result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', audio_path
            ], capture_output=True, text=True, check=True)
            
            probe_data = json.loads(probe_result.stdout)
            duration = float(probe_data['format']['duration'])
            
            print(f"🎵 Audio duration detected: {duration:.3f}s from {audio_path}")
            return duration
            
        except Exception as e:
            print(f"⚠️  Failed to detect audio duration: {e}")
            # Fallback to standard ad duration
            return 18.0
    
    def _adjust_architecture_to_audio_duration(self, architecture: Dict[str, Any], audio_duration: float) -> Dict[str, Any]:
        """Dynamically adjust scene architecture to match actual audio duration."""
        adjusted_architecture = architecture.copy()
        
        # Update total duration to match audio
        if 'scene_architecture' in adjusted_architecture:
            scene_arch = adjusted_architecture['scene_architecture']
            original_duration = scene_arch.get('total_duration', 18)
            
            print(f"🔄 Adjusting architecture: {original_duration:.1f}s → {audio_duration:.1f}s")
            
            # Update total duration
            scene_arch['total_duration'] = audio_duration
            
            # Proportionally adjust all scene durations
            scenes = scene_arch.get('scenes', [])
            if scenes:
                original_total = sum(scene.get('duration', 6) for scene in scenes)
                adjustment_factor = audio_duration / original_total if original_total > 0 else 1.0
                
                print(f"🎯 Scene duration adjustment factor: {adjustment_factor:.3f}x")
                
                for i, scene in enumerate(scenes):
                    original_scene_duration = scene.get('duration', 6)
                    new_scene_duration = original_scene_duration * adjustment_factor
                    scene['duration'] = new_scene_duration
                    
                    print(f"   Scene {i+1}: {original_scene_duration:.1f}s → {new_scene_duration:.2f}s")
        
        # Update audio architecture to match
        if 'audio_architecture' in adjusted_architecture:
            adjusted_architecture['audio_architecture']['total_duration'] = audio_duration
        
        print(f"✅ Architecture adjusted to audio-driven timing: {audio_duration:.2f}s total")
        return adjusted_architecture
    
    def create_video_from_simple_prompt(self, prompt: str, output_path: str, 
                                      duration: int = 18) -> Dict[str, Any]:
        """
        Create video from a simple text prompt.
        
        Args:
            prompt: Simple text description
            output_path: Path to save the final video
            duration: Video duration in seconds
            
        Returns:
            Dictionary with generation results
        """
        # Convert simple prompt to brand info
        brand_info = {
            'brand_name': 'Custom Video',
            'brand_description': prompt,
            'target_audience': 'general',
            'call_to_action': 'Learn more',
            'duration': duration
        }
        
        # Run preflight validation before expensive operations
        print("🛡️ Running preflight validation for simple prompt...")
        validation_result = self.preflight_validator.validate_complete_pipeline(brand_info)
        
        if not validation_result['is_valid']:
            error_msg = f"Preflight validation failed: {validation_result['error_message']}"
            print(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'validation_issues': validation_result.get('validation_issues', [])
            }
        
        print("✅ Preflight validation passed")
        return self.create_complete_video(brand_info, output_path)
    
    def switch_providers(self, video_provider: Optional[str] = None, 
                        audio_provider: Optional[str] = None,
                        text_provider: Optional[str] = None) -> None:
        """
        Switch providers at runtime.
        
        Args:
            video_provider: New video provider name
            audio_provider: New audio provider name
            text_provider: New text provider name
        """
        try:
            if video_provider:
                self.video_service.switch_provider(video_provider)
            if audio_provider:
                self.audio_service.switch_provider(audio_provider)
            if text_provider:
                self.planning_service.switch_provider(text_provider)
            
            print("Provider switching completed")
            
        except Exception as e:
            print(f"Provider switching failed: {e}")
            raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and configuration."""
        return {
            'providers': {
                'video': settings.VIDEO_PROVIDER,
                'audio': settings.AUDIO_PROVIDER,
                'text': settings.TEXT_PROVIDER
            },
            'configuration': {
                'max_concurrent_jobs': settings.MAX_CONCURRENT_JOBS,
                'job_timeout': settings.JOB_TIMEOUT,
                'output_dir': settings.OUTPUT_DIR,
                'temp_dir': settings.TEMP_DIR
            },
            'capabilities': {
                'video_providers': ['luma', 'runway'],
                'audio_providers': ['elevenlabs'],
                'text_providers': ['openai']
            }
        }
