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

from core.planning_service import PlanningService
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
        self.planning_service = PlanningService()
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
                brand_info, None, quality_mode
            )
            
            if not validation_result['is_valid']:
                error_msg = f"Preflight validation failed: {validation_result['error_message']}"
                print(f"❌ {error_msg}")
                for issue in validation_result.get('validation_issues', []):
                    print(f"   • {issue}")
                raise Exception(error_msg)
            
            print("✅ Preflight validation passed - proceeding with video generation")
            
            # Logo integration removed - proceed directly to video generation
            
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
                progress_callback(20, "Blueprint created, starting video generation...")
            
            # Step 3: Generate video scenes (20-70%)
            print("Generating video scenes...")
            with tempfile.TemporaryDirectory() as temp_dir:
                if progress_callback:
                    progress_callback(25, "Generating Scene 1/3...")
                
                video_results = self.video_service.generate_video_from_architecture(
                    architecture, temp_dir, progress_callback, None, quality_mode
                )
                
                if video_results['generated_scenes'] == 0:
                    raise Exception("No video scenes were generated successfully")
                
                if progress_callback:
                    progress_callback(70, "All video scenes generated successfully")
                
                # Step 4: Generate audio track with subtitle alignment (70-80%)
                if progress_callback:
                    progress_callback(75, "Generating voiceover and background music...")
                print("Generating audio track...")
                audio_path = os.path.join(temp_dir, "audio_track.mp3")
                audio_success = self.audio_service.generate_audio_from_architecture(
                    architecture, audio_path
                )
                
                if not audio_success:
                    raise Exception("Audio generation failed")
                
                # Step 5: Generate synchronized subtitles (80-85%)
                if progress_callback:
                    progress_callback(80, "Generating synchronized subtitles...")
                print("Generating synchronized subtitles...")
                
                from core.subtitle_service import subtitle_service
                # First try to generate subtitles from the script
                subtitle_segments = subtitle_service.generate_subtitles_from_script(architecture)
                
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
                
                # Assemble with subtitle overlays only (no logo) - pass architecture for script access
                assembly_success = self.assembly_service.assemble_final_video_with_script(
                    video_files, audio_path, output_path, target_duration,
                    subtitle_segments, None, quality_settings, architecture
                )
                
                if not assembly_success:
                    raise Exception("Video assembly failed")
                
                if progress_callback:
                    progress_callback(95, "Video assembly completed")
            
            # Step 7: Validate final output (95-100%)
            if progress_callback:
                progress_callback(98, "Validating final video...")
            print("Validating final video...")
            validation = self.assembly_service.validate_video_output(
                output_path, target_duration
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
                'target_duration': target_duration,
                'architecture': architecture,
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
                'video_providers': ['hailuo', 'luma'],
                'audio_providers': ['elevenlabs'],
                'text_providers': ['openai']
            }
        }
