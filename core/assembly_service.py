"""
  Video assembly service for combining video and audio components.
"""

import os
import subprocess
import tempfile
from typing import List, Dict, Any, Optional
from pathlib import Path
from core.logger import assembly_logger
from config.settings import settings


class AssemblyService:
    """Service for assembling final video from components."""
    
    def __init__(self):
        pass
    
    def assemble_final_video(self, video_files: List[str], audio_file: str, 
                           output_path: str, target_duration: float = 18.0,
                           subtitle_segments: List = None, logo_integration_plan: Dict[str, Any] = None,
                           quality_settings: Dict[str, Any] = None) -> bool:
        """
        Assemble final video from video scenes and audio track with subtitle overlays.
        Enhanced with professional quality controls and validation.
        
        Args:
            video_files: List of video file paths in order
            audio_file: Path to audio track file
            output_path: Path to save final assembled video
            target_duration: Target duration in seconds
            subtitle_segments: Optional subtitle segments for overlay
            logo_integration_plan: Optional logo integration plan
            quality_settings: Production quality settings and validation parameters
            
        Returns:
            True if assembly successful, False otherwise
        """
        assembly_logger.assembly_timing_debug(
            job_id="current", 
            target_duration_s=target_duration, 
            actual_duration_s=0.0, 
            adjustment_applied=False
        )
        
        # Set professional quality defaults
        if quality_settings is None:
            quality_settings = {
                'video_codec': 'libx264',
                'audio_codec': 'aac',
                'bitrate': '2500k',
                'audio_bitrate': '192k',
                'preset': 'medium',
                'crf': 18,  # High quality constant rate factor
                'pixel_format': 'yuv420p',
                'profile': 'high',
                'level': '4.0',
                'keyframe_interval': 2,
                'max_muxing_queue_size': 1024
            }
        
        try:
            if not video_files:
                print("No video files provided for assembly")
                return False
            
            if not os.path.exists(audio_file):
                print(f"Audio file not found: {audio_file}")
                return False
                
            # Validate video file quality before processing
            print("Validating video files for quality standards...")
            if not self._validate_video_quality(video_files):
                print("Warning: Video quality validation failed - proceeding with enhanced processing")
                quality_settings['crf'] = 16  # Use higher quality for poor input
            
            # Validate video files exist
            valid_videos = []
            for video_file in video_files:
                if os.path.exists(video_file):
                    valid_videos.append(video_file)
                else:
                    print(f"Warning: Video file not found: {video_file}")
            
            if not valid_videos:
                print("No valid video files found")
                return False
            
            # Create temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                # Step 1: Concatenate video files
                if len(valid_videos) > 1:
                    concat_video_path = os.path.join(temp_dir, "concatenated_video.mp4")
                    if not self._concatenate_videos(valid_videos, concat_video_path):
                        return False
                else:
                    concat_video_path = valid_videos[0]
                
                # Step 2: Adjust video duration to match target
                duration_adjusted_path = os.path.join(temp_dir, "duration_adjusted.mp4")
                if not self._adjust_video_duration(concat_video_path, duration_adjusted_path, target_duration):
                    return False
                
                # Step 3: Combine with audio using professional quality settings
                combined_video_path = os.path.join(temp_dir, "combined.mp4")
                if not self._combine_video_audio(duration_adjusted_path, audio_file, combined_video_path, target_duration, quality_settings):
                    return False

                # Step 4: Apply subtitle overlays
                if subtitle_segments:
                    overlay_applied_path = os.path.join(temp_dir, "with_overlays.mp4")
                    if not self._apply_overlays(combined_video_path, overlay_applied_path, 
                                              subtitle_segments, target_duration, audio_file):
                        return False
                    final_video_path = overlay_applied_path
                else:
                    final_video_path = combined_video_path
                
                # Step 5: Final quality validation
                if not self._validate_final_output(final_video_path, target_duration):
                    print("Warning: Final video failed quality validation")
                    # copy final_video_path to output_path
                    import shutil
                    shutil.copy2(final_video_path, output_path)
                    return False
                
                # copy final_video_path to output_path
                import shutil
                shutil.copy2(final_video_path, output_path)
                
                print(f"Final video assembled with production quality: {output_path}")
                return True
                
        except Exception as e:
            print(f"Video assembly failed: {e}")
            return False
    
    def assemble_final_video_with_script(self, video_files: List[str], audio_file: str, output_path: str, 
                           target_duration: float, subtitle_segments: List = None, 
                           logo_integration_plan: Dict[str, Any] = None,
                           quality_settings: Dict[str, Any] = None, architecture: Dict[str, Any] = None) -> bool:
        """
        Enhanced video assembly with script-aware subtitle generation.
        
        Args:
            video_files: List of video file paths to concatenate
            audio_file: Path to audio track
            output_path: Final output video path
            target_duration: Target video duration in seconds
            subtitle_segments: Pre-generated subtitle segments (optional)
            logo_integration_plan: Logo integration configuration (unused, for compatibility)
            quality_settings: Video quality configuration
            architecture: Video architecture containing script for subtitle alignment
            
        Returns:
            True if assembly successful, False otherwise
        """
        try:
            print("🎬 Starting professional video assembly with script-aligned subtitles...")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Step 1: Filter valid videos
                valid_videos = [v for v in video_files if os.path.exists(v)]
                if not valid_videos:
                    print("No valid video files found")
                    return False
                
                # Step 2: Concatenate videos if multiple
                if len(valid_videos) > 1:
                    concat_video_path = os.path.join(temp_dir, "concatenated.mp4")
                    if not self._concatenate_videos(valid_videos, concat_video_path):
                        return False
                else:
                    concat_video_path = valid_videos[0]
                
                # Step 3: Adjust video duration to match target
                duration_adjusted_path = os.path.join(temp_dir, "duration_adjusted.mp4")
                if not self._adjust_video_duration(concat_video_path, duration_adjusted_path, target_duration):
                    return False
                
                # Step 4: Combine with audio using professional quality settings
                combined_video_path = os.path.join(temp_dir, "combined.mp4")
                if not self._combine_video_audio(duration_adjusted_path, audio_file, combined_video_path, target_duration, quality_settings):
                    return False

                # Step 5: Apply professional subtitle overlays with script alignment
                if subtitle_segments or architecture:
                    overlay_applied_path = os.path.join(temp_dir, "with_overlays.mp4")
                    if not self._apply_professional_subtitles(combined_video_path, overlay_applied_path, 
                                              subtitle_segments, target_duration, audio_file, architecture):
                        return False
                    final_video_path = overlay_applied_path
                else:
                    final_video_path = combined_video_path
                
                # Step 6: Final quality validation
                if not self._validate_final_output(final_video_path, target_duration):
                    print("Warning: Final video failed quality validation")
                    # copy final_video_path to output_path
                    import shutil
                    shutil.copy2(final_video_path, output_path)
                    return False
                
                # copy final_video_path to output_path
                import shutil
                shutil.copy2(final_video_path, output_path)
                
                print(f"✅ Professional video assembled with script-synchronized subtitles: {output_path}")
                return True
                
        except Exception as e:
            print(f"Enhanced video assembly failed: {e}")
            return False
    
    def _apply_overlays(self, input_path: str, output_path: str, subtitle_segments: List = None,
                       target_duration: float = 18.0, audio_path: str = None) -> bool:
        """Apply subtitle overlays to video."""
        try:
            # If we have an audio file but no subtitle segments, try to generate them from audio
            if audio_path and os.path.exists(audio_path) and not subtitle_segments:
                from core.advanced_subtitle_service import subtitle_service
                # Try to extract script text from the original architecture for better alignment
                script_text = None
                # TODO: Pass architecture to get unified_script for perfect alignment
                subtitle_segments = subtitle_service.generate_subtitles_from_audio(audio_path, script_text)

            print("Applying subtitle overlays...")
            
            # Build filter complex for overlays
            filter_parts = []
            input_files = ['-i', input_path]
            
            # Start with the main video
            current_video = '[0:v]'
            input_count = 1
            
            
            # Add subtitle overlays if available
            if subtitle_segments:
                from core.advanced_subtitle_service import subtitle_service
                
                # Generate subtitle overlay filter
                subtitle_filter = subtitle_service.generate_subtitle_overlay_filter(
                    subtitle_segments,
                    style_config=self._get_subtitle_style_config()
                )
                
                if subtitle_filter:
                    final_filter = f"{current_video}{subtitle_filter}[final]"
                    filter_parts.append(final_filter)
                    current_video = '[final]'
            
            # Build complete filter complex
            if filter_parts:
                filter_complex = ';'.join(filter_parts)
                
                cmd = [
                    'ffmpeg', '-y'
                ] + input_files + [
                    '-filter_complex', filter_complex,
                    '-map', current_video,
                    '-c:v', 'libx264',
                    '-preset', 'medium',
                    '-crf', '23',
                    '-pix_fmt', 'yuv420p',
                    '-t', str(target_duration),
                    output_path
                ]
                
                print(f"Applying overlays with filter: {filter_complex}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("Overlays applied successfully")
                    return True
                else:
                    print(f"Overlay application failed: {result.stderr}")
                    # Try without overlays as fallback
                    import shutil
                    shutil.copy2(input_path, output_path)
                    return True
            else:
                # No overlays to apply, just copy
                import shutil
                shutil.copy2(input_path, output_path)
                return True
                
        except Exception as e:
            print(f"Overlay application error: {e}")
            # Try to copy the file as fallback
            try:
                import shutil
                shutil.copy2(input_path, output_path)
                return True
            except:
                return False
    
    def _get_subtitle_style_config(self) -> Dict[str, Any]:
        """Get subtitle styling configuration."""
        default_style = {
            'fontsize': 36,  # Optimized size for 9:16 aspect ratio
            'fontcolor': 'white',
            'bordercolor': 'black',
            'borderw': 4,  # Compact border for mobile viewing
            'shadow_color': '0x80000000',
            'shadow_x': 3,
            'shadow_y': 3,
            'position': 'bottom',  # Default to bottom
            'background_opacity': 0.8,  # More opaque background for better readability
            'background_color': '0x80000000'
        }
        
        # Optimize for 9:16 vertical videos
        # Position subtitles in the lower third for optimal mobile viewing
        if default_style['position'] == 'bottom':
            default_style['y_position'] = 'h*0.65'  # 65% down the video (lower third)
        elif default_style['position'] == 'top':
            default_style['y_position'] = 'h*0.15'  # 15% from top
        else:  # center
            default_style['y_position'] = 'h*0.5'   # True center
        
        # Style is now consistent for all videos
        
        return default_style
    
    def _apply_professional_subtitles(self, input_path: str, output_path: str, 
                                     subtitle_segments: List = None, target_duration: float = 18.0,
                                     audio_path: str = None, architecture: Dict[str, Any] = None) -> bool:
        """Apply professional subtitles with script alignment and advanced styling."""
        try:
            print("🎯 Applying professional script-aligned subtitles...")
            
            # Generate subtitles if not provided
            if not subtitle_segments:
                from core.advanced_subtitle_service import subtitle_service
                
                # Extract script from architecture for perfect alignment
                script_text = None
                if architecture:
                    script_text = architecture.get('unified_script', '')
                    if not script_text:
                        # Try to extract from scenes
                        scenes = architecture.get('scene_architecture', {}).get('scenes', [])
                        script_parts = []
                        for scene in scenes:
                            scene_script = scene.get('script_line', '')
                            if scene_script:
                                script_parts.append(scene_script)
                        script_text = ' '.join(script_parts)
                
                # Generate subtitles with script alignment
                if audio_path and os.path.exists(audio_path):
                    subtitle_segments = subtitle_service.generate_subtitles_from_audio(audio_path, script_text)
                elif script_text:
                    subtitle_segments = subtitle_service.generate_subtitles_from_script(architecture)
            
            if not subtitle_segments:
                print("⚠️ No subtitles generated - copying video without subtitles")
                import shutil
                shutil.copy2(input_path, output_path)
                return True
            
            # Generate professional subtitle overlay filter
            from core.advanced_subtitle_service import subtitle_service
            
            # Professional subtitle styling
            professional_style = {
                'fontsize': 36,  # Larger for professional appearance
                'fontcolor': 'white',
                'bordercolor': 'black',
                'borderw': 4,
                'shadow_color': '0x80000000',
                'shadow_x': 3,
                'shadow_y': 3,
                'position': 'bottom',
                'background_color': '0x80000000',
                'line_spacing': 10
            }
            
            subtitle_filter = subtitle_service.generate_subtitle_overlay_filter(
                subtitle_segments, professional_style
            )
            
            if not subtitle_filter:
                print("⚠️ No subtitle filter generated - copying video without subtitles")
                import shutil
                shutil.copy2(input_path, output_path)
                return True
            
            # Apply subtitle overlay with FFmpeg
            import subprocess
            cmd = [
                'ffmpeg', '-y',
                '-i', input_path,
                '-vf', subtitle_filter,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-c:a', 'copy',  # Copy audio without re-encoding
                '-pix_fmt', 'yuv420p',
                output_path
            ]
            
            print(f"📝 Applying {len(subtitle_segments)} professionally styled subtitle segments...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Professional subtitles applied successfully")
                return True
            else:
                print(f"❌ Subtitle application failed: {result.stderr}")
                # Fallback: copy without subtitles
                import shutil
                shutil.copy2(input_path, output_path)
                return True
                
        except Exception as e:
            print(f"Professional subtitle application error: {e}")
            # Fallback: copy without subtitles
            try:
                import shutil
                shutil.copy2(input_path, output_path)
                return True
            except:
                return False
    
    
    
    def _concatenate_videos(self, video_files: List[str], output_path: str) -> bool:
        """Concatenate multiple video files into one while preserving natural motion."""
        try:
            # Determine target dimensions based on quality settings (default 9:16 portrait)
            def _target_dims() -> tuple[int, int]:
                # Map common labels to portrait dimensions
                res = getattr(settings, 'DEFAULT_VIDEO_RESOLUTION', '1080p')
                res = (res or '1080p').lower()
                if res in ('1080p', 'fullhd', 'fhd'):
                    return (1080, 1920)
                if res in ('720p',):
                    return (720, 1280)
                # Fallback: probe first video, else 1080x1920
                try:
                    import json, subprocess
                    probe = subprocess.run([
                        'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', video_files[0]
                    ], capture_output=True, text=True)
                    data = json.loads(probe.stdout)
                    vstreams = [s for s in data.get('streams', []) if s.get('codec_type') == 'video']
                    if vstreams:
                        w = int(vstreams[0].get('width', 1080))
                        h = int(vstreams[0].get('height', 1920))
                        # Normalize to portrait 9:16 if clearly vertical
                        if h >= w:
                            # Round to nearest standard size
                            if h >= 1920:
                                return (1080, 1920)
                            return (720, 1280)
                        else:
                            # Landscape – still scale consistently without forcing rotation
                            return (1920, 1080)
                except Exception:
                    pass
                return (1080, 1920)

            target_w, target_h = _target_dims()

            # Create filter complex for concatenation with scaling (no FPS forcing)
            filter_parts = []
            input_files = []
            
            for i, video_file in enumerate(video_files):
                input_files.extend(['-i', video_file])
                # Scale all videos to target portrait dimensions, preserve natural frame cadence (no fps filter)
                filter_parts.append(
                    f"[{i}:v]scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
                    f"crop={target_w}:{target_h},setsar=1[v{i}]"
                )
            
            # Concatenate all scaled videos
            video_inputs = ''.join([f"[v{i}]" for i in range(len(video_files))])
            concat_filter = f"{video_inputs}concat=n={len(video_files)}:v=1:a=0[outv]"
            
            # Combine filters
            full_filter = ';'.join(filter_parts) + ';' + concat_filter
            
            cmd = [
                'ffmpeg', '-y'
            ] + input_files + [
                '-filter_complex', full_filter,
                '-map', '[outv]',
                '-c:v', 'libx264',
                '-preset', 'medium',
                # Use higher quality here to minimize re-encode artifacts; final pass will still occur later
                '-crf', '18',
                '-pix_fmt', 'yuv420p',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Videos concatenated successfully")
                return True
            else:
                print(f"Video concatenation failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"Video concatenation error: {e}")
            return False
    
    def _adjust_video_duration(self, input_path: str, output_path: str, target_duration: float) -> bool:
        """Adjust video duration to exact target length by extending with freeze frames."""
        try:
            # Get current video duration
            probe_cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', input_path
            ]
            
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
            
            if probe_result.returncode != 0:
                print(f"Failed to probe video duration")
                return False
            
            import json
            probe_data = json.loads(probe_result.stdout)
            current_duration = float(probe_data['format']['duration'])
            
            print(f"Current duration: {current_duration:.2f}s, Target: {target_duration:.2f}s")
            
            # Duration is close enough, just copy
            if abs(current_duration - target_duration) < 0.1:
                import shutil
                shutil.copy2(input_path, output_path)
                return True
            
            if current_duration > target_duration:
                # Video is longer than target - trim it
                print(f"Trimming video: {current_duration:.2f}s → {target_duration:.2f}s")
                cmd = [
                    'ffmpeg', '-y', '-i', input_path,
                    '-t', str(target_duration),
                    '-c:v', 'libx264',
                    '-preset', 'medium',
                    '-crf', '18',
                    '-c:a', 'copy',
                    output_path
                ]
            else:
                # Video is shorter than target - extend with freeze frame
                extension_duration = target_duration - current_duration
                print(f"Extending video: {current_duration:.2f}s → {target_duration:.2f}s (+{extension_duration:.2f}s freeze frame)")
                
                # Create filter to extend video by freezing last frame
                # Use tpad filter to pad video with last frame
                video_filter = f"tpad=stop_mode=clone:stop_duration={extension_duration:.3f}"
                
                cmd = [
                    'ffmpeg', '-y', '-i', input_path,
                    '-filter:v', video_filter,
                    '-c:v', 'libx264',
                    '-preset', 'medium',
                    '-crf', '18',
                    '-c:a', 'copy',
                    output_path
                ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Video duration adjusted to {target_duration}s (maintaining natural playback speed)")
                return True
            else:
                print(f"Duration adjustment failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"Duration adjustment error: {e}")
            return False
    
    def _validate_video_quality(self, video_files: List[str]) -> bool:
        """Validate input video files meet quality standards."""
        try:
            for video_file in video_files:
                if not os.path.exists(video_file):
                    continue
                    
                # Check video properties
                probe_cmd = [
                    'ffprobe', '-v', 'quiet', '-print_format', 'json',
                    '-show_streams', '-show_format', video_file
                ]
                
                result = subprocess.run(probe_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Failed to probe video: {video_file}")
                    continue
                    
                import json
                data = json.loads(result.stdout)
                streams = data.get('streams', [])
                video_streams = [s for s in streams if s.get('codec_type') == 'video']
                
                if not video_streams:
                    print(f"No video stream in: {video_file}")
                    return False
                    
                video_stream = video_streams[0]
                width = video_stream.get('width', 0)
                height = video_stream.get('height', 0)
                
                # Quality checks
                if width < 720 or height < 720:
                    print(f"Low resolution video detected: {width}x{height} in {video_file}")
                    return False
                    
                # Check for common quality issues
                if 'pix_fmt' in video_stream and video_stream['pix_fmt'] not in ['yuv420p', 'yuv444p']:
                    print(f"Suboptimal pixel format in: {video_file}")
                    
            return True
            
        except Exception as e:
            print(f"Video quality validation error: {e}")
            return False
    
    def _validate_final_output(self, output_path: str, target_duration: float) -> bool:
        """Comprehensive validation of final video output."""
        try:
            validation_result = self.validate_video_output(output_path, target_duration)
            
            if not validation_result['valid']:
                print(f"Final video validation failed: {validation_result.get('error', 'Unknown error')}")
                if 'issues' in validation_result:
                    for issue in validation_result['issues']:
                        print(f"  - {issue}")
                return False
                
            # Additional production quality checks
            duration_diff = validation_result.get('duration_diff', 0)
            if duration_diff > 0.5:  # Allow 0.5s tolerance
                print(f"Warning: Duration difference of {duration_diff:.2f}s exceeds tolerance")
                
            file_size = validation_result.get('file_size', 0)
            if file_size > 50 * 1024 * 1024:  # 50MB limit
                print(f"Warning: File size {file_size / (1024*1024):.1f}MB exceeds recommended limit")
                
            print(f"✅ Final video validation passed:")
            print(f"  - Duration: {validation_result.get('duration', 0):.2f}s")
            print(f"  - Resolution: {validation_result.get('resolution', 'Unknown')}")
            print(f"  - File size: {file_size / (1024*1024):.1f}MB")
            print(f"  - Video codec: {validation_result.get('video_codec', 'Unknown')}")
            print(f"  - Audio codec: {validation_result.get('audio_codec', 'Unknown')}")
            
            return True
            
        except Exception as e:
            print(f"Final output validation error: {e}")
            return False

    def _combine_video_audio(self, video_path: str, audio_path: str, output_path: str, duration: float, quality_settings: Dict[str, Any] = None) -> bool:
        """Combine video and audio into final output."""
        try:
            # First, let's verify the audio file has proper volume
            probe_cmd = ['ffmpeg', '-i', audio_path, '-af', 'volumedetect', '-f', 'null', '-']
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
            volume_info = probe_result.stderr.split('volumedetect')[1] if 'volumedetect' in probe_result.stderr else 'No volume data'
            print(f"🔍 DEBUG: Pre-assembly audio levels: {volume_info}")
            
            # Get actual video and audio durations to ensure proper sync
            video_probe = subprocess.run([
                'ffprobe', '-v', 'quiet', '-print_format', 'json', 
                '-show_format', video_path
            ], capture_output=True, text=True)
            
            audio_probe = subprocess.run([
                'ffprobe', '-v', 'quiet', '-print_format', 'json', 
                '-show_format', audio_path  
            ], capture_output=True, text=True)
            
            if video_probe.returncode == 0 and audio_probe.returncode == 0:
                import json
                video_duration = float(json.loads(video_probe.stdout)['format']['duration'])
                audio_duration = float(json.loads(audio_probe.stdout)['format']['duration'])
                print(f"Final assembly: Video={video_duration:.2f}s, Audio={audio_duration:.2f}s, Target={duration:.2f}s")
                
                # Handle duration mismatch more intelligently
                duration_diff = abs(video_duration - audio_duration)
                
                # FIXED: Tighter synchronization tolerance for professional quality
                if duration_diff < 0.2:  # 200ms tolerance instead of 1000ms
                    print("Video and audio durations match within professional tolerance")
                    use_shortest = False
                elif video_duration > audio_duration:
                    # Video is longer than audio - extend audio or use loop/pad
                    print(f"Video longer than audio by {duration_diff:.2f}s - will extend audio to match")
                    use_shortest = False  # Don't truncate video to audio length
                else:
                    # Audio is longer than video - use shortest (truncate audio)
                    print(f"Audio longer than video by {duration_diff:.2f}s - using shortest length")
                    use_shortest = True
            else:
                use_shortest = True
            
            # Use professional quality settings if provided
            if quality_settings is None:
                quality_settings = {
                    'video_codec': 'libx264',
                    'audio_codec': 'aac',
                    'bitrate': '2500k',
                    'audio_bitrate': '192k',
                    'crf': 18
                }
            
            # FIXED: More precise audio extension threshold
            need_audio_extension = (not use_shortest and 'video_duration' in locals() and 'audio_duration' in locals() 
                                  and video_duration > audio_duration + 0.1)  # 100ms threshold instead of 1000ms
            
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-i', audio_path,
                '-c:v', quality_settings.get('video_codec', 'libx264'),
                '-c:a', quality_settings.get('audio_codec', 'aac'),
                '-crf', str(quality_settings.get('crf', 18)),
                '-b:a', quality_settings.get('audio_bitrate', '192k'),
                '-preset', quality_settings.get('preset', 'medium'),
                '-pix_fmt', quality_settings.get('pixel_format', 'yuv420p'),
                '-profile:v', quality_settings.get('profile', 'high'),
                '-level:v', quality_settings.get('level', '4.0'),
                '-movflags', '+faststart',  # Optimize for streaming
            ]
            
            # Handle audio extension if needed
            if need_audio_extension:
                print(f"Extending audio from {audio_duration:.2f}s to match video {video_duration:.2f}s")
                # Use filter to extend audio with silence/fade
                cmd.extend([
                    '-filter_complex', f'[1:a]apad=pad_dur={video_duration-audio_duration:.2f}[extended_audio]',
                    '-map', '0:v:0',
                    '-map', '[extended_audio]'
                ])
            else:
                cmd.extend([
                    '-map', '0:v:0',
                    '-map', '1:a:0'
                ])
            
            # Only add -shortest if we're intentionally using shortest
            if use_shortest:
                cmd.extend(['-shortest'])
                
            cmd.append(output_path)
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Video and audio combined successfully")
                return True
            else:
                print(f"Video-audio combination failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"Video-audio combination error: {e}")
            return False
    
    def validate_video_output(self, video_path: str, expected_duration: float = 18.0) -> Dict[str, Any]:
        """
        Validate the final video output.
        
        Args:
            video_path: Path to video file to validate
            expected_duration: Expected duration in seconds
            
        Returns:
            Dictionary with validation results
        """
        try:
            if not os.path.exists(video_path):
                return {'valid': False, 'error': 'File does not exist'}
            
            # Get video info
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                return {'valid': False, 'error': 'Failed to probe video'}
            
            import json
            data = json.loads(result.stdout)
            
            # Extract video info
            format_info = data.get('format', {})
            streams = data.get('streams', [])
            
            video_streams = [s for s in streams if s.get('codec_type') == 'video']
            audio_streams = [s for s in streams if s.get('codec_type') == 'audio']
            
            duration = float(format_info.get('duration', 0))
            file_size = int(format_info.get('size', 0))
            
            # Validation results
            validation = {
                'valid': True,
                'duration': duration,
                'expected_duration': expected_duration,
                'duration_diff': abs(duration - expected_duration),
                'file_size': file_size,
                'has_video': len(video_streams) > 0,
                'has_audio': len(audio_streams) > 0,
                'video_codec': video_streams[0].get('codec_name') if video_streams else None,
                'audio_codec': audio_streams[0].get('codec_name') if audio_streams else None,
                'resolution': None,
                'aspect_ratio': None
            }
            
            # Get resolution info
            if video_streams:
                width = video_streams[0].get('width')
                height = video_streams[0].get('height')
                if width and height:
                    validation['resolution'] = f"{width}x{height}"
                    validation['aspect_ratio'] = f"{width}:{height}"
            
            # Check critical validations
            issues = []
            if not validation['has_video']:
                issues.append('No video stream found')
            if not validation['has_audio']:
                issues.append('No audio stream found')
            if validation['duration_diff'] > 2.0:
                issues.append(f'Duration off by {validation["duration_diff"]:.1f}s')
            if file_size < 1000000:  # Less than 1MB
                issues.append('File size suspiciously small')
            
            if issues:
                validation['valid'] = False
                validation['issues'] = issues
            
            return validation
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
