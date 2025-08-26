"""
  Video assembly service for combining video and audio components.
"""

import os
import subprocess
import tempfile
from typing import List, Dict, Any, Optional
from pathlib import Path
from core.logger import assembly_logger


class AssemblyService:
    """Service for assembling final video from components."""
    
    def __init__(self):
        pass
    
    def assemble_final_video(self, video_files: List[str], audio_file: str, 
                           output_path: str, target_duration: float = 18.0,
                           subtitle_segments: List = None, logo_integration_plan: Dict[str, Any] = None) -> bool:
        """
        Assemble final video from video scenes and audio track with subtitle and logo overlays.
        
        Args:
            video_files: List of video file paths in order
            audio_file: Path to audio track file
            output_path: Path to save final assembled video
            target_duration: Target duration in seconds
            subtitle_segments: Optional subtitle segments for overlay
            logo_integration_plan: Optional logo integration plan
            
        Returns:
            True if assembly successful, False otherwise
        """
        assembly_logger.assembly_timing_debug(
            job_id="current", 
            target_duration_s=target_duration, 
            actual_duration_s=0.0, 
            adjustment_applied=False
        )
        try:
            if not video_files:
                print("No video files provided for assembly")
                return False
            
            if not os.path.exists(audio_file):
                print(f"Audio file not found: {audio_file}")
                return False
            
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
                
                # Step 3: Apply subtitle and logo overlays
                if subtitle_segments or logo_integration_plan:
                    overlay_applied_path = os.path.join(temp_dir, "with_overlays.mp4")
                    if not self._apply_overlays(duration_adjusted_path, overlay_applied_path, 
                                              subtitle_segments, logo_integration_plan, target_duration):
                        return False
                    final_video_path = overlay_applied_path
                else:
                    final_video_path = duration_adjusted_path
                
                # Step 4: Combine with audio
                if not self._combine_video_audio(final_video_path, audio_file, output_path, target_duration):
                    return False
                
                print(f"Final video assembled: {output_path}")
                return True
                
        except Exception as e:
            print(f"Video assembly failed: {e}")
            return False
    
    def _apply_overlays(self, input_path: str, output_path: str, subtitle_segments: List = None,
                       logo_integration_plan: Dict[str, Any] = None, target_duration: float = 18.0) -> bool:
        """Apply subtitle and logo overlays to video."""
        try:
            print("Applying subtitle and logo overlays...")
            
            # Build filter complex for overlays
            filter_parts = []
            input_files = ['-i', input_path]
            
            # Start with the main video
            current_video = '[0:v]'
            input_count = 1
            
            # Add logo overlays if available
            if logo_integration_plan and 'logo_variations' in logo_integration_plan:
                logo_files = logo_integration_plan['logo_variations']
                logo_placements = logo_integration_plan.get('logo_placements', [])
                
                if logo_files and logo_placements:
                    # Use the first available logo variation
                    logo_path = next(iter(logo_files.values()))
                    input_files.extend(['-i', logo_path])
                    
                    # Create logo overlay filters
                    for i, placement in enumerate(logo_placements):
                        position_coords = self._calculate_logo_position(placement.get('position', 'bottom_right_corner'))
                        timing = placement.get('timing', [0, target_duration])
                        opacity = placement.get('opacity', 0.8)
                        
                        if i == 0:
                            # First overlay
                            overlay_filter = (
                                f"{current_video}[{input_count}:v]overlay="
                                f"{position_coords['x']}:{position_coords['y']}:"
                                f"enable='between(t,{timing[0]:.2f},{timing[1]:.2f})':"
                                f"alpha={opacity}[logo{i}]"
                            )
                        else:
                            # Additional overlays
                            overlay_filter = (
                                f"[logo{i-1}][{input_count}:v]overlay="
                                f"{position_coords['x']}:{position_coords['y']}:"
                                f"enable='between(t,{timing[0]:.2f},{timing[1]:.2f})':"
                                f"alpha={opacity}[logo{i}]"
                            )
                        
                        filter_parts.append(overlay_filter)
                    
                    current_video = f'[logo{len(logo_placements)-1}]'
                    input_count += 1
            
            # Add subtitle overlays if available
            if subtitle_segments:
                from core.subtitle_service import subtitle_service
                
                # Generate subtitle overlay filter
                subtitle_filter = subtitle_service.generate_subtitle_overlay_filter(
                    subtitle_segments,
                    style_config=self._get_subtitle_style_config(logo_integration_plan)
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
                    return False
            else:
                # No overlays to apply, just copy
                import shutil
                shutil.copy2(input_path, output_path)
                return True
                
        except Exception as e:
            print(f"Overlay application error: {e}")
            return False
    
    def _get_subtitle_style_config(self, logo_integration_plan: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get subtitle styling configuration based on logo integration."""
        default_style = {
            'fontsize': 28,
            'fontcolor': 'white',
            'bordercolor': 'black',
            'borderw': 3,
            'shadow_color': '0x80000000',
            'shadow_x': 2,
            'shadow_y': 2,
            'position': 'bottom'
        }
        
        # Adapt subtitle style based on logo colors
        if logo_integration_plan and 'brand_color_palette' in logo_integration_plan:
            palette = logo_integration_plan['brand_color_palette']
            primary_color = palette.get('primary', '#FFFFFF')
            
            # Use brand colors for subtle integration
            default_style['bordercolor'] = primary_color
            
            # Adjust position if logo is in bottom area
            logo_placements = logo_integration_plan.get('logo_placements', [])
            for placement in logo_placements:
                if 'bottom' in placement.get('position', ''):
                    default_style['position'] = 'top'
                    break
        
        return default_style
    
    def _apply_logo_overlay(self, video_path: str, logo_info: Dict[str, Any], output_path: str) -> bool:
        """
        Apply logo overlay to video using FFmpeg overlay filter.
        
        Args:
            video_path: Path to input video file
            logo_info: Dictionary containing logo configuration
            output_path: Path to save video with logo overlay
            
        Returns:
            True if logo overlay successful, False otherwise
        """
        try:
            logo_file_path = logo_info.get('logo_file_path')
            if not logo_file_path or not os.path.exists(logo_file_path):
                print(f"Logo file not found: {logo_file_path}")
                return False
            
            # Extract logo positioning parameters
            position = logo_info.get('logo_position', 'top-right')
            size = logo_info.get('logo_size', 'medium')
            opacity = float(logo_info.get('logo_opacity', 0.9))
            
            # Calculate position coordinates based on position string
            position_coords = self._calculate_logo_position(position)
            
            # Calculate logo size based on size parameter
            logo_scale = self._calculate_logo_scale(size)
            
            # Build FFmpeg overlay filter
            overlay_filter = (
                f"[1:v]scale={logo_scale}:-1:flags=lanczos,format=rgba,"
                f"colorchannelmixer=aa={opacity}[logo];"
                f"[0:v][logo]overlay={position_coords['x']}:{position_coords['y']}:format=auto"
            )
            
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-i', logo_file_path,
                '-filter_complex', overlay_filter,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                output_path
            ]
            
            print(f"Applying logo overlay: position={position}, size={size}, opacity={opacity}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("Logo overlay applied successfully")
                return True
            else:
                print(f"Logo overlay failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"Logo overlay error: {e}")
            return False
    
    def _calculate_logo_position(self, position: str) -> Dict[str, str]:
        """Calculate FFmpeg overlay position coordinates."""
        position_map = {
            'top-left': {'x': '20', 'y': '20'},
            'top-right': {'x': 'W-w-20', 'y': '20'},
            'bottom-left': {'x': '20', 'y': 'H-h-20'},
            'bottom-right': {'x': 'W-w-20', 'y': 'H-h-20'},
            'center': {'x': '(W-w)/2', 'y': '(H-h)/2'}
        }
        return position_map.get(position, position_map['top-right'])
    
    def _calculate_logo_scale(self, size: str) -> str:
        """Calculate logo scale for different sizes."""
        # Base on 720p width (720px) for cost-optimized format
        size_map = {
            'small': '72',    # 10% of width
            'medium': '108',  # 15% of width  
            'large': '144'    # 20% of width
        }
        return size_map.get(size, size_map['medium'])
    
    def _concatenate_videos(self, video_files: List[str], output_path: str) -> bool:
        """Concatenate multiple video files into one."""
        try:
            # Create filter complex for concatenation with scaling
            filter_parts = []
            input_files = []
            
            for i, video_file in enumerate(video_files):
                input_files.extend(['-i', video_file])
                # Scale all videos to cost-optimized 720p format
                filter_parts.append(f"[{i}:v]scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280,fps=30,setsar=1[v{i}]")
            
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
                '-crf', '23',
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
        """Adjust video duration to exact target length."""
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
            
            # Calculate speed adjustment
            if abs(current_duration - target_duration) < 0.1:
                # Duration is close enough, just copy
                import shutil
                shutil.copy2(input_path, output_path)
                return True
            
            if current_duration < target_duration:
                # Need to slow down video to stretch it longer
                # If current=15s, target=30s: setpts=2.0*PTS (slows down 2x)
                speed_factor = target_duration / current_duration
                filter_v = f'setpts={speed_factor}*PTS'
                print(f"Stretching video: {current_duration:.2f}s → {target_duration:.2f}s (factor: {speed_factor:.2f}x slower)")
            else:
                # Need to speed up video to make it shorter  
                # If current=45s, target=30s: setpts=0.67*PTS (speeds up 1.5x)
                speed_factor = target_duration / current_duration
                filter_v = f'setpts={speed_factor}*PTS'
                print(f"Compressing video: {current_duration:.2f}s → {target_duration:.2f}s (factor: {1/speed_factor:.2f}x faster)")
            
            # Apply speed adjustment
            cmd = [
                'ffmpeg', '-y', '-i', input_path,
                '-filter:v', filter_v,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                output_path  # Remove -t parameter to let the full stretched video render
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Video duration adjusted to {target_duration}s")
                return True
            else:
                print(f"Duration adjustment failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"Duration adjustment error: {e}")
            return False
    
    def _combine_video_audio(self, video_path: str, audio_path: str, output_path: str, duration: float) -> bool:
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
                
                # If video and audio durations are very close, don't use -shortest
                if abs(video_duration - audio_duration) < 1.0:
                    print("Video and audio durations match - using full length assembly")
                    use_shortest = False
                else:
                    print("Video and audio duration mismatch - using shortest length")
                    use_shortest = True
            else:
                use_shortest = True
            
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-i', audio_path,
                '-c:v', 'copy',
                '-c:a', 'aac', 
                '-b:a', '320k',
                '-map', '0:v:0',
                '-map', '1:a:0'
            ]
            
            # Only add -shortest if durations don't match
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
            if validation['duration_diff'] > 1.0:
                issues.append(f'Duration off by {validation["duration_diff"]:.1f}s')
            if file_size < 1000000:  # Less than 1MB
                issues.append('File size suspiciously small')
            
            if issues:
                validation['valid'] = False
                validation['issues'] = issues
            
            return validation
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
