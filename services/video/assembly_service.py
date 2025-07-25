"""
Video Assembly Service
Combines multiple video segments with audio using FFmpeg
"""
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List

class VideoAssemblyService:
    def __init__(self):
        self.temp_dir = "/tmp/video_assembly"
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def combine_videos_with_unified_audio(self, video_files: List[str], unified_audio_file: str, output_path: str, target_duration: int) -> bool:
        """
        Combine multiple video segments with ONE continuous audio track
        This ensures no audio jumps and exact duration control
        """
        try:
            print(f"🎬 Assembling {len(video_files)} video segments with unified {target_duration}s audio...")
            
            # Step 1: Concatenate videos first (without audio)
            temp_video_only = os.path.join(self.temp_dir, "concatenated_video_silent.mp4")
            
            if len(video_files) == 1:
                # Single video, just copy
                import shutil
                shutil.copy2(video_files[0], temp_video_only)
            else:
                # Multiple videos, concatenate
                if not self.concatenate_videos(video_files, temp_video_only):
                    return False
            
            # Step 2: Get video duration and trim/extend if needed
            actual_video_duration = self.get_video_duration(temp_video_only)
            print(f"📏 Video duration: {actual_video_duration:.1f}s, Target: {target_duration}s")
            
            # Step 3: Trim or extend video to exact target duration
            temp_video_exact = os.path.join(self.temp_dir, "video_exact_duration.mp4")
            if not self.adjust_video_to_exact_duration(temp_video_only, temp_video_exact, target_duration):
                return False
            
            # Step 4: Add unified audio overlay with exact duration
            if not self.add_unified_audio_overlay(temp_video_exact, unified_audio_file, output_path, target_duration):
                return False
            
            print(f"✅ Final video assembled: {output_path} ({target_duration}s)")
            return True
            
        except Exception as e:
            print(f"❌ Video assembly failed: {e}")
            return False
    
    def get_video_duration(self, video_file: str) -> float:
        """Get video duration in seconds"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', video_file
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return float(result.stdout.strip())
        except:
            return 0.0
    
    def adjust_video_to_exact_duration(self, input_file: str, output_file: str, target_duration: int) -> bool:
        """Adjust video to exact target duration by trimming or looping"""
        try:
            actual_duration = self.get_video_duration(input_file)
            
            if abs(actual_duration - target_duration) < 0.5:
                # Close enough, just trim to exact duration
                cmd = [
                    'ffmpeg', '-y', '-i', input_file,
                    '-t', str(target_duration),
                    '-c', 'copy',
                    output_file
                ]
            elif actual_duration < target_duration:
                # Video too short, loop it
                loops_needed = int(target_duration / actual_duration) + 1
                cmd = [
                    'ffmpeg', '-y', '-stream_loop', str(loops_needed), '-i', input_file,
                    '-t', str(target_duration),
                    '-c', 'copy',
                    output_file
                ]
            else:
                # Video too long, trim it
                cmd = [
                    'ffmpeg', '-y', '-i', input_file,
                    '-t', str(target_duration),
                    '-c', 'copy',
                    output_file
                ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
            
        except Exception as e:
            print(f"Error adjusting video duration: {e}")
            return False
    
    def add_unified_audio_overlay(self, video_file: str, audio_file: str, output_file: str, target_duration: int) -> bool:
        """Add continuous audio overlay to video with exact duration"""
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', video_file,
                '-i', audio_file,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-map', '0:v:0',  # Video from first input
                '-map', '1:a:0',  # Audio from second input
                '-t', str(target_duration),  # Exact duration
                '-af', 'afade=t=out:st={}:d=0.5'.format(target_duration - 0.5),  # Fade out at end
                output_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"FFmpeg audio overlay error: {result.stderr}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error adding audio overlay: {e}")
            return False
    
    def combine_videos_with_audio(self, video_files: List[str], audio_files: List[str], output_path: str) -> bool:
        """
        Combine multiple video segments with corresponding audio files
        """
        try:
            print(f"🎬 Assembling {len(video_files)} video segments with audio...")
            
            if len(video_files) != len(audio_files):
                print(f"Warning: Video count ({len(video_files)}) != Audio count ({len(audio_files)})")
            
            # Step 1: Combine videos with their audio
            video_with_audio_files = []
            for i, (video_file, audio_file) in enumerate(zip(video_files, audio_files)):
                output_segment = os.path.join(self.temp_dir, f"segment_{i}_with_audio.mp4")
                
                if self.add_audio_to_video(video_file, audio_file, output_segment):
                    video_with_audio_files.append(output_segment)
                else:
                    print(f"Failed to add audio to segment {i}")
                    return False
            
            # Step 2: Concatenate all segments
            if len(video_with_audio_files) == 1:
                # Single segment, just copy
                subprocess.run(['cp', video_with_audio_files[0], output_path])
            else:
                # Multiple segments, concatenate
                self.concatenate_videos(video_with_audio_files, output_path)
            
            print(f"✅ Final video assembled: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Video assembly failed: {e}")
            return False
    
    def add_audio_to_video(self, video_file: str, audio_file: str, output_file: str) -> bool:
        """
        Add audio track to video file
        """
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', video_file,  # Video input
                '-i', audio_file,  # Audio input
                '-c:v', 'copy',    # Copy video without re-encoding
                '-c:a', 'aac',     # Encode audio as AAC
                '-map', '0:v:0',   # Use video from first input
                '-map', '1:a:0',   # Use audio from second input
                '-shortest',       # End when shortest stream ends
                output_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"FFmpeg error: {result.stderr}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error adding audio to video: {e}")
            return False
    
    def concatenate_videos(self, video_files: List[str], output_file: str) -> bool:
        """
        Concatenate multiple video files
        """
        try:
            # Create concat file list
            concat_file = os.path.join(self.temp_dir, "concat_list.txt")
            with open(concat_file, 'w') as f:
                for video_file in video_files:
                    f.write(f"file '{video_file}'\n")
            
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_file,
                '-c', 'copy',
                output_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"FFmpeg concatenation error: {result.stderr}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error concatenating videos: {e}")
            return False
    
    def optimize_for_social_media(self, input_file: str, output_file: str, aspect_ratio: str = "9:16") -> bool:
        """
        Optimize video for social media platforms
        """
        try:
            print(f"📱 Optimizing video for social media ({aspect_ratio})...")
            
            cmd = [
                'ffmpeg', '-y',
                '-i', input_file,
                '-vf', f'scale=-2:1920,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black',  # 9:16 aspect ratio
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-c:a', 'aac',
                '-b:a', '128k',
                output_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Social media optimization error: {result.stderr}")
                return False
            
            print(f"✅ Video optimized for social media: {output_file}")
            return True
            
        except Exception as e:
            print(f"Error optimizing video: {e}")
            return False