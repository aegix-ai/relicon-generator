"""
Relicon AI Ad Creator - FFmpeg Service
Professional video assembly with mathematical precision
"""
import subprocess
import json
import time
import asyncio
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from core.settings import settings, derived
from core.models import MasterAdPlan, SceneComponent, GenerationAssets


class FFmpegService:
    """
    FFmpeg Service - Professional video assembly
    
    Handles ultra-precise video assembly, audio synchronization,
    and professional post-processing effects.
    """
    
    def __init__(self):
        self.temp_dir = Path(settings.OUTPUT_DIR) / "temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # FFmpeg quality settings
        self.video_settings = {
            "codec": "libx264",
            "preset": "slow",  # Higher quality
            "crf": "18",       # High quality (lower is better)
            "profile": "high",
            "level": "4.1",
            "pix_fmt": "yuv420p"
        }
        
        self.audio_settings = {
            "codec": "aac",
            "bitrate": "192k",
            "sample_rate": "48000",
            "channels": "2"
        }
    
    async def assemble_final_video(
        self, 
        master_plan: MasterAdPlan, 
        assets: GenerationAssets,
        context: Dict[str, Any]
    ) -> Optional[str]:
        """
        Assemble the final video with mathematical precision
        
        Args:
            master_plan: Ultra-detailed plan with exact timing
            assets: All generated audio, video, and image assets
            context: Brand and technical context
            
        Returns:
            Path to final assembled video or None if failed
        """
        print(f"🎬 FFmpeg: Starting final video assembly ({master_plan.total_duration}s)")
        
        try:
            # Create timeline with exact timing
            timeline = await self._create_precise_timeline(master_plan, assets)
            
            # Generate video segments for each scene
            scene_videos = await self._generate_scene_videos(master_plan, assets, timeline)
            
            # Create master audio track
            master_audio = await self._create_master_audio_track(master_plan, assets, timeline)
            
            # Assemble final video
            final_video = await self._assemble_with_effects(
                scene_videos, 
                master_audio, 
                master_plan, 
                context
            )
            
            # Apply final post-processing
            polished_video = await self._apply_final_polish(final_video, context)
            
            print(f"✅ FFmpeg: Final video assembled - {polished_video}")
            return polished_video
            
        except Exception as e:
            print(f"❌ FFmpeg: Assembly failed - {str(e)}")
            return None
    
    async def _create_precise_timeline(
        self, 
        master_plan: MasterAdPlan, 
        assets: GenerationAssets
    ) -> Dict[str, Any]:
        """Create mathematically precise timeline"""
        timeline = {
            "total_duration": master_plan.total_duration,
            "scenes": [],
            "audio_tracks": [],
            "effects": [],
            "transitions": []
        }
        
        current_time = 0.0
        
        for scene in master_plan.scenes:
            scene_timeline = {
                "scene_id": scene.scene_id,
                "start_time": scene.start_time,
                "duration": scene.duration,
                "end_time": scene.start_time + scene.duration,
                "components": []
            }
            
            # Process each component with exact timing
            for component in scene.components:
                component_timeline = {
                    "component_id": f"{scene.scene_id}_{component.visual_type}_{component.start_time}",
                    "start_time": component.start_time,
                    "duration": component.duration,
                    "end_time": component.end_time,
                    "type": component.visual_type,
                    "video_file": None,
                    "audio_file": None,
                    "effects": {
                        "entry": component.entry_effect,
                        "exit": component.exit_effect,
                        "overlays": component.overlay_effects
                    }
                }
                
                # Map assets to timeline
                if component.generated_asset_path:
                    if component.visual_type == "video":
                        component_timeline["video_file"] = component.generated_asset_path
                    elif component.visual_type in ["image", "logo"]:
                        component_timeline["image_file"] = component.generated_asset_path
                
                scene_timeline["components"].append(component_timeline)
            
            timeline["scenes"].append(scene_timeline)
        
        return timeline
    
    async def _generate_scene_videos(
        self, 
        master_plan: MasterAdPlan, 
        assets: GenerationAssets, 
        timeline: Dict[str, Any]
    ) -> List[str]:
        """Generate individual scene videos with precise timing"""
        scene_videos = []
        
        for scene_data in timeline["scenes"]:
            scene_video = await self._create_scene_video(scene_data, master_plan, assets)
            if scene_video:
                scene_videos.append(scene_video)
        
        return scene_videos
    
    async def _create_scene_video(
        self, 
        scene_data: Dict[str, Any], 
        master_plan: MasterAdPlan, 
        assets: GenerationAssets
    ) -> Optional[str]:
        """Create a single scene video with all components"""
        scene_id = scene_data["scene_id"]
        duration = scene_data["duration"]
        
        print(f"🎬 FFmpeg: Creating scene video - {scene_id} ({duration}s)")
        
        # Create base video (background)
        base_video = await self._create_base_video(scene_data, master_plan)
        if not base_video:
            return None
        
        # Add components layer by layer
        current_video = base_video
        
        for component_data in scene_data["components"]:
            if component_data.get("video_file"):
                # Overlay video component
                current_video = await self._overlay_video_component(
                    current_video, 
                    component_data, 
                    scene_data
                )
            elif component_data.get("image_file"):
                # Overlay image component
                current_video = await self._overlay_image_component(
                    current_video, 
                    component_data, 
                    scene_data
                )
        
        return current_video
    
    async def _create_base_video(
        self, 
        scene_data: Dict[str, Any], 
        master_plan: MasterAdPlan
    ) -> Optional[str]:
        """Create base video for the scene"""
        duration = scene_data["duration"]
        scene_id = scene_data["scene_id"]
        
        # Create filename
        base_video_path = self.temp_dir / f"{scene_id}_base.mp4"
        
        # Get scene from master plan for color information
        scene = next((s for s in master_plan.scenes if s.scene_id == scene_id), None)
        if not scene:
            return None
        
        # Use scene color palette or default
        bg_color = "0x1a1a2e"  # Default dark background
        if scene.color_palette:
            # Convert hex to FFmpeg color format
            hex_color = scene.color_palette[0].replace("#", "")
            bg_color = f"0x{hex_color}"
        
        # Create base video with color background
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=c={bg_color}:s={derived.video_width}x{derived.video_height}:d={duration}",
            "-c:v", self.video_settings["codec"],
            "-preset", self.video_settings["preset"],
            "-crf", self.video_settings["crf"],
            "-pix_fmt", self.video_settings["pix_fmt"],
            str(base_video_path)
        ]
        
        result = await self._run_ffmpeg_command(cmd)
        return str(base_video_path) if result else None
    
    async def _overlay_video_component(
        self, 
        base_video: str, 
        component_data: Dict[str, Any], 
        scene_data: Dict[str, Any]
    ) -> str:
        """Overlay a video component with precise timing and effects"""
        component_start = component_data["start_time"] - scene_data["start_time"]
        component_duration = component_data["duration"]
        video_file = component_data["video_file"]
        
        output_path = self.temp_dir / f"overlay_{int(time.time())}.mp4"
        
        # Build FFmpeg filter for overlay with effects
        filter_complex = []
        
        # Input video processing
        filter_complex.append(f"[1:v]scale={derived.video_width}:{derived.video_height}[scaled]")
        
        # Add entry effect
        entry_effect = component_data["effects"]["entry"]
        if entry_effect == "fade_in":
            filter_complex.append(f"[scaled]fade=t=in:st={component_start}:d=0.5[faded_in]")
            video_input = "[faded_in]"
        else:
            video_input = "[scaled]"
        
        # Add exit effect
        exit_effect = component_data["effects"]["exit"]
        if exit_effect == "fade_out":
            fade_start = component_start + component_duration - 0.5
            filter_complex.append(f"{video_input}fade=t=out:st={fade_start}:d=0.5[faded_out]")
            video_input = "[faded_out]"
        
        # Overlay onto base video
        filter_complex.append(
            f"[0:v]{video_input}overlay=0:0:enable='between(t,{component_start},{component_start + component_duration})'[output]"
        )
        
        cmd = [
            "ffmpeg", "-y",
            "-i", base_video,
            "-i", video_file,
            "-filter_complex", ";".join(filter_complex),
            "-map", "[output]",
            "-c:v", self.video_settings["codec"],
            "-preset", self.video_settings["preset"],
            "-crf", self.video_settings["crf"],
            str(output_path)
        ]
        
        result = await self._run_ffmpeg_command(cmd)
        return str(output_path) if result else base_video
    
    async def _overlay_image_component(
        self, 
        base_video: str, 
        component_data: Dict[str, Any], 
        scene_data: Dict[str, Any]
    ) -> str:
        """Overlay an image component with effects"""
        component_start = component_data["start_time"] - scene_data["start_time"]
        component_duration = component_data["duration"]
        image_file = component_data["image_file"]
        
        output_path = self.temp_dir / f"image_overlay_{int(time.time())}.mp4"
        
        # Determine image position and size based on component type
        if "logo" in component_data["component_id"]:
            # Logo positioning
            overlay_filter = f"overlay=W-w-50:H-h-50:enable='between(t,{component_start},{component_start + component_duration})'"
        else:
            # Full overlay
            overlay_filter = f"overlay=0:0:enable='between(t,{component_start},{component_start + component_duration})'"
        
        cmd = [
            "ffmpeg", "-y",
            "-i", base_video,
            "-i", image_file,
            "-filter_complex", f"[1:v]scale={derived.video_width}:{derived.video_height}[scaled];[0:v][scaled]{overlay_filter}[output]",
            "-map", "[output]",
            "-c:v", self.video_settings["codec"],
            "-preset", self.video_settings["preset"],
            "-crf", self.video_settings["crf"],
            str(output_path)
        ]
        
        result = await self._run_ffmpeg_command(cmd)
        return str(output_path) if result else base_video
    
    async def _create_master_audio_track(
        self, 
        master_plan: MasterAdPlan, 
        assets: GenerationAssets, 
        timeline: Dict[str, Any]
    ) -> Optional[str]:
        """Create master audio track with perfect synchronization"""
        print(f"🎵 FFmpeg: Creating master audio track ({master_plan.total_duration}s)")
        
        # Create silence base track
        master_audio_path = self.temp_dir / f"master_audio_{int(time.time())}.wav"
        
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"anullsrc=channel_layout=stereo:sample_rate=48000:duration={master_plan.total_duration}",
            str(master_audio_path)
        ]
        
        result = await self._run_ffmpeg_command(cmd)
        if not result:
            return None
        
        # Mix in all audio components
        current_audio = str(master_audio_path)
        
        for audio_file in assets.voiceover_files:
            if Path(audio_file).exists():
                current_audio = await self._mix_audio_track(
                    current_audio, 
                    audio_file, 
                    0.0,  # Will be positioned based on component timing
                    1.0   # Full volume for voiceover
                )
        
        # Add background music if available
        if assets.music_files:
            for music_file in assets.music_files:
                if Path(music_file).exists():
                    current_audio = await self._mix_audio_track(
                        current_audio, 
                        music_file, 
                        0.0, 
                        0.3  # Lower volume for background music
                    )
        
        return current_audio
    
    async def _mix_audio_track(
        self, 
        base_audio: str, 
        overlay_audio: str, 
        start_time: float, 
        volume: float
    ) -> str:
        """Mix an audio track into the base audio"""
        output_path = self.temp_dir / f"mixed_audio_{int(time.time())}.wav"
        
        cmd = [
            "ffmpeg", "-y",
            "-i", base_audio,
            "-i", overlay_audio,
            "-filter_complex", f"[1:a]volume={volume},adelay={int(start_time * 1000)}|{int(start_time * 1000)}[delayed];[0:a][delayed]amix=inputs=2:duration=first[output]",
            "-map", "[output]",
            "-c:a", "pcm_s16le",
            str(output_path)
        ]
        
        result = await self._run_ffmpeg_command(cmd)
        return str(output_path) if result else base_audio
    
    async def _assemble_with_effects(
        self, 
        scene_videos: List[str], 
        master_audio: str, 
        master_plan: MasterAdPlan, 
        context: Dict[str, Any]
    ) -> Optional[str]:
        """Assemble final video with professional effects"""
        if not scene_videos:
            return None
        
        print(f"🎬 FFmpeg: Assembling {len(scene_videos)} scenes with effects")
        
        # Create concatenation file
        concat_file = self.temp_dir / f"concat_{int(time.time())}.txt"
        with open(concat_file, "w") as f:
            for video in scene_videos:
                f.write(f"file '{Path(video).resolve()}'\n")
        
        # Concatenate videos
        concatenated_video = self.temp_dir / f"concatenated_{int(time.time())}.mp4"
        
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c:v", self.video_settings["codec"],
            "-preset", self.video_settings["preset"],
            "-crf", self.video_settings["crf"],
            str(concatenated_video)
        ]
        
        result = await self._run_ffmpeg_command(cmd)
        if not result:
            return None
        
        # Add master audio
        final_video = self.temp_dir / f"final_with_audio_{int(time.time())}.mp4"
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(concatenated_video),
            "-i", master_audio,
            "-c:v", "copy",
            "-c:a", self.audio_settings["codec"],
            "-b:a", self.audio_settings["bitrate"],
            "-ar", self.audio_settings["sample_rate"],
            "-ac", self.audio_settings["channels"],
            "-shortest",
            str(final_video)
        ]
        
        result = await self._run_ffmpeg_command(cmd)
        return str(final_video) if result else None
    
    async def _apply_final_polish(
        self, 
        video_path: str, 
        context: Dict[str, Any]
    ) -> Optional[str]:
        """Apply final polish and optimization"""
        brand_name = context.get("brand_name", "brand").lower().replace(" ", "_")
        timestamp = int(time.time())
        
        # Final output path
        output_dir = Path(settings.OUTPUT_DIR) / "final"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        final_path = output_dir / f"{brand_name}_ad_{timestamp}.mp4"
        
        # Apply final polish effects
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", "eq=contrast=1.1:brightness=0.02:saturation=1.1",  # Subtle color enhancement
            "-c:v", self.video_settings["codec"],
            "-preset", "medium",  # Balanced preset for final output
            "-crf", "20",         # Slightly compressed for smaller file size
            "-profile:v", self.video_settings["profile"],
            "-level", self.video_settings["level"],
            "-pix_fmt", self.video_settings["pix_fmt"],
            "-c:a", self.audio_settings["codec"],
            "-b:a", self.audio_settings["bitrate"],
            "-movflags", "+faststart",  # Optimize for web streaming
            str(final_path)
        ]
        
        result = await self._run_ffmpeg_command(cmd)
        return str(final_path) if result else None
    
    async def _run_ffmpeg_command(self, cmd: List[str]) -> bool:
        """Run FFmpeg command asynchronously"""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                return True
            else:
                print(f"❌ FFmpeg error: {stderr.decode()}")
                return False
                
        except Exception as e:
            print(f"❌ FFmpeg execution error: {str(e)}")
            return False
    
    async def create_preview_video(
        self, 
        video_path: str, 
        duration: float = 10.0
    ) -> Optional[str]:
        """Create a preview version of the video"""
        preview_path = video_path.replace(".mp4", "_preview.mp4")
        
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "28",
            "-vf", "scale=640:360",
            "-c:a", "aac",
            "-b:a", "128k",
            preview_path
        ]
        
        result = await self._run_ffmpeg_command(cmd)
        return preview_path if result else None
    
    async def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get detailed video information"""
        cmd = [
            "ffprobe", 
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                return json.loads(stdout.decode())
            else:
                return {}
                
        except Exception as e:
            print(f"❌ FFprobe error: {str(e)}")
            return {}


# Global FFmpeg service instance
ffmpeg_service = FFmpegService() 