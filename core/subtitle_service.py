"""
ElevenLabs-Integrated Subtitle Generation Service
Uses ElevenLabs TTS alignment data for perfect synchronization with generated audio.
"""

import os
import json
import tempfile
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from core.logger import get_logger
from providers.elevenlabs import ElevenLabsProvider

logger = get_logger(__name__)

@dataclass
class SubtitleSegment:
    """Represents a single subtitle segment with timing."""
    text: str
    start_time: float
    end_time: float
    scene_number: int
    confidence: float = 1.0

class SubtitleService:
    """ElevenLabs-integrated subtitle generation with perfect audio synchronization."""
    
    def __init__(self):
        self.elevenlabs = ElevenLabsProvider()
        
    def generate_subtitles_from_script(self, architecture: Dict[str, Any]) -> List[SubtitleSegment]:
        """
        Generate synchronized subtitles using ElevenLabs TTS alignment.
        
        Args:
            architecture: Video architecture containing script and scene timing
            
        Returns:
            List of synchronized subtitle segments
        """
        try:
            logger.info("Starting ElevenLabs subtitle generation", action="subtitle.generation.start")
            
            script = architecture.get('unified_script', '')
            scenes = architecture.get('scene_architecture', {}).get('scenes', [])
            audio_config = architecture.get('audio_architecture', {})
            
            if not script or not scenes:
                logger.error("Missing script or scenes in architecture")
                return []
            
            # Generate subtitles with ElevenLabs alignment
            subtitle_segments = self._generate_with_elevenlabs_alignment(
                script, scenes, audio_config
            )
            
            logger.info(
                f"Generated {len(subtitle_segments)} subtitle segments",
                action="subtitle.generation.complete",
                segment_count=len(subtitle_segments)
            )
            
            return subtitle_segments
            
        except Exception as e:
            logger.error(f"Subtitle generation failed: {e}", exc_info=True)
            return []
    
    def _generate_with_elevenlabs_alignment(self, script: str, scenes: List[Dict], audio_config: Dict) -> List[SubtitleSegment]:
        """Use ElevenLabs TTS with word-level timing for perfect alignment."""
        try:
            # Split script by scenes for better timing accuracy
            scene_scripts = self._split_script_by_scenes(script, scenes)
            
            subtitle_segments = []
            current_time = 0.0
            
            for scene_idx, (scene, scene_script) in enumerate(zip(scenes, scene_scripts)):
                if not scene_script.strip():
                    continue
                
                scene_duration = scene.get('duration', 6.0)
                
                # Get ElevenLabs alignment data for this scene
                alignment_data = self._get_elevenlabs_alignment(
                    scene_script, audio_config, scene_duration
                )
                
                if alignment_data:
                    # Convert alignment to subtitle segments
                    scene_segments = self._alignment_to_segments(
                        alignment_data, scene_idx + 1, current_time
                    )
                    subtitle_segments.extend(scene_segments)
                else:
                    # Fallback to time-based distribution
                    fallback_segments = self._create_fallback_segments(
                        scene_script, scene_duration, scene_idx + 1, current_time
                    )
                    subtitle_segments.extend(fallback_segments)
                
                current_time += scene_duration
            
            return subtitle_segments
            
        except Exception as e:
            logger.error(f"ElevenLabs alignment generation failed: {e}")
            return []
    
    def _get_elevenlabs_alignment(self, text: str, audio_config: Dict, duration: float) -> Optional[Dict]:
        """Get word-level alignment data from ElevenLabs TTS."""
        try:
            # Use ElevenLabs TTS with alignment enabled
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_audio = os.path.join(temp_dir, "temp_audio.mp3")
                
                # Generate audio with alignment data
                success = self.elevenlabs.generate_audio_with_alignment(
                    text, temp_audio, audio_config
                )
                
                if success and hasattr(self.elevenlabs, 'last_alignment_data'):
                    return self.elevenlabs.last_alignment_data
                
                return None
                
        except Exception as e:
            logger.warning(f"Failed to get ElevenLabs alignment: {e}")
            return None
    
    def _alignment_to_segments(self, alignment_data: Dict, scene_number: int, scene_start_time: float) -> List[SubtitleSegment]:
        """Convert ElevenLabs alignment data to subtitle segments."""
        segments = []
        
        # Process word-level alignment
        words = alignment_data.get('words', [])
        if not words:
            return segments
        
        # Group words into phrases for better readability
        current_phrase = []
        current_start = None
        
        for word_data in words:
            word = word_data.get('word', '')
            start_time = word_data.get('start', 0.0) + scene_start_time
            end_time = word_data.get('end', 0.0) + scene_start_time
            
            if current_start is None:
                current_start = start_time
            
            current_phrase.append(word)
            
            # Break phrases at natural points (punctuation or length)
            if (word.endswith('.') or word.endswith('!') or word.endswith('?') or 
                len(current_phrase) >= 6):
                
                phrase_text = ' '.join(current_phrase).strip()
                
                segments.append(SubtitleSegment(
                    text=phrase_text,
                    start_time=current_start,
                    end_time=end_time,
                    scene_number=scene_number,
                    confidence=1.0
                ))
                
                current_phrase = []
                current_start = None
        
        # Handle remaining words
        if current_phrase:
            phrase_text = ' '.join(current_phrase).strip()
            last_word = words[-1]
            
            segments.append(SubtitleSegment(
                text=phrase_text,
                start_time=current_start or scene_start_time,
                end_time=last_word.get('end', 0.0) + scene_start_time,
                scene_number=scene_number,
                confidence=1.0
            ))
        
        return segments
    
    def _create_fallback_segments(self, text: str, duration: float, scene_number: int, start_time: float) -> List[SubtitleSegment]:
        """Create time-based subtitle segments as fallback."""
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []
        
        segments = []
        sentence_duration = duration / len(sentences)
        
        for i, sentence in enumerate(sentences):
            segment_start = start_time + (i * sentence_duration)
            segment_end = segment_start + sentence_duration
            
            segments.append(SubtitleSegment(
                text=sentence.strip(),
                start_time=segment_start,
                end_time=segment_end,
                scene_number=scene_number,
                confidence=0.8  # Lower confidence for fallback
            ))
        
        return segments
    
    def _split_script_by_scenes(self, script: str, scenes: List[Dict]) -> List[str]:
        """Split unified script into scene-based parts."""
        if len(scenes) <= 1:
            return [script]
        
        # Extract scene scripts from architecture if available
        scene_scripts = []
        for scene in scenes:
            scene_script = scene.get('script_line', '')
            if scene_script:
                scene_scripts.append(scene_script)
        
        # Fallback to even distribution if no scene scripts
        if len(scene_scripts) != len(scenes):
            sentences = script.split('.')
            sentences = [s.strip() + '.' for s in sentences if s.strip()]
            
            scene_scripts = []
            sentences_per_scene = max(1, len(sentences) // len(scenes))
            
            for i in range(len(scenes)):
                start_idx = i * sentences_per_scene
                end_idx = start_idx + sentences_per_scene if i < len(scenes) - 1 else len(sentences)
                scene_scripts.append(' '.join(sentences[start_idx:end_idx]))
        
        return scene_scripts
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for subtitle timing."""
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def export_subtitles_srt(self, segments: List[SubtitleSegment], output_path: str) -> bool:
        """Export subtitles in SRT format."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, segment in enumerate(segments, 1):
                    start_time = self._format_srt_time(segment.start_time)
                    end_time = self._format_srt_time(segment.end_time)
                    
                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{segment.text}\n\n")
            
            logger.info(f"SRT subtitles exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export SRT subtitles: {e}")
            return False
    
    def _format_srt_time(self, seconds: float) -> str:
        """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
    def generate_subtitle_overlay_filter(self, segments: List[SubtitleSegment], style_config: Dict[str, Any] = None) -> str:
        """Generate FFmpeg subtitle filter for video overlay with style customization."""
        if not segments:
            return ""
        
        # Default style configuration
        default_style = {
            'fontsize': 28,
            'fontcolor': 'white',
            'bordercolor': 'black',
            'borderw': 3,
            'shadow_color': '0x80000000',
            'shadow_x': 2,
            'shadow_y': 2,
            'position': 'bottom'  # bottom, top, center
        }
        
        if style_config:
            default_style.update(style_config)
        
        # Position calculation
        if default_style['position'] == 'bottom':
            y_position = 'h-text_h-60'
        elif default_style['position'] == 'top':
            y_position = '60'
        else:  # center
            y_position = '(h-text_h)/2'
        
        # Create drawtext filters for each subtitle segment
        filters = []
        
        for segment in segments:
            # Escape text for FFmpeg
            escaped_text = segment.text.replace("'", "\\'").replace(":", "\\:").replace(",", "\\,")
            
            # Create drawtext filter with enhanced styling
            filter_str = (
                f"drawtext=text='{escaped_text}':"
                f"fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
                f"fontsize={default_style['fontsize']}:"
                f"fontcolor={default_style['fontcolor']}:"
                f"bordercolor={default_style['bordercolor']}:"
                f"borderw={default_style['borderw']}:"
                f"shadowcolor={default_style['shadow_color']}:"
                f"shadowx={default_style['shadow_x']}:"
                f"shadowy={default_style['shadow_y']}:"
                f"x=(w-text_w)/2:"
                f"y={y_position}:"
                f"enable='between(t,{segment.start_time:.2f},{segment.end_time:.2f})'"
            )
            filters.append(filter_str)
        
        return ','.join(filters) if filters else ""


# Global subtitle service instance
subtitle_service = SubtitleService()