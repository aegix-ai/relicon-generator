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
        try:
            self.elevenlabs = ElevenLabsProvider()
        except Exception as e:
            logger.warning(f"ElevenLabs provider not available: {e}")
            self.elevenlabs = None
        
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
                if self.elevenlabs:
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
        """Generate professional FFmpeg subtitle filter with enterprise-grade styling."""
        if not segments:
            return ""
        
        # Professional subtitle styling optimized for 9:16 vertical videos
        default_style = {
            'fontsize': 42,  # Larger for better readability on mobile/vertical screens
            'fontcolor': 'white',
            'bordercolor': 'black',
            'borderw': 5,  # Thicker border for better contrast
            'shadow_color': '0x80000000',
            'shadow_x': 4,
            'shadow_y': 4,
            'position': 'bottom',  # Default to bottom for vertical videos
            'background_opacity': 0.8,  # More opaque background for better readability
            'background_color': '0x80000000',
            'line_spacing': 10,  # More line spacing for readability
            'max_width': 0.90  # 90% of video width for better utilization
        }
        
        if style_config:
            default_style.update(style_config)
        
        # Dynamic font selection with fallbacks
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/System/Library/Fonts/Arial.ttf",  # macOS
            "/Windows/Fonts/arial.ttf",  # Windows
            "/usr/share/fonts/TTF/arial.ttf",  # Linux alternative
            # Fallback to default
            ""
        ]
        
        selected_font = ""
        import os
        for font_path in font_paths:
            if font_path == "" or os.path.exists(font_path):
                selected_font = font_path
                break
        
        # Position calculation optimized for 9:16 vertical videos
        # Position subtitles in the lower third but not too close to bottom
        if default_style['position'] == 'bottom':
            # For 9:16 vertical videos, position in lower third but not at very bottom
            y_position = 'h*0.75'  # 75% down the video (middle-bottom area)
        elif default_style['position'] == 'top':
            y_position = 'h*0.15'  # 15% from top (not too close to edge)
        else:  # center
            y_position = 'h*0.5'  # True center
        
        # Calculate maximum text width
        max_width_pixels = f"w*{default_style['max_width']}"
        
        # Create professional drawtext filters for each subtitle segment
        filters = []
        
        for segment in segments:
            # Professional text escaping and formatting
            text = segment.text.strip()
            
            # Break long lines for better readability
            words = text.split()
            formatted_lines = []
            current_line = []
            max_words_per_line = 8
            
            for word in words:
                current_line.append(word)
                if len(current_line) >= max_words_per_line:
                    formatted_lines.append(" ".join(current_line))
                    current_line = []
            
            if current_line:
                formatted_lines.append(" ".join(current_line))
            
            # Join lines with newline character
            formatted_text = "\\n".join(formatted_lines)
            
            # Escape special characters for FFmpeg
            escaped_text = (formatted_text
                          .replace('\\', '\\\\')
                          .replace('"', '\\"')
                          .replace("'", "\\'")
                          .replace(":", "\\:")
                          .replace(",", "\\,")
                          .replace("%", "\\%"))
            
            # Create professional drawtext filter with background box
            font_directive = f"fontfile={selected_font}:" if selected_font else ""
            
            filter_str = (
                f'drawtext=text="{escaped_text}":'
                f"{font_directive}"
                f"fontsize={default_style['fontsize']}:"
                f"fontcolor={default_style['fontcolor']}:"
                f"bordercolor={default_style['bordercolor']}:"
                f"borderw={default_style['borderw']}:"
                f"shadowcolor={default_style['shadow_color']}:"
                f"shadowx={default_style['shadow_x']}:"
                f"shadowy={default_style['shadow_y']}:"
                f"box=1:"
                f"boxcolor={default_style['background_color']}:"
                f"boxborderw=10:"
                f"x=(w-text_w)/2:"
                f"y={y_position}:"
                f"line_spacing={default_style['line_spacing']}:"
                f"enable='between(t,{segment.start_time:.3f},{segment.end_time:.3f})'"
            )
            filters.append(filter_str)
        
        return ','.join(filters) if filters else ""


    def generate_subtitles_from_audio(self, audio_path: str, script_text: str = None) -> List[SubtitleSegment]:
        """
        Generate synchronized subtitles by analyzing audio with script alignment.
        
        Args:
            audio_path: Path to the generated audio file
            script_text: Original script text for accurate content
            
        Returns:
            List of synchronized subtitle segments with proper text content
        """
        try:
            logger.info("Starting professional audio-based subtitle generation", action="subtitle.audio.generation.start")
            
            if not os.path.exists(audio_path):
                logger.error("Audio file not found for subtitle generation")
                return []
            
            # Get precise timing from audio analysis
            timing_segments = self._transcribe_audio_with_timing(audio_path)
            
            if script_text and timing_segments:
                # Align script content with audio timing for perfect synchronization
                subtitle_segments = self._align_script_with_timing(script_text, timing_segments)
            else:
                # Use timing segments as fallback
                subtitle_segments = timing_segments
            
            logger.info(
                f"Generated {len(subtitle_segments)} professionally synchronized subtitle segments",
                action="subtitle.audio.generation.complete",
                segment_count=len(subtitle_segments)
            )
            
            return subtitle_segments
            
        except Exception as e:
            logger.error(f"Audio-based subtitle generation failed: {e}", exc_info=True)
            return []

    def _transcribe_audio_with_timing(self, audio_path: str) -> List[SubtitleSegment]:
        """Transcribe audio using FFmpeg for precise timing extraction."""
        try:
            import subprocess
            import json
            
            # Use FFmpeg with silencedetect to find natural speech breaks
            silence_cmd = [
                'ffmpeg', '-i', audio_path, '-af', 
                'silencedetect=noise=-30dB:d=0.3', 
                '-f', 'null', '-'
            ]
            
            silence_result = subprocess.run(silence_cmd, capture_output=True, text=True)
            
            # Extract silence points from FFmpeg output
            silence_points = self._parse_silence_detection(silence_result.stderr)
            
            # Get audio duration
            duration_cmd = [
                'ffprobe', '-i', audio_path, '-show_entries', 
                'format=duration', '-v', 'quiet', '-of', 'csv=p=0'
            ]
            
            duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)
            total_duration = float(duration_result.stdout.strip())
            
            # Since we don't have the original script here, we'll create segments based on 
            # natural speech breaks for precise timing
            segments = self._create_segments_from_silence(silence_points, total_duration)
            
            logger.info(f"Created {len(segments)} audio-synchronized segments")
            return segments
            
        except Exception as e:
            logger.error(f"Audio transcription with timing failed: {e}")
            # Fallback to time-based segments
            return self._create_basic_timing_segments(audio_path)
    
    def _parse_silence_detection(self, ffmpeg_output: str) -> List[Tuple[float, float]]:
        """Parse FFmpeg silence detection output to find speech segments."""
        import re
        
        silence_points = []
        silence_start_pattern = r'silence_start: ([\d.]+)'
        silence_end_pattern = r'silence_end: ([\d.]+)'
        
        starts = re.findall(silence_start_pattern, ffmpeg_output)
        ends = re.findall(silence_end_pattern, ffmpeg_output)
        
        # Pair up silence starts and ends
        for i in range(min(len(starts), len(ends))):
            start_time = float(starts[i])
            end_time = float(ends[i])
            silence_points.append((start_time, end_time))
        
        return silence_points
    
    def _create_segments_from_silence(self, silence_points: List[Tuple[float, float]], total_duration: float) -> List[SubtitleSegment]:
        """Create subtitle segments based on detected silence gaps."""
        segments = []
        
        if not silence_points:
            # No silence detected, create basic segments
            return self._create_fallback_segments("", total_duration, 1, 0.0)
        
        # Create segments between silence points
        current_time = 0.0
        segment_count = 1
        
        for silence_start, silence_end in silence_points:
            if silence_start > current_time + 0.5:  # Minimum segment length
                segments.append(SubtitleSegment(
                    text=f"Segment {segment_count}",  # Placeholder text
                    start_time=current_time,
                    end_time=silence_start,
                    scene_number=1,
                    confidence=0.8
                ))
                segment_count += 1
            
            current_time = silence_end
        
        # Add final segment if there's remaining audio
        if current_time < total_duration - 0.5:
            segments.append(SubtitleSegment(
                text=f"Segment {segment_count}",
                start_time=current_time,
                end_time=total_duration,
                scene_number=1,
                confidence=0.8
            ))
        
        return segments
    
    def _create_basic_timing_segments(self, audio_path: str) -> List[SubtitleSegment]:
        """Create basic time-based segments when audio analysis fails."""
        try:
            import subprocess
            
            # Get audio duration
            duration_cmd = [
                'ffprobe', '-i', audio_path, '-show_entries', 
                'format=duration', '-v', 'quiet', '-of', 'csv=p=0'
            ]
            
            result = subprocess.run(duration_cmd, capture_output=True, text=True)
            total_duration = float(result.stdout.strip())
            
            # Create 3-4 second segments for better readability
            segment_duration = 3.5
            segments = []
            current_time = 0.0
            segment_count = 1
            
            while current_time < total_duration:
                end_time = min(current_time + segment_duration, total_duration)
                
                segments.append(SubtitleSegment(
                    text=f"Speech segment {segment_count}",
                    start_time=current_time,
                    end_time=end_time,
                    scene_number=1,
                    confidence=0.6
                ))
                
                current_time = end_time
                segment_count += 1
            
            return segments
            
        except Exception as e:
            logger.error(f"Basic timing segments creation failed: {e}")
            return []
    
    def _align_script_with_timing(self, script_text: str, timing_segments: List[SubtitleSegment]) -> List[SubtitleSegment]:
        """Align script content with precise audio timing segments."""
        try:
            # Clean and prepare script text
            script_text = script_text.strip()
            if not script_text:
                return timing_segments
            
            # Split script into sentences for alignment
            sentences = self._split_script_into_sentences(script_text)
            
            if not sentences:
                return timing_segments
            
            # Align sentences with timing segments
            aligned_segments = []
            
            # If we have more timing segments than sentences, group timing segments
            if len(timing_segments) > len(sentences):
                segments_per_sentence = len(timing_segments) // len(sentences)
                remainder = len(timing_segments) % len(sentences)
                
                sentence_idx = 0
                segment_idx = 0
                
                for sentence in sentences:
                    # Calculate how many timing segments to use for this sentence
                    segments_to_use = segments_per_sentence
                    if sentence_idx < remainder:
                        segments_to_use += 1
                    
                    if segments_to_use > 0 and segment_idx < len(timing_segments):
                        # Use the start time of first segment and end time of last segment
                        start_time = timing_segments[segment_idx].start_time
                        end_segment_idx = min(segment_idx + segments_to_use - 1, len(timing_segments) - 1)
                        end_time = timing_segments[end_segment_idx].end_time
                        
                        aligned_segments.append(SubtitleSegment(
                            text=sentence.strip(),
                            start_time=start_time,
                            end_time=end_time,
                            scene_number=1,
                            confidence=0.95
                        ))
                        
                        segment_idx += segments_to_use
                    
                    sentence_idx += 1
            
            # If we have more sentences than timing segments, distribute evenly
            elif len(sentences) > len(timing_segments):
                total_duration = timing_segments[-1].end_time if timing_segments else 18.0
                sentence_duration = total_duration / len(sentences)
                
                for i, sentence in enumerate(sentences):
                    start_time = i * sentence_duration
                    end_time = min((i + 1) * sentence_duration, total_duration)
                    
                    aligned_segments.append(SubtitleSegment(
                        text=sentence.strip(),
                        start_time=start_time,
                        end_time=end_time,
                        scene_number=1,
                        confidence=0.9
                    ))
            
            # Equal number of sentences and timing segments - perfect alignment
            else:
                for sentence, timing_segment in zip(sentences, timing_segments):
                    aligned_segments.append(SubtitleSegment(
                        text=sentence.strip(),
                        start_time=timing_segment.start_time,
                        end_time=timing_segment.end_time,
                        scene_number=timing_segment.scene_number,
                        confidence=1.0
                    ))
            
            return aligned_segments if aligned_segments else timing_segments
            
        except Exception as e:
            logger.error(f"Script alignment failed: {e}")
            return timing_segments
    
    def _split_script_into_sentences(self, script_text: str) -> List[str]:
        """Split script into natural sentence segments for subtitle alignment."""
        import re
        
        # Split on sentence endings but preserve them
        sentences = re.split(r'([.!?]+)', script_text)
        
        # Rejoin sentence endings with their sentences
        cleaned_sentences = []
        current_sentence = ""
        
        for i, part in enumerate(sentences):
            if re.match(r'^[.!?]+$', part):
                # This is punctuation, add to current sentence
                current_sentence += part
                if current_sentence.strip():
                    cleaned_sentences.append(current_sentence.strip())
                current_sentence = ""
            else:
                # This is text content
                current_sentence += part
        
        # Handle any remaining text
        if current_sentence.strip():
            cleaned_sentences.append(current_sentence.strip())
        
        # Filter out empty sentences and very short ones
        meaningful_sentences = []
        for sentence in cleaned_sentences:
            if len(sentence.split()) >= 3:  # At least 3 words
                meaningful_sentences.append(sentence)
        
        # If no meaningful sentences found, split by commas as fallback
        if not meaningful_sentences:
            comma_splits = [s.strip() for s in script_text.split(',') if s.strip()]
            meaningful_sentences = [s for s in comma_splits if len(s.split()) >= 2]
        
        return meaningful_sentences

# Global subtitle service instance
subtitle_service = SubtitleService()