"""
Advanced Subtitle Service - Reels-Optimized with Perfect Audio Sync
Designed for 9:16 aspect ratio with black background, white text, and millisecond-perfect audio synchronization.
"""

import os
import json
import tempfile
import subprocess
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from core.logger import get_logger

logger = get_logger(__name__)

class AnimationType(Enum):
    """Subtitle animation types."""
    FADE = "fade"
    SLIDE_UP = "slide_up"  
    SLIDE_DOWN = "slide_down"
    SLIDE_LEFT = "slide_left"
    SLIDE_RIGHT = "slide_right"
    BOUNCE = "bounce"
    ZOOM_IN = "zoom_in"
    TYPEWRITER = "typewriter"
    WORD_BY_WORD = "word_by_word"

class HighlightStyle(Enum):
    """Word highlight styles."""
    COLOR_CHANGE = "color_change"
    GRADIENT = "gradient" 
    OUTLINE = "outline"
    BACKGROUND = "background"
    SCALE_UP = "scale_up"
    GLOW = "glow"

@dataclass 
class AnimatedWord:
    """Individual word with animation timing and effects."""
    text: str
    start_ms: int
    end_ms: int
    highlight: bool = False
    highlight_style: HighlightStyle = HighlightStyle.COLOR_CHANGE
    highlight_color: str = "#FFD700"  # Gold
    animation: AnimationType = AnimationType.FADE
    delay_ms: int = 0  # Animation delay from segment start

@dataclass
class MusicSyncPoint:
    """Music synchronization point for subtitle timing."""
    time_ms: int
    beat_strength: float  # 0.0 to 1.0
    is_emphasis: bool = False  # Major beat/emphasis point
    bpm_at_point: Optional[float] = None

@dataclass
class PreciseSubtitleSegment:
    """Millisecond-precise subtitle segment with animation support."""
    text: str
    start_ms: int  # Milliseconds for precision
    end_ms: int
    scene_number: int
    word_timings: List[Tuple[str, int, int]] = field(default_factory=list)  # Legacy format
    animated_words: List[AnimatedWord] = field(default_factory=list)  # Enhanced format
    confidence: float = 1.0
    animation_type: AnimationType = AnimationType.WORD_BY_WORD
    music_sync_points: List[MusicSyncPoint] = field(default_factory=list)
    is_emphasis_segment: bool = False  # For highlighting important phrases

@dataclass
class ModernSubtitleStyle:
    """Modern animated subtitle styling for social media."""
    # Position - centered and prominent for 9:16 mobile viewing
    x: str = "(w-text_w)/2"  # Perfect center horizontally
    y: str = "h*0.68"  # Positioned for 9:16 aspect ratio with smaller font
    
    # Typography - bold and modern, optimized for 9:16 mobile viewing
    fontsize: int = 32  # Smaller for 9:16 mobile screens - appropriate for phone viewing
    fontfile: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    fontcolor: str = "white"
    
    # Dynamic background - modern style optimized for 9:16
    box: int = 1
    boxcolor: str = "black@0.7"  # Slightly more transparent
    boxborderw: int = 8  # Compact padding for smaller 9:16 mobile font
    
    # Modern text effects - scaled for smaller 9:16 mobile font
    enable_outline: bool = True
    outline_color: str = "black"
    outline_width: int = 2  # Proportional outline for smaller font
    enable_shadow: bool = True
    shadow_color: str = "black@0.5"  # Moderate shadow
    shadow_offset: str = "2:2"  # Proportional shadow for smaller font
    
    # Animation settings
    word_by_word: bool = True
    animation_type: AnimationType = AnimationType.WORD_BY_WORD
    word_delay_ms: int = 150  # Time between words
    entrance_animation: AnimationType = AnimationType.SLIDE_UP
    exit_animation: AnimationType = AnimationType.FADE
    
    # Highlight settings for important words
    highlight_keywords: List[str] = field(default_factory=lambda: [
        "new", "amazing", "incredible", "perfect", "best", "free", "save", "now",
        "today", "discover", "transform", "revolutionize", "breakthrough"
    ])
    highlight_color: str = "#FFD700"  # Gold
    highlight_style: HighlightStyle = HighlightStyle.GLOW
    
    # Music sync
    enable_beat_sync: bool = True
    beat_emphasis_scale: float = 1.1  # Scale text on beats
    beat_emphasis_duration: float = 0.1  # 100ms emphasis

class AdvancedSubtitleService:
    """Advanced subtitle service with millisecond precision and reels optimization."""
    
    def __init__(self, use_modern_style: bool = True):
        self.modern_style = ModernSubtitleStyle() if use_modern_style else None
        self.legacy_style = None  # For backward compatibility
        
    def generate_precise_subtitles(self, 
                                 audio_file_path: str, 
                                 script_text: str,
                                 scene_timings: List[Dict[str, Any]]) -> List[PreciseSubtitleSegment]:
        
        # ADDED: Validate scene timing consistency before subtitle generation
        if scene_timings:
            total_scene_duration = sum(scene.get('duration', 6.0) for scene in scene_timings)
            print(f"📝 Subtitle timing: Total scene duration = {total_scene_duration:.2f}s across {len(scene_timings)} scenes")
            
            # Log per-scene durations for debugging
            for i, scene in enumerate(scene_timings):
                duration = scene.get('duration', 6.0)
                print(f"📝 Scene {i+1}: {duration:.2f}s duration")
        """
        Generate millisecond-precise subtitles using audio analysis.
        
        Args:
            audio_file_path: Path to the generated audio file
            script_text: Full script text
            scene_timings: Scene timing breakdown
            
        Returns:
            List of precise subtitle segments with word-level timing
        """
        try:
            logger.info("Generating precise subtitles with audio alignment", 
                       action="subtitle.precise.start")
            
            # CRITICAL: First detect actual audio duration for perfect timing
            actual_audio_duration = self._get_precise_audio_duration(audio_file_path)
            print(f"📝 CRITICAL: Actual audio duration detected = {actual_audio_duration:.3f}s")
            
            # Step 1: Extract audio timing using Whisper or similar
            word_timings = self._extract_word_timings_from_audio(audio_file_path, script_text)
            
            # Step 2: Validate and adjust word timings to match actual audio duration
            if word_timings:
                last_word_end = word_timings[-1][2] / 1000.0  # Convert to seconds
                if abs(last_word_end - actual_audio_duration) > 0.5:  # >500ms difference
                    print(f"📝 TIMING ADJUSTMENT: Word timings end at {last_word_end:.2f}s, audio is {actual_audio_duration:.2f}s")
                    word_timings = self._adjust_word_timings_to_audio_duration(word_timings, actual_audio_duration)
            
            # Step 3: Group words into subtitle segments
            subtitle_segments = self._create_segments_from_word_timings(word_timings, scene_timings)
            
            # Step 3: Optimize for reels viewing (break long sentences, etc.)
            optimized_segments = self._optimize_for_reels(subtitle_segments)
            
            logger.info(f"Generated {len(optimized_segments)} precise subtitle segments",
                       action="subtitle.precise.complete")
            
            return optimized_segments
            
        except Exception as e:
            logger.error(f"Precise subtitle generation failed: {e}", exc_info=True)
            # Fallback to basic timing with actual audio duration
            actual_audio_duration = self._get_precise_audio_duration(audio_file_path)
            return self._generate_fallback_subtitles_with_duration(script_text, scene_timings, actual_audio_duration)
    
    def _get_precise_audio_duration(self, audio_path: str) -> float:
        """Get precise audio duration using ffprobe for perfect subtitle sync."""
        try:
            if not audio_path or not os.path.exists(audio_path):
                print(f"📝 Audio file not found: {audio_path}")
                return 18.0  # Fallback to standard duration
            
            cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", 
                   "-of", "csv=p=0", audio_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and result.stdout.strip():
                duration = float(result.stdout.strip())
                print(f"📝 Precise audio duration: {duration:.3f}s from {audio_path}")
                return duration
            else:
                print(f"📝 Failed to get audio duration: {result.stderr}")
                return 18.0
        except Exception as e:
            print(f"📝 Audio duration detection error: {e}")
            return 18.0
    
    def _adjust_word_timings_to_audio_duration(self, word_timings: List[Tuple[str, int, int]], 
                                             target_duration: float) -> List[Tuple[str, int, int]]:
        """Adjust word timings to fit exact audio duration for perfect sync."""
        if not word_timings:
            return word_timings
        
        # Calculate current timing span
        current_duration_ms = word_timings[-1][2]  # End time of last word
        target_duration_ms = int(target_duration * 1000)
        
        # Calculate scaling factor with 2% buffer at end
        scale_factor = (target_duration_ms * 0.98) / current_duration_ms if current_duration_ms > 0 else 1.0
        
        print(f"📝 Adjusting word timings: scale_factor = {scale_factor:.3f} "
              f"({current_duration_ms}ms → {target_duration_ms}ms)")
        
        # Apply scaling to all word timings
        adjusted_timings = []
        for word, start_ms, end_ms in word_timings:
            new_start = int(start_ms * scale_factor)
            new_end = int(end_ms * scale_factor)
            adjusted_timings.append((word, new_start, new_end))
        
        return adjusted_timings
    
    def _extract_word_timings_from_audio(self, audio_path: str, script_text: str) -> List[Tuple[str, int, int]]:
        """Extract word-level timing from audio using speech recognition."""
        try:
            # Check if Whisper is available
            whisper_check = subprocess.run(["which", "whisper"], capture_output=True, text=True)
            if whisper_check.returncode != 0:
                logger.info("Whisper not available, using fallback timing method")
                return self._generate_fallback_word_timings(script_text, audio_path)
            
            # Use Whisper for high-accuracy word-level timing
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                output_file = os.path.join(temp_dir, "whisper_output")
                
                cmd = [
                    "whisper", audio_path,
                    "--model", "base",
                    "--output_format", "json",
                    "--word_timestamps", "True",
                    "--language", "en",
                    "--output_dir", temp_dir,
                    "--verbose", "False"
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                
                if result.returncode != 0:
                    logger.warning(f"Whisper failed with return code {result.returncode}, using fallback timing method")
                    return self._generate_fallback_word_timings(script_text, audio_path)
                
                # Find the JSON output file
                json_files = [f for f in os.listdir(temp_dir) if f.endswith('.json')]
                if not json_files:
                    logger.warning("No Whisper JSON output found, using fallback timing method")
                    return self._generate_fallback_word_timings(script_text, audio_path)
                
                json_file_path = os.path.join(temp_dir, json_files[0])
                
                # Parse Whisper output
                with open(json_file_path, 'r') as f:
                    whisper_data = json.load(f)
                
                word_timings = []
                
                for segment in whisper_data.get('segments', []):
                    for word_info in segment.get('words', []):
                        word = word_info['word'].strip()
                        start_ms = int(word_info['start'] * 1000)
                        end_ms = int(word_info['end'] * 1000)
                        word_timings.append((word, start_ms, end_ms))
                
                if word_timings:
                    logger.info(f"Extracted {len(word_timings)} word timings from audio using Whisper")
                    return word_timings
                else:
                    logger.warning("No word timings found in Whisper output, using fallback method")
                    return self._generate_fallback_word_timings(script_text, audio_path)
            
        except Exception as e:
            logger.warning(f"Audio timing extraction failed: {e}")
            return self._generate_fallback_word_timings(script_text, audio_path)
    
    def _generate_fallback_word_timings(self, script_text: str, audio_path: str) -> List[Tuple[str, int, int]]:
        """Generate fallback word timings based on audio duration and word count."""
        try:
            # Get audio duration
            cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", 
                   "-of", "csv=p=0", audio_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0 or not result.stdout.strip():
                logger.warning("Could not get audio duration from ffprobe, using default duration")
                duration_seconds = 18.0  # Default duration (18s standard for ads)
            else:
                try:
                    duration_seconds = float(result.stdout.strip())
                except ValueError:
                    logger.warning("Invalid duration from ffprobe, using default duration") 
                    duration_seconds = 18.0
            
            duration_ms = int(duration_seconds * 1000)
            
            # Calculate words per minute (assuming average speech rate)
            words = script_text.split() if script_text else ["Generated", "content"]
            if not words:
                words = ["Generated", "content"]
            
            words_per_minute = 138  # Professional speech rate: 2.3 words/second = 138 words/minute
            ms_per_word = 60000 / words_per_minute
            
            word_timings = []
            current_time_ms = 0
            
            # Calculate total theoretical time needed
            total_theoretical_time = 0
            word_durations = []
            for word in words:
                word_duration = max(200, int(ms_per_word * (len(word) / 5)))
                word_durations.append(word_duration)
                total_theoretical_time += word_duration + 50  # Include gaps
            
            # Scale to fit actual audio duration with 5% buffer at end
            scale_factor = (duration_ms * 0.95) / total_theoretical_time if total_theoretical_time > 0 else 1.0
            print(f"📝 Subtitle timing: Scaling factor = {scale_factor:.3f} to fit {duration_seconds:.2f}s audio")
            
            for i, word in enumerate(words):
                word_duration = int(word_durations[i] * scale_factor)
                gap_duration = int(50 * scale_factor)  # Scale gaps too
                
                word_timings.append((word, current_time_ms, current_time_ms + word_duration))
                current_time_ms += word_duration + gap_duration
            
            print(f"📝 Generated {len(word_timings)} word timings covering {current_time_ms/1000:.2f}s (target: {duration_seconds:.2f}s)")
            return word_timings
            
        except Exception as e:
            logger.error(f"Fallback timing generation failed: {e}")
            return []
    
    def _create_segments_from_word_timings(self, 
                                         word_timings: List[Tuple[str, int, int]], 
                                         scene_timings: List[Dict[str, Any]]) -> List[PreciseSubtitleSegment]:
        """Create subtitle segments from word timings, optimized for readability."""
        segments = []
        current_segment_words = []
        current_segment_start = 0
        scene_number = 1
        
        # Parameters for optimal 9:16 mobile subtitle segmentation
        max_words_per_segment = 4  # Shorter for 9:16 mobile readability with larger font
        max_segment_duration_ms = 2500  # 2.5 seconds maximum for mobile attention span
        min_segment_duration_ms = 800   # 0.8 seconds minimum
        
        for word, start_ms, end_ms in word_timings:
            if not current_segment_words:
                current_segment_start = start_ms
            
            current_segment_words.append((word, start_ms, end_ms))
            
            # Determine if we should end the current segment
            should_end_segment = (
                len(current_segment_words) >= max_words_per_segment or
                (end_ms - current_segment_start) >= max_segment_duration_ms or
                word.endswith('.') or word.endswith('!') or word.endswith('?')
            )
            
            if should_end_segment and (end_ms - current_segment_start) >= min_segment_duration_ms:
                # Create segment
                segment_text = ' '.join([w[0] for w in current_segment_words])
                segment = PreciseSubtitleSegment(
                    text=segment_text,
                    start_ms=current_segment_start,
                    end_ms=end_ms,
                    scene_number=scene_number,
                    word_timings=current_segment_words.copy()
                )
                segments.append(segment)
                
                # Reset for next segment
                current_segment_words = []
                
                # Update scene number based on timing
                for scene in scene_timings:
                    scene_start = scene.get('start_time', 0) * 1000
                    scene_end = scene.get('end_time', 18) * 1000
                    if scene_start <= end_ms <= scene_end:
                        scene_number = scene.get('scene_number', 1)
                        break
        
        # Handle remaining words
        if current_segment_words:
            segment_text = ' '.join([w[0] for w in current_segment_words])
            segment = PreciseSubtitleSegment(
                text=segment_text,
                start_ms=current_segment_start,
                end_ms=current_segment_words[-1][2],
                scene_number=scene_number,
                word_timings=current_segment_words
            )
            segments.append(segment)
        
        return segments
    
    def _optimize_for_reels(self, segments: List[PreciseSubtitleSegment]) -> List[PreciseSubtitleSegment]:
        """Optimize subtitle segments specifically for vertical/reels format."""
        optimized = []
        
        for segment in segments:
            # Split long segments for better 9:16 mobile readability with larger font
            if len(segment.text) > 45:  # Characters - shorter for 9:16 with larger font
                words = segment.text.split()
                mid_point = len(words) // 2
                
                # Find a good break point (sentence boundary)
                break_point = mid_point
                for i in range(max(1, mid_point - 2), min(len(words), mid_point + 3)):
                    if words[i-1].endswith(('.', '!', '?', ',')):
                        break_point = i
                        break
                
                # Create two segments
                first_half = ' '.join(words[:break_point])
                second_half = ' '.join(words[break_point:])
                
                # Calculate timing split
                duration = segment.end_ms - segment.start_ms
                split_time = segment.start_ms + (duration * break_point // len(words))
                
                optimized.append(PreciseSubtitleSegment(
                    text=first_half,
                    start_ms=segment.start_ms,
                    end_ms=split_time,
                    scene_number=segment.scene_number,
                    word_timings=segment.word_timings[:break_point]
                ))
                
                optimized.append(PreciseSubtitleSegment(
                    text=second_half,
                    start_ms=split_time + 100,  # Small gap
                    end_ms=segment.end_ms,
                    scene_number=segment.scene_number,
                    word_timings=segment.word_timings[break_point:]
                ))
            else:
                optimized.append(segment)
        
        return optimized
    
    def generate_reels_subtitle_filter(self, segments: List[PreciseSubtitleSegment]) -> str:
        """Generate FFmpeg filter for reels-optimized subtitles with black background."""
        if not segments:
            return ""
        
        filters = []
        
        for segment in segments:
            # Escape text for FFmpeg
            escaped_text = (segment.text
                          .replace('\\', '\\\\')
                          .replace('"', '\\"')
                          .replace("'", "\\'")
                          .replace(":", "\\:")
                          .replace(",", "\\,"))
            
            # Convert milliseconds to seconds for FFmpeg
            start_time = segment.start_ms / 1000.0
            end_time = segment.end_ms / 1000.0
            
            # Create drawtext filter with reels-optimized styling
            if not self.modern_style:
                continue
                
            filter_components = [
                f"text='{escaped_text}'"
            ]
            
            # Only add fontfile if it exists
            if os.path.exists(self.modern_style.fontfile):
                filter_components.append(f'fontfile={self.modern_style.fontfile}')
            
            filter_components.extend([
                f'fontsize={self.modern_style.fontsize}',
                f'fontcolor={self.modern_style.fontcolor}',
                f'x={self.modern_style.x}',
                f'y={self.modern_style.y}',
                f'box={self.modern_style.box}',
                f'boxcolor={self.modern_style.boxcolor}',
                f'boxborderw={self.modern_style.boxborderw}',
            ])
            
            # Add timing - enable parameter needs quotes for between() function
            filter_components.append(f"enable='between(t,{start_time},{end_time})'")
            
            filter_str = f"drawtext={':'.join([c for c in filter_components if c])}"
            filters.append(filter_str)
        
        # Combine all filters - use comma separation for multiple drawtext filters
        # This creates a filter chain: input -> drawtext1 -> drawtext2 -> ... -> output
        return ','.join(filters) if filters else ""
    
    def _generate_fallback_subtitles(self, script_text: str, scene_timings: List[Dict[str, Any]]) -> List[PreciseSubtitleSegment]:
        """Generate basic subtitles when advanced methods fail."""
        segments = []
        sentences = script_text.split('.')
        
        # FIXED: Use consistent duration fallback with video/audio services (18s standard)
        if scene_timings:
            total_duration = sum(scene.get('duration', 6.0) for scene in scene_timings) * 1000
        else:
            total_duration = 18000  # 18 seconds in milliseconds
        time_per_sentence = total_duration / len(sentences) if sentences else 3000
        
        current_time = 0
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                segment = PreciseSubtitleSegment(
                    text=sentence.strip(),
                    start_ms=int(current_time),
                    end_ms=int(current_time + time_per_sentence),
                    scene_number=1,
                    word_timings=[]
                )
                segments.append(segment)
                current_time += time_per_sentence
        
        return segments
    
    def _generate_fallback_subtitles_with_duration(self, script_text: str, scene_timings: List[Dict[str, Any]], 
                                                  actual_duration: float) -> List[PreciseSubtitleSegment]:
        """Generate precise fallback subtitles using actual audio duration."""
        segments = []
        sentences = [s.strip() for s in script_text.split('.') if s.strip()] if script_text else ["Generated content"]
        
        if not sentences:
            sentences = ["Generated content"]
        
        # Use actual audio duration for perfect timing
        total_duration_ms = int(actual_duration * 1000)
        time_per_sentence = total_duration_ms / len(sentences)
        
        print(f"📝 FALLBACK: Creating {len(sentences)} subtitle segments for {actual_duration:.2f}s audio")
        print(f"📝 Time per sentence: {time_per_sentence/1000:.2f}s")
        
        current_time = 0
        for i, sentence in enumerate(sentences):
            start_ms = int(current_time)
            end_ms = int(current_time + time_per_sentence)
            
            # Ensure last segment doesn't exceed audio duration
            if i == len(sentences) - 1:
                end_ms = min(end_ms, total_duration_ms - 100)  # 100ms buffer
            
            segment = PreciseSubtitleSegment(
                text=sentence,
                start_ms=start_ms,
                end_ms=end_ms,
                scene_number=i + 1,
                word_timings=[]
            )
            segments.append(segment)
            current_time += time_per_sentence
            
            print(f"📝 Segment {i+1}: '{sentence[:30]}...' from {start_ms/1000:.2f}s to {end_ms/1000:.2f}s")
        
        return segments
    
    def generate_subtitles_from_script(self, architecture: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Compatibility method for legacy subtitle generation."""
        try:
            script_text = architecture.get('unified_script', '')
            scene_timings = architecture.get('scene_architecture', {}).get('scenes', [])
            
            # Generate precise subtitles using the advanced method
            precise_segments = self.generate_precise_subtitles('', script_text, scene_timings)
            
            # Convert to legacy format for compatibility
            legacy_segments = []
            for segment in precise_segments:
                legacy_segment = {
                    'text': segment.text,
                    'start_time': segment.start_ms / 1000.0,  # Convert to seconds
                    'end_time': segment.end_ms / 1000.0,
                    'scene_number': segment.scene_number
                }
                legacy_segments.append(legacy_segment)
            
            return legacy_segments
            
        except Exception as e:
            logger.error(f"Legacy subtitle generation failed: {e}")
            return []
    
    def generate_subtitles_from_audio(self, audio_file_path: str, script_text: str = None) -> List[Dict[str, Any]]:
        """Generate subtitles from audio file with optional script alignment."""
        try:
            logger.info("Generating subtitles from audio file", action="subtitle.audio.start")
            
            # CRITICAL: Get precise audio duration first
            actual_audio_duration = self._get_precise_audio_duration(audio_file_path)
            print(f"📝 PRECISE SYNC: Audio duration = {actual_audio_duration:.3f}s for subtitle generation")
            
            # Extract word timings from audio
            word_timings = self._extract_word_timings_from_audio(audio_file_path, script_text or "")
            
            # Adjust word timings to match actual audio duration
            if word_timings:
                last_word_end = word_timings[-1][2] / 1000.0
                if abs(last_word_end - actual_audio_duration) > 0.5:
                    print(f"📝 TIMING FIX: Adjusting word timings from {last_word_end:.2f}s to {actual_audio_duration:.2f}s")
                    word_timings = self._adjust_word_timings_to_audio_duration(word_timings, actual_audio_duration)
            
            if not word_timings:
                logger.warning("No word timings extracted, generating precise fallback subtitles")
                return self._generate_basic_subtitles_from_precise_duration(audio_file_path, script_text or "", actual_audio_duration)
            
            # Create segments from word timings
            segments = self._create_segments_from_word_timings(word_timings, [])
            
            # Convert to legacy format
            legacy_segments = []
            for segment in segments:
                legacy_segment = {
                    'text': segment.text,
                    'start_time': segment.start_ms / 1000.0,
                    'end_time': segment.end_ms / 1000.0,
                    'scene_number': segment.scene_number
                }
                legacy_segments.append(legacy_segment)
            
            logger.info(f"Generated {len(legacy_segments)} audio-based subtitle segments",
                       action="subtitle.audio.complete")
            
            return legacy_segments
            
        except Exception as e:
            logger.error(f"Audio-based subtitle generation failed: {e}")
            # Use precise duration even in error case
            actual_audio_duration = self._get_precise_audio_duration(audio_file_path)
            return self._generate_basic_subtitles_from_precise_duration(audio_file_path, script_text or "", actual_audio_duration)
    
    def _generate_basic_subtitles_from_duration(self, audio_file_path: str, script_text: str) -> List[Dict[str, Any]]:
        """Generate basic subtitles based on audio duration when other methods fail."""
        try:
            # Get audio duration
            cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", 
                   "-of", "csv=p=0", audio_file_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0 or not result.stdout.strip():
                logger.warning("Could not get audio duration, using default")
                duration_seconds = 18.0  # Default duration (18s standard for ads)
            else:
                duration_seconds = float(result.stdout.strip())
            
            # Split script into sentences or use default text
            if script_text:
                sentences = [s.strip() for s in script_text.split('.') if s.strip()]
            else:
                sentences = ["Generated content"]
            
            if not sentences:
                sentences = ["Generated content"]
            
            # Create evenly distributed segments
            segment_duration = duration_seconds / len(sentences)
            segments = []
            
            for i, sentence in enumerate(sentences):
                start_time = i * segment_duration
                end_time = min((i + 1) * segment_duration, duration_seconds)
                
                segments.append({
                    'text': sentence,
                    'start_time': start_time,
                    'end_time': end_time,
                    'scene_number': i + 1
                })
            
            logger.info(f"Generated {len(segments)} basic subtitle segments",
                       action="subtitle.basic.complete")
            
            return segments
            
        except Exception as e:
            logger.error(f"Basic subtitle generation failed: {e}")
            return []
    
    def _generate_basic_subtitles_from_precise_duration(self, audio_file_path: str, script_text: str, 
                                                       duration_seconds: float) -> List[Dict[str, Any]]:
        """Generate basic subtitles using precise audio duration for perfect sync."""
        try:
            # Split script into sentences or use default text
            if script_text:
                sentences = [s.strip() for s in script_text.split('.') if s.strip()]
            else:
                sentences = ["Generated content"]
            
            if not sentences:
                sentences = ["Generated content"]
            
            # Create evenly distributed segments with precise timing
            segment_duration = duration_seconds / len(sentences)
            segments = []
            
            print(f"📝 PRECISE BASIC: Creating {len(sentences)} segments for {duration_seconds:.3f}s audio")
            
            for i, sentence in enumerate(sentences):
                start_time = i * segment_duration
                end_time = (i + 1) * segment_duration
                
                # Ensure last segment doesn't exceed audio duration
                if i == len(sentences) - 1:
                    end_time = duration_seconds - 0.1  # 100ms buffer
                
                segments.append({
                    'text': sentence,
                    'start_time': start_time,
                    'end_time': end_time,
                    'scene_number': i + 1
                })
                
                print(f"📝 Segment {i+1}: '{sentence[:25]}...' {start_time:.2f}s-{end_time:.2f}s")
            
            logger.info(f"Generated {len(segments)} precise basic subtitle segments",
                       action="subtitle.precise.basic.complete")
            
            return segments
            
        except Exception as e:
            logger.error(f"Precise basic subtitle generation failed: {e}")
            return []
    
    def create_animated_subtitles(self, 
                                audio_file_path: str,
                                script_text: str, 
                                scene_timings: List[Dict[str, Any]],
                                music_file_path: Optional[str] = None) -> List[PreciseSubtitleSegment]:
        """Create modern animated subtitles with word-by-word timing and music sync."""
        try:
            logger.info("Creating modern animated subtitles", action="subtitle.animated.start")
            
            # Step 1: Extract precise word timings from audio
            word_timings = self._extract_word_timings_from_audio(audio_file_path, script_text)
            
            # Step 2: Analyze music for beat sync if provided
            music_sync_points = []
            if music_file_path and os.path.exists(music_file_path):
                music_sync_points = self._analyze_music_beats(music_file_path)
            
            # Step 3: Create animated segments with enhanced timing
            animated_segments = self._create_animated_segments(
                word_timings, scene_timings, music_sync_points, script_text
            )
            
            logger.info(f"Created {len(animated_segments)} animated subtitle segments",
                       action="subtitle.animated.complete")
            
            return animated_segments
            
        except Exception as e:
            logger.error(f"Animated subtitle creation failed: {e}", exc_info=True)
            # Fallback to basic subtitles
            return self.generate_precise_subtitles(audio_file_path, script_text, scene_timings)
    
    def _analyze_music_beats(self, music_file_path: str) -> List[MusicSyncPoint]:
        """Analyze music file to extract beat timing for subtitle sync."""
        try:
            # Use ffprobe to analyze audio for beat detection
            cmd = [
                "ffprobe", "-v", "quiet", "-show_entries", "frame=best_effort_timestamp_time",
                "-select_streams", "a:0", "-of", "csv=p=0", music_file_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # Simple beat detection - every 0.5 seconds for now
                # In a real implementation, you'd use proper beat detection algorithms
                sync_points = []
                duration_cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", 
                              "-of", "csv=p=0", music_file_path]
                duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)
                
                if duration_result.returncode == 0:
                    duration_seconds = float(duration_result.stdout.strip())
                    
                    # Create beat points every 500ms (120 BPM equivalent)
                    for i in range(int(duration_seconds * 2)):
                        time_ms = i * 500
                        is_strong_beat = i % 4 == 0  # Every 2 seconds is emphasis
                        
                        sync_points.append(MusicSyncPoint(
                            time_ms=time_ms,
                            beat_strength=0.8 if is_strong_beat else 0.5,
                            is_emphasis=is_strong_beat,
                            bpm_at_point=120.0
                        ))
                
                return sync_points
            
        except Exception as e:
            logger.warning(f"Music beat analysis failed: {e}")
        
        return []  # Return empty list if analysis fails
    
    def _create_animated_segments(self,
                                word_timings: List[Tuple[str, int, int]],
                                scene_timings: List[Dict[str, Any]],
                                music_sync_points: List[MusicSyncPoint],
                                script_text: str) -> List[PreciseSubtitleSegment]:
        """Create enhanced animated subtitle segments."""
        if not self.modern_style:
            return []  # Need modern style for animations
        
        segments = []
        words_per_segment = 4  # Shorter segments for better animation
        
        # Group words into animated segments
        for i in range(0, len(word_timings), words_per_segment):
            segment_words = word_timings[i:i + words_per_segment]
            if not segment_words:
                continue
            
            segment_text = ' '.join([word[0] for word in segment_words])
            start_time = segment_words[0][1]
            end_time = segment_words[-1][2]
            
            # Create animated words with individual timing
            animated_words = []
            for j, (word, word_start, word_end) in enumerate(segment_words):
                # Check if word should be highlighted
                is_highlighted = self._should_highlight_word(word)
                
                # Add animation delay for word-by-word effect
                delay_ms = j * self.modern_style.word_delay_ms
                
                animated_word = AnimatedWord(
                    text=word,
                    start_ms=word_start,
                    end_ms=word_end,
                    highlight=is_highlighted,
                    highlight_style=self.modern_style.highlight_style,
                    highlight_color=self.modern_style.highlight_color,
                    animation=self.modern_style.entrance_animation,
                    delay_ms=delay_ms
                )
                animated_words.append(animated_word)
            
            # Find relevant music sync points for this segment
            segment_sync_points = [
                sync for sync in music_sync_points
                if start_time <= sync.time_ms <= end_time
            ]
            
            # Determine if this is an emphasis segment
            is_emphasis = any(sync.is_emphasis for sync in segment_sync_points)
            
            segment = PreciseSubtitleSegment(
                text=segment_text,
                start_ms=start_time,
                end_ms=end_time,
                scene_number=1,  # Will be updated based on scene_timings
                word_timings=[(w[0], w[1], w[2]) for w in segment_words],  # Legacy format
                animated_words=animated_words,
                animation_type=AnimationType.WORD_BY_WORD,
                music_sync_points=segment_sync_points,
                is_emphasis_segment=is_emphasis
            )
            
            segments.append(segment)
        
        return segments
    
    def _should_highlight_word(self, word: str) -> bool:
        """Determine if a word should be highlighted based on importance."""
        if not self.modern_style or not self.modern_style.highlight_keywords:
            return False
            
        word_lower = word.lower().strip('.,!?')
        return word_lower in self.modern_style.highlight_keywords
    
    def generate_modern_subtitle_filter(self, segments: List[PreciseSubtitleSegment]) -> str:
        """Generate FFmpeg filter for modern animated subtitles."""
        if not segments or not self.modern_style:
            return ""
        
        filters = []
        
        for segment in segments:
            if not segment.animated_words:
                # Fallback to regular segment display
                filters.append(self._create_basic_segment_filter(segment))
                continue
            
            # Create word-by-word animation filters
            for word in segment.animated_words:
                word_filter = self._create_word_animation_filter(word, segment)
                if word_filter:
                    filters.append(word_filter)
        
        return ','.join(filters) if filters else ""
    
    def generate_subtitle_overlay_filter(self, subtitle_segments: List, style: Dict[str, Any] = None) -> str:
        """Generate subtitle overlay filter compatible with assembly service.
        
        This method provides compatibility with the existing assembly service
        while using the modern animated subtitle system.
        """
        try:
            # Convert legacy subtitle segments to PreciseSubtitleSegment format
            if not subtitle_segments:
                return ""
            
            # Handle both legacy dict format and modern PreciseSubtitleSegment format
            precise_segments = []
            
            for segment in subtitle_segments:
                if isinstance(segment, dict):
                    # Legacy format conversion
                    start_time = segment.get('start_time', 0)
                    end_time = segment.get('end_time', start_time + 3)
                    text = segment.get('text', '')
                    
                    # Convert to milliseconds if needed
                    if start_time < 100:  # Assume seconds if small number
                        start_time *= 1000
                        end_time *= 1000
                    
                    # Create PreciseSubtitleSegment with modern styling
                    precise_segment = PreciseSubtitleSegment(
                        text=text,
                        start_ms=int(start_time),
                        end_ms=int(end_time),
                        scene_number=0,
                        animation_type=AnimationType.WORD_BY_WORD
                    )
                    precise_segments.append(precise_segment)
                    
                elif isinstance(segment, PreciseSubtitleSegment):
                    # Already in correct format
                    precise_segments.append(segment)
            
            # Use the modern subtitle filter generation
            if precise_segments:
                return self.generate_modern_subtitle_filter(precise_segments)
            else:
                return ""
                
        except Exception as e:
            print(f"Error generating subtitle overlay filter: {e}")
            # Fallback to basic subtitle generation
            return self._generate_basic_subtitle_filter(subtitle_segments, style)
    
    def _generate_basic_subtitle_filter(self, subtitle_segments: List, style: Dict[str, Any] = None) -> str:
        """Generate basic subtitle filter as fallback."""
        if not subtitle_segments or not isinstance(subtitle_segments, list):
            return ""
        
        # Default style optimized for 9:16 mobile viewing
        if not style:
            style = {
                'fontsize': 32,  # Smaller font for 9:16 mobile screens
                'fontcolor': 'white',
                'bordercolor': 'black',
                'borderw': 4,  # Proportional border for smaller font
                'position': 'bottom'
            }
        
        filter_parts = []
        for segment in subtitle_segments:
            if isinstance(segment, dict):
                text = segment.get('text', '').replace("'", "\\'").replace('"', '\\"')
                start_time = segment.get('start_time', 0)
                end_time = segment.get('end_time', start_time + 3)
                
                # Convert to seconds if needed
                if start_time > 1000:  # Assume milliseconds
                    start_time /= 1000
                    end_time /= 1000
                
                # Ensure start_time and end_time are valid numbers
                try:
                    start_time = float(start_time)
                    end_time = float(end_time)
                except (ValueError, TypeError):
                    continue
                
                if text.strip():
                    # Position subtitles: optimized for 9:16 aspect ratio mobile viewing
                    # y=h*0.68 positions subtitles for smaller font and better mobile readability
                    filter_part = f"drawtext=text='{text}':fontsize={style['fontsize']}:fontcolor={style['fontcolor']}:bordercolor={style['bordercolor']}:borderw={style['borderw']}:x=(w-text_w)/2:y=h*0.68:enable='between(t,{start_time},{end_time})'"
                    filter_parts.append(filter_part)
        
        return ','.join(filter_parts) if filter_parts else ""
    
    def _create_word_animation_filter(self, word: AnimatedWord, segment: PreciseSubtitleSegment) -> str:
        """Create FFmpeg filter for individual animated word."""
        try:
            # Escape text for FFmpeg
            escaped_text = (word.text
                          .replace('\\', '\\\\')
                          .replace('"', '\\"')
                          .replace("'", "\\'")
                          .replace(":", "\\:")
                          .replace(",", "\\,"))
            
            # Calculate timing with animation delay
            actual_start = (word.start_ms + word.delay_ms) / 1000.0
            actual_end = word.end_ms / 1000.0
            
            # Base filter components
            filter_components = [
                f"text='{escaped_text}'",
                f'fontfile={self.modern_style.fontfile}',
                f'fontsize={self.modern_style.fontsize}',
                f'x={self.modern_style.x}',
                f'y={self.modern_style.y}',
                f'box={self.modern_style.box}',
                f'boxcolor={self.modern_style.boxcolor}',
                f'boxborderw={self.modern_style.boxborderw}',
            ]
            
            # Apply highlighting if needed
            if word.highlight:
                if word.highlight_style == HighlightStyle.COLOR_CHANGE:
                    filter_components.append(f'fontcolor={word.highlight_color}')
                elif word.highlight_style == HighlightStyle.GLOW:
                    filter_components.extend([
                        f'fontcolor={word.highlight_color}',
                        'shadowcolor=white@0.8',
                        'shadowx=0',
                        'shadowy=0'
                    ])
                else:
                    filter_components.append(f'fontcolor={self.modern_style.fontcolor}')
            else:
                filter_components.append(f'fontcolor={self.modern_style.fontcolor}')
            
            # Add animation timing
            if word.animation == AnimationType.SLIDE_UP:
                # Slide up animation
                start_y = f"{self.modern_style.y}"
                filter_components.extend([
                    f'y=if(between(t\\,{actual_start}\\,{actual_start+0.3})\\,{start_y}-(t-{actual_start})*167,{self.modern_style.y})',
                    f'alpha=if(between(t\\,{actual_start}\\,{actual_start+0.1}),(t-{actual_start})*10,if(between(t\\,{actual_end-0.1}\\,{actual_end}),({actual_end}-t)*10,1))'
                ])
            elif word.animation == AnimationType.BOUNCE:
                # Bounce animation
                filter_components.extend([
                    f'fontsize=if(between(t\\,{actual_start}\\,{actual_start+0.2}),{self.modern_style.fontsize}+sin((t-{actual_start})*15)*10,{self.modern_style.fontsize})',
                    f'alpha=if(between(t\\,{actual_start}\\,{actual_end}),1,0)'
                ])
            else:
                # Default fade animation
                filter_components.extend([
                    f'alpha=if(between(t\\,{actual_start}\\,{actual_start+0.1}),(t-{actual_start})*10,if(between(t\\,{actual_end-0.1}\\,{actual_end}),({actual_end}-t)*10,1))'
                ])
            
            # Enable timing - quotes required for between() function
            filter_components.append(f"enable='between(t,{actual_start},{actual_end})'")
            
            return f"drawtext={':'.join([c for c in filter_components if c])}"
            
        except Exception as e:
            logger.warning(f"Failed to create word animation filter: {e}")
            return ""
    
    def _create_basic_segment_filter(self, segment: PreciseSubtitleSegment) -> str:
        """Create basic filter for segments without word animations."""
        if not self.modern_style:
            return ""
            
        try:
            # Escape text for FFmpeg - use single quotes to contain text, escape single quotes inside
            escaped_text = segment.text.replace("'", "'\\''").replace('"', '\\"')
            
            start_time = segment.start_ms / 1000.0
            end_time = segment.end_ms / 1000.0
            
            # Ensure start_time and end_time are valid numbers
            try:
                start_time = float(start_time)
                end_time = float(end_time)
            except (ValueError, TypeError):
                return ""
            
            # Build filter with proper escaping - avoid issues with font file paths
            filter_parts = []
            filter_parts.append(f"text='{escaped_text}'")
            
            # Only add fontfile if it exists and is valid
            if os.path.exists(self.modern_style.fontfile):
                filter_parts.append(f'fontfile={self.modern_style.fontfile}')
            
            filter_parts.extend([
                f'fontsize={self.modern_style.fontsize}',
                f'fontcolor={self.modern_style.fontcolor}',
                f'x={self.modern_style.x}',
                f'y={self.modern_style.y}',
                f'box={self.modern_style.box}',
                f'boxcolor={self.modern_style.boxcolor}',
                f'boxborderw={self.modern_style.boxborderw}',
                f"enable='between(t,{start_time},{end_time})'"
            ])
            
            # Join with colons and ensure no empty parts
            filter_string = ':'.join([part for part in filter_parts if part])
            return f"drawtext={filter_string}"
            
        except Exception as e:
            logger.warning(f"Failed to create basic segment filter: {e}")
            return ""

# Create modern service instance by default
subtitle_service = AdvancedSubtitleService(use_modern_style=True)

# Convenience function for easy integration
def create_modern_subtitles(audio_file_path: str, 
                          script_text: str,
                          scene_timings: List[Dict[str, Any]] = None,
                          music_file_path: str = None) -> str:
    """
    Convenience function to create modern animated subtitles.
    
    Returns FFmpeg filter string for direct use in video processing.
    """
    try:
        service = AdvancedSubtitleService(use_modern_style=True)
        
        if not scene_timings:
            scene_timings = []
        
        # Create animated subtitles
        segments = service.create_animated_subtitles(
            audio_file_path=audio_file_path,
            script_text=script_text, 
            scene_timings=scene_timings,
            music_file_path=music_file_path
        )
        
        # Generate modern filter
        filter_string = service.generate_modern_subtitle_filter(segments)
        
        logger.info(f"Created modern subtitle filter with {len(segments)} segments",
                   action="subtitle.modern.filter.created")
        
        return filter_string
        
    except Exception as e:
        logger.error(f"Modern subtitle creation failed: {e}", exc_info=True)
        return ""