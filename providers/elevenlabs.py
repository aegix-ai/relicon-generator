"""
  ElevenLabs audio generation provider implementation.
"""

import os
import requests
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, List
from interfaces.audio_generator import AudioGenerator


class ElevenLabsProvider(AudioGenerator):
    """ElevenLabs text-to-speech service implementation with subtitle alignment support."""
    
    def __init__(self):
        self.api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY environment variable is required")
        
        self.base_url = "https://api.elevenlabs.io/v1/text-to-speech"
        self.headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }
        
        # Store alignment data for subtitle generation
        self.last_alignment_data = None
    
    def generate_audio(self, text: str, output_path: str, audio_config: Dict[str, Any]) -> bool:
        """Generate audio from text using ElevenLabs."""
        try:
            voice_gender = audio_config.get('voice_gender', 'male')
            voice_tone = audio_config.get('voice_tone', 'energetic')
            
            # Make text more energetic if needed
            if audio_config.get('energy_level') == 'high':
                energetic_text = self._make_text_energetic(text)
            else:
                energetic_text = text
            
            print(f"Generating ElevenLabs voice ({voice_gender}, {voice_tone})")
            return self._generate_elevenlabs_audio(energetic_text, output_path, voice_gender, voice_tone)
            
        except Exception as e:
            print(f"TTS generation failed: {e}")
            return False
    
    def _make_text_energetic(self, text: str) -> str:
        """Transform text to be more energetic and advertisement-style."""
        energetic_patterns = [
            ("Are you", "Have you ever wondered if you're"),
            ("Do you", "Have you ever thought about whether you"),
            ("This is", "This is exactly what you've been looking for!"),
            ("We offer", "Get ready to experience"),
            ("Our product", "Discover the revolutionary"),
            ("You can", "You're about to"),
            ("It helps", "Watch how it transforms"),
            ("Benefits include", "Get ready for incredible benefits like"),
            ("Join", "Don't miss out - join"),
            ("Try", "Ready to try"),
            (".", "!"),
        ]
        
        energetic_text = text
        for old, new in energetic_patterns:
            energetic_text = energetic_text.replace(old, new)
        
        # Add hook questions at the start if not present
        if not any(starter in energetic_text.lower() for starter in ["have you", "are you", "do you", "ready to", "discover"]):
            if "tired" in energetic_text.lower() or "problem" in energetic_text.lower():
                energetic_text = "Have you ever faced this problem? " + energetic_text
            elif "solution" in energetic_text.lower() or "help" in energetic_text.lower():
                energetic_text = "Ready for the solution you've been waiting for? " + energetic_text
            else:
                energetic_text = "Have you ever wondered about this? " + energetic_text
        
        return energetic_text
    
    def generate_background_music(self, duration: float, tone: str, output_path: str) -> bool:
        """Generate background music using ElevenLabs music generation."""
        try:
            # ElevenLabs music generation prompts based on tone
            music_prompts = {
                "friendly": "Warm, uplifting corporate background music with soft piano and gentle strings, professional and welcoming tone",
                "energetic": "High-energy upbeat commercial music with driving beats and bright melodies, motivating and exciting",
                "inspiring": "Inspiring motivational background music with soaring melodies and uplifting harmonies, empowering and positive",
                "calm": "Calm professional background music with soft ambient tones and subtle harmonies, peaceful and focused",
                "professional": "Sophisticated corporate background music with elegant piano and refined orchestration, trustworthy and polished"
            }
            
            prompt = music_prompts.get(tone, music_prompts["friendly"])
            print(f"Generating ElevenLabs background music: {tone} tone for {duration}s")
            
            # ElevenLabs Music Generation API
            music_url = "https://api.elevenlabs.io/v1/sound-generation"
            music_headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.api_key
            }
            
            music_data = {
                "text": prompt,
                "duration_seconds": min(duration, 22),  # ElevenLabs max duration
                "prompt_influence": 0.3
            }
            
            print(f"Calling ElevenLabs Music API with prompt: {prompt}")
            response = requests.post(
                music_url,
                json=music_data,
                headers=music_headers,
                timeout=60
            )
            
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                print(f"ElevenLabs background music generated: {output_path}")
                return True
            else:
                print(f"ElevenLabs Music API error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"ElevenLabs music generation failed: {e}")
            return False
    
    def _generate_elevenlabs_audio(self, text: str, output_path: str, voice_gender: str, voice_tone: str) -> bool:
        """Generate audio using ElevenLabs API."""
        try:
            # ElevenLabs premium voice selection
            elevenlabs_voices = {
                ('male', 'energetic'): 'pNInz6obpgDQGcFmaJgB',      # Adam
                ('male', 'calm'): 'VR6AewLTigWG4xSOukaG',           # Arnold
                ('male', 'authoritative'): 'ErXwobaYiN019PkySvjV',    # Antoni
                ('female', 'energetic'): 'EXAVITQu4vr4xnSDxMaL',     # Bella
                ('female', 'friendly'): 'MF3mGyEYCl7XYWbV9V6O',      # Elli
                ('female', 'professional'): 'ThT5KcBeYPX3keUQqHPh',  # Dorothy
                ('neutral', 'energetic'): 'EXAVITQu4vr4xnSDxMaL'     # Default to Bella
            }
            
            voice_id = elevenlabs_voices.get((voice_gender, voice_tone), 'EXAVITQu4vr4xnSDxMaL')
            
            data = {
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {
                    "stability": 0.75,
                    "similarity_boost": 0.85,
                    "style": 0.8,
                    "use_speaker_boost": True
                }
            }
            
            response = requests.post(
                f"{self.base_url}/{voice_id}",
                json=data,
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                print(f"ElevenLabs audio generated: {output_path}")
                return True
            else:
                print(f"ElevenLabs API error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"ElevenLabs generation failed: {e}")
            return False
    
    def enhance_audio(self, input_file: str, output_file: str, audio_config: Dict[str, Any]) -> None:
        """Enhance audio based on AI architectural decisions."""
        try:
            energy_level = audio_config.get('energy_level', 'medium')
            
            if energy_level == 'high':
                volume_boost = '12dB'
                compand_settings = 'attacks=0.2:decays=0.6:points=-80/-900|-45/-12|-27/-6:gain=6'
            elif energy_level == 'medium':
                volume_boost = '8dB'
                compand_settings = 'attacks=0.3:decays=0.8:points=-80/-900|-45/-15|-27/-9:gain=5'
            else:
                volume_boost = '4dB'
                compand_settings = 'attacks=0.5:decays=1.0:points=-80/-900|-45/-18|-27/-12:gain=3'
            
            print(f"Enhancing audio with {energy_level} energy profile...")
            
            cmd = [
                'ffmpeg', '-y', '-i', input_file,
                '-af', f'volume={volume_boost},compand={compand_settings}',
                '-c:a', 'libmp3lame', '-b:a', '320k',
                output_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Warning: Audio enhancement failed: {result.stderr}")
                shutil.copy2(input_file, output_file)
            else:
                print(f"Audio enhanced: {output_file}")
                
        except Exception as e:
            print(f"Warning: Audio enhancement failed: {e}")
            shutil.copy2(input_file, output_file)

    def create_unified_audio(self, audio_segments: List[Dict[str, Any]], target_duration: float, output_path: str) -> bool:
        """Create unified audio track with precise timing."""
        try:
            if not audio_segments:
                print("No audio segments provided")
                return False
            
            total_duration = 0
            for seg in audio_segments:
                try:
                    probe_cmd = [
                        'ffprobe', '-v', 'quiet', '-print_format', 'json',
                        '-show_format', seg['file']
                    ]
                    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
                    if probe_result.returncode == 0:
                        import json
                        probe_data = json.loads(probe_result.stdout)
                        total_duration += float(probe_data['format']['duration'])
                except Exception as e:
                    print(f"Could not get duration of audio segment: {e}")
            
            if total_duration == 0:
                print("Total audio duration is 0")
                return False
                
            if target_duration <= 0:
                print(f"Invalid target duration: {target_duration}")
                return False
            
            # Create filter complex for concatenation and timing
            filter_parts = []
            input_files = []
            
            for i, segment in enumerate(audio_segments):
                input_files.extend(['-i', segment['file']])
                filter_parts.append(f"[{i}:a]")
            
            # Concatenate all segments
            concat_filter = f"{''.join(filter_parts)}concat=n={len(audio_segments)}:v=0:a=1[concatenated]"
            
            # Adjust speed to fit target duration
            speed_factor = total_duration / target_duration
            
            # For voice audio, we should never slow it down below 1.0x as it sounds unnatural
            # Instead, we should pad with silence or use other techniques
            if speed_factor < 1.0:
                # If audio is shorter than target, pad with silence instead of slowing down
                # This preserves natural voice speed
                silence_duration = target_duration - total_duration
                speed_filter = f"[concatenated]apad=pad_dur={silence_duration:.3f}[adjusted]"
            elif speed_factor > 100.0:
                # If audio is too long, speed it up to maximum
                clamped_speed = 100.0
                speed_filter = f"[concatenated]atempo={clamped_speed:.3f}[adjusted]"
            else:
                # Normal case - speed up if needed (but not slow down)
                clamped_speed = max(1.0, speed_factor)  # Ensure we don't slow down voice
                speed_filter = f"[concatenated]atempo={clamped_speed:.3f}[adjusted]"
            
            # Add fade out at the end
            fade_filter = f"[adjusted]afade=out:st={target_duration-1:.1f}:d=1[final]"
            
            # Combine all filters
            full_filter = f"{concat_filter};{speed_filter};{fade_filter}"
            
            cmd = [
                'ffmpeg', '-y'
            ] + input_files + [
                '-filter_complex', full_filter,
                '-map', '[final]',
                '-c:a', 'libmp3lame', '-b:a', '320k',
                '-t', str(target_duration),
                output_path
            ]
            
            print(f"Creating unified {target_duration}s audio from {len(audio_segments)} segments...")
            print(f"Audio duration: total={total_duration:.3f}s, target={target_duration:.3f}s (keeping normal voice speed)")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Unified audio created: {output_path}")
                return True
            else:
                print(f"Audio unification failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"Audio unification error: {e}")
            return False
    
    def generate_audio_with_alignment(self, text: str, output_path: str, audio_config: Dict[str, Any]) -> bool:
        """Generate audio with word-level alignment data for subtitle synchronization."""
        try:
            voice_gender = audio_config.get('voice_gender', 'male')
            voice_tone = audio_config.get('voice_tone', 'energetic')
            
            # Make text more energetic if needed
            if audio_config.get('energy_level') == 'high':
                energetic_text = self._make_text_energetic(text)
            else:
                energetic_text = text
            
            print(f"Generating ElevenLabs audio with alignment data ({voice_gender}, {voice_tone})")
            
            # Generate alignment data
            alignment_success = self._generate_alignment_data(energetic_text, voice_gender, voice_tone)
            
            # Generate actual audio
            audio_success = self._generate_elevenlabs_audio(energetic_text, output_path, voice_gender, voice_tone)
            
            return audio_success and alignment_success
            
        except Exception as e:
            print(f"TTS with alignment generation failed: {e}")
            return False
    
    def _generate_alignment_data(self, text: str, voice_gender: str, voice_tone: str) -> bool:
        """Generate word-level timing alignment data."""
        try:
            # Split text into words for timing estimation
            words = text.split()
            if not words:
                return False
            
            # Estimate timing based on word length and speech rate
            # Average speech rate: 150 words per minute = 2.5 words per second
            speech_rate = 2.2  # Slightly slower for better clarity
            
            alignment_data = {
                'text': text,
                'words': []
            }
            
            current_time = 0.0
            
            for word in words:
                # Estimate word duration based on length and complexity
                word_chars = len(word)
                
                # Base duration calculation
                if word_chars <= 3:
                    word_duration = 0.3
                elif word_chars <= 6:
                    word_duration = 0.4
                elif word_chars <= 9:
                    word_duration = 0.5
                else:
                    word_duration = 0.6
                
                # Adjust for punctuation (longer pause)
                if word.endswith(('.', '!', '?')):
                    word_duration += 0.2
                elif word.endswith((',', ';')):
                    word_duration += 0.1
                
                # Add small pause between words
                pause_duration = 0.1
                
                # Store word timing
                alignment_data['words'].append({
                    'word': word,
                    'start': round(current_time, 2),
                    'end': round(current_time + word_duration, 2)
                })
                
                current_time += word_duration + pause_duration
            
            # Store alignment data for subtitle service
            self.last_alignment_data = alignment_data
            
            print(f"Generated alignment data for {len(words)} words")
            return True
            
        except Exception as e:
            print(f"Alignment data generation failed: {e}")
            self.last_alignment_data = None
            return False
