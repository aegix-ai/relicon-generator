"""
Text-to-Speech Service using ElevenLabs with OpenAI fallback
Generates professional, ultra-realistic voiceovers for advertisements
"""
import os
import subprocess
import shutil
import requests
from pathlib import Path
from typing import Dict, Any
from openai import OpenAI

class TTSService:
    def __init__(self):
        # Primary: ElevenLabs
        self.elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY")
        # Fallback: OpenAI with compatibility fix
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            try:
                # Use basic client initialization to avoid httpx conflicts
                self.openai_client = OpenAI(
                    api_key=openai_key,
                    timeout=60.0
                )
            except Exception as e:
                print(f"⚠️ OpenAI client initialization failed: {e}")
                self.openai_client = None
        else:
            self.openai_client = None
        
        # ElevenLabs configuration
        self.elevenlabs_url = "https://api.elevenlabs.io/v1/text-to-speech"
        self.elevenlabs_headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.elevenlabs_api_key
        }
    
    def make_text_energetic(self, text: str) -> str:
        """Transform text to be more energetic and advertisement-style"""
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
            (".", "!"),  # Make statements more exciting
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
    
    def generate_autonomous_audio(self, text: str, output_path: str, audio_config: Dict[str, Any]) -> bool:
        """
        Generate autonomous audio with AI-selected voice characteristics
        Primary: ElevenLabs, Fallback: OpenAI TTS
        """
        try:
            # AI-selected voice characteristics
            voice_gender = audio_config.get('voice_gender', 'male')
            voice_tone = audio_config.get('voice_tone', 'energetic')
            
            # Make text more energetic if needed
            if audio_config.get('energy_level') == 'high':
                energetic_text = self.make_text_energetic(text)
            else:
                energetic_text = text
            
            # Try ElevenLabs first (primary)
            if self.elevenlabs_api_key:
                print(f"🎙️ Generating ultra-realistic audio with ElevenLabs ({voice_gender}, {voice_tone})")
                if self._generate_elevenlabs_audio(energetic_text, output_path, voice_gender, voice_tone):
                    return True
                print("⚠️ ElevenLabs failed, falling back to OpenAI TTS...")
            
            # Fallback to OpenAI TTS
            print(f"🎙️ Generating audio with OpenAI TTS fallback ({voice_gender}, {voice_tone})")
            return self._generate_openai_audio(energetic_text, output_path, voice_gender, voice_tone)
            
        except Exception as e:
            print(f"❌ TTS generation failed: {e}")
            return False
    
    def _generate_elevenlabs_audio(self, text: str, output_path: str, voice_gender: str, voice_tone: str) -> bool:
        """Generate audio using ElevenLabs API"""
        try:
            # ElevenLabs premium voice selection (ultra-realistic)
            elevenlabs_voices = {
                ('male', 'energetic'): 'pNInz6obpgDQGcFmaJgB',      # Adam (energetic male)
                ('male', 'calm'): 'VR6AewLTigWG4xSOukaG',           # Arnold (calm male)
                ('male', 'authoritative'): 'ErXwobaYiN019PkySvjV',    # Antoni (authoritative)
                ('female', 'energetic'): 'EXAVITQu4vr4xnSDxMaL',     # Bella (energetic female)
                ('female', 'friendly'): 'MF3mGyEYCl7XYWbV9V6O',      # Elli (friendly female)
                ('female', 'professional'): 'ThT5KcBeYPX3keUQqHPh',  # Dorothy (professional)
                ('neutral', 'energetic'): 'EXAVITQu4vr4xnSDxMaL'     # Default to Bella
            }
            
            voice_id = elevenlabs_voices.get((voice_gender, voice_tone), 'EXAVITQu4vr4xnSDxMaL')
            
            # ElevenLabs request payload
            data = {
                "text": text,
                "model_id": "eleven_multilingual_v2",  # Latest model
                "voice_settings": {
                    "stability": 0.75,
                    "similarity_boost": 0.85,
                    "style": 0.8,
                    "use_speaker_boost": True
                }
            }
            
            # Make request to ElevenLabs
            response = requests.post(
                f"{self.elevenlabs_url}/{voice_id}",
                json=data,
                headers=self.elevenlabs_headers,
                timeout=30
            )
            
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                print(f"✅ ElevenLabs audio generated: {output_path}")
                return True
            else:
                print(f"❌ ElevenLabs API error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ ElevenLabs generation failed: {e}")
            return False
    
    def _generate_openai_audio(self, text: str, output_path: str, voice_gender: str, voice_tone: str) -> bool:
        """Generate audio using OpenAI TTS as fallback"""
        try:
            # OpenAI voice mapping
            openai_voices = {
                ('male', 'energetic'): 'onyx',
                ('male', 'calm'): 'echo', 
                ('male', 'authoritative'): 'onyx',
                ('female', 'energetic'): 'nova',
                ('female', 'friendly'): 'alloy',
                ('female', 'professional'): 'shimmer',
                ('neutral', 'energetic'): 'alloy'
            }
            
            selected_voice = openai_voices.get((voice_gender, voice_tone), 'alloy')
            
            # Generate speech with OpenAI TTS
            response = self.openai_client.audio.speech.create(
                model="tts-1-hd",  # High quality model
                voice=selected_voice,
                input=text,
                speed=1.1  # Slightly faster for energy
            )
            
            # Save to file
            response.stream_to_file(output_path)
            
            print(f"✅ OpenAI TTS audio generated: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ OpenAI TTS generation failed: {e}")
            return False

    def enhance_autonomous_audio(self, input_file: str, output_file: str, audio_config: Dict[str, Any]):
        """Enhance audio based on AI architectural decisions"""
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
            
            print(f"🔊 Enhancing audio with {energy_level} energy profile...")
            
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
                print(f"✅ Audio enhanced: {output_file}")
                
        except Exception as e:
            print(f"Warning: Audio enhancement failed: {e}")
            shutil.copy2(input_file, output_file)

    def create_unified_audio(self, audio_segments: list, target_duration: float, output_path: str) -> bool:
        """Create unified 15-second audio track from segments"""
        try:
            if not audio_segments:
                print("❌ No audio segments provided")
                return False
            
            # Calculate total duration of segments
            total_duration = sum(seg.get('duration', 0) for seg in audio_segments)
            
            if total_duration == 0:
                print("❌ Total audio duration is 0")
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
            speed_filter = f"[concatenated]atempo={speed_factor:.3f}[adjusted]"
            
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
            
            print(f"🎵 Creating unified {target_duration}s audio from {len(audio_segments)} segments...")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Unified audio created: {output_path}")
                return True
            else:
                print(f"❌ Audio unification failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Audio unification error: {e}")
            return False