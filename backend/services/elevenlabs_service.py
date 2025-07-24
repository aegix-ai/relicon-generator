"""
Relicon AI Ad Creator - ElevenLabs Service  
Ultra-realistic voice synthesis with advanced audio processing
"""
import asyncio
import aiohttp
import json
import time
from typing import Optional, Dict, Any, List
from pathlib import Path
from core.settings import settings
from core.models import SceneComponent


class ElevenLabsService:
    """
    ElevenLabs Service - Professional voice synthesis
    
    Handles ultra-realistic voice generation with advanced audio processing,
    emotion control, and perfect timing synchronization.
    """
    
    def __init__(self):
        self.api_key = settings.ELEVENLABS_API_KEY
        self.base_url = "https://api.elevenlabs.io/v1"
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Premium voice IDs (these would be your actual ElevenLabs voice IDs)
        self.premium_voices = {
            "neutral": {
                "professional": "21m00Tcm4TlvDq8ikWAM",  # Rachel - professional female
                "energetic": "AZnzlk1XvdvUeBnXmlld",     # Domi - energetic
                "casual": "EXAVITQu4vr4xnSDxMaL"        # Bella - casual
            },
            "female": {
                "professional": "21m00Tcm4TlvDq8ikWAM",  # Rachel
                "energetic": "ThT5KcBeYPX3keUQqHPh",     # Dorothy - energetic
                "casual": "XrExE9yKIg1WjnnlVkGX"         # Matilda - casual
            },
            "male": {
                "professional": "pNInz6obpgDQGcFmaJgB",  # Adam - professional
                "energetic": "onwK4e9ZLuTAKqWW03F9",     # Daniel - energetic  
                "casual": "IKne3meq5aSn9XLyUdCD"         # Charlie - casual
            }
        }
        
        # Voice settings for different contexts
        self.voice_settings = {
            "professional": {"stability": 0.5, "similarity_boost": 0.8, "style": 0.2},
            "energetic": {"stability": 0.3, "similarity_boost": 0.9, "style": 0.8},
            "casual": {"stability": 0.7, "similarity_boost": 0.7, "style": 0.4},
            "urgent": {"stability": 0.2, "similarity_boost": 0.9, "style": 0.9}
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            headers={
                "xi-api-key": self.api_key,
                "Content-Type": "application/json"
            },
            timeout=aiohttp.ClientTimeout(total=120)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def generate_component_audio(
        self, 
        component: SceneComponent, 
        context: Dict[str, Any]
    ) -> Optional[str]:
        """
        Generate high-quality audio for a scene component
        
        Args:
            component: Scene component with voiceover requirements
            context: Brand and technical context
            
        Returns:
            Path to generated audio file or None if failed
        """
        if not self.api_key or not component.has_voiceover or not component.voiceover_text:
            return None
        
        print(f"🎙️ ElevenLabs: Generating audio for {component.duration}s")
        
        try:
            # Select optimal voice
            voice_id = self._select_voice(component, context)
            
            # Optimize text for target duration
            optimized_text = self._optimize_text_for_duration(
                component.voiceover_text, 
                component.duration
            )
            
            # Generate audio with advanced settings
            audio_data = await self._generate_speech(
                optimized_text, 
                voice_id, 
                component, 
                context
            )
            
            if not audio_data:
                return None
            
            # Save audio file
            audio_path = await self._save_audio(audio_data, component, context)
            
            # Post-process audio for perfect timing
            final_path = await self._post_process_audio(
                audio_path, 
                component.duration, 
                context
            )
            
            print(f"✅ ElevenLabs: Audio generated - {final_path}")
            return final_path
            
        except Exception as e:
            print(f"❌ ElevenLabs: Generation failed - {str(e)}")
            return None
    
    async def generate_batch_audio(
        self, 
        components: List[SceneComponent], 
        context: Dict[str, Any],
        max_concurrent: int = 5
    ) -> Dict[str, Optional[str]]:
        """
        Generate multiple audio files concurrently
        
        Args:
            components: List of scene components with voiceover
            context: Brand and technical context
            max_concurrent: Maximum concurrent generations
            
        Returns:
            Dictionary mapping component IDs to audio file paths
        """
        voiceover_components = [c for c in components if c.has_voiceover and c.voiceover_text]
        
        if not voiceover_components:
            return {}
        
        print(f"🎙️ ElevenLabs: Starting batch generation of {len(voiceover_components)} audio files")
        
        # Create semaphore for rate limiting
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_with_limit(component: SceneComponent) -> tuple[str, Optional[str]]:
            async with semaphore:
                component_id = f"audio_{component.start_time}"
                result = await self.generate_component_audio(component, context)
                # Small delay to respect rate limits
                await asyncio.sleep(0.5)
                return component_id, result
        
        # Execute all generations
        tasks = [generate_with_limit(comp) for comp in voiceover_components]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        audio_paths = {}
        successful = 0
        for result in results:
            if isinstance(result, Exception):
                print(f"❌ Batch audio error: {result}")
                continue
            
            component_id, audio_path = result
            audio_paths[component_id] = audio_path
            if audio_path:
                successful += 1
        
        print(f"✅ ElevenLabs: Batch complete - {successful}/{len(voiceover_components)} audio files generated")
        return audio_paths
    
    def _select_voice(self, component: SceneComponent, context: Dict[str, Any]) -> str:
        """Select optimal voice based on component and context"""
        # Get voice preference from context
        voice_preference = context.get("voice_preference", "neutral")
        
        # Determine voice style based on component tone
        voice_tone = component.voice_tone or "professional"
        
        # Map tone to style
        tone_to_style = {
            "confident": "professional",
            "engaging": "energetic", 
            "supportive": "casual",
            "urgent": "energetic",
            "friendly": "casual"
        }
        
        voice_style = tone_to_style.get(voice_tone, "professional")
        
        # Get voice ID
        voice_category = self.premium_voices.get(voice_preference, self.premium_voices["neutral"])
        voice_id = voice_category.get(voice_style, voice_category["professional"])
        
        return voice_id
    
    def _optimize_text_for_duration(self, text: str, target_duration: float) -> str:
        """Optimize text to fit target duration"""
        # Average speaking rate: 2.5 words per second for natural speech
        target_words = int(target_duration * 2.5)
        words = text.split()
        
        if len(words) <= target_words:
            return text
        
        # Truncate to fit duration while maintaining meaning
        optimized_words = words[:target_words]
        
        # Ensure we end on a complete thought
        for i in range(len(optimized_words) - 1, 0, -1):
            if optimized_words[i].endswith(('.', '!', '?')):
                optimized_words = optimized_words[:i+1]
                break
        
        return " ".join(optimized_words)
    
    async def _generate_speech(
        self, 
        text: str, 
        voice_id: str, 
        component: SceneComponent, 
        context: Dict[str, Any]
    ) -> Optional[bytes]:
        """Generate speech with advanced settings"""
        if not self.session:
            await self.__aenter__()
        
        # Get voice settings based on component tone
        voice_tone = component.voice_tone or "professional"
        settings = self.voice_settings.get(voice_tone, self.voice_settings["professional"])
        
        # Prepare request
        request_data = {
            "text": text,
            "model_id": "eleven_multilingual_v2",  # High-quality model
            "voice_settings": settings
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/text-to-speech/{voice_id}",
                json=request_data
            ) as response:
                
                if response.status == 200:
                    audio_data = await response.read()
                    print(f"🎙️ ElevenLabs: Speech generated ({len(audio_data)} bytes)")
                    return audio_data
                else:
                    error_text = await response.text()
                    print(f"❌ ElevenLabs: Generation failed - {response.status}: {error_text}")
                    return None
                    
        except Exception as e:
            print(f"❌ ElevenLabs: Request failed - {str(e)}")
            return None
    
    async def _save_audio(
        self, 
        audio_data: bytes, 
        component: SceneComponent, 
        context: Dict[str, Any]
    ) -> str:
        """Save audio data to file"""
        # Create filename
        brand_name = context.get("brand_name", "brand").lower().replace(" ", "_")
        timestamp = int(time.time())
        filename = f"elevenlabs_{brand_name}_{component.start_time}_{timestamp}.mp3"
        
        # Ensure output directory exists
        output_dir = Path(settings.OUTPUT_DIR) / "audio" / "elevenlabs"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = output_dir / filename
        
        # Save audio file
        with open(file_path, "wb") as f:
            f.write(audio_data)
        
        return str(file_path)
    
    async def _post_process_audio(
        self, 
        audio_path: str, 
        target_duration: float, 
        context: Dict[str, Any]
    ) -> str:
        """Post-process audio for perfect timing and quality"""
        try:
            # Import audio processing library
            from pydub import AudioSegment
            
            # Load audio
            audio = AudioSegment.from_mp3(audio_path)
            current_duration = len(audio) / 1000.0  # Convert to seconds
            
            # Adjust duration if needed (with tolerance)
            if abs(current_duration - target_duration) > 0.5:
                if current_duration > target_duration:
                    # Trim audio
                    audio = audio[:int(target_duration * 1000)]
                else:
                    # Add silence padding
                    silence_needed = int((target_duration - current_duration) * 1000)
                    silence = AudioSegment.silent(duration=silence_needed)
                    audio = audio + silence
            
            # Apply audio enhancements
            audio = self._enhance_audio_quality(audio, context)
            
            # Save processed audio
            processed_path = audio_path.replace(".mp3", "_processed.mp3")
            audio.export(processed_path, format="mp3", bitrate="192k")
            
            return processed_path
            
        except ImportError:
            print("⚠️ Audio processing library not available, using original audio")
            return audio_path
        except Exception as e:
            print(f"⚠️ Audio post-processing failed: {e}, using original audio")
            return audio_path
    
    def _enhance_audio_quality(self, audio: 'AudioSegment', context: Dict[str, Any]) -> 'AudioSegment':
        """Enhance audio quality with professional processing"""
        try:
            # Normalize audio levels
            audio = audio.normalize()
            
            # Apply subtle compression for consistency
            # This is a simplified version - in production you'd use more sophisticated processing
            
            # Ensure consistent volume
            target_dBFS = -20.0
            change_in_dBFS = target_dBFS - audio.dBFS
            audio = audio.apply_gain(change_in_dBFS)
            
            return audio
            
        except Exception as e:
            print(f"⚠️ Audio enhancement failed: {e}")
            return audio
    
    async def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get list of available voices"""
        if not self.session:
            await self.__aenter__()
        
        try:
            async with self.session.get(f"{self.base_url}/voices") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("voices", [])
                else:
                    return []
                    
        except Exception as e:
            print(f"❌ ElevenLabs: Failed to get voices - {str(e)}")
            return []
    
    async def get_voice_settings(self, voice_id: str) -> Dict[str, Any]:
        """Get current settings for a voice"""
        if not self.session:
            await self.__aenter__()
        
        try:
            async with self.session.get(
                f"{self.base_url}/voices/{voice_id}/settings"
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {}
                    
        except Exception as e:
            print(f"❌ ElevenLabs: Failed to get voice settings - {str(e)}")
            return {}


# Global ElevenLabs service instance
elevenlabs_service = ElevenLabsService() 
