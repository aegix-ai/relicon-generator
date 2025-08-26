"""
  Central provider manager for AI service abstraction.
  Enables seamless switching between providers via configuration.
"""

import os
from typing import Optional, Type
from config.settings import settings
from interfaces.video_generator import VideoGenerator
from interfaces.audio_generator import AudioGenerator
from interfaces.text_generator import TextGenerator


class ProviderManager:
    """Central manager for AI service providers."""
    
    def __init__(self):
        self._video_provider: Optional[VideoGenerator] = None
        self._audio_provider: Optional[AudioGenerator] = None
        self._text_provider: Optional[TextGenerator] = None
    
    def get_video_generator(self, provider_name: Optional[str] = None) -> VideoGenerator:
        """Get the configured video generation provider."""
        if self._video_provider is None or (provider_name and provider_name != self._video_provider.__class__.__name__.lower().replace('provider', '')):
            if provider_name:
                self._video_provider = self._create_video_provider(provider_name)
            else:
                provider_name = getattr(settings, 'VIDEO_PROVIDER', 'hailuo').lower()
                self._video_provider = self._create_video_provider(provider_name)
        return self._video_provider
    
    def get_video_provider(self) -> VideoGenerator:
        """Alias for get_video_generator for compatibility."""
        return self.get_video_generator()
    
    def get_audio_generator(self) -> AudioGenerator:
        """Get the configured audio generation provider."""
        if self._audio_provider is None:
            provider_name = getattr(settings, 'AUDIO_PROVIDER', 'elevenlabs').lower()
            self._audio_provider = self._create_audio_provider(provider_name)
        return self._audio_provider
    
    def get_text_generator(self) -> TextGenerator:
        """Get the configured text generation provider."""
        if self._text_provider is None:
            provider_name = getattr(settings, 'TEXT_PROVIDER', 'openai').lower()
            self._text_provider = self._create_text_provider(provider_name)
        return self._text_provider
    
    def _create_video_provider(self, provider_name: str) -> VideoGenerator:
        """Create video provider instance based on configuration."""
        if provider_name == 'runway':
            from providers.runway import RunwayProvider
            return RunwayProvider()
        elif provider_name == 'hailuo':
            from providers.hailuo import HailuoProvider
            return HailuoProvider()
        elif provider_name == 'luma':
            from providers.luma import LumaProvider
            return LumaProvider()
        else:
            raise ValueError(f"Unknown video provider: {provider_name}")
    
    def _create_audio_provider(self, provider_name: str) -> AudioGenerator:
        """Create audio provider instance based on configuration."""
        if provider_name == 'elevenlabs':
            from providers.elevenlabs import ElevenLabsProvider
            return ElevenLabsProvider()
        else:
            raise ValueError(f"Unknown audio provider: {provider_name}")
    
    def _create_text_provider(self, provider_name: str) -> TextGenerator:
        """Create text provider instance based on configuration."""
        if provider_name == 'openai':
            from providers.openai import OpenAIProvider
            return OpenAIProvider()
        else:
            raise ValueError(f"Unknown text provider: {provider_name}")
    
    def set_video_provider(self, provider_name: str) -> None:
        """Change video provider at runtime."""
        self._video_provider = self._create_video_provider(provider_name)
    
    def set_audio_provider(self, provider_name: str) -> None:
        """Change audio provider at runtime."""
        self._audio_provider = self._create_audio_provider(provider_name)
    
    def set_text_provider(self, provider_name: str) -> None:
        """Change text provider at runtime."""
        self._text_provider = self._create_text_provider(provider_name)


# Global provider manager instance
provider_manager = ProviderManager()
