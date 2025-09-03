"""
  Luma AI video generation provider implementation.
"""

import os
import requests
import time
from typing import Dict, Any, Optional
from interfaces.video_generator import VideoGenerator


class LumaProvider(VideoGenerator):
    """Luma AI video generation service implementation."""
    
    def __init__(self):
        self.api_key = os.environ.get("LUMA_API_KEY")
        if not self.api_key:
            raise ValueError("LUMA_API_KEY environment variable is required")
        
        self.base_url = "https://api.lumalabs.ai/dream-machine/v1/generations"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def generate_video(self, prompt: str, aspect_ratio: str = "9:16", 
                      image_url: Optional[str] = None, force_unique: bool = False, **kwargs) -> str:
        """Generate video using Luma AI with enhanced prompts."""
        
        # Get duration from kwargs, default to 9s (Luma only supports 5s or 9s)
        duration = kwargs.get('duration', 9)
        # Luma API only accepts '5s' or '9s' duration
        if duration <= 5:
            scene_duration = 5
        else:
            scene_duration = 9
        
        # Enhance prompt for hyper-realistic professional promotional quality
        if force_unique:
            # Hyperrealistic cinematography elements focused on natural movement and quality
            cinematic_elements = [
                "hyperrealistic cinematography with smooth natural camera movements, no jitter or artificial motion artifacts",
                "photorealistic lighting with accurate shadow casting and natural reflections, perfect exposure balance",
                "cinematic depth of field with sharp focus transitions, natural bokeh effects, professional lens quality",
                "fluid camera tracking with stabilized footage, organic movement patterns, no robotic or unnatural motion",
                "realistic environmental lighting with proper color temperature, natural illumination consistency",
                "seamless temporal coherence across frames, consistent visual quality with no morphing or distortion"
            ]
            
            # Hyperrealistic technical elements focused on natural human behavior and physics
            technical_elements = [
                "natural human expressions and gestures with realistic facial micro-movements, authentic body language",
                "accurate object physics with realistic material properties, proper weight and momentum simulation",
                "consistent character appearance throughout with no face morphing or body distortions",
                "realistic environmental interactions with proper lighting response and shadow dynamics",
                "authentic muscle movement and natural walking patterns, no unnatural or robotic gestures",
                "perfect temporal consistency with smooth motion interpolation, fluid character movements"
            ]
            
            # Hyperrealistic quality modifiers focused on visual fidelity
            quality_modifiers = [
                "ultra-high definition 8K quality with perfect edge definition and zero compression artifacts",
                "photorealistic human skin texture with accurate pore detail and natural color variation",
                "flawless motion interpolation with smooth 60fps movement quality, no stuttering or frame drops",
                "cinema-grade visual fidelity with professional color accuracy and natural contrast ratios",
                "hyperrealistic material rendering with accurate surface properties and realistic wear patterns"
            ]
            
            import random
            selected_cinematic = random.choice(cinematic_elements)
            selected_technical = random.choice(technical_elements)
            selected_quality = random.choice(quality_modifiers)
            
            enhanced_prompt = f"{prompt}, {selected_cinematic}, {selected_technical}, {selected_quality}"
        else:
            enhanced_prompt = prompt
            
        payload = {
            "prompt": enhanced_prompt,
            "aspect_ratio": aspect_ratio,
            "model": "ray-2",  # Project preference: best-quality Ray 2
            "resolution": "720p",  # Project preference: 720p output
            "duration": f"{scene_duration}s"  # Dynamic duration based on target
        }
        
        if image_url:
            # Luma keyframes structure for initial frame input
            payload["keyframes"] = {"frame0": {"type": "image", "url": image_url}}
        
        print(f"Starting Luma generation with prompt: {enhanced_prompt[:100]}...")
        print(f"Luma payload: {payload}")
        
        # Add connection timeout and retry logic for DNS resolution issues
        max_connection_retries = 3
        for connection_attempt in range(max_connection_retries):
            try:
                response = requests.post(
                    self.base_url, 
                    json=payload, 
                    headers=self.headers,
                    timeout=(10, 30)  # 10s connection timeout, 30s read timeout
                )
                break  # Success, exit retry loop
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as conn_error:
                if connection_attempt < max_connection_retries - 1:
                    wait_time = (connection_attempt + 1) * 2  # 2s, 4s, 6s
                    print(f"🔄 Connection attempt {connection_attempt + 1} failed, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Connection failed after {max_connection_retries} attempts: {conn_error}")
        
        if response.status_code != 201:
            error_text = response.text
            # Check for specific error conditions
            if response.status_code == 400:
                try:
                    error_data = response.json()
                    detail = error_data.get("detail", "")
                    if "Insufficient credits" in detail or "insufficient credits" in detail.lower():
                        raise Exception(f"Insufficient credits in Luma account. Please add credits to continue. Details: {detail}")
                    elif "rate limit" in detail.lower() or "too many requests" in detail.lower():
                        raise Exception(f"Rate limit exceeded for Luma API. Please wait before retrying. Details: {detail}")
                    else:
                        raise Exception(f"Luma API validation error: {detail}")
                except (ValueError, KeyError):
                    pass
            elif response.status_code == 429:
                raise Exception("Rate limit exceeded for Luma API. Please wait before retrying.")
            elif response.status_code == 402:
                raise Exception("Payment required for Luma API. Please add credits to your account.")
            
            raise Exception(f"Failed to create Luma generation: {response.status_code} - {error_text}")
        
        generation_data = response.json()
        generation_id = generation_data.get("id")
        
        if not generation_id:
            raise Exception("Could not extract generation ID from Luma response")
        
        print(f"Luma generation started with ID: {generation_id}")
        
        # Poll for completion with exponential backoff
        max_attempts = 60  # Allow more attempts but with smarter timing
        attempt = 0
        
        while attempt < max_attempts:
            # Exponential backoff: 3s, 5s, 8s, 10s, then 10s intervals
            if attempt < 5:
                sleep_time = 3 + attempt * 1.5  # 3s, 4.5s, 6s, 7.5s, 9s
            else:
                sleep_time = 10  # 10s for later attempts
            
            time.sleep(sleep_time)
            attempt += 1
            
            status_url = f"{self.base_url}/{generation_id}"
            
            # Add connection retry for status checks too
            try:
                status_response = requests.get(
                    status_url, 
                    headers=self.headers,
                    timeout=(5, 15)  # Shorter timeout for status checks
                )
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as status_error:
                print(f"🔄 Status check connection failed (attempt {attempt}): {status_error}")
                continue  # Continue to next attempt
            
            if status_response.status_code != 200:
                print(f"Warning: Luma status check failed: {status_response.status_code}")
                continue
            
            status_data = status_response.json()
            state = status_data.get("state", "unknown")
            
            print(f"Luma generation {generation_id} status: {state} (attempt {attempt}/{max_attempts})")
            
            if state == "completed":
                video_url = status_data.get("assets", {}).get("video")
                if video_url:
                    print(f"Luma video generation completed! URL: {video_url}")
                    return video_url
                else:
                    raise Exception("Generation completed but no video URL found")
            
            elif state in ["failed", "error"]:
                error_msg = status_data.get("failure_reason", "Unknown error")
                raise Exception(f"Luma video generation failed: {error_msg}")
        
        raise Exception(f"Luma video generation timed out after {max_attempts} attempts")
    
    def generate_professional_promotional_video(self, prompt: str, aspect_ratio: str = "9:16", 
                                               image_url: Optional[str] = None, **kwargs) -> str:
        """Generate video using Luma's best model with professional promotional quality enhancements."""
        print(f"🎬 Generating professional promotional video with Luma ray-2 (best quality)")
        print(f"📱 Resolution: 720p | Aspect Ratio: {aspect_ratio} | Model: ray-2 (best)")
        
        # Always use force_unique for professional promotional content
        # Remove force_unique from kwargs to avoid duplicate parameter
        kwargs_copy = kwargs.copy()
        kwargs_copy.pop('force_unique', None)  # Remove if it exists
        
        return self.generate_video(
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            image_url=image_url,
            force_unique=True,  # Always enhance for professional quality
            **kwargs_copy
        )
    
    def download_video(self, video_url: str, output_path: str, max_retries: int = 3) -> bool:
        """Download video from URL to local file."""
        for attempt in range(max_retries):
            try:
                print(f"Downloading Luma video (attempt {attempt + 1}/{max_retries})...")
                
                response = requests.get(video_url, stream=True, timeout=120)
                response.raise_for_status()
                
                with open(output_path, 'wb') as f:
                    total_size = 0
                    for chunk in response.iter_content(chunk_size=16384):
                        if chunk:
                            f.write(chunk)
                            total_size += len(chunk)
                
                print(f"Downloaded {total_size:,} bytes")
                
                if total_size > 100000:  # Basic validation
                    print(f"Video download successful!")
                    return True
                else:
                    print(f"Downloaded file too small, retrying...")
                    continue
                
            except Exception as e:
                print(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
        
        print(f"All {max_retries} download attempts failed")
        return False
    
    def text_to_video(self, prompt: str, aspect_ratio: str = "9:16") -> str:
        """Text to video generation."""
        return self.generate_video(prompt, aspect_ratio, image_url=None)
    
    def image_to_video(self, prompt: str, image_url: str, aspect_ratio: str = "9:16") -> str:
        """Image to video generation."""
        return self.generate_video(prompt, aspect_ratio, image_url=image_url)
