"""
Luma AI Video Generation Service
Handles video generation using Luma AI API
"""
import os
import requests
import time
from typing import Dict, Any, Optional

class LumaVideoService:
    def __init__(self):
        self.api_key = os.environ.get("LUMA_API_KEY")
        if not self.api_key:
            raise ValueError("LUMA_API_KEY environment variable is required")
        
        self.base_url = "https://api.lumalabs.ai/dream-machine/v1/generations"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def generate_video(self, prompt: str, aspect_ratio: str = "9:16", force_unique: bool = True) -> str:
        """
        Generate ultra-realistic advertisement video using Luma AI
        Returns the video URL when complete
        """
        # Add uniqueness and realism to prompt
        if force_unique:
            import random
            import time
            
            # Add uniqueness tokens to prevent reuse
            unique_elements = [
                f"timestamp_{int(time.time())}",
                f"seed_{random.randint(10000, 99999)}",
                "ultra realistic",
                "advertisement quality",
                "professional cinematography",
                "commercial grade"
            ]
            
            enhanced_prompt = f"{prompt}, {', '.join(unique_elements[:3])}"
        else:
            enhanced_prompt = prompt
        
        # Create generation
        payload = {
            "prompt": enhanced_prompt,
            "aspect_ratio": aspect_ratio,
            "model": "ray-1-6"  # Latest model
        }
        
        print(f"🎬 Starting Luma generation with prompt: {prompt[:100]}...")
        
        response = requests.post(self.base_url, json=payload, headers=self.headers)
        
        if response.status_code != 201:
            raise Exception(f"Failed to create generation: {response.status_code} - {response.text}")
        
        generation_data = response.json()
        generation_id = generation_data["id"]
        
        print(f"✓ Generation started with ID: {generation_id}")
        
        # Poll for completion
        max_attempts = 60  # 10 minutes max
        attempt = 0
        
        while attempt < max_attempts:
            time.sleep(10)  # Wait 10 seconds between checks
            attempt += 1
            
            # Check status
            status_response = requests.get(f"{self.base_url}/{generation_id}", headers=self.headers)
            
            if status_response.status_code != 200:
                print(f"Warning: Status check failed: {status_response.status_code}")
                continue
            
            status_data = status_response.json()
            state = status_data.get("state", "unknown")
            
            print(f"🔄 Generation {generation_id} status: {state} (attempt {attempt}/{max_attempts})")
            
            if state == "completed":
                video_url = status_data.get("assets", {}).get("video")
                if video_url:
                    print(f"✅ Video generation completed! URL: {video_url}")
                    return video_url
                else:
                    raise Exception("Generation completed but no video URL found")
            
            elif state == "failed":
                failure_reason = status_data.get("failure_reason", "Unknown error")
                raise Exception(f"Video generation failed: {failure_reason}")
        
        raise Exception(f"Video generation timed out after {max_attempts} attempts")
    
    def download_video(self, video_url: str, output_path: str) -> bool:
        """
        Download video from URL to local file
        """
        try:
            print(f"📥 Downloading video from {video_url}")
            response = requests.get(video_url, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✅ Video downloaded to {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False