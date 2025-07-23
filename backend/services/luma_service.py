"""
Relicon AI Ad Creator - Luma AI Service
Ultra-high quality video generation using Luma Dream Machine
"""
import asyncio
import aiohttp
import time
from typing import Optional, Dict, Any, List
from pathlib import Path
from core.settings import settings
from core.models import SceneComponent


class LumaAIService:
    """
    Luma AI Service - Professional video generation
    
    Handles ultra-high quality video generation using Luma Dream Machine API
    with advanced prompt optimization and quality monitoring.
    """
    
    def __init__(self):
        self.api_key = settings.LUMA_API_KEY
        self.base_url = "https://api.lumalabs.ai/dream-machine/v1"
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Quality settings
        self.default_settings = {
            "aspect_ratio": "16:9",
            "loop": False,
            "keyframes": {
                "frame0": {
                    "type": "generation",
                    "url": None
                }
            }
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            timeout=aiohttp.ClientTimeout(total=300)  # 5 minute timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def generate_scene_video(
        self, 
        component: SceneComponent, 
        context: Dict[str, Any]
    ) -> Optional[str]:
        """
        Generate high-quality video for a scene component
        
        Args:
            component: Scene component with detailed specifications
            context: Brand and technical context
            
        Returns:
            Path to generated video file or None if failed
        """
        if not self.api_key:
            print("⚠️ Luma API key not configured, skipping video generation")
            return None
        
        print(f"🎬 Luma AI: Generating {component.visual_type} for {component.duration}s")
        
        try:
            # Optimize prompt for Luma AI
            optimized_prompt = self._optimize_prompt(component, context)
            
            # Create generation request
            generation_request = {
                "prompt": optimized_prompt,
                "aspect_ratio": self._get_aspect_ratio(context),
                "loop": False,
                "keyframes": {
                    "frame0": {
                        "type": "generation",
                        "url": None
                    }
                }
            }
            
            # Start generation
            generation_id = await self._start_generation(generation_request)
            if not generation_id:
                return None
            
            # Monitor progress and get result
            video_url = await self._wait_for_completion(generation_id, component.duration * 30)  # 30s per video second
            if not video_url:
                return None
            
            # Download and save video
            local_path = await self._download_video(video_url, component, context)
            
            print(f"✅ Luma AI: Generated video saved to {local_path}")
            return local_path
            
        except Exception as e:
            print(f"❌ Luma AI: Generation failed - {str(e)}")
            return None
    
    async def generate_batch_videos(
        self, 
        components: List[SceneComponent], 
        context: Dict[str, Any],
        max_concurrent: int = 3
    ) -> Dict[str, Optional[str]]:
        """
        Generate multiple videos concurrently with rate limiting
        
        Args:
            components: List of scene components
            context: Brand and technical context
            max_concurrent: Maximum concurrent generations
            
        Returns:
            Dictionary mapping component IDs to video paths
        """
        print(f"🎬 Luma AI: Starting batch generation of {len(components)} videos")
        
        # Create semaphore for rate limiting
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_with_limit(component: SceneComponent) -> tuple[str, Optional[str]]:
            async with semaphore:
                component_id = f"{component.visual_type}_{component.start_time}"
                result = await self.generate_scene_video(component, context)
                # Add delay between requests to respect rate limits
                await asyncio.sleep(2)
                return component_id, result
        
        # Execute all generations concurrently
        tasks = [generate_with_limit(comp) for comp in components]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        video_paths = {}
        successful = 0
        for result in results:
            if isinstance(result, Exception):
                print(f"❌ Batch generation error: {result}")
                continue
            
            component_id, video_path = result
            video_paths[component_id] = video_path
            if video_path:
                successful += 1
        
        print(f"✅ Luma AI: Batch complete - {successful}/{len(components)} videos generated")
        return video_paths
    
    def _optimize_prompt(self, component: SceneComponent, context: Dict[str, Any]) -> str:
        """
        Optimize prompt for Luma AI with advanced techniques
        
        Creates ultra-specific prompts that maximize video quality and relevance
        """
        base_prompt = component.luma_prompt or component.visual_prompt
        
        # Add technical optimizations
        optimizations = []
        
        # Duration-specific optimizations
        if component.duration <= 3:
            optimizations.append("quick, dynamic movement")
        elif component.duration <= 6:
            optimizations.append("smooth, controlled motion")
        else:
            optimizations.append("slow, cinematic movement")
        
        # Visual type optimizations
        if component.visual_type == "video":
            optimizations.append("professional cinematography")
        elif component.visual_type == "image":
            optimizations.append("high-resolution product shot")
        
        # Style optimizations
        brand_style = context.get("style", "professional")
        style_map = {
            "professional": "corporate, clean, polished lighting",
            "energetic": "vibrant colors, dynamic angles, high energy",
            "minimal": "clean composition, subtle movement, soft lighting",
            "cinematic": "dramatic lighting, film-quality, artistic composition"
        }
        optimizations.append(style_map.get(brand_style, "professional quality"))
        
        # Brand-specific optimizations
        if context.get("brand_colors"):
            color_hint = f"color scheme: {', '.join(context['brand_colors'][:2])}"
            optimizations.append(color_hint)
        
        # Combine optimizations
        optimization_text = ", ".join(optimizations)
        optimized_prompt = f"{base_prompt}, {optimization_text}, 4K quality, professional advertising"
        
        # Ensure prompt is within Luma's limits (typically 500 characters)
        if len(optimized_prompt) > 400:
            optimized_prompt = optimized_prompt[:397] + "..."
        
        return optimized_prompt
    
    def _get_aspect_ratio(self, context: Dict[str, Any]) -> str:
        """Get optimal aspect ratio for the context"""
        platform = context.get("platform", "universal")
        
        aspect_ratios = {
            "tiktok": "9:16",
            "instagram": "1:1", 
            "facebook": "16:9",
            "youtube_shorts": "9:16",
            "universal": "16:9"
        }
        
        return aspect_ratios.get(platform, "16:9")
    
    async def _start_generation(self, request: Dict[str, Any]) -> Optional[str]:
        """Start video generation and return generation ID"""
        if not self.session:
            await self.__aenter__()
        
        try:
            async with self.session.post(
                f"{self.base_url}/generations",
                json=request
            ) as response:
                
                if response.status == 201:
                    data = await response.json()
                    generation_id = data.get("id")
                    print(f"🎬 Luma AI: Generation started - ID: {generation_id}")
                    return generation_id
                else:
                    error_text = await response.text()
                    print(f"❌ Luma AI: Generation failed - {response.status}: {error_text}")
                    return None
                    
        except Exception as e:
            print(f"❌ Luma AI: Request failed - {str(e)}")
            return None
    
    async def _wait_for_completion(
        self, 
        generation_id: str, 
        timeout_seconds: int = 180
    ) -> Optional[str]:
        """Wait for generation completion and return video URL"""
        start_time = time.time()
        check_interval = 5  # Check every 5 seconds
        
        print(f"⏳ Luma AI: Waiting for generation {generation_id}")
        
        while (time.time() - start_time) < timeout_seconds:
            try:
                async with self.session.get(
                    f"{self.base_url}/generations/{generation_id}"
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        state = data.get("state")
                        
                        if state == "completed":
                            video_url = data.get("assets", {}).get("video")
                            if video_url:
                                print(f"✅ Luma AI: Generation completed - {generation_id}")
                                return video_url
                            else:
                                print(f"❌ Luma AI: No video URL in completed generation")
                                return None
                        
                        elif state == "failed":
                            failure_reason = data.get("failure_reason", "Unknown error")
                            print(f"❌ Luma AI: Generation failed - {failure_reason}")
                            return None
                        
                        elif state in ["queued", "dreaming"]:
                            print(f"⏳ Luma AI: Status - {state}")
                        
                        else:
                            print(f"🔄 Luma AI: Unknown state - {state}")
                    
                    else:
                        print(f"❌ Luma AI: Status check failed - {response.status}")
                        
            except Exception as e:
                print(f"❌ Luma AI: Status check error - {str(e)}")
            
            # Wait before next check
            await asyncio.sleep(check_interval)
        
        print(f"⏰ Luma AI: Generation timeout after {timeout_seconds}s")
        return None
    
    async def _download_video(
        self, 
        video_url: str, 
        component: SceneComponent, 
        context: Dict[str, Any]
    ) -> Optional[str]:
        """Download generated video to local storage"""
        try:
            # Create filename
            brand_name = context.get("brand_name", "brand").lower().replace(" ", "_")
            timestamp = int(time.time())
            filename = f"luma_{brand_name}_{component.visual_type}_{timestamp}.mp4"
            
            # Ensure output directory exists
            output_dir = Path(settings.OUTPUT_DIR) / "videos" / "luma"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = output_dir / filename
            
            # Download video
            async with self.session.get(video_url) as response:
                if response.status == 200:
                    with open(file_path, "wb") as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
                    
                    print(f"📁 Luma AI: Video downloaded - {file_path}")
                    return str(file_path)
                else:
                    print(f"❌ Luma AI: Download failed - {response.status}")
                    return None
                    
        except Exception as e:
            print(f"❌ Luma AI: Download error - {str(e)}")
            return None
    
    async def get_generation_status(self, generation_id: str) -> Dict[str, Any]:
        """Get current status of a generation"""
        if not self.session:
            await self.__aenter__()
        
        try:
            async with self.session.get(
                f"{self.base_url}/generations/{generation_id}"
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"Status check failed: {response.status}"}
                    
        except Exception as e:
            return {"error": f"Status check error: {str(e)}"}
    
    async def cancel_generation(self, generation_id: str) -> bool:
        """Cancel an ongoing generation"""
        if not self.session:
            await self.__aenter__()
        
        try:
            async with self.session.delete(
                f"{self.base_url}/generations/{generation_id}"
            ) as response:
                return response.status == 204
                
        except Exception as e:
            print(f"❌ Luma AI: Cancel error - {str(e)}")
            return False


# Global Luma AI service instance
luma_service = LumaAIService() 