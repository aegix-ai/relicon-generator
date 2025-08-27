"""
  OpenAI text generation provider implementation with professional prompt engineering.
  Enhanced with ML-powered brand intelligence, quality validation, and monitoring.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from openai import OpenAI
from interfaces.text_generator import TextGenerator
from core.enhanced_planning_service import EnhancedPlanningService
from core.brand_intelligence import brand_intelligence
from core.validators import validator
from core.monitoring import monitoring
from core.cache import cache
from core.logger import get_logger

logger = get_logger(__name__)


class OpenAIProvider(TextGenerator):
    """
    Enhanced OpenAI GPT text generation service with ML-powered capabilities.
    
    Features:
    - Neural brand intelligence integration
    - ML-powered quality validation
    - Real-time performance monitoring
    - Intelligent caching with ML optimization
    - Professional prompt engineering with brand alignment
    """
    
    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        try:
            self.client = OpenAI(api_key=api_key)
            self.enhanced_planner = EnhancedPlanningService()
            
            # Performance metrics tracking
            self.performance_metrics = {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'average_response_time': 0.0,
                'cache_hits': 0,
                'ml_enhancement_usage': 0,
                'quality_validation_runs': 0
            }
            
            logger.info(
                "Enhanced OpenAI Provider initialized successfully",
                action="openai.provider.init",
                ml_capabilities_available=True,
                brand_intelligence_enabled=True,
                quality_validation_enabled=True
            )
            
        except Exception as e:
            logger.error(f"OpenAI client initialization failed: {e}", exc_info=True)
            raise ValueError(f"Failed to initialize OpenAI client: {e}")
    
    def architect_complete_video(
        self, 
        brand_info: Dict[str, Any], 
        enable_ml_enhancements: bool = True,
        enable_quality_validation: bool = True,
        logo_file_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create complete video architecture with ML-powered enhancements.
        
        Features:
        - Neural brand intelligence and competitive analysis
        - ML-powered prompt optimization
        - Quality validation with scoring
        - Performance monitoring and caching
        - Brand archetype classification
        
        Args:
            brand_info: Brand information dictionary
            enable_ml_enhancements: Enable ML-powered brand intelligence
            enable_quality_validation: Enable quality validation network
            logo_file_path: Optional logo file for visual analysis
            
        Returns:
            Enhanced video architecture with quality metrics
        """
        import time
        start_time = time.time()
        
        try:
            # Update metrics
            self.performance_metrics['total_requests'] += 1
            
            # Step 1: Input validation with quality scoring
            logger.info("Starting ML-enhanced video architecture creation", 
                       action="openai.architect.start",
                       brand_name=brand_info.get('brand_name', 'unknown'),
                       ml_enabled=enable_ml_enhancements,
                       quality_validation=enable_quality_validation)
            
            validation_result, quality_score = validator.validate_with_quality_scoring(
                {'brand_info': brand_info, 'logo_file_path': logo_file_path},
                enable_quality_validation
            )
            
            if not validation_result.is_valid:
                raise ValueError(f"Input validation failed: {'; '.join(validation_result.errors)}")
            
            sanitized_brand_info = validation_result.sanitized_value.get('brand_info', {})
            
            if quality_score and enable_quality_validation:
                self.performance_metrics['quality_validation_runs'] += 1
                logger.info("Input quality assessment completed",
                           action="openai.quality.input",
                           quality_score=quality_score.overall_score,
                           quality_level=quality_score.details.get('quality_level', 'unknown'))
            
            # Step 2: Check cache first
            cache_key = f"video_architecture:{hash(str(sanitized_brand_info))}"
            cached_result = cache.get(cache_key)
            
            if cached_result:
                self.performance_metrics['cache_hits'] += 1
                logger.info("Architecture retrieved from cache",
                           action="openai.cache.hit",
                           cache_key=cache_key[:16])
                return cached_result
            
            # Step 3: ML-powered brand intelligence (if enabled)
            brand_analysis = None
            if enable_ml_enhancements:
                try:
                    self.performance_metrics['ml_enhancement_usage'] += 1
                    
                    # Get brand intelligence analysis
                    brand_analysis = brand_intelligence.analyze_brand_comprehensive(
                        brand_name=sanitized_brand_info.get('brand_name', ''),
                        brand_description=sanitized_brand_info.get('brand_description', '')
                    )
                    
                    logger.info("Brand intelligence analysis completed",
                               action="openai.brand_intelligence.complete",
                               niche=brand_analysis.get('niche_classification', {}).get('primary_niche', 'unknown'),
                               archetype=brand_analysis.get('brand_archetype', 'unknown'),
                               competitive_position=brand_analysis.get('competitive_intelligence', {}).get('market_position', 'unknown'))
                    
                except Exception as e:
                    logger.warning(f"Brand intelligence analysis failed, continuing without: {e}")
                    brand_analysis = None
            
            target_duration = min(sanitized_brand_info.get('duration', 30), 30)
            
            # Step 4: Create architecture using enhanced planning service
            try:
                professional_blueprint = self.enhanced_planner.create_professional_video_blueprint(
                    brand_info=sanitized_brand_info,
                    target_duration=target_duration,
                    service_type="luma",
                    logo_file_path=logo_file_path,
                    enable_quality_validation=enable_quality_validation,
                    enable_prompt_optimization=enable_ml_enhancements
                )
                
                # Step 5: Integrate brand intelligence results
                if brand_analysis:
                    professional_blueprint['brand_intelligence'] = brand_analysis
                    
                    # Enhance scenes with brand insights
                    scenes = professional_blueprint.get('scene_architecture', {}).get('scenes', [])
                    for scene in scenes:
                        # Add brand alignment hints to prompts
                        if brand_analysis.get('brand_archetype'):
                            archetype = brand_analysis['brand_archetype']
                            scene['brand_archetype_alignment'] = archetype
                        
                        # Add competitive positioning context
                        competitive_info = brand_analysis.get('competitive_intelligence', {})
                        if competitive_info.get('market_position'):
                            scene['market_positioning'] = competitive_info['market_position']
                
                # Step 6: Quality validation of complete architecture
                if enable_quality_validation:
                    arch_validation, quality_details = validator.validate_architecture_with_quality(
                        professional_blueprint, 
                        enable_quality_validation
                    )
                    
                    if quality_details:
                        professional_blueprint['quality_validation'] = quality_details
                        
                        logger.info("Architecture quality validation completed",
                                   action="openai.quality.architecture",
                                   overall_quality=quality_details.get('overall_quality_level', 'unknown'))
                
                # Step 7: Cache the successful result
                cache.set(cache_key, professional_blueprint, ttl_seconds=3600)  # Cache for 1 hour
                
                # Step 8: Update performance metrics
                processing_time = time.time() - start_time
                self._update_performance_metrics(processing_time, True)
                
                # Step 9: Monitor quality and performance
                monitoring.track_quality_metrics({
                    'provider': 'openai',
                    'architecture_quality': quality_details.get('overall_quality_level', 'unknown') if quality_details else 'not_assessed',
                    'processing_time': processing_time,
                    'ml_enhanced': enable_ml_enhancements,
                    'brand_intelligence_used': brand_analysis is not None,
                    'cache_used': False
                })
                
                logger.info("ML-enhanced video architecture created successfully",
                           action="openai.architect.success",
                           brand_name=sanitized_brand_info.get('brand_name'),
                           processing_time=processing_time,
                           ml_enhanced=enable_ml_enhancements,
                           quality_validated=enable_quality_validation)
                
                return professional_blueprint
                
            except Exception as enhanced_error:
                logger.error(f"Enhanced planning failed: {enhanced_error}",
                           action="openai.enhanced.error",
                           exc_info=True)
                
                # Fallback to legacy method
                logger.info("Falling back to legacy architecture method")
                legacy_result = self._legacy_architect_complete_video(sanitized_brand_info)
                
                # Still perform quality validation on legacy result if requested
                if enable_quality_validation:
                    try:
                        arch_validation, quality_details = validator.validate_architecture_with_quality(
                            legacy_result, enable_quality_validation
                        )
                        if quality_details:
                            legacy_result['quality_validation'] = quality_details
                    except Exception as quality_error:
                        logger.warning(f"Quality validation of legacy result failed: {quality_error}")
                
                # Add fallback metadata
                legacy_result['production_metadata'] = {
                    'fallback_mode': True,
                    'enhanced_planning_error': str(enhanced_error),
                    'ml_enhanced': False
                }
                
                processing_time = time.time() - start_time
                self._update_performance_metrics(processing_time, False)
                
                return legacy_result
        
        except Exception as e:
            processing_time = time.time() - start_time
            self.performance_metrics['failed_requests'] += 1
            
            logger.error(f"Video architecture creation failed: {e}",
                        action="openai.architect.error",
                        processing_time=processing_time,
                        exc_info=True)
            
            raise
    
    def _legacy_architect_complete_video(self, brand_info: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy video architecture method (fallback only)."""
        target_duration = brand_info.get('duration', 30)
        
        # Expert modern advertising strategist prompt - product-focused commercial direction
        creative_prompt = f"""You are a world-class modern advertising strategist specializing in product-focused commercials that drive conversions.

Your mission: Create a blueprint for **one single modern advertisement** that showcases the product brilliantly with minimal talking heads.

Critical rules:
- The ad must follow: Hook → Pain → Solution → Call to Action structure
- Exactly **3 scenes**, each **5-6 seconds long** (total 15-18 seconds)
- Focus on PRODUCT DEMONSTRATION and lifestyle integration
- Minimize face-focused shots - prioritize product, environment, and action
- Show the product in use, its benefits, and real-world impact

Brand Information:
- Brand: {brand_info.get('brand_name', 'Brand')}
- Product/Service: {brand_info.get('brand_description', 'Product/service')}
- Target Audience: {brand_info.get('target_audience', 'general')}
- Key Benefits: {brand_info.get('key_benefits', 'benefits')}
- Call to Action: {brand_info.get('call_to_action', 'Take action')}

MODERN AD STRUCTURE REQUIREMENTS:

Scene 1 - HOOK (5-6s): Eye-catching product introduction or lifestyle moment
- Show product in action, stunning environment, or intriguing situation
- Focus on visual impact, not faces - wide shots, product close-ups, lifestyle context
- Create immediate curiosity about the product/service

Scene 2 - PAIN/PROBLEM (5-6s): Show the problem your product solves
- Demonstrate frustration, inefficiency, or limitation WITHOUT close-up faces
- Focus on situations, environments, actions that show the pain point
- Show hands, body language, problematic situations, environmental challenges

Scene 3 - SOLUTION + CTA (5-6s): Product solving the problem + clear call to action
- Demonstrate product benefits through action, results, transformation
- Show the product working, environment improving, lifestyle enhancing
- End with strong product branding and clear call to action

VISUAL DIRECTION REQUIREMENTS:
 Product Focus: Always show the actual product, its features, and benefits in action
 Lifestyle Integration: Show how product fits into real life scenarios
 Cinematic Quality: Professional lighting, camera movement, composition
 Environmental Storytelling: Use settings and situations to tell the story
 Action-Oriented: Show hands using product, situations changing, results happening
 Visual Variety: Mix wide shots, product close-ups, environmental shots, action sequences

For each scene, generate complete modern commercial architecture:
- scene_id: numeric index (1, 2, 3)
- duration: exactly 5-6 seconds (must total 15-18s)
- visual_concept: detailed cinematic description with camera, lighting, emotion, environment
- luma_prompt: cinema-quality prompt with specific cinematography direction, human expressions, lighting, and realistic details
- script_line: powerful narration aligned with 5-6s timing (15-18 words per scene max)
- audio_cues: cinematic sound design, ambient audio that enhances realism
- music_cues: emotional music direction that supports the visual storytelling

Audio Architecture Requirements:
- Total audio duration: exactly 15-18 seconds
- Total script: 45-55 words maximum across all 3 scenes
- Narration/Voiceover must fit within each 5-6s scene (15-18 words per scene)
- Music Layer: 15-18s emotional arc (Hook → Problem/Solution → Resolve)
- Sound FX: transitions between 5-6s segments
- All audio layers perfectly synchronized to 3 × 5-6s scene structure

Validation Requirements:
- Exactly 3 scenes, each exactly 5-6 seconds
- Total duration = 15-18 seconds (video + audio)
- Total script: 45-55 words maximum
- Each script_line fits within 5-6s scene timing (15-18 words max)
- No overspill between scenes
- One cohesive story arc across 15-18 seconds

Respond in JSON format:
{{
    "creative_vision": {{
        "overall_concept": "complete brand narrative concept",
        "visual_style": "cinematic style choice",
        "color_palette": ["primary_color", "secondary_color", "accent_color"],
        "mood": "emotional tone and energy",
        "brand_story_arc": "beginning-middle-end emotional journey"
    }},
    "audio_architecture": {{
        "voice_gender": "male/female",
        "voice_tone": "energetic/professional/warm",
        "energy_level": "high/medium/low",
        "music_style": "cinematic/electronic/orchestral",
        "tempo_bpm": 120,
        "total_duration": 18
    }},
    "scene_architecture": {{
        "total_duration": 18,
        "scenes": [
            {{
                "scene_id": 1,
                "duration": 6,
                "visual_concept": "modern commercial visual concept focusing on product introduction, lifestyle context, environmental storytelling, and eye-catching hook without face focus",
                "luma_prompt": "modern commercial prompt featuring product demonstration, lifestyle integration, professional cinematography, environmental action, and visual impact without talking heads",
                "script_line": "precise 6-second narration for hook scene (15-18 words max)",
                "audio_cues": "background sounds and ambient audio",
                "music_cues": "tempo, mood, instruments for this 6s segment"
            }},
            {{
                "scene_id": 2,
                "duration": 6,
                "visual_concept": "modern commercial visual concept showing problem/pain point through environmental situations, product absence, and lifestyle challenges without face-focused shots",
                "luma_prompt": "modern commercial prompt demonstrating problems through action, environmental context, situational challenges, and lifestyle pain points with minimal face shots",
                "script_line": "precise 6-second narration for problem/solution scene (15-18 words max)",
                "audio_cues": "background sounds and ambient audio", 
                "music_cues": "tempo, mood, instruments for this 6s segment"
            }},
            {{
                "scene_id": 3,
                "duration": 6,
                "visual_concept": "modern commercial visual concept showing product solution in action, lifestyle transformation, clear call-to-action, and product branding without face focus",
                "luma_prompt": "modern commercial prompt featuring product benefits demonstration, lifestyle improvement, clear branding, compelling call-to-action, and transformation results with minimal face shots",
                "script_line": "precise 6-second narration for resolution/CTA scene (15-18 words max)",
                "audio_cues": "background sounds and ambient audio",
                "music_cues": "tempo, mood, instruments for this 6s segment"
            }}
        ]
    }},
    "unified_script": "Complete 18-second narration script combining all 3 scene script lines, perfectly timed for 18s duration (45-55 words total)"
}}"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": creative_prompt}],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        
        content = response.choices[0].message.content
        
        # Use safe JSON parsing with better error messages
        try:
            if not content or not content.strip():
                raise ValueError("Empty architecture response")
            
            content = content.strip()
            
            # Try to extract JSON if response contains other text
            if not content.startswith('{'):
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    content = json_match.group()
                else:
                    print(f"No JSON found in OpenAI response: {content[:200]}...")
                    raise ValueError("No valid JSON found in architecture response")
            
            architecture = json.loads(content)
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error from OpenAI architecture: {e}")
            print(f"Raw response content: {content[:500] if content else 'No content'}")
            raise ValueError(f"Failed to parse architecture JSON: {e}")
        
        print(f"DEBUG: OpenAI raw response total_duration = {architecture.get('scene_architecture', {}).get('total_duration', 'MISSING')}")
        
        # CRITICAL: Force exactly 18s architecture regardless of OpenAI response
        scenes = architecture.get('scene_architecture', {}).get('scenes', [])
        
        # Ensure exactly 3 scenes with 6s each (3×6s = 18s total)
        while len(scenes) < 3:
            scenes.append({
                'scene_id': len(scenes) + 1,
                'duration': 6,
                'visual_concept': f'Professional scene {len(scenes) + 1}',
                'luma_prompt': f'High-quality scene {len(scenes) + 1}',
                'script_line': f'Scene {len(scenes) + 1} narration',
                'audio_cues': 'Professional audio',
                'music_cues': 'Enterprise music'
            })
        scenes = scenes[:3]  # Limit to exactly 3 scenes
        
        # Force exactly 6s per scene (3×6s = 18s total)
        for i, scene in enumerate(scenes):
            scene['duration'] = 6
            scene['scene_id'] = i + 1
        
        # Force total duration to 18s (3×6s scenes)
        architecture['scene_architecture']['scenes'] = scenes
        architecture['scene_architecture']['total_duration'] = 18
        architecture['audio_architecture']['total_duration'] = 18
        
        print(f"DEBUG: OpenAI forced total_duration = {architecture['scene_architecture']['total_duration']}")
        
        return architecture
    
    def create_cutting_edge_prompts(self, architecture: Dict[str, Any], service_type: str = "hailuo") -> List[Dict[str, Any]]:
        """Generate service-specific prompts for each scene."""
        scenes = architecture.get('scene_architecture', {}).get('scenes', [])
        enhanced_scenes = []
        
        for i, scene in enumerate(scenes):
            # Use existing luma_prompt or visual_concept
            prompt = scene.get('luma_prompt') or scene.get('visual_concept', 'Professional commercial scene')
            
            # Optimize prompt length for service type
            if service_type.lower() == "hailuo":
                # Hailuo prefers shorter, concise prompts
                words = prompt.split()[:20]  # Max 20 words
                optimized_prompt = " ".join(words)
            else:
                # Luma can handle longer prompts
                optimized_prompt = prompt[:300]  # Max 300 chars
            
            enhanced_scene = {
                **scene,
                'hailuo_prompt': optimized_prompt,
                'luma_prompt': prompt,
                'scene_number': i + 1
            }
            enhanced_scenes.append(enhanced_scene)
        
        return enhanced_scenes
    
    def optimize_scene_prompts(self, scenes: List[Dict[str, Any]], service_type: str = "hailuo") -> List[Dict[str, Any]]:
        """
        Optimize scene prompts for specific video generation services.
        Removes legacy prompt generation and focuses on efficiency.
        
        Args:
            scenes: List of scene dictionaries
            service_type: Target service for optimization
            
        Returns:
            Optimized scenes with service-specific prompts
        """
        optimized_scenes = []
        
        for scene in scenes:
            # Get base prompt from luma_prompt or visual_concept
            base_prompt = scene.get('luma_prompt') or scene.get('visual_concept', '')
            
            if service_type.lower() == "hailuo":
                # Hailuo optimization: extract product-focused commercial essence
                # Prioritize: product, action, demonstration, environment, benefits
                words = base_prompt.split()
                
                # Extract essential commercial elements
                essential_keywords = []
                for word in words:
                    if any(key in word.lower() for key in ['product', 'demonstration', 'lifestyle', 'action', 'benefits', 'commercial', 'branding', 'transformation', 'solution', 'professional', 'modern', 'showcase']):
                        essential_keywords.append(word)
                
                # Build concise but product-focused prompt
                core_action = ' '.join(words[:8])  # Core product/action
                commercial_elements = ' '.join(essential_keywords[:4])  # Key commercial elements
                
                if commercial_elements:
                    scene['hailuo_prompt'] = f"{core_action}, {commercial_elements}"
                else:
                    scene['hailuo_prompt'] = ' '.join(words[:12])  # Fallback to first 12 words
                    
                scene['luma_prompt'] = base_prompt  # Keep full commercial description
            else:
                # Luma optimization: detailed descriptions
                scene['luma_prompt'] = base_prompt[:250]  # Optimal length for Luma
                scene['hailuo_prompt'] = " ".join(base_prompt.split()[:15])
            
            optimized_scenes.append(scene)
        
        return optimized_scenes
    
    def _update_performance_metrics(self, response_time: float, success: bool = True):
        """Update performance metrics for monitoring."""
        try:
            # Simple performance tracking
            if not hasattr(self, 'performance_metrics'):
                self.performance_metrics = {
                    'total_requests': 0,
                    'successful_requests': 0,
                    'average_response_time': 0.0,
                    'last_updated': time.time()
                }
            
            self.performance_metrics['total_requests'] += 1
            if success:
                self.performance_metrics['successful_requests'] += 1
            
            # Update average response time
            current_avg = self.performance_metrics['average_response_time']
            total_requests = self.performance_metrics['total_requests']
            new_avg = ((current_avg * (total_requests - 1)) + response_time) / total_requests
            self.performance_metrics['average_response_time'] = new_avg
            self.performance_metrics['last_updated'] = time.time()
            
        except Exception as e:
            logger.warning(f"Performance metrics update failed: {e}")
