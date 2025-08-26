"""
  ML-Enhanced High-level planning service for video architecture.
  Orchestrates creative planning using ML-powered providers with neural prompt optimization.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from core.provider_manager import provider_manager
from core.enhanced_planning_service import EnhancedPlanningService
from core.logger import get_logger
from core.monitoring import monitoring
from config.settings import settings

logger = get_logger(__name__)


class PlanningService:
    """ML-enhanced high-level planning orchestration service with neural optimization."""
    
    def __init__(self):
        self.provider_manager = provider_manager
        self.enhanced_planner = EnhancedPlanningService()
        self.performance_metrics = {
            'total_blueprints': 0,
            'success_rate': 0.0,
            'average_quality': 0.0,
            'ml_optimization_usage': 0.0
        }
    
    def create_video_architecture(self, brand_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        DEPRECATED: Use create_enterprise_blueprint() for new implementations.
        Legacy method maintained for backward compatibility.
        """
        print("Using deprecated method. Consider switching to create_enterprise_blueprint()")
        return self.create_enterprise_blueprint(brand_info)
    
    def validate_architecture(self, architecture: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and correct video architecture with enterprise-grade precision.
        
        Args:
            architecture: Video architecture to validate
            
        Returns:
            Validated and corrected architecture with 30-second precision
        """
        try:
            # Validate required enterprise architecture fields
            required_fields = ['creative_vision', 'audio_architecture', 'scene_architecture', 'unified_script']
            for field in required_fields:
                if field not in architecture:
                    raise ValueError(f"Missing required field: {field}")
            
            # Validate scene architecture structure
            scene_arch = architecture.get('scene_architecture', {})
            scenes = scene_arch.get('scenes', [])
            target_duration = scene_arch.get('total_duration', 30)
            
            if not scenes:
                raise ValueError("No scenes found in architecture")
            
            # Validate 30-second precision timing
            actual_total = sum(scene.get('duration', 0) for scene in scenes)
            if actual_total != target_duration:
                print(f"Duration precision error: expected {target_duration}s, got {actual_total}s - auto-correcting")
                architecture = self._correct_scene_timing(architecture, target_duration)
            
            # Validate enterprise scene requirements
            for i, scene in enumerate(scenes):
                scene_id = i + 1
                required_scene_fields = ['visual_concept', 'script_line', 'brand_alignment', 'duration']
                
                for field in required_scene_fields:
                    if not scene.get(field):
                        print(f"Scene {scene_id} missing required field: {field}")
                
                # Ensure visual prompts exist (luma_prompt is primary)
                if not scene.get('luma_prompt') and not scene.get('visual_concept'):
                    print(f"Scene {scene_id} missing visual prompts")
                
                # Validate scene duration precision
                duration = scene.get('duration', 0)
                if duration <= 0 or duration > target_duration:
                    print(f"Scene {scene_id} invalid duration: {duration}s")
            
            # Validate audio architecture
            audio_arch = architecture.get('audio_architecture', {})
            if audio_arch.get('total_duration', 0) != target_duration:
                audio_arch['total_duration'] = target_duration
                print(f"Corrected audio duration to {target_duration}s")
            
            # Validate unified script quality
            script = architecture.get('unified_script', '').strip()
            if len(script) < 20:
                print("Script too short - may not fill 30 seconds")
            
            print(f"Architecture validated: {len(scenes)} scenes, {target_duration}s precision")
            return architecture
            
        except Exception as e:
            logger.error(f"Architecture validation failed: {e}", exc_info=True)
            raise
    
    def _correct_scene_timing(self, architecture: Dict[str, Any], target_duration: int) -> Dict[str, Any]:
        """
        Correct scene timing with intelligent distribution for narrative flow.
        Maintains storytelling structure: Hook-Build-Transform-Resolve.
        """
        scenes = architecture.get('scene_architecture', {}).get('scenes', [])
        scene_count = len(scenes)
        
        if scene_count == 0:
            return architecture
        
        # Force exactly 3 scenes for optimal 18s structure  
        if scene_count != 3:
            print(f"Adjusting from {scene_count} to exactly 3 scenes for 18s structure")
            # Take first 3 scenes if more than 3, or duplicate if less than 3
            if scene_count > 3:
                scenes = scenes[:3]
                scene_count = 3
            elif scene_count < 3:
                # Duplicate scenes to reach 3
                while len(scenes) < 3:
                    scenes.append(scenes[0].copy())
                scene_count = 3
        
        # Perfect 30s distribution for 3 scenes: Hook(10s) - Problem/Solution(10s) - Resolve(10s)
        # Using 10s per scene
        durations = [10, 10, 10]
        
        # Apply corrected durations and update scenes array
        for i, scene in enumerate(scenes):
            scene['duration'] = durations[i] if i < len(durations) else durations[0]
        
        # Update the scenes array in architecture
        architecture['scene_architecture']['scenes'] = scenes
        architecture['scene_architecture']['total_duration'] = target_duration
        
        actual_total = sum(scene.get('duration', 0) for scene in scenes)
        print(f"Applied narrative-optimized timing: {scene_count} scenes, {actual_total}s total")
        return architecture
    
    def create_enterprise_blueprint(
        self, 
        brand_info: Dict[str, Any], 
        enable_ml_optimization: bool = True,
        enable_quality_validation: bool = True,
        logo_file_path: Optional[str] = None,
        video_provider: Optional[str] = None,
        creative_brief_mode: str = "professional"  # professional, luxury, innovative
    ) -> Dict[str, Any]:
        """
        Create ML-enhanced enterprise-grade video blueprint with neural optimization.
        Uses advanced brand intelligence and competitive analysis for maximum accuracy.
        
        Args:
            brand_info: Brand information dictionary
            enable_ml_optimization: Enable neural prompt optimization
            enable_quality_validation: Enable quality validation network
            logo_file_path: Optional logo file path for visual analysis
            
        Returns:
            ML-enhanced enterprise video blueprint with quality scoring
        """
        try:
            start_time = datetime.utcnow()
            
            logger.info(
                "Starting ML-enhanced enterprise blueprint creation",
                action="enterprise.blueprint.start",
                brand_name=brand_info.get('brand_name', 'unknown'),
                ml_optimization=enable_ml_optimization,
                quality_validation=enable_quality_validation
            )
            
            # Validate input requirements with enhanced validation
            self._validate_enterprise_input(brand_info)
            
            # Enforce 30-second maximum constraint
            target_duration = min(brand_info.get('duration', 30), 30)
            brand_info['duration'] = target_duration
            
            # Set enterprise defaults with ML insights
            brand_info.setdefault('target_audience', 'professional')
            brand_info.setdefault('call_to_action', 'Learn more')
            
            # Monitor system resources before processing
            monitoring.get_system_metrics()
            
            # Use ML-enhanced planning service for enterprise results
            try:
                blueprint = self.enhanced_planner.create_professional_video_blueprint(
                    brand_info=brand_info,
                    target_duration=target_duration,
                    service_type=video_provider or "hailuo",  # Use provided or Hailuo as default
                    logo_file_path=logo_file_path,
                    enable_quality_validation=enable_quality_validation,
                    enable_prompt_optimization=enable_ml_optimization,
                    creative_brief_mode=creative_brief_mode  # Professional GPT-4o creative level
                )
                
                # Calculate processing time
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                
                # Update performance metrics
                self._update_performance_metrics(blueprint, processing_time, enable_ml_optimization)
                
                logger.info(
                    "ML-enhanced enterprise blueprint created successfully",
                    action="enterprise.blueprint.complete",
                    brand_name=brand_info.get('brand_name'),
                    duration=target_duration,
                    processing_time_seconds=processing_time,
                    quality_score=blueprint.get('quality_validation', {}).get('overall_quality_score', 0),
                    ml_enhanced=True
                )
                
                return blueprint
                
            except Exception as enhanced_error:
                logger.error(
                    f"ML-enhanced planning failed: {enhanced_error}",
                    action="enterprise.blueprint.ml_error",
                    brand_name=brand_info.get('brand_name'),
                    exc_info=True
                )
                
                # Fallback to legacy AI director method
                try:
                    text_generator = self.provider_manager.get_text_generator()
                    blueprint = text_generator.architect_complete_video(brand_info)
                    
                    # Enterprise validation and optimization
                    blueprint = self.validate_architecture(blueprint)
                    
                    # Add production metadata
                    blueprint['production_metadata'] = {
                        'architect_version': '2.0_legacy_fallback',
                        'target_duration': target_duration,
                        'scene_count': len(blueprint.get('scene_architecture', {}).get('scenes', [])),
                        'validated': True,
                        'enterprise_grade': True,
                        'fallback_mode': True,
                        'ml_optimization_failed': True,
                        'fallback_reason': str(enhanced_error)
                    }
                    
                    processing_time = (datetime.utcnow() - start_time).total_seconds()
                    
                    logger.warning(
                        "Legacy enterprise blueprint created as fallback",
                        action="enterprise.blueprint.fallback",
                        brand_name=brand_info.get('brand_name'),
                        scene_count=blueprint['production_metadata']['scene_count'],
                        duration=target_duration,
                        processing_time_seconds=processing_time
                    )
                    
                    return blueprint
                    
                except Exception as fallback_error:
                    logger.error(
                        f"Fallback blueprint creation also failed: {fallback_error}",
                        action="enterprise.blueprint.total_failure",
                        exc_info=True
                    )
                    raise enhanced_error  # Raise original ML error
            
        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            logger.error(
                f"Enterprise blueprint creation failed: {e}",
                action="enterprise.blueprint.error",
                brand_name=brand_info.get('brand_name', 'unknown'),
                processing_time_seconds=processing_time,
                exc_info=True
            )
            
            # Update failure metrics
            self.performance_metrics['total_blueprints'] += 1
            
            raise
    
    def _validate_enterprise_input(self, brand_info: Dict[str, Any]) -> None:
        """Enhanced enterprise input validation."""
        required_brand_fields = ['brand_name', 'brand_description']
        for field in required_brand_fields:
            if not brand_info.get(field):
                raise ValueError(f"Missing required brand field: {field}")
        
        # Validate brand name
        brand_name = brand_info.get('brand_name', '').strip()
        if len(brand_name) < 2:
            raise ValueError("Brand name must be at least 2 characters")
        
        # Validate brand description
        brand_description = brand_info.get('brand_description', '').strip()
        if len(brand_description) < 20:
            raise ValueError("Brand description must be at least 20 characters for accurate analysis")
        
        if len(brand_description) > 2000:
            raise ValueError("Brand description too long (max 2000 characters)")
    
    def _update_performance_metrics(self, blueprint: Dict[str, Any], processing_time: float, ml_enabled: bool) -> None:
        """Update performance metrics with blueprint results."""
        try:
            self.performance_metrics['total_blueprints'] += 1
            total_blueprints = self.performance_metrics['total_blueprints']
            
            # Update success rate
            current_success_rate = self.performance_metrics['success_rate']
            self.performance_metrics['success_rate'] = (
                (current_success_rate * (total_blueprints - 1) + 1.0) / total_blueprints
            )
            
            # Update average quality
            quality_score = blueprint.get('quality_validation', {}).get('overall_quality_score', 0.5)
            current_avg_quality = self.performance_metrics['average_quality']
            self.performance_metrics['average_quality'] = (
                (current_avg_quality * (total_blueprints - 1) + quality_score) / total_blueprints
            )
            
            # Update ML optimization usage
            current_ml_usage = self.performance_metrics['ml_optimization_usage']
            ml_value = 1.0 if ml_enabled else 0.0
            self.performance_metrics['ml_optimization_usage'] = (
                (current_ml_usage * (total_blueprints - 1) + ml_value) / total_blueprints
            )
            
            logger.debug(
                "Performance metrics updated",
                action="metrics.update",
                total_blueprints=total_blueprints,
                success_rate=self.performance_metrics['success_rate'],
                avg_quality=self.performance_metrics['average_quality'],
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.warning(f"Failed to update performance metrics: {e}")
    
    def get_performance_insights(self) -> Dict[str, Any]:
        """Get comprehensive performance insights and recommendations."""
        try:
            insights = {
                'current_metrics': self.performance_metrics.copy(),
                'enhanced_planner_stats': self.enhanced_planner.get_ml_performance_stats(),
                'optimization_insights': self.enhanced_planner.get_optimization_insights(),
                'system_health': monitoring.get_health_status(),
                'recommendations': []
            }
            
            # Generate recommendations
            avg_quality = self.performance_metrics['average_quality']
            ml_usage = self.performance_metrics['ml_optimization_usage']
            
            if avg_quality < 0.7:
                insights['recommendations'].append("Consider enabling ML optimization for all blueprints")
            
            if ml_usage < 0.5:
                insights['recommendations'].append("Increase ML optimization usage for better quality")
            
            if avg_quality > 0.85:
                insights['recommendations'].append("Quality is excellent - consider A/B testing for further optimization")
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to get performance insights: {e}")
            return {'error': str(e)}
    
    def create_ab_test_blueprint(
        self,
        brand_info: Dict[str, Any],
        test_name: str,
        enable_ml_a: bool = True,
        enable_ml_b: bool = False
    ) -> Dict[str, Any]:
        """Create A/B test comparing ML vs non-ML blueprint generation."""
        try:
            variant_a_config = {
                'target_duration': 30,
                'service_type': 'luma',
                'enable_quality_validation': True,
                'enable_prompt_optimization': enable_ml_a
            }
            
            variant_b_config = {
                'target_duration': 30,
                'service_type': 'luma', 
                'enable_quality_validation': True,
                'enable_prompt_optimization': enable_ml_b
            }
            
            return self.enhanced_planner.create_ab_test_blueprint(
                brand_info=brand_info,
                test_name=test_name,
                variant_a_config=variant_a_config,
                variant_b_config=variant_b_config
            )
            
        except Exception as e:
            logger.error(f"A/B test blueprint creation failed: {e}", exc_info=True)
            raise
    
    def switch_provider(self, provider_name: str) -> None:
        """
        Switch planning provider at runtime with enhanced logging.
        
        Args:
            provider_name: Name of the provider ('openai', etc.)
        """
        try:
            self.provider_manager.set_text_provider(provider_name)
            logger.info(
                f"Switched to {provider_name} planning provider",
                action="provider.switch",
                provider=provider_name
            )
        except Exception as e:
            logger.error(
                f"Failed to switch to {provider_name}: {e}",
                action="provider.switch.error",
                provider=provider_name,
                exc_info=True
            )
            raise
    
    def get_planning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive planning service statistics."""
        return {
            'performance_metrics': self.performance_metrics,
            'enhanced_planner_active': bool(self.enhanced_planner),
            'ml_optimization_available': hasattr(self.enhanced_planner, 'prompt_optimizer'),
            'quality_validation_available': hasattr(self.enhanced_planner, 'quality_validator'),
            'ab_testing_available': hasattr(self.enhanced_planner, 'ab_testing')
        }
