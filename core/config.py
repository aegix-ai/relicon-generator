"""
Professional Configuration Management System
Environment-aware configuration with validation and type safety.
"""

import os
import json
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
from core.exceptions import ConfigurationError

class Environment(Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"

class ServiceTier(Enum):
    """Service tier configurations."""
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"

@dataclass
class VideoConfig:
    """Video generation configuration."""
    max_duration_seconds: int = 30
    default_scenes: int = 3
    scene_duration_seconds: int = 10
    default_service: str = "hailuo"
    fallback_service: str = "luma"
    quality_preset: str = "professional"
    resolution: str = "1080p"
    frame_rate: int = 30
    supported_formats: List[str] = field(default_factory=lambda: ["mp4", "mov", "avi"])

@dataclass
class LogoConfig:
    """Logo analysis configuration."""
    max_file_size_mb: int = 10
    supported_formats: List[str] = field(default_factory=lambda: ["PNG", "JPG", "JPEG", "GIF", "BMP", "WEBP"])
    analysis_timeout_seconds: int = 30
    color_extraction_limit: int = 5
    enable_style_detection: bool = True
    enable_font_suggestions: bool = True

@dataclass
class BrandConfig:
    """Brand intelligence configuration."""
    min_description_length: int = 20
    max_description_length: int = 2000
    confidence_threshold: float = 0.7
    enable_visual_keywords: bool = True
    enable_personality_analysis: bool = True
    cache_analysis_results: bool = True

@dataclass
class APIConfig:
    """API configuration."""
    request_timeout_seconds: int = 300
    max_retries: int = 3
    retry_delay_seconds: int = 2
    rate_limit_requests_per_minute: int = 60
    enable_cors: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])

@dataclass
class DatabaseConfig:
    """PostgreSQL database configuration."""
    enable_database: bool = True
    host: str = "localhost"
    port: int = 5432
    database: str = "relicon"
    username: str = "relicon_user"
    password: str = "relicon_password"
    
    # Connection pooling settings
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600  # 1 hour
    pool_pre_ping: bool = True
    
    # SSL settings
    ssl_mode: str = "prefer"  # prefer, require, disable
    ssl_cert: Optional[str] = None
    ssl_key: Optional[str] = None
    ssl_ca: Optional[str] = None
    
    # Performance settings
    statement_timeout: int = 30000  # 30 seconds in milliseconds
    lock_timeout: int = 10000  # 10 seconds in milliseconds
    idle_in_transaction_session_timeout: int = 60000  # 60 seconds
    
    # Backup and maintenance
    enable_backup_retention: bool = True
    backup_retention_days: int = 30
    enable_auto_vacuum: bool = True
    vacuum_cost_delay: int = 20
    
    @property
    def connection_url(self) -> str:
        """Generate SQLAlchemy connection URL."""
        url = f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        if self.ssl_mode != "disable":
            url += f"?sslmode={self.ssl_mode}"
        return url

@dataclass
class CacheConfig:
    """Caching configuration."""
    enable_redis: bool = False
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    default_ttl_seconds: int = 3600
    brand_analysis_ttl_seconds: int = 86400  # 24 hours
    logo_analysis_ttl_seconds: int = 604800  # 7 days

@dataclass
class CostConfig:
    """Cost tracking and optimization configuration."""
    enable_cost_tracking: bool = True
    cost_alert_threshold: float = 100.0
    budget_limit_daily: Optional[float] = None
    enable_cost_optimization: bool = True
    service_cost_weights: Dict[str, float] = field(default_factory=lambda: {
        "luma": 1.0,
        "hailuo": 0.7,
        "openai": 0.1
    })

@dataclass
class MLModelConfig:
    """Machine Learning model configuration."""
    enable_ml_optimization: bool = True
    enable_neural_classification: bool = True
    enable_quality_validation: bool = True
    enable_anomaly_detection: bool = True
    
    # Model performance settings
    classification_confidence_threshold: float = 0.7
    quality_validation_threshold: float = 0.6
    anomaly_detection_sensitivity: float = 0.1
    
    # Training and inference settings
    model_update_interval_hours: int = 168  # 1 week
    inference_batch_size: int = 32
    max_inference_time_seconds: float = 5.0
    
    # Neural network architecture
    hidden_layer_sizes: List[int] = field(default_factory=lambda: [128, 64, 32])
    activation_function: str = "relu"
    learning_rate: float = 0.001
    max_iterations: int = 1000
    
    # Feature engineering
    max_text_features: int = 5000
    ngram_range: Tuple[int, int] = (1, 3)
    min_document_frequency: int = 2
    max_document_frequency: float = 0.95

@dataclass
class QualityConfig:
    """Quality monitoring and validation configuration."""
    enable_quality_monitoring: bool = True
    enable_regression_detection: bool = True
    enable_automated_alerts: bool = True
    
    # Quality thresholds
    minimum_quality_score: float = 0.6
    good_quality_threshold: float = 0.75
    excellent_quality_threshold: float = 0.9
    
    # Brand intelligence thresholds
    minimum_brand_confidence: float = 0.7
    niche_detection_confidence: float = 0.8
    competitive_analysis_threshold: float = 0.6
    
    # Regression detection settings
    baseline_sample_size: int = 100
    regression_detection_window: int = 20
    acceptable_quality_deviation: float = 0.05  # 5%
    warning_deviation_threshold: float = 0.10   # 10%
    critical_deviation_threshold: float = 0.20  # 20%
    
    # Performance thresholds
    max_acceptable_processing_time: float = 30.0  # seconds
    minimum_success_rate: float = 0.85
    maximum_error_rate: float = 0.05
    
    # Quality trend analysis
    trend_analysis_window: int = 50
    volatility_threshold: float = 0.2
    improvement_detection_ratio: float = 0.8
    
    # Automated response settings
    enable_automated_fallback: bool = True
    critical_regression_threshold: float = 0.3
    emergency_mode_threshold: float = 0.5

@dataclass
class MonitoringConfig:
    """Enhanced monitoring and health check configuration."""
    enable_health_checks: bool = True
    health_check_interval_seconds: int = 30
    enable_metrics: bool = True
    metrics_port: int = 9090
    enable_distributed_tracing: bool = True
    
    # Quality monitoring
    enable_quality_monitoring: bool = True
    quality_metrics_retention_hours: int = 168  # 1 week
    quality_alert_cooldown_minutes: int = 15
    
    # Alert configuration
    enable_email_alerts: bool = False
    enable_slack_alerts: bool = False
    enable_webhook_alerts: bool = False
    alert_webhook_url: Optional[str] = None
    
    # Performance monitoring
    enable_performance_profiling: bool = True
    profile_slow_requests: bool = True
    slow_request_threshold_seconds: float = 5.0

@dataclass
class AdPlatformConfig:
    """Ad platform API integration configuration for Phase 2."""
    
    # Platform API enablement
    enable_meta_integration: bool = False
    enable_tiktok_integration: bool = False
    enable_google_ads_integration: bool = False
    
    # Meta (Facebook/Instagram) configuration
    meta_app_id: Optional[str] = None
    meta_app_secret: Optional[str] = None
    meta_access_token: Optional[str] = None
    meta_api_version: str = "v18.0"
    meta_base_url: str = "https://graph.facebook.com"
    
    # TikTok for Business configuration
    tiktok_app_id: Optional[str] = None
    tiktok_secret: Optional[str] = None
    tiktok_access_token: Optional[str] = None
    tiktok_api_version: str = "v1.3"
    tiktok_base_url: str = "https://business-api.tiktok.com"
    
    # Google Ads configuration
    google_ads_customer_id: Optional[str] = None
    google_ads_developer_token: Optional[str] = None
    google_ads_client_id: Optional[str] = None
    google_ads_client_secret: Optional[str] = None
    google_ads_refresh_token: Optional[str] = None
    google_ads_api_version: str = "v14"
    google_ads_base_url: str = "https://googleads.googleapis.com"
    
    # Common API settings
    request_timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: int = 2
    rate_limit_requests_per_minute: int = 100
    
    # Performance data sync settings
    sync_performance_data: bool = True
    sync_interval_hours: int = 1
    batch_size: int = 1000
    historical_data_days: int = 30
    
    # Creative sync settings
    sync_creative_data: bool = True
    sync_creative_performance: bool = True
    auto_create_campaigns: bool = False
    auto_upload_creatives: bool = False
    
    # Performance prediction settings
    enable_performance_predictions: bool = True
    prediction_confidence_threshold: float = 0.7
    min_historical_data_points: int = 100

@dataclass
class ReliconConfig:
    """Main application configuration with ML enhancements."""
    environment: Environment = Environment.DEVELOPMENT
    service_tier: ServiceTier = ServiceTier.PROFESSIONAL
    debug: bool = False
    log_level: str = "INFO"
    
    # Component configurations
    video: VideoConfig = field(default_factory=VideoConfig)
    logo: LogoConfig = field(default_factory=LogoConfig)
    brand: BrandConfig = field(default_factory=BrandConfig)
    api: APIConfig = field(default_factory=APIConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Enhanced ML and quality configurations
    ml_models: MLModelConfig = field(default_factory=MLModelConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    
    # Phase 2: Ad platform integration
    ad_platforms: 'AdPlatformConfig' = field(default_factory=lambda: AdPlatformConfig())

class ConfigManager:
    """Professional configuration manager with environment support."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._detect_config_path()
        self._config: Optional[ReliconConfig] = None
        self._load_config()
    
    def _detect_config_path(self) -> str:
        """Detect configuration file path based on environment."""
        env = os.getenv("RELICON_ENV", "development").lower()
        
        config_files = [
            f"config/relicon.{env}.json",
            f"config/relicon.json",
            "relicon.config.json",
            ".relicon.json"
        ]
        
        for config_file in config_files:
            if Path(config_file).exists():
                return config_file
        
        # Return default path
        return "config/relicon.json"
    
    def _load_config(self):
        """Load configuration from file and environment variables."""
        # Start with default config
        self._config = ReliconConfig()
        
        # Load from file if exists
        if Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    file_config = json.load(f)
                self._merge_config(file_config)
            except Exception as e:
                raise ConfigurationError(f"Failed to load config file {self.config_path}: {e}")
        
        # Override with environment variables
        self._load_env_overrides()
        
        # Validate configuration
        self._validate_config()
    
    def _merge_config(self, file_config: Dict[str, Any]):
        """Merge file configuration with default configuration."""
        
        # Environment and tier
        if "environment" in file_config:
            self._config.environment = Environment(file_config["environment"])
        
        if "service_tier" in file_config:
            self._config.service_tier = ServiceTier(file_config["service_tier"])
        
        if "debug" in file_config:
            self._config.debug = bool(file_config["debug"])
        
        if "log_level" in file_config:
            self._config.log_level = file_config["log_level"]
        
        # Component configurations
        for component in ["video", "logo", "brand", "api", "database", "cache", "cost", "monitoring", "ml_models", "quality", "ad_platforms"]:
            if component in file_config:
                self._merge_component_config(component, file_config[component])
    
    def _merge_component_config(self, component: str, component_config: Dict[str, Any]):
        """Merge component-specific configuration."""
        current_config = getattr(self._config, component)
        
        for key, value in component_config.items():
            if hasattr(current_config, key):
                setattr(current_config, key, value)
    
    def _load_env_overrides(self):
        """Load configuration overrides from environment variables."""
        
        # Main config overrides
        env_mappings = {
            "RELICON_ENV": ("environment", lambda x: Environment(x)),
            "RELICON_DEBUG": ("debug", lambda x: x.lower() == "true"),
            "RELICON_LOG_LEVEL": ("log_level", str),
            
            # Video config
            "RELICON_VIDEO_MAX_DURATION": ("video.max_duration_seconds", int),
            "RELICON_VIDEO_DEFAULT_SERVICE": ("video.default_service", str),
            "RELICON_VIDEO_QUALITY": ("video.quality_preset", str),
            
            # API config
            "RELICON_API_TIMEOUT": ("api.request_timeout_seconds", int),
            "RELICON_API_MAX_RETRIES": ("api.max_retries", int),
            "RELICON_API_RATE_LIMIT": ("api.rate_limit_requests_per_minute", int),
            
            # Database config
            "RELICON_DATABASE_ENABLE": ("database.enable_database", lambda x: x.lower() == "true"),
            "RELICON_DATABASE_HOST": ("database.host", str),
            "RELICON_DATABASE_PORT": ("database.port", int),
            "RELICON_DATABASE_NAME": ("database.database", str),
            "RELICON_DATABASE_USER": ("database.username", str),
            "RELICON_DATABASE_PASSWORD": ("database.password", str),
            "RELICON_DATABASE_POOL_SIZE": ("database.pool_size", int),
            "RELICON_DATABASE_SSL_MODE": ("database.ssl_mode", str),
            
            # Cache config
            "RELICON_REDIS_ENABLE": ("cache.enable_redis", lambda x: x.lower() == "true"),
            "RELICON_REDIS_HOST": ("cache.redis_host", str),
            "RELICON_REDIS_PORT": ("cache.redis_port", int),
            
            # Cost config
            "RELICON_COST_TRACKING": ("cost.enable_cost_tracking", lambda x: x.lower() == "true"),
            "RELICON_COST_ALERT_THRESHOLD": ("cost.cost_alert_threshold", float),
            "RELICON_BUDGET_DAILY": ("cost.budget_limit_daily", float),
            
            # ML model config
            "RELICON_ML_OPTIMIZATION": ("ml_models.enable_ml_optimization", lambda x: x.lower() == "true"),
            "RELICON_NEURAL_CLASSIFICATION": ("ml_models.enable_neural_classification", lambda x: x.lower() == "true"),
            "RELICON_QUALITY_VALIDATION": ("ml_models.enable_quality_validation", lambda x: x.lower() == "true"),
            "RELICON_ML_CONFIDENCE_THRESHOLD": ("ml_models.classification_confidence_threshold", float),
            "RELICON_MAX_INFERENCE_TIME": ("ml_models.max_inference_time_seconds", float),
            
            # Quality config
            "RELICON_QUALITY_MONITORING": ("quality.enable_quality_monitoring", lambda x: x.lower() == "true"),
            "RELICON_REGRESSION_DETECTION": ("quality.enable_regression_detection", lambda x: x.lower() == "true"),
            "RELICON_MIN_QUALITY_SCORE": ("quality.minimum_quality_score", float),
            "RELICON_QUALITY_THRESHOLD": ("quality.good_quality_threshold", float),
            "RELICON_MAX_PROCESSING_TIME": ("quality.max_acceptable_processing_time", float),
            "RELICON_MIN_SUCCESS_RATE": ("quality.minimum_success_rate", float),
            
            # Ad platform config
            "RELICON_ENABLE_META_INTEGRATION": ("ad_platforms.enable_meta_integration", lambda x: x.lower() == "true"),
            "RELICON_ENABLE_TIKTOK_INTEGRATION": ("ad_platforms.enable_tiktok_integration", lambda x: x.lower() == "true"),
            "RELICON_ENABLE_GOOGLE_ADS_INTEGRATION": ("ad_platforms.enable_google_ads_integration", lambda x: x.lower() == "true"),
            
            # Meta (Facebook/Instagram) credentials
            "RELICON_META_APP_ID": ("ad_platforms.meta_app_id", str),
            "RELICON_META_APP_SECRET": ("ad_platforms.meta_app_secret", str),
            "RELICON_META_ACCESS_TOKEN": ("ad_platforms.meta_access_token", str),
            "RELICON_META_API_VERSION": ("ad_platforms.meta_api_version", str),
            
            # TikTok for Business credentials
            "RELICON_TIKTOK_APP_ID": ("ad_platforms.tiktok_app_id", str),
            "RELICON_TIKTOK_SECRET": ("ad_platforms.tiktok_secret", str),
            "RELICON_TIKTOK_ACCESS_TOKEN": ("ad_platforms.tiktok_access_token", str),
            "RELICON_TIKTOK_API_VERSION": ("ad_platforms.tiktok_api_version", str),
            
            # Google Ads credentials
            "RELICON_GOOGLE_ADS_CUSTOMER_ID": ("ad_platforms.google_ads_customer_id", str),
            "RELICON_GOOGLE_ADS_DEVELOPER_TOKEN": ("ad_platforms.google_ads_developer_token", str),
            "RELICON_GOOGLE_ADS_CLIENT_ID": ("ad_platforms.google_ads_client_id", str),
            "RELICON_GOOGLE_ADS_CLIENT_SECRET": ("ad_platforms.google_ads_client_secret", str),
            "RELICON_GOOGLE_ADS_REFRESH_TOKEN": ("ad_platforms.google_ads_refresh_token", str),
            
            # Ad platform performance settings
            "RELICON_AD_SYNC_PERFORMANCE": ("ad_platforms.sync_performance_data", lambda x: x.lower() == "true"),
            "RELICON_AD_SYNC_INTERVAL": ("ad_platforms.sync_interval_hours", int),
            "RELICON_AD_BATCH_SIZE": ("ad_platforms.batch_size", int),
            "RELICON_AD_HISTORICAL_DAYS": ("ad_platforms.historical_data_days", int),
            "RELICON_AD_PERFORMANCE_PREDICTIONS": ("ad_platforms.enable_performance_predictions", lambda x: x.lower() == "true")
        }
        
        for env_var, (config_path, type_converter) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    converted_value = type_converter(env_value)
                    self._set_nested_config(config_path, converted_value)
                except (ValueError, TypeError) as e:
                    raise ConfigurationError(f"Invalid environment variable {env_var}={env_value}: {e}")
    
    def _set_nested_config(self, config_path: str, value: Any):
        """Set nested configuration value using dot notation."""
        parts = config_path.split('.')
        current = self._config
        
        for part in parts[:-1]:
            current = getattr(current, part)
        
        setattr(current, parts[-1], value)
    
    def _validate_config(self):
        """Validate configuration values."""
        
        # Validate video config
        if self._config.video.max_duration_seconds > 300:
            raise ConfigurationError("Maximum video duration cannot exceed 300 seconds")
        
        if self._config.video.default_scenes < 1 or self._config.video.default_scenes > 10:
            raise ConfigurationError("Number of scenes must be between 1 and 10")
        
        # Validate logo config
        if self._config.logo.max_file_size_mb > 50:
            raise ConfigurationError("Maximum logo file size cannot exceed 50MB")
        
        # Validate brand config
        if self._config.brand.min_description_length < 10:
            raise ConfigurationError("Minimum brand description length must be at least 10 characters")
        
        # Validate API config
        if self._config.api.request_timeout_seconds < 10:
            raise ConfigurationError("API timeout must be at least 10 seconds")
        
        # Validate cost config
        if self._config.cost.cost_alert_threshold < 0:
            raise ConfigurationError("Cost alert threshold must be non-negative")
        
        # Validate ML model config
        if self._config.ml_models.classification_confidence_threshold < 0.0 or self._config.ml_models.classification_confidence_threshold > 1.0:
            raise ConfigurationError("Classification confidence threshold must be between 0.0 and 1.0")
        
        if self._config.ml_models.max_inference_time_seconds <= 0:
            raise ConfigurationError("Max inference time must be positive")
        
        if not self._config.ml_models.hidden_layer_sizes:
            raise ConfigurationError("Hidden layer sizes cannot be empty")
        
        # Validate quality config
        if self._config.quality.minimum_quality_score < 0.0 or self._config.quality.minimum_quality_score > 1.0:
            raise ConfigurationError("Minimum quality score must be between 0.0 and 1.0")
        
        if self._config.quality.good_quality_threshold < self._config.quality.minimum_quality_score:
            raise ConfigurationError("Good quality threshold must be >= minimum quality score")
        
        if self._config.quality.excellent_quality_threshold < self._config.quality.good_quality_threshold:
            raise ConfigurationError("Excellent quality threshold must be >= good quality threshold")
        
        if self._config.quality.max_acceptable_processing_time <= 0:
            raise ConfigurationError("Max acceptable processing time must be positive")
        
        if self._config.quality.minimum_success_rate < 0.0 or self._config.quality.minimum_success_rate > 1.0:
            raise ConfigurationError("Minimum success rate must be between 0.0 and 1.0")
    
    @property
    def config(self) -> ReliconConfig:
        """Get the current configuration."""
        return self._config
    
    def reload(self):
        """Reload configuration from file and environment."""
        self._load_config()
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self._config.environment == Environment.DEVELOPMENT
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self._config.environment == Environment.PRODUCTION
    
    def get_service_url(self, service_name: str) -> str:
        """Get service URL based on environment."""
        base_urls = {
            Environment.DEVELOPMENT: {
                "luma": "https://api.dev.lumalabs.ai",
                "hailuo": "https://api.dev.hailuoai.com"
            },
            Environment.STAGING: {
                "luma": "https://api.staging.lumalabs.ai", 
                "hailuo": "https://api.staging.hailuoai.com"
            },
            Environment.PRODUCTION: {
                "luma": "https://api.lumalabs.ai",
                "hailuo": "https://api.hailuoai.com"
            }
        }
        
        return base_urls.get(self._config.environment, {}).get(service_name, "")
    
    def get_ml_config_summary(self) -> Dict[str, Any]:
        """Get ML configuration summary for monitoring and diagnostics."""
        return {
            'ml_optimization_enabled': self._config.ml_models.enable_ml_optimization,
            'neural_classification_enabled': self._config.ml_models.enable_neural_classification,
            'quality_validation_enabled': self._config.ml_models.enable_quality_validation,
            'anomaly_detection_enabled': self._config.ml_models.enable_anomaly_detection,
            'confidence_threshold': self._config.ml_models.classification_confidence_threshold,
            'quality_threshold': self._config.quality.minimum_quality_score,
            'regression_detection_enabled': self._config.quality.enable_regression_detection,
            'automated_alerts_enabled': self._config.quality.enable_automated_alerts,
            'max_inference_time': self._config.ml_models.max_inference_time_seconds,
            'quality_monitoring_enabled': self._config.quality.enable_quality_monitoring
        }
    
    def get_quality_thresholds(self) -> Dict[str, float]:
        """Get quality threshold configuration."""
        return {
            'minimum_quality_score': self._config.quality.minimum_quality_score,
            'good_quality_threshold': self._config.quality.good_quality_threshold,
            'excellent_quality_threshold': self._config.quality.excellent_quality_threshold,
            'minimum_brand_confidence': self._config.quality.minimum_brand_confidence,
            'niche_detection_confidence': self._config.quality.niche_detection_confidence,
            'competitive_analysis_threshold': self._config.quality.competitive_analysis_threshold,
            'acceptable_deviation': self._config.quality.acceptable_quality_deviation,
            'warning_threshold': self._config.quality.warning_deviation_threshold,
            'critical_threshold': self._config.quality.critical_deviation_threshold,
            'max_processing_time': self._config.quality.max_acceptable_processing_time,
            'minimum_success_rate': self._config.quality.minimum_success_rate,
            'maximum_error_rate': self._config.quality.maximum_error_rate
        }
    
    def is_ml_optimization_enabled(self) -> bool:
        """Check if ML optimization is enabled."""
        return self._config.ml_models.enable_ml_optimization
    
    def is_quality_monitoring_enabled(self) -> bool:
        """Check if quality monitoring is enabled."""
        return self._config.quality.enable_quality_monitoring
    
    def get_neural_network_config(self) -> Dict[str, Any]:
        """Get neural network configuration for model initialization."""
        return {
            'hidden_layer_sizes': self._config.ml_models.hidden_layer_sizes,
            'activation': self._config.ml_models.activation_function,
            'learning_rate': self._config.ml_models.learning_rate,
            'max_iter': self._config.ml_models.max_iterations,
            'max_features': self._config.ml_models.max_text_features,
            'ngram_range': self._config.ml_models.ngram_range,
            'min_df': self._config.ml_models.min_document_frequency,
            'max_df': self._config.ml_models.max_document_frequency
        }
    
    def get_ad_platform_config(self) -> Dict[str, Any]:
        """Get ad platform configuration summary."""
        return {
            'meta_enabled': self._config.ad_platforms.enable_meta_integration,
            'tiktok_enabled': self._config.ad_platforms.enable_tiktok_integration,
            'google_ads_enabled': self._config.ad_platforms.enable_google_ads_integration,
            'sync_performance_data': self._config.ad_platforms.sync_performance_data,
            'sync_interval_hours': self._config.ad_platforms.sync_interval_hours,
            'performance_predictions_enabled': self._config.ad_platforms.enable_performance_predictions,
            'request_timeout': self._config.ad_platforms.request_timeout_seconds,
            'batch_size': self._config.ad_platforms.batch_size,
            'historical_data_days': self._config.ad_platforms.historical_data_days
        }
    
    def is_ad_platform_enabled(self, platform: str) -> bool:
        """Check if specific ad platform is enabled."""
        platform_map = {
            'meta': self._config.ad_platforms.enable_meta_integration,
            'tiktok': self._config.ad_platforms.enable_tiktok_integration,
            'google_ads': self._config.ad_platforms.enable_google_ads_integration
        }
        return platform_map.get(platform, False)
    
    def export_config(self, file_path: str):
        """Export current configuration to file including ML and quality settings."""
        config_dict = {
            "environment": self._config.environment.value,
            "service_tier": self._config.service_tier.value,
            "debug": self._config.debug,
            "log_level": self._config.log_level,
            "video": {
                "max_duration_seconds": self._config.video.max_duration_seconds,
                "default_scenes": self._config.video.default_scenes,
                "scene_duration_seconds": self._config.video.scene_duration_seconds,
                "default_service": self._config.video.default_service,
                "fallback_service": self._config.video.fallback_service,
                "quality_preset": self._config.video.quality_preset,
                "resolution": self._config.video.resolution,
                "frame_rate": self._config.video.frame_rate,
                "supported_formats": self._config.video.supported_formats
            },
            "logo": {
                "max_file_size_mb": self._config.logo.max_file_size_mb,
                "supported_formats": self._config.logo.supported_formats,
                "analysis_timeout_seconds": self._config.logo.analysis_timeout_seconds,
                "color_extraction_limit": self._config.logo.color_extraction_limit,
                "enable_style_detection": self._config.logo.enable_style_detection,
                "enable_font_suggestions": self._config.logo.enable_font_suggestions
            },
            "ml_models": {
                "enable_ml_optimization": self._config.ml_models.enable_ml_optimization,
                "enable_neural_classification": self._config.ml_models.enable_neural_classification,
                "enable_quality_validation": self._config.ml_models.enable_quality_validation,
                "classification_confidence_threshold": self._config.ml_models.classification_confidence_threshold,
                "quality_validation_threshold": self._config.ml_models.quality_validation_threshold,
                "max_inference_time_seconds": self._config.ml_models.max_inference_time_seconds,
                "hidden_layer_sizes": self._config.ml_models.hidden_layer_sizes,
                "activation_function": self._config.ml_models.activation_function,
                "learning_rate": self._config.ml_models.learning_rate
            },
            "quality": {
                "enable_quality_monitoring": self._config.quality.enable_quality_monitoring,
                "enable_regression_detection": self._config.quality.enable_regression_detection,
                "minimum_quality_score": self._config.quality.minimum_quality_score,
                "good_quality_threshold": self._config.quality.good_quality_threshold,
                "excellent_quality_threshold": self._config.quality.excellent_quality_threshold,
                "minimum_brand_confidence": self._config.quality.minimum_brand_confidence,
                "max_acceptable_processing_time": self._config.quality.max_acceptable_processing_time,
                "minimum_success_rate": self._config.quality.minimum_success_rate,
                "acceptable_quality_deviation": self._config.quality.acceptable_quality_deviation,
                "warning_deviation_threshold": self._config.quality.warning_deviation_threshold,
                "critical_deviation_threshold": self._config.quality.critical_deviation_threshold
            },
            "ad_platforms": {
                "enable_meta_integration": self._config.ad_platforms.enable_meta_integration,
                "enable_tiktok_integration": self._config.ad_platforms.enable_tiktok_integration,
                "enable_google_ads_integration": self._config.ad_platforms.enable_google_ads_integration,
                "sync_performance_data": self._config.ad_platforms.sync_performance_data,
                "sync_interval_hours": self._config.ad_platforms.sync_interval_hours,
                "enable_performance_predictions": self._config.ad_platforms.enable_performance_predictions,
                "request_timeout_seconds": self._config.ad_platforms.request_timeout_seconds,
                "batch_size": self._config.ad_platforms.batch_size,
                "historical_data_days": self._config.ad_platforms.historical_data_days
            }
        }
        
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

# Global configuration manager
config_manager = ConfigManager()
config = config_manager.config