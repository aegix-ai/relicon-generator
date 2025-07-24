"""
Relicon AI Ad Creator - Core Module
Core system components including models, database, and settings
"""

# Import from the new modular models package
from .models import (
    # Enums
    AdPlatform, AdStyle, JobStatus,
    
    # Request Models
    AdCreationRequest, FileUpload,
    
    # Response Models
    AdCreationResponse, JobStatusResponse, HealthCheckResponse, ErrorResponse,
    
    # Planning Models
    MasterAdPlan, AdScene, SceneComponent, PlanningContext,
    
    # Asset Models
    GenerationAssets
)

# Database and settings
from .database import get_database, init_database, db_manager
from .settings import settings, derived

__all__ = [
    # Models - Enums
    'AdPlatform', 'AdStyle', 'JobStatus',
    
    # Models - Requests
    'AdCreationRequest', 'FileUpload',
    
    # Models - Responses
    'AdCreationResponse', 'JobStatusResponse', 'HealthCheckResponse', 'ErrorResponse',
    
    # Models - Planning
    'MasterAdPlan', 'AdScene', 'SceneComponent', 'PlanningContext',
    
    # Models - Assets
    'GenerationAssets',
    
    # Database
    'get_database', 'init_database', 'db_manager',
    
    # Settings
    'settings', 'derived'
] 