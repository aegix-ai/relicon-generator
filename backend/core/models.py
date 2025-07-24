"""
Relicon AI Ad Creator - Data Models (Legacy Import)
This file now imports from the modular models package for backward compatibility
"""

# Import all models from the new modular structure
from .models import *

# This file is kept for backward compatibility
# All models are now organized in the core/models/ package:
# - enums.py: AdPlatform, AdStyle, JobStatus
# - requests.py: AdCreationRequest, FileUpload  
# - responses.py: AdCreationResponse, JobStatusResponse, HealthCheckResponse, ErrorResponse
# - planning.py: MasterAdPlan, AdScene, SceneComponent, PlanningContext
# - assets.py: GenerationAssets 