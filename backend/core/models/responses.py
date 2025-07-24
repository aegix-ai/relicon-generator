"""
Relicon AI Ad Creator - Response Models
Pydantic models for API responses with comprehensive status tracking
"""
from datetime import datetime
from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field

from .enums import JobStatus
from .planning import MasterAdPlan


class AdCreationResponse(BaseModel):
    """
    Response for ad creation request initiation
    
    Returns job tracking information and initial status for the
    revolutionary AI ad creation workflow.
    """
    
    # Core Response Fields
    success: bool = Field(..., description="Whether the request was successfully received")
    job_id: str = Field(..., description="Unique identifier for tracking this ad creation job")
    status: JobStatus = Field(..., description="Current processing status")
    message: str = Field(..., description="Human-readable status message")
    
    # Timing Information
    estimated_completion: Optional[datetime] = Field(
        None, 
        description="Estimated completion time based on current queue and complexity"
    )
    
    # Progress Tracking
    progress_percentage: int = Field(
        0, 
        ge=0, 
        le=100,
        description="Completion percentage (0-100)"
    )
    current_step: Optional[str] = Field(
        None,
        description="Current processing step in human-readable format"
    )
    
    # Results (populated when completed)
    video_url: Optional[str] = Field(
        None,
        description="Download URL for the final ad video (available when completed)"
    )
    master_plan: Optional[MasterAdPlan] = Field(
        None,
        description="Detailed AI-generated plan (available during/after planning)"
    )
    generation_stats: Optional[Dict[str, Any]] = Field(
        None,
        description="Performance metrics and generation statistics"
    )


class JobStatusResponse(BaseModel):
    """
    Detailed job status response for progress tracking
    
    Provides comprehensive information about the current state
    of an ad creation job, including detailed progress metrics.
    """
    
    # Core Identification
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current processing status")
    
    # Progress Information
    progress_percentage: int = Field(
        ge=0, 
        le=100,
        description="Overall completion percentage"
    )
    current_step: Optional[str] = Field(
        None,
        description="Current processing step"
    )
    message: str = Field(..., description="Detailed status message")
    
    # Timestamps
    created_at: datetime = Field(..., description="Job creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    completed_at: Optional[datetime] = Field(
        None,
        description="Completion timestamp (if finished)"
    )
    
    # Results
    video_url: Optional[str] = Field(
        None,
        description="Final video download URL (if completed)"
    )
    error_details: Optional[str] = Field(
        None,
        description="Error information (if failed)"
    )
    
    # Detailed Progress Breakdown
    planning_complete: bool = Field(
        False,
        description="Whether AI planning phase is complete"
    )
    script_complete: bool = Field(
        False,
        description="Whether script generation is complete"
    )
    audio_complete: bool = Field(
        False,
        description="Whether audio generation is complete"
    )
    video_complete: bool = Field(
        False,
        description="Whether video generation is complete"
    )
    assembly_complete: bool = Field(
        False,
        description="Whether final assembly is complete"
    )


class HealthCheckResponse(BaseModel):
    """
    System health check response
    
    Provides comprehensive status of all system components
    for monitoring and debugging purposes.
    """
    
    # Overall System Status
    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ...,
        description="Overall system health status"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Health check timestamp"
    )
    version: str = Field(..., description="Application version")
    
    # Service Status
    services: Dict[str, Literal["healthy", "unhealthy"]] = Field(
        default_factory=dict,
        description="Individual service health status"
    )
    
    # System Metrics
    uptime_seconds: int = Field(..., description="System uptime in seconds")
    active_jobs: int = Field(..., description="Number of currently active jobs")
    queue_size: int = Field(..., description="Number of jobs in queue")


class ErrorResponse(BaseModel):
    """
    Standardized error response format
    
    Provides consistent error information across all API endpoints
    with detailed debugging information.
    """
    
    # Core Error Information
    success: bool = Field(False, description="Always false for error responses")
    error_code: str = Field(..., description="Machine-readable error code")
    error_message: str = Field(..., description="Human-readable error description")
    
    # Additional Context
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error context and debugging information"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error occurrence timestamp"
    )
    
    class Config:
        """Pydantic configuration for error responses"""
        schema_extra = {
            "example": {
                "success": False,
                "error_code": "VALIDATION_ERROR",
                "error_message": "Invalid brand name: must be between 1 and 100 characters",
                "details": {
                    "field": "brand_name",
                    "provided_value": "",
                    "constraints": {"min_length": 1, "max_length": 100}
                },
                "timestamp": "2024-01-15T10:30:00Z"
            }
        } 