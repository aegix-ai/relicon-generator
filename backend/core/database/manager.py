"""
Relicon AI Ad Creator - Database Manager
High-level database manager that orchestrates repository operations
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from .repositories import JobRepository, AssetRepository, AnalyticsRepository
from .models import AdJob, UploadedAsset, GeneratedAsset, AdAnalytics
from ..models.enums import JobStatus


class DatabaseManager:
    """
    High-level database manager
    
    Provides a unified interface for all database operations by orchestrating
    the various repository classes. This is the main entry point for database
    operations throughout the application.
    """
    
    def __init__(self):
        self.job_repo = JobRepository()
        self.asset_repo = AssetRepository()
        self.analytics_repo = AnalyticsRepository()
    
    # Job Operations
    def create_job(self, job_data: Dict[str, Any]) -> AdJob:
        """Create a new ad creation job"""
        return self.job_repo.create_job(job_data)
    
    def get_job(self, job_id: str) -> Optional[AdJob]:
        """Get job by ID"""
        return self.job_repo.get_job_by_id(job_id)
    
    def update_job(self, job_id: str, updates: Dict[str, Any]) -> Optional[AdJob]:
        """Update job with new data"""
        return self.job_repo.update_job(job_id, updates)
    
    def get_recent_jobs(self, limit: int = 50) -> List[AdJob]:
        """Get recent jobs"""
        return self.job_repo.get_recent_jobs(limit)
    
    def get_active_jobs(self) -> List[AdJob]:
        """Get all currently active jobs"""
        return self.job_repo.get_active_jobs()
    
    # Asset Operations  
    def add_uploaded_asset(self, asset_data: Dict[str, Any]) -> UploadedAsset:
        """Add uploaded asset record"""
        return self.asset_repo.create_uploaded_asset(asset_data)
    
    def add_generated_asset(self, asset_data: Dict[str, Any]) -> GeneratedAsset:
        """Add generated asset record"""
        return self.asset_repo.create_generated_asset(asset_data)
    
    # Analytics Operations
    def record_analytics(self, analytics_data: Dict[str, Any]) -> AdAnalytics:
        """Record job analytics"""
        return self.analytics_repo.create_job_analytics(analytics_data) 