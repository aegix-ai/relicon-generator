"""
Relicon AI Ad Creator - Database Repositories
Repository pattern implementations for clean database access and operations
"""
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, or_, func

from .connection import SessionLocal
from .models import AdJob, UploadedAsset, GeneratedAsset, AdAnalytics, SystemMetrics
from ..models.enums import JobStatus


class BaseRepository:
    """
    Base repository class with common database operations
    
    Provides common CRUD operations and session management
    that can be inherited by specific repositories.
    """
    
    def __init__(self):
        self.SessionLocal = SessionLocal
    
    def get_session(self) -> Session:
        """Get a new database session"""
        return self.SessionLocal()
    
    def close_session(self, session: Session) -> None:
        """Close a database session"""
        session.close()


class JobRepository(BaseRepository):
    """
    Repository for ad job operations
    
    Handles all database operations related to ad creation jobs,
    including CRUD operations, status updates, and queries.
    """
    
    def create_job(self, job_data: Dict[str, Any]) -> AdJob:
        """
        Create a new ad job
        
        Args:
            job_data: Dictionary containing job information
            
        Returns:
            AdJob: Created job instance
        """
        with self.get_session() as session:
            job = AdJob(**job_data)
            session.add(job)
            session.commit()
            session.refresh(job)
            return job
    
    def get_job_by_id(self, job_id: str) -> Optional[AdJob]:
        """
        Get job by job ID
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            AdJob or None if not found
        """
        with self.get_session() as session:
            return session.query(AdJob).filter(AdJob.job_id == job_id).first()
    
    def update_job(self, job_id: str, updates: Dict[str, Any]) -> Optional[AdJob]:
        """
        Update job with new data
        
        Args:
            job_id: Job identifier
            updates: Dictionary of fields to update
            
        Returns:
            Updated AdJob or None if not found
        """
        with self.get_session() as session:
            job = session.query(AdJob).filter(AdJob.job_id == job_id).first()
            if job:
                for key, value in updates.items():
                    setattr(job, key, value)
                job.updated_at = datetime.utcnow()
                session.commit()
                session.refresh(job)
            return job
    
    def update_job_status(self, job_id: str, status: JobStatus, message: str = "", progress: int = 0) -> bool:
        """
        Update job status and progress
        
        Args:
            job_id: Job identifier
            status: New job status
            message: Status message
            progress: Progress percentage (0-100)
            
        Returns:
            bool: True if updated successfully
        """
        updates = {
            'status': status,
            'message': message,
            'progress_percentage': progress
        }
        
        if status == JobStatus.COMPLETED:
            updates['completed_at'] = datetime.utcnow()
        
        job = self.update_job(job_id, updates)
        return job is not None
    
    def get_recent_jobs(self, limit: int = 50, status: Optional[JobStatus] = None) -> List[AdJob]:
        """
        Get recent jobs with optional status filter
        
        Args:
            limit: Maximum number of jobs to return
            status: Optional status filter
            
        Returns:
            List of AdJob instances
        """
        with self.get_session() as session:
            query = session.query(AdJob).order_by(desc(AdJob.created_at))
            
            if status:
                query = query.filter(AdJob.status == status)
            
            return query.limit(limit).all()
    
    def get_active_jobs(self) -> List[AdJob]:
        """
        Get all currently active (non-completed, non-failed) jobs
        
        Returns:
            List of active AdJob instances
        """
        active_statuses = [
            JobStatus.PENDING,
            JobStatus.PLANNING,
            JobStatus.GENERATING_SCRIPT,
            JobStatus.GENERATING_AUDIO,
            JobStatus.GENERATING_VIDEO,
            JobStatus.ASSEMBLING
        ]
        
        with self.get_session() as session:
            return session.query(AdJob).filter(
                AdJob.status.in_(active_statuses)
            ).order_by(AdJob.created_at).all()
    
    def get_jobs_by_brand(self, brand_name: str, limit: int = 20) -> List[AdJob]:
        """
        Get jobs for a specific brand
        
        Args:
            brand_name: Brand name to search for
            limit: Maximum results
            
        Returns:
            List of AdJob instances for the brand
        """
        with self.get_session() as session:
            return session.query(AdJob).filter(
                AdJob.brand_name.ilike(f"%{brand_name}%")
            ).order_by(desc(AdJob.created_at)).limit(limit).all()
    
    def get_job_stats(self) -> Dict[str, Any]:
        """
        Get job statistics
        
        Returns:
            Dictionary with job statistics
        """
        with self.get_session() as session:
            total_jobs = session.query(func.count(AdJob.id)).scalar()
            completed_jobs = session.query(func.count(AdJob.id)).filter(
                AdJob.status == JobStatus.COMPLETED
            ).scalar()
            failed_jobs = session.query(func.count(AdJob.id)).filter(
                AdJob.status == JobStatus.FAILED
            ).scalar()
            
            # Jobs in last 24 hours
            yesterday = datetime.utcnow() - timedelta(days=1)
            recent_jobs = session.query(func.count(AdJob.id)).filter(
                AdJob.created_at >= yesterday
            ).scalar()
            
            success_rate = (completed_jobs / total_jobs * 100) if total_jobs > 0 else 0
            
            return {
                'total_jobs': total_jobs,
                'completed_jobs': completed_jobs,
                'failed_jobs': failed_jobs,
                'active_jobs': total_jobs - completed_jobs - failed_jobs,
                'recent_jobs_24h': recent_jobs,
                'success_rate': round(success_rate, 2)
            }


class AssetRepository(BaseRepository):
    """
    Repository for asset operations
    
    Handles uploaded and generated assets with metadata tracking.
    """
    
    def create_uploaded_asset(self, asset_data: Dict[str, Any]) -> UploadedAsset:
        """Create a new uploaded asset record"""
        with self.get_session() as session:
            asset = UploadedAsset(**asset_data)
            session.add(asset)
            session.commit()
            session.refresh(asset)
            return asset
    
    def create_generated_asset(self, asset_data: Dict[str, Any]) -> GeneratedAsset:
        """Create a new generated asset record"""
        with self.get_session() as session:
            asset = GeneratedAsset(**asset_data)
            session.add(asset)
            session.commit()
            session.refresh(asset)
            return asset
    
    def get_job_uploaded_assets(self, job_id: str) -> List[UploadedAsset]:
        """Get all uploaded assets for a job"""
        with self.get_session() as session:
            return session.query(UploadedAsset).filter(
                UploadedAsset.job_id == job_id
            ).all()
    
    def get_job_generated_assets(self, job_id: str) -> List[GeneratedAsset]:
        """Get all generated assets for a job"""
        with self.get_session() as session:
            return session.query(GeneratedAsset).filter(
                GeneratedAsset.job_id == job_id
            ).all()
    
    def get_generated_assets_by_type(self, job_id: str, asset_type: str) -> List[GeneratedAsset]:
        """Get generated assets by type for a job"""
        with self.get_session() as session:
            return session.query(GeneratedAsset).filter(
                and_(
                    GeneratedAsset.job_id == job_id,
                    GeneratedAsset.asset_type == asset_type
                )
            ).all()
    
    def update_asset_processing_status(self, asset_id: str, status: str, notes: str = "") -> bool:
        """Update asset processing status"""
        with self.get_session() as session:
            asset = session.query(UploadedAsset).filter(
                UploadedAsset.id == asset_id
            ).first()
            
            if asset:
                asset.processing_status = status
                asset.processing_notes = notes
                if status == "processed":
                    asset.processed_at = datetime.utcnow()
                session.commit()
                return True
            return False


class AnalyticsRepository(BaseRepository):
    """
    Repository for analytics and metrics operations
    
    Handles performance tracking, cost analysis, and system metrics.
    """
    
    def create_job_analytics(self, analytics_data: Dict[str, Any]) -> AdAnalytics:
        """Create analytics record for a job"""
        with self.get_session() as session:
            analytics = AdAnalytics(**analytics_data)
            session.add(analytics)
            session.commit()
            session.refresh(analytics)
            return analytics
    
    def update_job_analytics(self, job_id: str, updates: Dict[str, Any]) -> Optional[AdAnalytics]:
        """Update job analytics"""
        with self.get_session() as session:
            analytics = session.query(AdAnalytics).filter(
                AdAnalytics.job_id == job_id
            ).first()
            
            if analytics:
                for key, value in updates.items():
                    setattr(analytics, key, value)
                session.commit()
                session.refresh(analytics)
            return analytics
    
    def get_job_analytics(self, job_id: str) -> Optional[AdAnalytics]:
        """Get analytics for a specific job"""
        with self.get_session() as session:
            return session.query(AdAnalytics).filter(
                AdAnalytics.job_id == job_id
            ).first()
    
    def create_system_metrics(self, metrics_data: Dict[str, Any]) -> SystemMetrics:
        """Create system metrics record"""
        with self.get_session() as session:
            metrics = SystemMetrics(**metrics_data)
            session.add(metrics)
            session.commit()
            session.refresh(metrics)
            return metrics
    
    def get_recent_system_metrics(self, hours: int = 24) -> List[SystemMetrics]:
        """Get system metrics for the last N hours"""
        since = datetime.utcnow() - timedelta(hours=hours)
        
        with self.get_session() as session:
            return session.query(SystemMetrics).filter(
                SystemMetrics.timestamp >= since
            ).order_by(desc(SystemMetrics.timestamp)).all()
    
    def get_cost_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get cost summary for the last N days"""
        since = datetime.utcnow() - timedelta(days=days)
        
        with self.get_session() as session:
            result = session.query(
                func.sum(AdAnalytics.total_cost).label('total_cost'),
                func.sum(AdAnalytics.openai_cost).label('openai_cost'),
                func.sum(AdAnalytics.luma_cost).label('luma_cost'),
                func.sum(AdAnalytics.elevenlabs_cost).label('elevenlabs_cost'),
                func.count(AdAnalytics.id).label('job_count')
            ).filter(AdAnalytics.created_at >= since).first()
            
            return {
                'total_cost': float(result.total_cost or 0),
                'openai_cost': float(result.openai_cost or 0),
                'luma_cost': float(result.luma_cost or 0),
                'elevenlabs_cost': float(result.elevenlabs_cost or 0),
                'job_count': result.job_count or 0,
                'average_cost_per_job': float(result.total_cost or 0) / max(result.job_count or 1, 1)
            }
    
    def get_performance_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get performance summary for the last N days"""
        since = datetime.utcnow() - timedelta(days=days)
        
        with self.get_session() as session:
            result = session.query(
                func.avg(AdAnalytics.total_generation_time).label('avg_total_time'),
                func.avg(AdAnalytics.planning_time).label('avg_planning_time'),
                func.avg(AdAnalytics.audio_generation_time).label('avg_audio_time'),
                func.avg(AdAnalytics.video_generation_time).label('avg_video_time'),
                func.avg(AdAnalytics.assembly_time).label('avg_assembly_time'),
                func.avg(AdAnalytics.overall_quality_score).label('avg_quality')
            ).filter(AdAnalytics.created_at >= since).first()
            
            return {
                'avg_total_generation_time': float(result.avg_total_time or 0),
                'avg_planning_time': float(result.avg_planning_time or 0),
                'avg_audio_generation_time': float(result.avg_audio_time or 0),
                'avg_video_generation_time': float(result.avg_video_time or 0),
                'avg_assembly_time': float(result.avg_assembly_time or 0),
                'avg_quality_score': float(result.avg_quality or 0)
            } 