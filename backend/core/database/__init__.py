"""
Relicon AI Ad Creator - Database Package
Modular database layer with repository pattern and clean separation of concerns
"""

from .connection import engine, SessionLocal, get_database, init_database
from .models import Base, AdJob, UploadedAsset, GeneratedAsset, AdAnalytics, SystemMetrics
from .repositories import JobRepository, AssetRepository, AnalyticsRepository
from .manager import DatabaseManager

# Create global instances
db_manager = DatabaseManager()
job_repo = JobRepository()
asset_repo = AssetRepository()
analytics_repo = AnalyticsRepository()

__all__ = [
    # Database Connection
    'engine', 'SessionLocal', 'get_database', 'init_database',
    
    # Database Models
    'Base', 'AdJob', 'UploadedAsset', 'GeneratedAsset', 'AdAnalytics', 'SystemMetrics',
    
    # Repositories
    'JobRepository', 'AssetRepository', 'AnalyticsRepository',
    
    # Manager
    'DatabaseManager', 'db_manager',
    
    # Global Repository Instances
    'job_repo', 'asset_repo', 'analytics_repo'
] 