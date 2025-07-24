"""
Relicon AI Ad Creator - Database Layer
SQLAlchemy models and database connection management
"""
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import (
    create_engine, Column, String, Integer, Float, Boolean, 
    DateTime, Text, JSON, Enum as SQLEnum, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
from core.settings import settings
from core.models import JobStatus, AdPlatform, AdStyle

# Database setup
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=settings.DEBUG
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Database Models
class AdJob(Base):
    """Main ad creation job tracking"""
    
    __tablename__ = "ad_jobs"
    
    # Primary Fields
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(String(50), unique=True, nullable=False, index=True)
    status = Column(SQLEnum(JobStatus), default=JobStatus.PENDING, nullable=False)
    
    # Request Data
    brand_name = Column(String(100), nullable=False)
    brand_description = Column(Text, nullable=False)
    product_name = Column(String(100))
    target_audience = Column(Text)
    unique_selling_point = Column(Text)
    call_to_action = Column(String(100))
    
    # Technical Specs
    duration = Column(Integer, default=30)
    platform = Column(SQLEnum(AdPlatform), default=AdPlatform.UNIVERSAL)
    style = Column(SQLEnum(AdStyle), default=AdStyle.PROFESSIONAL)
    
    # Processing Info
    progress_percentage = Column(Integer, default=0)
    current_step = Column(String(100))
    message = Column(Text)
    
    # Results
    video_url = Column(String(500))
    master_plan = Column(JSON)
    generation_stats = Column(JSON)
    error_details = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime)
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('job_id', name='uq_job_id'),
    )


class UploadedAsset(Base):
    """Uploaded user assets"""
    
    __tablename__ = "uploaded_assets"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(String(50), nullable=False, index=True)
    
    # File Info
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)
    file_size = Column(Integer, nullable=False)
    file_path = Column(String(500), nullable=False)
    
    # Metadata
    file_metadata = Column(JSON)
    processing_status = Column(String(50), default="uploaded")
    
    # Timestamps
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime)


class GeneratedAsset(Base):
    """AI-generated assets for ads"""
    
    __tablename__ = "generated_assets"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(String(50), nullable=False, index=True)
    
    # Asset Info
    asset_type = Column(String(50), nullable=False)  # video, audio, image, etc.
    asset_purpose = Column(String(100))  # scene_1_video, background_music, etc.
    file_path = Column(String(500), nullable=False)
    
    # Generation Details
    generation_prompt = Column(Text)
    generation_parameters = Column(JSON)
    generation_service = Column(String(50))  # luma, elevenlabs, openai, etc.
    
    # Quality Metrics
    quality_score = Column(Float)
    generation_time_seconds = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)


class AdAnalytics(Base):
    """Analytics and performance tracking"""
    
    __tablename__ = "ad_analytics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(String(50), nullable=False, index=True)
    
    # Performance Metrics
    total_generation_time = Column(Float)
    planning_time = Column(Float)
    script_generation_time = Column(Float)
    audio_generation_time = Column(Float)
    video_generation_time = Column(Float)
    assembly_time = Column(Float)
    
    # Quality Scores
    overall_quality_score = Column(Float)
    script_quality_score = Column(Float)
    audio_quality_score = Column(Float)
    video_quality_score = Column(Float)
    
    # Resource Usage
    cpu_usage_percent = Column(Float)
    memory_usage_mb = Column(Float)
    gpu_usage_percent = Column(Float)
    
    # Costs
    openai_cost = Column(Float)
    luma_cost = Column(Float)
    elevenlabs_cost = Column(Float)
    total_cost = Column(Float)
    
    # User Feedback
    user_rating = Column(Integer)  # 1-5 stars
    user_feedback = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)


class SystemMetrics(Base):
    """Overall system performance metrics"""
    
    __tablename__ = "system_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # System Stats
    timestamp = Column(DateTime, default=datetime.utcnow)
    active_jobs = Column(Integer, default=0)
    queue_size = Column(Integer, default=0)
    total_jobs_processed = Column(Integer, default=0)
    success_rate = Column(Float, default=0.0)
    
    # Performance
    average_generation_time = Column(Float)
    system_uptime_seconds = Column(Integer)
    
    # Resource Usage
    cpu_usage = Column(Float)
    memory_usage = Column(Float)
    disk_usage = Column(Float)
    
    # Service Health
    database_healthy = Column(Boolean, default=True)
    redis_healthy = Column(Boolean, default=True)
    openai_healthy = Column(Boolean, default=True)
    luma_healthy = Column(Boolean, default=True)
    elevenlabs_healthy = Column(Boolean, default=True)


# Database operations
class DatabaseManager:
    """Database connection and operations manager"""
    
    def __init__(self):
        self.SessionLocal = SessionLocal
    
    def get_db(self) -> Session:
        """Get database session"""
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    def create_tables(self):
        """Create all database tables"""
        Base.metadata.create_all(bind=engine)
    
    def drop_tables(self):
        """Drop all database tables (use with caution!)"""
        Base.metadata.drop_all(bind=engine)
    
    def get_job(self, job_id: str) -> Optional[AdJob]:
        """Get job by ID"""
        with self.SessionLocal() as db:
            return db.query(AdJob).filter(AdJob.job_id == job_id).first()
    
    def create_job(self, job_data: Dict[str, Any]) -> AdJob:
        """Create new ad job"""
        with self.SessionLocal() as db:
            job = AdJob(**job_data)
            db.add(job)
            db.commit()
            db.refresh(job)
            return job
    
    def update_job(self, job_id: str, updates: Dict[str, Any]) -> Optional[AdJob]:
        """Update job status and data"""
        with self.SessionLocal() as db:
            job = db.query(AdJob).filter(AdJob.job_id == job_id).first()
            if job:
                for key, value in updates.items():
                    setattr(job, key, value)
                job.updated_at = datetime.utcnow()
                db.commit()
                db.refresh(job)
            return job
    
    def get_recent_jobs(self, limit: int = 100) -> List[AdJob]:
        """Get recent jobs"""
        with self.SessionLocal() as db:
            return db.query(AdJob).order_by(AdJob.created_at.desc()).limit(limit).all()
    
    def add_uploaded_asset(self, asset_data: Dict[str, Any]) -> UploadedAsset:
        """Add uploaded asset record"""
        with self.SessionLocal() as db:
            asset = UploadedAsset(**asset_data)
            db.add(asset)
            db.commit()
            db.refresh(asset)
            return asset
    
    def add_generated_asset(self, asset_data: Dict[str, Any]) -> GeneratedAsset:
        """Add generated asset record"""
        with self.SessionLocal() as db:
            asset = GeneratedAsset(**asset_data)
            db.add(asset)
            db.commit()
            db.refresh(asset)
            return asset
    
    def record_analytics(self, analytics_data: Dict[str, Any]) -> AdAnalytics:
        """Record job analytics"""
        with self.SessionLocal() as db:
            analytics = AdAnalytics(**analytics_data)
            db.add(analytics)
            db.commit()
            db.refresh(analytics)
            return analytics


# Global database manager instance
db_manager = DatabaseManager()

# Dependency for FastAPI
def get_database() -> Session:
    """FastAPI dependency for database sessions"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Initialize database
def init_database():
    """Initialize database with tables"""
    print("🗄️ Initializing database...")
    try:
        db_manager.create_tables()
        print("✅ Database initialized successfully!")
    except Exception as e:
        print(f"⚠️ Database initialization failed: {e}")
        print("⚠️ Continuing without database (testing mode)")
        return False
    return True


if __name__ == "__main__":
    init_database() 