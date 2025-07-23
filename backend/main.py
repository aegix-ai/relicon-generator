"""
Relicon AI Ad Creator - Main Application
Revolutionary AI-powered ad creation system with ultra-detailed planning
"""
import asyncio
import uuid
import time
from datetime import datetime
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

# Core imports
from core.settings import settings
from core.models import (
    AdCreationRequest, AdCreationResponse, JobStatusResponse, 
    HealthCheckResponse, JobStatus, ErrorResponse
)
from core.database import get_database, init_database, db_manager

# AI Agents
from agents.master_planner import master_planner
from agents.scene_architect import scene_architect

# Services
from services.luma_service import luma_service
from services.elevenlabs_service import elevenlabs_service
from services.ffmpeg_service import ffmpeg_service

# Background tasks
from tasks.ad_creation_task import create_ad_background_task


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    print("🚀 Relicon AI Ad Creator - Starting up...")
    
    # Initialize database
    init_database()
    
    # Check services
    await check_service_health()
    
    print("✅ Relicon AI Ad Creator - Started successfully!")
    
    yield
    
    # Shutdown
    print("🔄 Relicon AI Ad Creator - Shutting down...")
    print("✅ Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Relicon AI Ad Creator",
    description="Revolutionary AI-powered ad creation system with ultra-detailed planning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/outputs", StaticFiles(directory=settings.OUTPUT_DIR), name="outputs")
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")


# Health check endpoint
@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Comprehensive system health check"""
    try:
        # Check services
        services_status = await check_service_health()
        
        # Get system metrics
        active_jobs = len(await get_active_jobs())
        
        # Overall status
        unhealthy_services = [k for k, v in services_status.items() if v == "unhealthy"]
        overall_status = "unhealthy" if unhealthy_services else "healthy"
        
        return HealthCheckResponse(
            status=overall_status,
            version=settings.APP_VERSION,
            services=services_status,
            uptime_seconds=int(time.time() - app.state.start_time) if hasattr(app.state, 'start_time') else 0,
            active_jobs=active_jobs,
            queue_size=0  # Would be from Redis in production
        )
        
    except Exception as e:
        return HealthCheckResponse(
            status="unhealthy",
            version=settings.APP_VERSION,
            services={"error": "unhealthy"},
            uptime_seconds=0,
            active_jobs=0,
            queue_size=0
        )


# Main ad creation endpoint
@app.post("/create-ad", response_model=AdCreationResponse)
async def create_ad(
    request: AdCreationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_database)
):
    """
    Create a revolutionary AI-powered ad with ultra-detailed planning
    
    This is the main endpoint that orchestrates the entire AI agent system
    to create professional-quality ads with mathematical precision.
    """
    try:
        # Generate unique job ID
        job_id = f"relicon_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        
        print(f"🎯 New ad creation request - Job ID: {job_id}")
        print(f"📋 Brand: {request.brand_name}")
        print(f"⏱️ Duration: {request.duration}s")
        print(f"🎨 Style: {request.style}")
        print(f"📱 Platform: {request.platform}")
        
        # Create job record in database
        job_data = {
            "job_id": job_id,
            "status": JobStatus.PENDING,
            "brand_name": request.brand_name,
            "brand_description": request.brand_description,
            "product_name": request.product_name,
            "target_audience": request.target_audience,
            "unique_selling_point": request.unique_selling_point,
            "call_to_action": request.call_to_action,
            "duration": request.duration,
            "platform": request.platform,
            "style": request.style,
            "progress_percentage": 0,
            "current_step": "Initializing AI planning system",
            "message": "Your ad creation request has been received and is being processed by our revolutionary AI system."
        }
        
        db_job = db_manager.create_job(job_data)
        
        # Start background processing
        background_tasks.add_task(
            create_ad_background_task,
            job_id,
            request,
            {"brand_name": request.brand_name, "style": request.style, "platform": request.platform}
        )
        
        # Calculate estimated completion time (based on duration and complexity)
        base_time = 180  # 3 minutes base
        duration_factor = request.duration * 2  # 2 seconds per ad second
        complexity_factor = 60 if request.style == "cinematic" else 30
        estimated_seconds = base_time + duration_factor + complexity_factor
        
        estimated_completion = datetime.utcnow().timestamp() + estimated_seconds
        
        return AdCreationResponse(
            success=True,
            job_id=job_id,
            status=JobStatus.PENDING,
            message="🚀 Revolutionary AI planning initiated! Your ultra-detailed ad is being architected with mathematical precision.",
            estimated_completion=datetime.fromtimestamp(estimated_completion),
            progress_percentage=0,
            current_step="Initializing Master Planner AI Agent"
        )
        
    except Exception as e:
        print(f"❌ Ad creation request failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create ad: {str(e)}"
        )


# Job status endpoint
@app.get("/job-status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str, db: Session = Depends(get_database)):
    """Get detailed status of an ad creation job"""
    try:
        job = db_manager.get_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return JobStatusResponse(
            job_id=job.job_id,
            status=job.status,
            progress_percentage=job.progress_percentage,
            current_step=job.current_step,
            message=job.message,
            created_at=job.created_at,
            updated_at=job.updated_at,
            completed_at=job.completed_at,
            video_url=job.video_url,
            error_details=job.error_details,
            planning_complete=job.progress_percentage >= 20,
            script_complete=job.progress_percentage >= 40,
            audio_complete=job.progress_percentage >= 60,
            video_complete=job.progress_percentage >= 80,
            assembly_complete=job.progress_percentage >= 95
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Job status check failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get job status: {str(e)}"
        )


# File upload endpoint
@app.post("/upload-asset")
async def upload_asset(
    file: UploadFile = File(...),
    job_id: Optional[str] = None,
    db: Session = Depends(get_database)
):
    """Upload brand assets for ad creation"""
    try:
        # Validate file
        if file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE / (1024*1024):.1f}MB"
            )
        
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed. Allowed types: {', '.join(settings.ALLOWED_EXTENSIONS)}"
            )
        
        # Create unique filename
        timestamp = int(time.time())
        unique_filename = f"{timestamp}_{file.filename}"
        
        # Save file
        upload_path = Path(settings.UPLOAD_DIR) / unique_filename
        with open(upload_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Record in database if job_id provided
        if job_id:
            asset_data = {
                "job_id": job_id,
                "filename": unique_filename,
                "original_filename": file.filename,
                "file_type": file.content_type,
                "file_size": file.size,
                "file_path": str(upload_path),
                "file_metadata": {
                    "extension": file_extension,
                    "uploaded_at": datetime.utcnow().isoformat()
                }
            }
            db_manager.add_uploaded_asset(asset_data)
        
        return {
            "success": True,
            "filename": unique_filename,
            "file_path": str(upload_path),
            "file_size": file.size,
            "message": "Asset uploaded successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ File upload failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )


# Get recent jobs endpoint
@app.get("/recent-jobs")
async def get_recent_jobs(limit: int = 20, db: Session = Depends(get_database)):
    """Get recent ad creation jobs"""
    try:
        jobs = db_manager.get_recent_jobs(limit)
        
        jobs_data = []
        for job in jobs:
            jobs_data.append({
                "job_id": job.job_id,
                "brand_name": job.brand_name,
                "status": job.status,
                "progress_percentage": job.progress_percentage,
                "created_at": job.created_at.isoformat(),
                "duration": job.duration,
                "platform": job.platform,
                "video_url": job.video_url
            })
        
        return {
            "success": True,
            "jobs": jobs_data,
            "total": len(jobs_data)
        }
        
    except Exception as e:
        print(f"❌ Recent jobs query failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get recent jobs: {str(e)}"
        )


# Download video endpoint
@app.get("/download/{job_id}")
async def download_video(job_id: str, db: Session = Depends(get_database)):
    """Download the generated video"""
    try:
        job = db_manager.get_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job.status != JobStatus.COMPLETED or not job.video_url:
            raise HTTPException(status_code=404, detail="Video not available")
        
        # Extract file path from URL
        video_path = job.video_url.replace("/outputs/", "")
        full_path = Path(settings.OUTPUT_DIR) / video_path
        
        if not full_path.exists():
            raise HTTPException(status_code=404, detail="Video file not found")
        
        return FileResponse(
            path=str(full_path),
            media_type="video/mp4",
            filename=f"{job.brand_name}_ad.mp4"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Video download failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Download failed: {str(e)}"
        )


# System statistics endpoint
@app.get("/stats")
async def get_system_stats(db: Session = Depends(get_database)):
    """Get system statistics and performance metrics"""
    try:
        recent_jobs = db_manager.get_recent_jobs(100)
        
        # Calculate statistics
        total_jobs = len(recent_jobs)
        completed_jobs = len([j for j in recent_jobs if j.status == JobStatus.COMPLETED])
        failed_jobs = len([j for j in recent_jobs if j.status == JobStatus.FAILED])
        active_jobs = len([j for j in recent_jobs if j.status not in [JobStatus.COMPLETED, JobStatus.FAILED]])
        
        success_rate = (completed_jobs / total_jobs * 100) if total_jobs > 0 else 0
        
        # Platform distribution
        platform_stats = {}
        for job in recent_jobs:
            platform = job.platform.value if job.platform else "unknown"
            platform_stats[platform] = platform_stats.get(platform, 0) + 1
        
        return {
            "success": True,
            "statistics": {
                "total_jobs": total_jobs,
                "completed_jobs": completed_jobs,
                "failed_jobs": failed_jobs,
                "active_jobs": active_jobs,
                "success_rate": round(success_rate, 1),
                "platform_distribution": platform_stats
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Stats query failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get statistics: {str(e)}"
        )


# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    print(f"❌ Unhandled exception: {str(exc)}")
    
    return ErrorResponse(
        error_code="INTERNAL_ERROR",
        error_message="An unexpected error occurred",
        details={"error": str(exc)}
    )


# Helper functions
async def check_service_health() -> Dict[str, str]:
    """Check health of all external services"""
    services = {
        "database": "healthy",  # Assume healthy if we got here
        "openai": "healthy" if settings.OPENAI_API_KEY else "unhealthy",
        "luma": "healthy" if settings.LUMA_API_KEY else "unhealthy",
        "elevenlabs": "healthy" if settings.ELEVENLABS_API_KEY else "unhealthy",
        "ffmpeg": "healthy"  # Would check FFmpeg availability in production
    }
    
    return services


async def get_active_jobs() -> list:
    """Get list of currently active jobs"""
    try:
        all_jobs = db_manager.get_recent_jobs(50)
        active_jobs = [
            job for job in all_jobs 
            if job.status not in [JobStatus.COMPLETED, JobStatus.FAILED]
        ]
        return active_jobs
    except:
        return []


# Initialize start time
@app.on_event("startup")
async def set_start_time():
    """Set application start time"""
    app.state.start_time = time.time()


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Relicon AI Ad Creator...")
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1
    ) 