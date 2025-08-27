"""
  Clean Relicon API Server
  Main FastAPI application for the clean relicon system
"""

import sys
import os
import time
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn
import uuid
import shutil
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ensure we're running from relicon directory
relicon_root = Path(__file__).parent.parent.parent
if relicon_root.name != 'relicon':
    # We're being called from outside relicon, change to relicon directory
    actual_relicon = Path.cwd() / 'relicon'
    if actual_relicon.exists():
        os.chdir(str(actual_relicon))
        relicon_root = actual_relicon
    else:
        # We're already in relicon
        relicon_root = Path.cwd()

sys.path.append(str(relicon_root))

from backend.core.job_manager import JobManager
from core.cost_tracker import cost_tracker
from core.logger import get_logger, set_trace_context

# Initialize structured logger
api_logger = get_logger("api_server")

app = FastAPI(title="Relicon Clean API", version="1.0.0")

# Initialize job manager
job_manager = JobManager()

class VideoRequest(BaseModel):
    brand_name: str = Field(..., description="Name of the brand/company")
    brand_description: str = Field(..., description="Description of the brand, product, or service")
    target_audience: Optional[str] = Field(default="general audience", description="Target audience for the video")
    tone: Optional[str] = Field(default="friendly", description="Tone of the video")
    duration: Optional[int] = Field(default=30, description="Duration of the video in seconds")
    call_to_action: Optional[str] = Field(default="Take action now", description="Call to action text")

@app.get("/api/health")
async def health_check():
    """Health check endpoint with cost optimization info"""
    cost_estimate = cost_tracker.estimate_video_cost()
    return {
        "message": "Relicon Clean System API",
        "status": "running",
        "version": "2.0.0",
        "architecture": "cost-optimized relicon",
        "pricing": {
            "estimated_cost_per_video": f"${cost_estimate.total_estimated_cost:.2f}",
            "resolution": cost_estimate.resolution,
            "videos_per_20_dollars": int(20 / cost_estimate.total_estimated_cost),
            "optimization_status": "720p cost-optimized"
        }
    }

@app.get("/api/cost-analysis")
async def cost_analysis():
    """Provide detailed cost analysis and optimization recommendations"""
    cost_estimate = cost_tracker.estimate_video_cost()
    optimization_recs = cost_tracker.get_cost_optimization_recommendations(current_resolution="1080p")
    analytics = cost_tracker.get_cost_analytics()
    
    return {
        "current_pricing": {
            "cost_per_video": cost_estimate.total_estimated_cost,
            "cost_breakdown": {
                "video_generation": cost_estimate.total_video_cost,
                "audio_generation": cost_estimate.audio_cost,
                "ai_planning": cost_estimate.planning_cost
            },
            "resolution": cost_estimate.resolution,
            "scenes_per_video": cost_estimate.video_scenes,
            "duration_seconds": cost_estimate.duration_seconds
        },
        "budget_efficiency": {
            "videos_per_20_dollars": int(20 / cost_estimate.total_estimated_cost),
            "cost_per_scene": cost_estimate.video_cost_per_scene,
            "optimization_status": "720p cost-optimized",
            "savings_vs_1080p": optimization_recs["potential_savings"]
        },
        "performance_metrics": analytics,
        "luma_pricing_compliance": {
            "luma_cost_per_5s_720p": "$0.40",
            "our_cost_per_scene": f"${cost_estimate.video_cost_per_scene:.2f}",
            "compliance_status": "Optimized" if cost_estimate.video_cost_per_scene <= 0.40 else "Over budget"
        }
    }


@app.post("/api/generate")
async def generate_video(request: VideoRequest, quality_mode: str = "professional"):
    """Generate professional dynamic video ad with quality control.
    Quality modes: professional (broadcast quality), standard (high quality), economy (cost-optimized)."""
    try:
        # Validate quality mode
        valid_quality_modes = ['professional', 'standard', 'economy']
        if quality_mode not in valid_quality_modes:
            quality_mode = 'professional'  # Default to professional
        
        # Request data without logo integration
        request_data = {
            "brand_name": request.brand_name,
            "brand_description": request.brand_description,
            "target_audience": request.target_audience or "general audience",
            "tone": request.tone or "friendly",
            "call_to_action": request.call_to_action or "Take action now",
            "duration": request.duration or 30,
            "quality_mode": quality_mode
        }
        
        try:
            job_id = job_manager.create_job(request_data)
            
            # Start generation process with enhanced system
            job_manager.start_generation(job_id, request_data)
            
            return {
                "job_id": job_id,
                "status": "queued",
                "message": "Professional video generation started",
                "quality_mode": quality_mode
            }
        except Exception as generation_error:
            raise HTTPException(status_code=500, detail=f"Generation startup failed: {str(generation_error)}")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    """Get job status"""
    status = job_manager.get_job_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    return status

@app.get("/api/video/{filename}")
async def get_video(filename: str):
    """Serve generated video files"""
    video_path = Path.cwd() / "outputs" / filename
    if video_path.exists():
        return FileResponse(video_path, media_type="video/mp4")
    raise HTTPException(status_code=404, detail="Video not found")

# Serve static files from outputs directory  
outputs_dir = Path.cwd() / "outputs"
outputs_dir.mkdir(exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(outputs_dir)), name="outputs")

# Serve the frontend HTML file
@app.get("/")
async def serve_frontend():
    """Serve the simple GPT-4o powered frontend"""
    # Serve simple frontend with only brand name and description
    frontend_path = Path.cwd() / "frontend" / "index.html"
    if frontend_path.exists():
        return FileResponse(frontend_path)
    else:
        return {"error": "Simple frontend not found", "path": str(frontend_path), "cwd": str(Path.cwd())}

if __name__ == "__main__":
    print("Starting Relicon Enterprise API Server")
    print("Running from relicon directory")
    print("Frontend available at http://localhost:5000")
    print("Enterprise system ready")
    uvicorn.run(app, host="0.0.0.0", port=5000, ws="none")
