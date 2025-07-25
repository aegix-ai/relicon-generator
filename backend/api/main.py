#!/usr/bin/env python3
"""
Clean Relicon API Server
Main FastAPI application for the clean relicon-rewrite system
"""
import sys
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from backend.core.job_manager import JobManager
# from ai.generation.video_generator import VideoGenerator  # TODO: Re-enable when needed

app = FastAPI(title="Relicon Clean API", version="1.0.0")

# Initialize job manager
job_manager = JobManager()

class VideoRequest(BaseModel):
    brand_name: str
    brand_description: str
    target_audience: str = "general audience"
    tone: str = "friendly"
    duration: int = 10
    call_to_action: str = "Take action now"

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Relicon Clean System API",
        "status": "running",
        "version": "1.0.0",
        "architecture": "clean relicon-rewrite"
    }

@app.post("/api/generate")
async def generate_video(request: VideoRequest):
    """Generate video using clean system"""
    try:
        # Create job
        job_id = job_manager.create_job({
            "brand_name": request.brand_name,
            "brand_description": request.brand_description,
            "target_audience": request.target_audience,
            "tone": request.tone,
            "duration": request.duration,
            "call_to_action": request.call_to_action
        })
        
        # Start generation process
        job_manager.start_generation(job_id)
        
        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Video generation started with clean system"
        }
        
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
    video_path = project_root / "outputs" / filename
    if video_path.exists():
        return FileResponse(video_path, media_type="video/mp4")
    raise HTTPException(status_code=404, detail="Video not found")

# Serve static files from outputs directory
app.mount("/outputs", StaticFiles(directory=str(project_root / "outputs")), name="outputs")

# Serve the frontend HTML file
@app.get("/", response_class=FileResponse)
async def serve_frontend():
    """Serve the clean system frontend"""
    return FileResponse(project_root / "index.html")

if __name__ == "__main__":
    print("🚀 Starting Relicon Clean System API Server")
    print("📁 Running from relicon-rewrite directory")
    print("🌐 Frontend available at http://localhost:5000")
    print("🔥 Clean system frontend will be served")
    uvicorn.run(app, host="0.0.0.0", port=5000)