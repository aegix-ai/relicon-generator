#!/usr/bin/env python3
"""
Simple working server for clean relicon-rewrite system
"""
from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import uuid
import time
import os
from urllib.parse import urlparse, parse_qs
import threading
from pathlib import Path

# In-memory job storage
jobs = {}

class ReliconHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        
        if path == '/health':
            # Health check endpoint for Docker
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {"status": "healthy", "service": "relicon-ai-video-generator"}
            self.wfile.write(json.dumps(response).encode())
            
        elif path == '/':
            # Serve the frontend HTML
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Relicon Clean System</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>body { background: linear-gradient(135deg, #1e3a8a 0%, #7c3aed 100%); }</style>
</head>
<body class="min-h-screen p-8">
    <div class="max-w-4xl mx-auto">
        <header class="text-center mb-12">
            <h1 class="text-5xl font-bold text-white mb-4">Relicon Clean System</h1>
            <p class="text-xl text-blue-200">AI-Powered Video Generation - Clean Architecture</p>
            <div class="text-sm text-green-400 mt-2 font-mono">✓ Running from relicon-rewrite/ directory ONLY</div>
        </header>
        
        <div class="bg-white/10 backdrop-blur-md rounded-lg p-8 shadow-xl">
            <form id="videoForm" class="space-y-6">
                <div>
                    <label class="block text-white text-sm font-medium mb-2">Brand Name</label>
                    <input type="text" id="brandName" class="w-full px-4 py-3 bg-white/20 text-white placeholder-white/60 border border-white/30 rounded-lg" placeholder="Enter your brand name" required />
                </div>
                <div>
                    <label class="block text-white text-sm font-medium mb-2">Brand Description</label>
                    <textarea id="brandDescription" rows="4" class="w-full px-4 py-3 bg-white/20 text-white placeholder-white/60 border border-white/30 rounded-lg" placeholder="Describe your brand" required></textarea>
                </div>
                <button type="submit" id="submitBtn" class="w-full bg-gradient-to-r from-blue-500 to-purple-600 text-white py-3 px-6 rounded-lg font-semibold">Generate Professional Video</button>
            </form>
            
            <div id="statusDiv" class="mt-8 p-6 bg-white/10 rounded-lg hidden">
                <h3 class="text-white text-lg font-semibold mb-4">Generation Status</h3>
                <div class="space-y-2">
                    <p class="text-white">Status: <span id="status">-</span></p>
                    <p class="text-white">Progress: <span id="progress">0</span>%</p>
                    <div class="w-full bg-white/20 rounded-full h-2 mt-2">
                        <div id="progressBar" class="bg-gradient-to-r from-blue-500 to-purple-600 h-2 rounded-full transition-all duration-300" style="width: 0%"></div>
                    </div>
                    <p class="text-white">Message: <span id="message">-</span></p>
                </div>
            </div>
            
            <div id="videoResult" class="mt-8 p-6 bg-white/10 rounded-lg hidden">
                <h3 class="text-white text-lg font-semibold mb-4">Your Professional Video</h3>
                <div class="aspect-w-9 aspect-h-16 max-w-sm mx-auto">
                    <video id="resultVideo" controls class="w-full h-full rounded-lg" style="aspect-ratio: 9/16;">
                        Your browser does not support video playback.
                    </video>
                </div>
                <div class="mt-4 text-center">
                    <p class="text-green-400 text-sm" id="videoInfo">Video details will appear here</p>
                    <a id="downloadLink" href="#" download class="inline-block mt-2 px-4 py-2 bg-gradient-to-r from-green-500 to-blue-600 text-white rounded-lg hover:opacity-90 transition-opacity">
                        Download Video
                    </a>
                </div>
            </div>
        </div>
        
        <div class="mt-8 text-center text-white/60 text-sm">
            <p>🚀 Powered by Clean Relicon-Rewrite Architecture</p>
            <p>📁 Organized: ai/, backend/, services/, config/, tests/, scripts/</p>
        </div>
    </div>
    
    <script>
        document.getElementById('videoForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const brandName = document.getElementById('brandName').value;
            const brandDescription = document.getElementById('brandDescription').value;
            const submitBtn = document.getElementById('submitBtn');
            const statusDiv = document.getElementById('statusDiv');
            const videoResult = document.getElementById('videoResult');
            
            submitBtn.textContent = 'Generating Professional Video...';
            submitBtn.disabled = true;
            statusDiv.classList.remove('hidden');
            videoResult.classList.add('hidden');
            
            try {
                const response = await fetch('/api/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        brand_name: brandName,
                        brand_description: brandDescription,
                        duration: 15
                    })
                });
                
                const data = await response.json();
                document.getElementById('status').textContent = data.status;
                document.getElementById('message').textContent = data.message;
                
                if (data.job_id) {
                    const pollInterval = setInterval(async () => {
                        try {
                            const statusResponse = await fetch(`/api/status/${data.job_id}`);
                            const statusData = await statusResponse.json();
                            
                            document.getElementById('status').textContent = statusData.status;
                            document.getElementById('progress').textContent = statusData.progress || 0;
                            document.getElementById('progressBar').style.width = `${statusData.progress || 0}%`;
                            document.getElementById('message').textContent = statusData.message;
                            
                            if (statusData.status === 'completed') {
                                clearInterval(pollInterval);
                                
                                // Show video result
                                if (statusData.video_url) {
                                    const videoElement = document.getElementById('resultVideo');
                                    const videoInfo = document.getElementById('videoInfo');
                                    const downloadLink = document.getElementById('downloadLink');
                                    
                                    videoElement.src = statusData.video_url;
                                    downloadLink.href = statusData.video_url;
                                    
                                    if (statusData.result) {
                                        videoInfo.textContent = `Revolutionary AI Video Created`;
                                    }
                                    
                                    videoResult.classList.remove('hidden');
                                }
                                
                                submitBtn.textContent = 'Generate Another Video';
                                submitBtn.disabled = false;
                                
                            } else if (statusData.status === 'failed') {
                                clearInterval(pollInterval);
                                submitBtn.textContent = 'Try Again';
                                submitBtn.disabled = false;
                            }
                        } catch (error) {
                            console.error('Status check error:', error);
                        }
                    }, 2000);
                }
            } catch (error) {
                console.error('Error:', error);
                submitBtn.textContent = 'Try Again';
                submitBtn.disabled = false;
            }
        });
    </script>
</body>
</html>
            """
            self.wfile.write(html_content.encode())
            
        elif path.startswith('/api/status/'):
            # Get job status
            job_id = path.split('/')[-1]
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            if job_id in jobs:
                response = jobs[job_id]
            else:
                response = {"error": "Job not found"}
                
            self.wfile.write(json.dumps(response).encode())
            
        elif path.startswith('/outputs/'):
            # Serve generated videos
            video_filename = path.split('/')[-1]
            video_path = os.path.join(os.path.dirname(__file__), 'outputs', video_filename)
            
            if os.path.exists(video_path):
                self.send_response(200)
                self.send_header('Content-type', 'video/mp4')
                self.send_header('Content-Length', str(os.path.getsize(video_path)))
                self.end_headers()
                
                with open(video_path, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
            
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        if self.path == '/api/generate':
            # Generate video with REAL AI system
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            
            job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            
            # Initialize job
            jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "progress": 0,
                "message": "Starting sophisticated AI video generation...",
                "video_url": None
            }
            
            # Start REAL video generation
            def generate_real_video():
                try:
                    import sys
                    import os
                    
                    # Add path for imports
                    sys.path.insert(0, os.path.dirname(__file__))
                    
                    from ai.generation.video_generator import CompleteVideoGenerator
                    
                    # Extract brand info
                    brand_info = {
                        'brand_name': data.get('brand_name', 'Unknown Brand'),
                        'brand_description': data.get('brand_description', 'A great brand'),
                        'target_audience': data.get('target_audience', 'general audience'),
                        'tone': data.get('tone', 'professional'),
                        'duration': data.get('duration', 15),
                        'call_to_action': data.get('call_to_action', 'Learn more')
                    }
                    
                    # Create output path
                    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, f"{job_id}.mp4")
                    
                    # Generate video
                    generator = CompleteVideoGenerator()
                    
                    # Custom progress handler
                    original_progress = generator.progress_update
                    def track_progress(progress, message):
                        jobs[job_id].update({
                            "status": "processing" if progress < 100 else "completed",
                            "progress": progress,
                            "message": message
                        })
                        original_progress(progress, message)
                    
                    generator.progress_update = track_progress
                    
                    # Run generation
                    result = generator.generate_complete_video(brand_info, output_path)
                    
                    if result['success']:
                        jobs[job_id].update({
                            "status": "completed",
                            "progress": 100,
                            "message": f"✅ Professional video generated! {result['file_size_mb']}MB, {result['scenes']} scenes",
                            "video_url": f"/outputs/{job_id}.mp4",
                            "result": result
                        })
                    else:
                        jobs[job_id].update({
                            "status": "failed",
                            "progress": 0,
                            "message": f"❌ Generation failed: {result.get('error', 'Unknown error')}"
                        })
                        
                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    print(f"Generation error: {error_details}")
                    jobs[job_id].update({
                        "status": "failed",
                        "progress": 0,
                        "message": f"❌ System error: {str(e)}"
                    })
            
            thread = threading.Thread(target=generate_real_video)
            thread.daemon = True
            thread.start()
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            response = {
                "job_id": job_id,
                "status": "queued", 
                "message": "Sophisticated AI video generation started - this will create a REAL video!"
            }
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

def run_server():
    server_address = ('0.0.0.0', 5000)
    httpd = HTTPServer(server_address, ReliconHandler)
    print("🚀 Starting Relicon Clean System Server")
    print("📁 Running from relicon-rewrite directory")
    print("🌐 Frontend available at http://localhost:5000")
    print("✨ Clean system frontend active")
    httpd.serve_forever()

if __name__ == "__main__":
    run_server()