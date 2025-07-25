#!/usr/bin/env python3
"""
Development startup script for Relicon
Starts both backend and frontend in development mode
"""
import os
import sys
import subprocess
import time
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔧 Checking dependencies...")
    
    # Check Python dependencies
    try:
        import fastapi
        import openai
        print("✅ Python dependencies found")
    except ImportError as e:
        print(f"❌ Missing Python dependency: {e}")
        print("Run: pip install -r backend/requirements.txt")
        return False
    
    # Check Node.js
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Node.js found: {result.stdout.strip()}")
        else:
            print("❌ Node.js not found")
            return False
    except FileNotFoundError:
        print("❌ Node.js not found")
        return False
    
    return True

def check_environment():
    """Check if required environment variables are set"""
    print("🔑 Checking environment variables...")
    
    required_vars = ['OPENAI_API_KEY', 'LUMA_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("Please set these in your .env file or environment")
        return False
    
    print("✅ All required environment variables found")
    return True

def start_backend():
    """Start the FastAPI backend"""
    print("🚀 Starting FastAPI backend...")
    
    backend_dir = Path(__file__).parent.parent / "backend"
    
    cmd = [
        sys.executable, "-m", "uvicorn",
        "api.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ]
    
    return subprocess.Popen(cmd, cwd=backend_dir)

def start_frontend():
    """Start the React frontend"""
    print("🚀 Starting React frontend...")
    
    frontend_dir = Path(__file__).parent.parent / "frontend"
    
    # Check if node_modules exists
    if not (frontend_dir / "node_modules").exists():
        print("📦 Installing frontend dependencies...")
        subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)
    
    cmd = ["npm", "start"]
    return subprocess.Popen(cmd, cwd=frontend_dir)

def main():
    """Main startup function"""
    print("🎬 Relicon Development Startup")
    print("=" * 40)
    
    # Check prerequisites
    if not check_dependencies():
        sys.exit(1)
    
    if not check_environment():
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Start services
        backend_process = start_backend()
        time.sleep(3)  # Give backend time to start
        
        frontend_process = start_frontend()
        
        print("\n🎉 Relicon is starting up!")
        print("📱 Frontend: http://localhost:3000")
        print("🔧 Backend API: http://localhost:8000")
        print("📚 API Docs: http://localhost:8000/docs")
        print("\nPress Ctrl+C to stop all services")
        
        # Wait for processes
        try:
            backend_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping services...")
            backend_process.terminate()
            frontend_process.terminate()
            
            # Wait for graceful shutdown
            try:
                backend_process.wait(timeout=5)
                frontend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                backend_process.kill()
                frontend_process.kill()
            
            print("✅ All services stopped")
    
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()