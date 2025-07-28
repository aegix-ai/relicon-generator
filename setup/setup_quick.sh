#!/bin/bash
# Quick Local Setup for Relicon AI Video Generator
# Fixes all dependency conflicts and gets you running fast

echo "🚀 Relicon AI - Quick Local Setup"
echo "================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" < "3.11" ]]; then
    echo "❌ Python 3.11+ required. Current: $PYTHON_VERSION"
    echo "Install Python 3.11+ and try again"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip first
echo "📈 Upgrading pip..."
pip install --upgrade pip

# Install Python 3.13 compatible versions
echo "📦 Installing Python 3.13 compatible dependencies..."
pip install --no-cache-dir \
    fastapi>=0.104.0 \
    "uvicorn[standard]>=0.24.0" \
    openai==1.30.5 \
    httpx>=0.24.0 \
    requests>=2.31.0 \
    aiofiles>=23.2.0 \
    structlog>=23.2.0 \
    "pydantic>=2.5.0" \
    "pydantic-settings>=2.1.0" \
    python-dotenv>=1.0.0

# Check if FFmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠️  FFmpeg not found. Installing..."
    
    # Try different package managers
    if command -v apt &> /dev/null; then
        sudo apt update && sudo apt install -y ffmpeg
    elif command -v brew &> /dev/null; then
        brew install ffmpeg
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y ffmpeg
    else
        echo "❌ Please install FFmpeg manually:"
        echo "   Ubuntu/Debian: sudo apt install ffmpeg"
        echo "   macOS: brew install ffmpeg"
        echo "   Windows: Download from https://ffmpeg.org"
        exit 1
    fi
fi

# Verify FFmpeg installation
if command -v ffmpeg &> /dev/null; then
    echo "✅ FFmpeg installed: $(ffmpeg -version | head -n1)"
else
    echo "❌ FFmpeg installation failed"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from example..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your API keys:"
    echo "   - OPENAI_API_KEY"
    echo "   - ELEVENLABS_API_KEY" 
    echo "   - LUMA_API_KEY"
fi

# Test the setup
echo "🧪 Testing installation..."
python3 -c "
import fastapi, uvicorn, openai, requests, aiofiles, structlog, pydantic
from dotenv import load_dotenv
print('✅ All Python packages imported successfully')

import subprocess
result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
if result.returncode == 0:
    print('✅ FFmpeg working correctly')
else:
    print('❌ FFmpeg test failed')
"

echo ""
echo "🎯 Setup Complete!"
echo "==================="
echo "✅ Virtual environment: venv/"
echo "✅ Compatible dependencies installed"
echo "✅ FFmpeg ready for video processing"
echo ""
echo "🚀 To start the server:"
echo "   source venv/bin/activate"
echo "   python3 simple_server.py"
echo ""
echo "🌐 Then open: http://localhost:5000"
echo ""
echo "📝 Don't forget to add your API keys to .env file!"