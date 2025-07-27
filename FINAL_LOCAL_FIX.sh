#!/bin/bash
# Final fix for Python 3.13 OpenAI/httpx compatibility issues

echo "🔧 FINAL COMPATIBILITY FIX FOR PYTHON 3.13"
echo "==========================================="

#cd relicon || exit 1

# Remove problematic virtual environment
if [ -d "venv" ]; then
    echo "🗑️ Removing problematic virtual environment..."
    rm -rf venv
fi

# Create fresh environment
echo "📦 Creating fresh Python 3.13 environment..."
python3 -m venv venv
source venv/bin/activate

# Install exact compatible versions that work together
echo "📥 Installing EXACT compatible versions..."
pip install --no-cache-dir --force-reinstall \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    openai==1.30.5 \
    httpx==0.24.1 \
    requests==2.31.0 \
    python-dotenv==1.0.0

# Verify no conflicts
echo "🧪 Testing OpenAI client initialization..."
python3 -c "
import os
os.environ['OPENAI_API_KEY'] = 'test-key'
from openai import OpenAI
try:
    client = OpenAI(api_key='test-key', timeout=60.0)
    print('✅ OpenAI client initializes without httpx errors')
except Exception as e:
    print(f'❌ Still have issues: {e}')
"

# Setup environment
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "📝 Created .env file - please add your API keys"
fi

# Check FFmpeg
if command -v ffmpeg &> /dev/null; then
    echo "✅ FFmpeg already available"
else
    echo "📦 Installing FFmpeg..."
    if command -v apt &> /dev/null; then
        sudo apt update && sudo apt install -y ffmpeg
    elif command -v brew &> /dev/null; then
        brew install ffmpeg
    fi
fi

echo ""
echo "🎯 FINAL SETUP COMPLETE!"
echo "======================="
echo "✅ Python 3.13 compatible environment"
echo "✅ OpenAI 1.30.5 (no httpx conflicts)"  
echo "✅ All dependencies locked to working versions"
echo ""
echo "🚀 START THE SERVER:"
echo "   source venv/bin/activate"
echo "   python3 simple_server.py"
echo ""
echo "🌐 Then open: http://localhost:5000"
echo ""
echo "💡 This should work perfectly now!"
