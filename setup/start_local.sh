#!/bin/bash
# Quick start script for local development

echo "🚀 Starting Relicon AI Video Generator"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Run ./QUICK_LOCAL_SETUP.sh first"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if .env exists and has API keys
if [ ! -f ".env" ]; then
    echo "❌ .env file not found. Copy from .env.example and add your API keys"
    exit 1
fi

# Check for required API keys
if ! grep -q "OPENAI_API_KEY=sk-" .env; then
    echo "❌ OPENAI_API_KEY not found in .env file"
    echo "   Add: OPENAI_API_KEY=sk-your-key-here"
    exit 1
fi

if ! grep -q "LUMA_API_KEY=" .env && ! grep -q "LUMA_API_KEY=luma-" .env; then
    echo "⚠️  LUMA_API_KEY not found in .env file"
    echo "   Add: LUMA_API_KEY=your-luma-key-here"
fi

echo "✅ Environment ready"
echo "🌐 Starting server at http://localhost:5000"
echo "📺 Ready to generate AI videos!"
echo ""

# Start the server
python3 simple_server.py