# Relicon AI Video Generator - Complete Setup Guide

## System Overview
This is the world's most advanced autonomous AI video generation system with 14 Python scripts totaling 2,066 lines of clean code. The system creates revolutionary 15-second advertisement videos with full AI autonomy, featuring ElevenLabs premium TTS and ultra-realistic content generation.

## Setup Options

Choose your preferred deployment method:
- **🐳 Docker (Recommended)** - Production-ready, containerized deployment
- **🐍 Native Python** - Direct installation for development

---

## 🐳 DOCKER SETUP (RECOMMENDED)

### Prerequisites
- Docker 20.10+
- Docker Compose 2.0+

### 1. Download and Extract Code
```bash
# If from Git repository
git clone [your-repo-url] relicon-ai
cd relicon-ai/relicon-rewrite

# If from ZIP download
unzip relicon-code.zip
cd relicon-ai/relicon-rewrite
```

### 2. Configure Environment Variables
```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your API keys
nano .env
```

Add your API keys:
```bash
OPENAI_API_KEY=sk-your-openai-api-key-here
ELEVENLABS_API_KEY=your-elevenlabs-api-key-here
LUMA_API_KEY=your-luma-api-key-here
DEBUG=false
PORT=8080
```

### 3. Start with Docker Compose
```bash
# Build and start the system
docker-compose up --build

# Or run in background
docker-compose up -d --build
```

### 4. Access the Application
- Open browser to `http://localhost:8080`
- Health check: `http://localhost:8080/health`
- Generated videos in `./outputs/` directory

### 5. Docker Management Commands
```bash
# View logs
docker-compose logs -f

# Stop the system
docker-compose down

# Rebuild after changes
docker-compose up --build --force-recreate

# Remove everything (including volumes)
docker-compose down -v
```

---

## 🐍 NATIVE PYTHON SETUP

### Prerequisites
- Python 3.11+ 
- FFmpeg (for video processing)
- Git

### 1. Download and Extract Code
```bash
# If from Git repository
git clone [your-repo-url] relicon-ai
cd relicon-ai/relicon-rewrite

# If from ZIP download
unzip relicon-code.zip
cd relicon-ai/relicon-rewrite
```

### 2. Install Python Dependencies
```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 3. Install System Dependencies
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download FFmpeg from https://ffmpeg.org/download.html
# Add to PATH environment variable
```

### 4. Configure Environment Variables
Create a `.env` file in the `relicon-rewrite` directory:
```bash
# Required API Keys
OPENAI_API_KEY=sk-your-openai-api-key-here
LUMA_API_KEY=your-luma-api-key-here

# Optional Configuration
PORT=8080
DEBUG=true
```

### 5. Verify Installation
```bash
# Test Python dependencies
python3 -c "import fastapi, openai, requests; print('✅ Dependencies OK')"

# Test FFmpeg
ffmpeg -version

# Test system
python3 -c "from ai.generation.video_generator import CompleteVideoGenerator; print('✅ System Ready')"
```

### 6. Start the System
```bash
# Method 1: Direct Python execution
python3 simple_server.py

# Method 2: Using the startup script
python3 start_clean_system.py

# The server will start on http://localhost:8080
```

### 7. Access the Application
1. Open your browser
2. Navigate to `http://localhost:8080`
3. You should see the Relicon AI Video Generator interface
4. Fill in brand information and click "Generate Video"

## API Keys Setup

### OpenAI API Key
1. Go to https://platform.openai.com/
2. Create account or sign in
3. Navigate to API Keys section
4. Create new secret key
5. Copy the key (starts with `sk-`)

### Luma AI API Key
1. Go to https://lumalabs.ai/
2. Sign up for Dream Machine API access
3. Get your API key from dashboard
4. Copy the key

## Troubleshooting

### Common Issues

**FFmpeg not found:**
```bash
# Verify FFmpeg installation
which ffmpeg  # Should return path
ffmpeg -version  # Should show version info
```

**Import errors:**
```bash
# Ensure you're in the correct directory
pwd  # Should end with /relicon-rewrite
ls  # Should show ai/, services/, simple_server.py
```

**API key errors:**
```bash
# Check environment variables
python3 -c "import os; print('OpenAI:', bool(os.getenv('OPENAI_API_KEY'))); print('Luma:', bool(os.getenv('LUMA_API_KEY')))"
```

**Port already in use:**
```bash
# Kill existing process
lsof -ti:8080 | xargs kill -9

# Or use different port
PORT=8081 python3 simple_server.py
```

## System Architecture

### Core Components
- **AI Planning**: `ai/planning/autonomous_architect.py` - Revolutionary AI decision maker
- **Video Generation**: `ai/generation/video_generator.py` - Complete pipeline orchestrator  
- **Services**: `services/` - Luma AI, TTS, Video Assembly
- **Backend**: `backend/` - FastAPI server and job management
- **Server**: `simple_server.py` - Main application server

### File Structure
```
relicon-rewrite/
├── ai/
│   ├── planning/
│   │   ├── autonomous_architect.py    # AI architect (520 lines)
│   │   └── master_planner.py         # Legacy planner
│   └── generation/
│       └── video_generator.py        # Main generator (280 lines)
├── services/
│   ├── luma/
│   │   └── video_service.py          # Luma AI integration (160 lines)
│   ├── audio/
│   │   └── tts_service.py            # OpenAI TTS (180 lines)
│   └── video/
│       └── assembly_service.py       # FFmpeg video assembly (220 lines)
├── backend/
│   ├── main.py                       # FastAPI server (150 lines)
│   └── job_manager.py               # Job management (120 lines)
├── config/
│   └── settings.py                   # Configuration (80 lines)
└── simple_server.py                 # Main server (340 lines)
```

## Development Mode

### Running with Hot Reload
```bash
# Install development dependencies
pip install uvicorn[standard]

# Run with auto-reload
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8080
```

### Testing the System
```bash
# Quick test
python3 simple_test.py

# Complete system test
python3 test_complete_system.py
```

## Production Deployment

### Docker Option (Optional)
If you prefer Docker:
```bash
# Create Dockerfile
cat > Dockerfile << EOF
FROM python:3.11-slim

RUN apt-get update && apt-get install -y ffmpeg

WORKDIR /app
COPY . .

RUN pip install fastapi uvicorn openai requests aiofiles structlog pydantic pydantic-settings

EXPOSE 8080

CMD ["python3", "simple_server.py"]
EOF

# Build and run
docker build -t relicon-ai .
docker run -p 8080:8080 --env-file .env relicon-ai
```

### System Requirements
- **CPU**: 2+ cores (AI processing)
- **RAM**: 4GB+ (video assembly)
- **Storage**: 10GB+ (temporary video files)
- **Network**: Stable internet (API calls)

## Cost Estimation
- **Per 15-second video**: $3-5
- **OpenAI GPT-4o**: ~$0.02 per video
- **ElevenLabs TTS**: ~$0.05 per video (premium quality)
- **Luma AI**: ~$1.50 per scene (2-3 scenes = $3-4.50)

## Docker Features

### Container Architecture
- **Main Container**: Complete Relicon system with all dependencies
- **Volume Persistence**: Generated videos saved to `./outputs/`
- **Health Monitoring**: Built-in health checks at `/health`
- **Environment Isolation**: Secure API key management
- **Auto-restart**: Container restarts on failure

### Production Benefits
- **Consistent Environment**: Same setup across development/production
- **Easy Scaling**: Ready for horizontal scaling with load balancers
- **Version Control**: Lock exact Python/FFmpeg versions
- **Security**: Isolated container environment
- **Monitoring**: Health checks and log aggregation ready

### Development Workflow
```bash
# Make code changes
nano services/audio/tts_service.py

# Rebuild and restart
docker-compose up --build

# View real-time logs
docker-compose logs -f relicon

# Debug inside container
docker-compose exec relicon bash
```

## Support
For issues or questions:
1. Check logs in terminal output
2. Verify API keys are working
3. Ensure FFmpeg is properly installed
4. Check network connectivity for API calls

## Next Steps
Once running, experiment with different brand inputs to see the AI's autonomous creative decisions in action. Each video will be unique with AI-controlled timing, visuals, voice characteristics, and scene composition.