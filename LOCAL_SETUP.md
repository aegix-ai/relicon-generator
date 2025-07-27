# Local Setup Fix - Step by Step

## The Problem
You're getting version conflicts because:
1. **Python version mismatch** - System needs Python 3.11+
2. **Missing FFmpeg** - Required for video processing
3. **Exact version pinning** - requirements.txt was too strict

## Solution 1: Docker (Easiest)

Skip all the Python version headaches:

```bash
# Check if Docker is installed
docker --version

# If not installed, install Docker Desktop first
# Then run:
cd relicon-rewrite
docker-compose up --build
```

This installs everything in a container - no version conflicts.

## Solution 2: Fix Local Python Setup

### Step 1: Check Python Version
```bash
python3 --version
# You need Python 3.11 or higher
```

### Step 2: Install Python 3.11 (if needed)
```bash
# Ubuntu/Debian:
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev

# macOS with Homebrew:
brew install python@3.11

# Windows: Download from python.org
```

### Step 3: Install FFmpeg
```bash
# Ubuntu/Debian:
sudo apt install ffmpeg

# macOS:
brew install ffmpeg

# Windows: Download from ffmpeg.org
```

### Step 4: Create Virtual Environment with Python 3.11
```bash
cd relicon-rewrite

# Use specific Python version
python3.11 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip
```

### Step 5: Install Requirements (Fixed Version)
```bash
# Install with flexible versioning
pip install -r requirements.txt

# If still fails, install one by one:
pip install fastapi uvicorn openai requests aiofiles structlog pydantic pydantic-settings
```

## Solution 3: Manual Minimal Setup

If requirements still fail, install just what you need:

```bash
pip install fastapi uvicorn openai requests
```

The system will work with just these core packages.

## How Replit Works (Why It Works Here)

Replit uses **Nix package management** which:
- Pre-installs Python 3.11 and all dependencies
- Includes FFmpeg and system libraries
- Handles all version conflicts automatically
- Uses the main package.json to manage Python packages

That's why it "just works" here but fails locally.

## Test Your Setup

Once installed, test with:

```bash
cd relicon-rewrite
python simple_server.py
```

Should see:
```
🚀 Starting Relicon Clean System Server
📁 Running from relicon-rewrite directory
🌐 Frontend available at http://localhost:5000
✨ Clean system frontend active
```

## Still Failing?

**Use Docker** - it's literally designed to solve exactly this problem. The Dockerfile already has everything configured correctly.

```bash
docker-compose up --build
# Wait 2-3 minutes for build
# Open http://localhost:8080
```

Zero configuration, zero version conflicts, zero headaches.