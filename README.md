# Relicon - AI Video Ad Generation Platform

A clean, modular AI-powered video generation platform that creates professional short-form promotional videos.

## Architecture

```
relicon/
.
├── ai/
│   ├── generation/
│   └── planning/
├── backend/
│   ├── api/
│   ├── core/
│   └── requirements.txt
├── CHANGELOG.md
├── config/
│   ├── env.example
│   ├── settings.py
│   └── version.py
├── database/
├── docs/
│   ├── COMPATIBILITY_FIX.md
│   ├── LEARNING_ROADMAP.md
│   ├── LOCAL_SETUP.md
│   ├── SETUP_GUIDE.md
│   └── VERIFICATION.md
├── frontend/
│   ├── index.html
│   ├── package.json
│   └── src/
├── infra/
│   ├── docker-compose.yml
│   ├── Dockerfile
│   └── docker-test.sh
├── main_clean.py
├── outputs/
│   ├── clean_system_test.mp4
│   ├── job_1753431371_64a397cc.mp4
│   ├── job_1753431990_e738ecd8.mp4
│   ├── job_1753432616_b90820bf.mp4
│   ├── job_1753433048_7dfb41f2.mp4
│   ├── job_1753436032_9a236ea8.mp4
│   └── job_1753529630_69b4d0de.mp4
├── package.json
├── README.md
├── requirements.txt
├── scripts/
│   ├── simple_test.py
│   └── start_dev.py
├── services/
│   ├── audio/
│   ├── luma/
│   └── video/
├── setup/
│   ├── fix_final_local.sh
│   ├── PYTHON313_FIX.sh
│   ├── run_clean_system.sh
│   ├── setup_quick.sh
│   └── start_local.sh
├── simple_server.py
├── tests/
│   └── test_complete_system.py
└── venv/
    ├── bin/
    ├── include/
    ├── lib/
    ├── lib64/ -> lib/
    └── pyvenv.cfg

26 directories, 36 files
```

## Key Features

- AI-powered video concept and script generation
- Professional voiceover synthesis
- Luma AI video generation
- Automatic video assembly with FFmpeg
- Real-time progress tracking
- Cost-optimized generation

## Technology Stack

- **Frontend**: React + TypeScript + Tailwind CSS
- **Backend**: FastAPI + Python
- **AI**: OpenAI GPT-4o + LangChain
- **Video**: Luma AI + FFmpeg
- **Audio**: OpenAI TTS / ElevenLabs
- **Database**: PostgreSQL + Drizzle ORM
- **Queue**: Celery + Redis

## Quick Start

1. **Set API Keys**: Copy `config/env.example` to `.env` and add your API keys
2. **Install Dependencies**: `pip install -r backend/requirements.txt`
3. **Test System**: `python scripts/simple_test.py`
4. **Full Test**: `python tests/test_complete_system.py`
5. **Start Development**: `python scripts/start_dev.py`

## Cost Optimization

- Intelligent segment limiting based on duration
- Efficient prompt engineering  
- Cost per video: $2.42-4.80
- Successfully tested: 10-second video for $2.64

## Verified Features ✅

- **AI Planning**: GPT-4o powered video strategy and scene breakdown
- **Video Generation**: Luma AI ray-1-6 model integration
- **Audio Processing**: OpenAI TTS with professional enhancement
- **Video Assembly**: FFmpeg-based final compilation
- **Real Testing**: Successfully generated 4.84MB test video
- **Clean Architecture**: Fully modular, readable codebase
