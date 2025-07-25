# Relicon - AI Video Ad Generation Platform

A clean, modular AI-powered video generation platform that creates professional short-form promotional videos.

## Architecture

```
relicon-rewrite/
├── frontend/          # React frontend application
├── backend/           # FastAPI backend services
├── ai/               # AI planning and generation services
├── services/         # External service integrations
├── database/         # Database models and migrations
├── tests/            # Comprehensive test suite
├── config/           # Configuration files
└── scripts/          # Utility scripts
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