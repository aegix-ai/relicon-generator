# Relicon - Self-improving AI-driven short-form video ad creator system.
**Version** => **[`0.6.6`]**

A modular AI-driven platform for crafting high-conversion short-form promo videos — fast, polished, and brand-ready.

<p align="center">
  <img src="assets/relicon_potential_logo.png" alt="Relicon Logo" width="400">
</p>

## Architecture

```
relicon/ [v0.6.6]
.
├── ai
│   ├── generation
│   └── planning
├── archived
│   ├── cli_video_generator.py
│   ├── main_clean.py
│   └── version.py
├── assets
│   └── relicon_potential_logo.png
├── backend
│   ├── api
│   └── core
├── config
│   ├── env.example
│   └── settings.py
├── database
├── docs
│   ├── CHANGELOG.md
│   ├── COMPATIBILITY_FIX.md
│   ├── LEARNING_ROADMAP.md
│   ├── LOCAL_SETUP.md
│   ├── SETUP_GUIDE.md
│   └── VERIFICATION.md
├── frontend
│   ├── index.html
│   ├── package.json
│   └── src
├── infra
│   ├── docker-compose.yml
│   ├── Dockerfile
│   └── docker-test.sh
├── README.md
├── requirements.txt
├── scripts
│   ├── simple_test.py
│   └── start_dev.py
├── services
│   ├── audio
│   ├── luma
│   └── video
├── setup
│   ├── fix_final_local.sh
│   ├── PYTHON313_FIX.sh
│   ├── run_clean_system.sh
│   ├── setup_quick.sh
│   └── start_local.sh
├── simple_server.py
├── tests
│   └── test_complete_system.py
└── venv
    ├── bin
    ├── include
    ├── lib
    ├── lib64 -> lib
    └── pyvenv.cfg

28 directories, 37 files
```

## Key Features

- AI-powered video concept and script generation
- Professional voiceover synthesis
- AI video generation
- Automatic video assembly with FFmpeg
- Real-time progress tracking
- Cost-optimized generation

## Technology Stack

- **Frontend**: React + TypeScript + Tailwind CSS
- **Backend**: FastAPI + Python
- **AI**: OpenAI GPT-4o + LangChain
- **Video**: Luma AI + FFmpeg
- **Audio**: ElevenLabs / OpenAI TTS
- **Database**: PostgreSQL + Drizzle ORM
- **Queue**: Celery + Redis

## Quick Start

1. **Add API Keys**:Add your API keys to `.env`
2. **Install**: `pip install -r requirements.txt`
3. **Quick Test**: `python scripts/simple_test.py`
4. **Full Test**: `python tests/test_complete_system.py`
5. **Start**: `python scripts/start_dev.py` → `http://localhost:5000`

## Changelog

```
## [1.0.0] - 2025-07-30
- AI Ad Creator (v1.0) **current version**

## [2.0.0] - YYYY-MM-DD
- Feedback Loop Integration (v2.0)

## [3.0.0] - YYYY-MM-DD
- Self-Improvement (v3.0)

## [4.0.0] - YYYY-MM-DD
- AI Chat Workspace (v4.0)
```

## Cost Optimization

- Intelligent segment limiting based on duration
- Efficient prompt engineering  
- Cost per video: $2.00 - $4.00

## Verified Features ✅

- **AI Planning**: GPT-4o powered video strategy and scene breakdown
- **Video Generation**: Luma AI ray-1-6 model integration
- **Audio Processing**: OpenAI TTS with professional enhancement
- **Video Assembly**: FFmpeg-based final compilation
- **Real Testing**: Successfully generated 4.84MB test video
- **Clean Architecture**: Fully modular, readable codebase

---

## Contact Us

**Website:** [relicon.co](https://relicon.co)  
**Email:** [contact@relicon.co](mailto:contact@relicon.co)  
**Twitter:** [@reliconAI](https://twitter.com/reliconAI)

---

© Aegix, 2025
