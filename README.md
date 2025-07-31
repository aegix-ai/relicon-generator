# [Relicon](https://relicon.co)

### **Self-Improving AI-Driven Short-Form Video Ad Creator System**

![GitHub version](https://img.shields.io/badge/version-0.6.6-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)
![FastAPI](https://img.shields.io/badge/framework-FastAPI-teal.svg)
![Frontend](https://img.shields.io/badge/frontend-React%20%2B%20TS%20%2B%20Tailwind-9cf.svg)
![AI Integration](https://img.shields.io/badge/AI-GPT--4o%20%2B%20LangChain-orange.svg)
![CI](https://img.shields.io/badge/tests-passing-brightgreen.svg)

A powerful and modular AI-driven platform designed to rapidly create high-conversion, polished, and brand-ready short-form promotional videos.

---

<p align="center">
  <img src="assets/relicon_potential_logo.png" alt="Relicon Logo" width="400">
</p>

---

## Overview

Relicon leverages advanced AI to streamline the creation of engaging short-form video advertisements, automating the process from ideation to final delivery.

## Key Features

* **AI-Powered Video Planning:** Automatic generation of engaging video concepts and scripts.
* **Professional Voiceover:** Integrated, high-quality AI voice synthesis.
* **AI-Generated Visuals:** Cutting-edge video rendering powered by Luma AI.
* **Seamless Assembly:** Automatic and precise video compilation with FFmpeg.
* **Continuous Improvement:** Real-time feedback and optimization loops for video performance.
* **Cost Efficiency:** Optimized for budget-conscious video generation (\$2-\$4 per video).

## Technology Stack

| Area           | Technology                               |
| -------------- | ---------------------------------------- |
| **Frontend**   | Next.js, React, TypeScript, Tailwind CSS |
| **Backend**    | FastAPI, Python, PostgreSQL, Drizzle ORM |
| **AI & ML**    | GPT-4o, LangChain, Claude                |
| **Video**      | Hailuo AI, Luma AI, FFmpeg               |
| **Audio**      | ElevenLabs, OpenAI TTS                   |
| **Music**      | Artlist, Mubert AI                       |
| **Task Queue** | Celery, Redis                            |
| **Deployment** | Docker, Docker Compose                   |

## Project Structure

```
relicon/ [v0.6.6]
.
├── ai/
│   ├── generation/
│   └── planning/
├── assets/
│   └── relicon_potential_logo.png
├── backend/
│   ├── api/
│   └── core/
├── frontend/
│   ├── index.html
│   └── src/
├── infra/
│   ├── docker-compose.yml
│   └── Dockerfile
├── scripts/
│   ├── simple_test.py
│   └── start_dev.py
├── services/
│   ├── audio/
│   ├── luma/
│   └── video/
├── setup/
│   ├── fix_final_local.sh
│   └── setup_quick.sh
├── tests/
│   └── test_complete_system.py
├── README.md
├── requirements.txt
└── venv/
```

## Quick Start

1. **Clone the repository:**

```bash
git clone https://github.com/aegix-ai/relicon.git
cd relicon
```

2. **Setup environment:**

```bash
pip install -r requirements.txt
cp config/env.example .env
# Add your API keys in .env
```

3. **Run Tests:**

```bash
python scripts/simple_test.py
python tests/test_complete_system.py
```

4. **Launch Development Server:**

```bash
python scripts/start_dev.py
# Server running at http://localhost:5000
```

## Development Roadmap

| Version | Release Date | Major Features                    |
| ------- | ------------ | --------------------------------- |
| 1.0.0   | 2025-07-30   | AI Ad Creator (**current**)       |
| 2.0.0   | TBD          | Feedback Loop Integration         |
| 3.0.0   | TBD          | Self-Improvement and Optimization |
| 4.0.0   | TBD          | Synapsite Implementation          |

## Verified Functionality

* **AI Planning:** Video strategy and scene breakdown via GPT-4o
* **Video Rendering:** Integrated with Luma AI models
* **Audio Production:** Professional voice enhancements via OpenAI
* **Video Compilation:** Automated assembly using FFmpeg
* **Robust Testing:** Verified through comprehensive system tests
* **Modularity:** Clear, structured, and maintainable codebase

## Documentation

* [Local Setup](docs/LOCAL_SETUP.md)
* [Setup Guide](docs/SETUP_GUIDE.md)
* [Compatibility & Fixes](docs/COMPATIBILITY_FIX.md)
* [Learning Roadmap](docs/LEARNING_ROADMAP.md)
* [Verification](docs/VERIFICATION.md)
* [Changelog](docs/CHANGELOG.md)

## Contact & Community

* **Website:** [relicon.co](https://relicon.co)
* **Email:** [contact@relicon.co](mailto:contact@relicon.co)
* **Twitter:** [@reliconAI](https://twitter.com/reliconAI)

---

> © 2025 Relicon by Aegix Group. All rights reserved.
> 
