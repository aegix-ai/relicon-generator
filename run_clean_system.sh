#!/bin/bash
cd "$(dirname "$0")"
echo "🚀 Starting Relicon Clean System from relicon-rewrite directory"
echo "📁 Working directory: $(pwd)"
PYTHONPATH=. python3 backend/api/main.py