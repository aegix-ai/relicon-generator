#!/usr/bin/env python3
"""
Start the clean Relicon system
This script runs ONLY the clean relicon-rewrite system
"""
import subprocess
import sys
import os
from pathlib import Path

def main():
    # Change to relicon-rewrite directory
    os.chdir(Path(__file__).parent)
    
    print("🚀 Starting Clean Relicon System")
    print("📁 Working directory:", os.getcwd())
    print("🌐 API will be available at http://localhost:8000")
    print("✨ Running ONLY from relicon-rewrite folder")
    print("-" * 50)
    
    # Start the FastAPI server
    try:
        subprocess.run([sys.executable, "backend/api/main.py"], check=True)
    except KeyboardInterrupt:
        print("\n✅ Clean system stopped")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()