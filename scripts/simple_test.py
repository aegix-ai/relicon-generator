#!/usr/bin/env python3
"""
Simple test script for quick validation
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def quick_test():
    """Quick test of core functionality"""
    
    print("🧪 Relicon Quick Test")
    print("=" * 30)
    
    # Test imports
    try:
        from ai.planning.autonomous_architect import AutonomousVideoArchitect
        from services.luma.video_service import LumaVideoService
        from services.audio.tts_service import TTSService
        from ai.generation.video_generator import CompleteVideoGenerator
        print("✅ All imports successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test configuration
    try:
        from config.settings import settings
        errors = settings.validate()
        if errors:
            print(f"⚠️  Configuration issues: {errors}")
        else:
            print("✅ Configuration valid")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False
    
    # Test basic AI planning
    try:
        planner = VideoPlanner()
        test_brand = {
            "brand_name": "QuickTest",
            "brand_description": "A test brand for validation",
            "duration": 10
        }
        
        plan = planner.create_master_plan(test_brand)
        if plan and "core_message" in plan:
            print("✅ AI planning working")
        else:
            print("❌ AI planning failed")
            return False
    except Exception as e:
        print(f"❌ AI planning error: {e}")
        return False
    
    print("🎉 Quick test passed!")
    return True

if __name__ == "__main__":
    success = quick_test()
    if not success:
        sys.exit(1)