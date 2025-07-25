#!/usr/bin/env python3
"""
Complete System Test for Relicon
Tests the entire video generation pipeline
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ai.generation.video_generator import VideoGenerator
from config.settings import settings


def test_complete_system():
    """Test complete video generation workflow"""
    
    print("🧪 Testing Complete Relicon System")
    print("=" * 50)
    
    # Validate configuration
    print("🔧 Validating configuration...")
    errors = settings.validate()
    if errors:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    print("✅ Configuration valid")
    
    # Test brand information
    brand_info = {
        "brand_name": "TestBrand",
        "brand_description": "A revolutionary productivity app that helps busy professionals organize their daily tasks and boost efficiency through AI-powered insights.",
        "target_audience": "professionals",
        "tone": "energetic", 
        "duration": 15,  # Short test for cost efficiency
        "call_to_action": "Download now and transform your workflow!"
    }
    
    print(f"🎬 Testing video generation for: {brand_info['brand_name']}")
    
    # Create output path
    output_path = Path(settings.OUTPUT_DIR) / "test_complete_system.mp4"
    
    try:
        # Initialize generator
        generator = VideoGenerator()
        
        # Test cost estimation
        print("💰 Estimating costs...")
        cost_estimate = generator.estimate_cost(brand_info)
        print(f"   Estimated cost: ${cost_estimate['total_estimated']:.2f}")
        print(f"   Segments: {cost_estimate['segments']}")
        
        if cost_estimate['total_estimated'] > 10.0:
            print("⚠️  Cost too high for test, reducing duration...")
            brand_info['duration'] = 10
        
        # Generate video
        print("🎥 Starting video generation...")
        result_path = generator.generate_video(brand_info, str(output_path), "test_job")
        
        # Verify output
        if os.path.exists(result_path):
            file_size = os.path.getsize(result_path) / (1024 * 1024)  # MB
            print(f"✅ Video generated successfully!")
            print(f"   Output: {result_path}")
            print(f"   Size: {file_size:.2f} MB")
            
            # Basic validation
            if file_size > 0.1:  # At least 100KB
                print("✅ File size check passed")
                return True
            else:
                print("❌ File too small, may be corrupted")
                return False
        else:
            print("❌ Output file not found")
            return False
            
    except Exception as e:
        print(f"❌ System test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_individual_components():
    """Test individual components"""
    
    print("\n🔬 Testing Individual Components")
    print("=" * 50)
    
    try:
        # Test video planner
        print("🧠 Testing AI Planner...")
        from ai.planning.autonomous_architect import AutonomousVideoArchitect
        planner = AutonomousVideoArchitect()
        
        test_brand = {
            "brand_name": "TestApp",
            "brand_description": "A simple test app",
            "duration": 15
        }
        
        plan = planner.create_complete_plan(test_brand)
        if plan and 'detailed_scenes' in plan:
            print(f"   ✅ Planner working: {len(plan['detailed_scenes'])} scenes")
        else:
            print("   ❌ Planner failed")
            return False
        
        # Test Luma service (if API key available)
        if settings.LUMA_API_KEY:
            print("🎥 Testing Luma Service...")
            from services.luma.video_service import LumaVideoService
            luma = LumaVideoService()
            print("   ✅ Luma service initialized")
        else:
            print("   ⚠️  Luma API key not available, skipping")
        
        # Test TTS service
        print("🎤 Testing TTS Service...")
        from services.audio.tts_service import TTSService
        tts = TTSService()
        
        test_audio = Path("test_tts.mp3")
        tts.generate_voiceover("Hello, this is a test!", str(test_audio))
        
        if test_audio.exists():
            print("   ✅ TTS working")
            test_audio.unlink()  # Clean up
        else:
            print("   ❌ TTS failed")
            return False
        
        print("✅ All components working")
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Relicon System Test Suite")
    print("=" * 50)
    
    # Test components first
    components_ok = test_individual_components()
    
    if components_ok:
        # Test complete system
        system_ok = test_complete_system()
        
        if system_ok:
            print("\n🎉 ALL TESTS PASSED!")
            print("✅ Relicon system is working correctly")
        else:
            print("\n❌ SYSTEM TEST FAILED")
            sys.exit(1)
    else:
        print("\n❌ COMPONENT TESTS FAILED")
        sys.exit(1)