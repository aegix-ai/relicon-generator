"""
Pre-flight Validation System
Validates all components before expensive API calls to prevent token waste
"""

import os
import sys
import subprocess
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import time

class PreflightValidator:
    """Comprehensive validation before expensive video generation"""
    
    def __init__(self):
        self.validation_results = {}
        self.critical_failures = []
        self.warnings = []
        
    def validate_all_systems(self) -> Dict[str, Any]:
        """Run all validations and return comprehensive report"""
        print("🔍 PREFLIGHT VALIDATION - Preventing Token Waste")
        print("=" * 60)
        
        # Run all validation checks
        validations = [
            ("API Keys", self._validate_api_keys),
            ("System Dependencies", self._validate_dependencies),
            ("File System", self._validate_filesystem),
            ("FFmpeg Functionality", self._validate_ffmpeg),
            ("Audio Processing", self._validate_audio_pipeline),
            ("Video Processing", self._validate_video_pipeline),
            ("OpenAI Connection", self._validate_openai_basic),
            ("Template System", self._validate_templates),
            ("Assembly Pipeline", self._validate_assembly_pipeline)
        ]
        
        for check_name, validation_func in validations:
            print(f"\n🧪 {check_name}...")
            try:
                result = validation_func()
                self.validation_results[check_name] = result
                if result['status'] == 'pass':
                    print(f"✅ {check_name}: PASSED")
                elif result['status'] == 'warning':
                    print(f"⚠️  {check_name}: WARNING - {result.get('message', '')}")
                    self.warnings.append(f"{check_name}: {result.get('message', '')}")
                else:
                    print(f"❌ {check_name}: FAILED - {result.get('error', '')}")
                    self.critical_failures.append(f"{check_name}: {result.get('error', '')}")
            except Exception as e:
                print(f"💥 {check_name}: CRASHED - {e}")
                self.critical_failures.append(f"{check_name}: System error - {e}")
        
        # Generate final report
        return self._generate_report()
    
    def _validate_api_keys(self) -> Dict[str, Any]:
        """Validate all required API keys are present"""
        required_keys = {
            'OPENAI_API_KEY': 'OpenAI GPT-4o (scene generation)',
            'ELEVENLABS_API_KEY': 'ElevenLabs (audio generation)', 
            'HAILUO_API_KEY': 'Hailuo/MiniMax (video generation)'
        }
        
        missing_keys = []
        for key, description in required_keys.items():
            if not os.environ.get(key):
                missing_keys.append(f"{key} ({description})")
        
        if missing_keys:
            return {
                'status': 'fail',
                'error': f"Missing API keys: {', '.join(missing_keys)}",
                'fix': 'Set environment variables or update .env file'
            }
        
        return {'status': 'pass', 'message': 'All API keys present'}
    
    def _validate_dependencies(self) -> Dict[str, Any]:
        """Check system dependencies"""
        dependencies = ['ffmpeg', 'ffprobe']
        missing = []
        
        for dep in dependencies:
            try:
                subprocess.run([dep, '-version'], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                missing.append(dep)
        
        if missing:
            return {
                'status': 'fail',
                'error': f"Missing dependencies: {', '.join(missing)}",
                'fix': 'Install FFmpeg: apt-get install ffmpeg'
            }
        
        return {'status': 'pass', 'message': 'All dependencies available'}
    
    def _validate_filesystem(self) -> Dict[str, Any]:
        """Check file system permissions and space"""
        issues = []
        
        # Check output directory
        output_dir = Path('outputs')
        if not output_dir.exists():
            try:
                output_dir.mkdir(exist_ok=True)
            except Exception as e:
                issues.append(f"Cannot create output directory: {e}")
        
        # Check write permissions
        test_file = output_dir / 'preflight_test.txt'
        try:
            test_file.write_text('test')
            test_file.unlink()
        except Exception as e:
            issues.append(f"No write permission to outputs: {e}")
        
        # Check temp directory
        try:
            with tempfile.NamedTemporaryFile() as tmp:
                pass
        except Exception as e:
            issues.append(f"Cannot create temp files: {e}")
        
        if issues:
            return {'status': 'fail', 'error': '; '.join(issues)}
        
        return {'status': 'pass', 'message': 'File system ready'}
    
    def _validate_ffmpeg(self) -> Dict[str, Any]:
        """Test FFmpeg functionality with actual operations"""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Test audio generation
                audio_file = Path(temp_dir) / 'test_audio.mp3'
                cmd = [
                    'ffmpeg', '-y', '-f', 'lavfi',
                    '-i', 'sine=frequency=440:duration=1',
                    '-c:a', 'libmp3lame', '-b:a', '192k',
                    str(audio_file)
                ]
                result = subprocess.run(cmd, capture_output=True)
                
                if result.returncode != 0:
                    return {'status': 'fail', 'error': f'FFmpeg audio test failed: {result.stderr.decode()[:200]}'}
                
                if not audio_file.exists() or audio_file.stat().st_size < 1000:
                    return {'status': 'fail', 'error': 'FFmpeg produced invalid audio file'}
                
                # Test video generation
                video_file = Path(temp_dir) / 'test_video.mp4'
                cmd = [
                    'ffmpeg', '-y', '-f', 'lavfi',
                    '-i', 'testsrc=duration=1:size=1280x720:rate=30',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    str(video_file)
                ]
                result = subprocess.run(cmd, capture_output=True)
                
                if result.returncode != 0:
                    return {'status': 'fail', 'error': f'FFmpeg video test failed: {result.stderr.decode()[:200]}'}
                
                if not video_file.exists() or video_file.stat().st_size < 10000:
                    return {'status': 'fail', 'error': 'FFmpeg produced invalid video file'}
                
                return {'status': 'pass', 'message': 'FFmpeg audio/video generation working'}
        
        except Exception as e:
            return {'status': 'fail', 'error': f'FFmpeg test crashed: {e}'}
    
    def _validate_audio_pipeline(self) -> Dict[str, Any]:
        """Test the audio processing pipeline without expensive API calls"""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create test audio segments
                segments = []
                for i in range(3):
                    segment_file = Path(temp_dir) / f'segment_{i:02d}.mp3'
                    
                    # Generate test audio segment
                    cmd = [
                        'ffmpeg', '-y', '-f', 'lavfi',
                        '-i', f'sine=frequency={220 + i*110}:duration=3',
                        '-c:a', 'libmp3lame', '-b:a', '128k',
                        str(segment_file)
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True)
                    if result.returncode != 0:
                        return {'status': 'fail', 'error': f'Test audio segment {i} generation failed'}
                    
                    segments.append({'file': str(segment_file)})
                
                # Test the audio unification process
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from providers.elevenlabs import ElevenLabsProvider
                
                provider = ElevenLabsProvider()
                output_file = Path(temp_dir) / 'unified_test.mp3'
                
                success = provider.create_unified_audio(segments, 10.0, str(output_file))
                
                if not success:
                    return {'status': 'fail', 'error': 'Audio unification process failed'}
                
                if not output_file.exists():
                    return {'status': 'fail', 'error': 'Unified audio file not created'}
                
                # Verify duration
                probe_cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', str(output_file)]
                probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
                
                if probe_result.returncode == 0:
                    probe_data = json.loads(probe_result.stdout)
                    duration = float(probe_data['format']['duration'])
                    if abs(duration - 10.0) > 1.0:
                        return {'status': 'warning', 'message': f'Duration slightly off: {duration:.2f}s vs 10.0s'}
                
                return {'status': 'pass', 'message': 'Audio pipeline working correctly'}
        
        except Exception as e:
            return {'status': 'fail', 'error': f'Audio pipeline test failed: {e}'}
    
    def _validate_video_pipeline(self) -> Dict[str, Any]:
        """Test video assembly without expensive API calls"""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create test video
                video_file = Path(temp_dir) / 'test_video.mp4'
                cmd = [
                    'ffmpeg', '-y', '-f', 'lavfi',
                    '-i', 'testsrc=duration=5:size=1280x720:rate=30',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    str(video_file)
                ]
                
                result = subprocess.run(cmd, capture_output=True)
                if result.returncode != 0:
                    return {'status': 'fail', 'error': 'Test video creation failed'}
                
                # Create test audio
                audio_file = Path(temp_dir) / 'test_audio.mp3'
                cmd = [
                    'ffmpeg', '-y', '-f', 'lavfi',
                    '-i', 'sine=frequency=440:duration=5',
                    '-c:a', 'libmp3lame', '-b:a', '192k',
                    str(audio_file)
                ]
                
                result = subprocess.run(cmd, capture_output=True)
                if result.returncode != 0:
                    return {'status': 'fail', 'error': 'Test audio creation failed'}
                
                # Test video-audio combination
                output_file = Path(temp_dir) / 'combined.mp4'
                cmd = [
                    'ffmpeg', '-y',
                    '-i', str(video_file),
                    '-i', str(audio_file),
                    '-c:v', 'libx264', '-c:a', 'aac',
                    str(output_file)
                ]
                
                result = subprocess.run(cmd, capture_output=True)
                if result.returncode != 0:
                    return {'status': 'fail', 'error': f'Video-audio combination failed: {result.stderr.decode()[:200]}'}
                
                if not output_file.exists() or output_file.stat().st_size < 50000:
                    return {'status': 'fail', 'error': 'Combined video file invalid'}
                
                return {'status': 'pass', 'message': 'Video pipeline working correctly'}
        
        except Exception as e:
            return {'status': 'fail', 'error': f'Video pipeline test failed: {e}'}
    
    def _validate_openai_basic(self) -> Dict[str, Any]:
        """Test OpenAI connection with minimal token usage"""
        try:
            if not os.environ.get('OPENAI_API_KEY'):
                return {'status': 'fail', 'error': 'OpenAI API key not set'}
            
            from openai import OpenAI
            client = OpenAI()
            
            # Minimal test request (very few tokens)
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Use cheaper model for testing
                messages=[{"role": "user", "content": "Reply with just: OK"}],
                max_tokens=5,
                temperature=0
            )
            
            content = response.choices[0].message.content.strip()
            if not content or content.lower() != 'ok':
                return {'status': 'warning', 'message': f'OpenAI responded but content unexpected: {content}'}
            
            return {'status': 'pass', 'message': 'OpenAI API working (used ~3 tokens)'}
        
        except Exception as e:
            return {'status': 'fail', 'error': f'OpenAI connection failed: {e}'}
    
    def _validate_templates(self) -> Dict[str, Any]:
        """Test template system without API calls"""
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from core.brand_intelligence import BusinessNiche, BusinessType
            from core.niche_prompt_templates import NichePromptTemplateEngine
            
            engine = NichePromptTemplateEngine()
            
            # Test fallback template
            template = engine.get_fallback_template(BusinessNiche.TECHNOLOGY, 'hailuo')
            
            if not template or len(template) == 0:
                return {'status': 'fail', 'error': 'Fallback template system not working'}
            
            # Check required fields
            required_fields = ['visual_concept', 'script_line', 'character_description']
            for scene in template:
                for field in required_fields:
                    if field not in scene or not scene[field]:
                        return {'status': 'fail', 'error': f'Template missing field: {field}'}
            
            return {'status': 'pass', 'message': f'Template system working ({len(template)} scenes)'}
        
        except Exception as e:
            return {'status': 'fail', 'error': f'Template system test failed: {e}'}
    
    def _validate_assembly_pipeline(self) -> Dict[str, Any]:
        """Test the full assembly pipeline logic"""
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from core.assembly_service import AssemblyService
            
            service = AssemblyService()
            
            # Test validation method
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create test video file
                test_video = Path(temp_dir) / 'test.mp4'
                cmd = [
                    'ffmpeg', '-y', '-f', 'lavfi',
                    '-i', 'testsrc=duration=10:size=1280x720:rate=30',
                    '-i', 'sine=frequency=440:duration=10',
                    '-c:v', 'libx264', '-c:a', 'aac', '-pix_fmt', 'yuv420p',
                    str(test_video)
                ]
                
                result = subprocess.run(cmd, capture_output=True)
                if result.returncode != 0:
                    return {'status': 'fail', 'error': 'Cannot create test video for assembly validation'}
                
                # Test validation
                validation = service.validate_video_output(str(test_video), 10.0)
                
                if not validation.get('valid', False):
                    return {'status': 'fail', 'error': f"Assembly validation failed: {validation.get('error', 'Unknown')}"}
                
                return {'status': 'pass', 'message': 'Assembly pipeline validation working'}
        
        except Exception as e:
            return {'status': 'fail', 'error': f'Assembly pipeline test failed: {e}'}
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate final validation report"""
        
        print(f"\n{'='*60}")
        print("🎯 PREFLIGHT VALIDATION REPORT")
        print(f"{'='*60}")
        
        total_checks = len(self.validation_results)
        passed_checks = sum(1 for r in self.validation_results.values() if r['status'] == 'pass')
        warning_checks = sum(1 for r in self.validation_results.values() if r['status'] == 'warning')
        failed_checks = sum(1 for r in self.validation_results.values() if r['status'] == 'fail')
        
        print(f"📊 Results: {passed_checks} passed, {warning_checks} warnings, {failed_checks} failed")
        
        if self.critical_failures:
            print(f"\n❌ CRITICAL FAILURES ({len(self.critical_failures)}):")
            for failure in self.critical_failures:
                print(f"   • {failure}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   • {warning}")
        
        # Determine overall status
        if self.critical_failures:
            status = 'FAILED'
            print(f"\n🚫 PREFLIGHT CHECK: FAILED")
            print("   ❌ DO NOT START VIDEO GENERATION")
            print("   💰 This will waste API tokens and fail")
            print("   🔧 Fix the issues above first")
            safe_to_proceed = False
        elif self.warnings:
            status = 'WARNING'
            print(f"\n⚠️  PREFLIGHT CHECK: PASSED WITH WARNINGS") 
            print("   ✅ Safe to proceed but may have issues")
            print("   💡 Consider fixing warnings for optimal results")
            safe_to_proceed = True
        else:
            status = 'PASSED'
            print(f"\n🎉 PREFLIGHT CHECK: PASSED")
            print("   ✅ All systems ready for video generation")
            print("   💰 API tokens will be used efficiently")
            print("   🚀 Proceed with confidence")
            safe_to_proceed = True
        
        return {
            'status': status,
            'safe_to_proceed': safe_to_proceed,
            'total_checks': total_checks,
            'passed': passed_checks,
            'warnings': warning_checks,
            'failures': failed_checks,
            'critical_failures': self.critical_failures,
            'warning_messages': self.warnings,
            'detailed_results': self.validation_results
        }
    
    def validate_complete_pipeline(self, brand_info: Dict[str, Any], quality_mode: str = 'professional') -> Dict[str, Any]:
        """
        Validate complete pipeline before expensive API calls
        
        Args:
            brand_info: Brand information for video generation
            quality_mode: Video quality mode
            
        Returns:
            Dictionary with validation results
        """
        validation_issues = []
        
        # Validate brand info
        if not brand_info:
            return {
                'is_valid': False,
                'error_message': 'Brand information is required',
                'validation_issues': ['Missing brand information']
            }
        
        # Check required brand fields
        required_fields = ['brand_name', 'brand_description']
        for field in required_fields:
            if not brand_info.get(field) or not str(brand_info.get(field)).strip():
                validation_issues.append(f'Missing or empty required field: {field}')
        
        # Logo validation removed
        
        # Run critical system validations
        api_keys_result = self._validate_api_keys()
        if api_keys_result['status'] == 'fail':
            validation_issues.extend(api_keys_result.get('details', [api_keys_result.get('message', 'API key validation failed')]))
        
        ffmpeg_result = self._validate_ffmpeg()
        if ffmpeg_result['status'] == 'fail':
            validation_issues.append('FFmpeg validation failed - video processing will not work')
        
        deps_result = self._validate_dependencies()
        if deps_result['status'] == 'fail':
            validation_issues.append('System dependencies validation failed')
        
        filesystem_result = self._validate_filesystem()
        if filesystem_result['status'] == 'fail':
            validation_issues.append('File system validation failed - cannot create output directories')
        
        # Return results
        if validation_issues:
            return {
                'is_valid': False,
                'error_message': f'Preflight validation failed with {len(validation_issues)} issue(s)',
                'validation_issues': validation_issues
            }
        else:
            return {
                'is_valid': True,
                'error_message': None,
                'validation_issues': []
            }
    
    def validate_ffmpeg(self) -> Dict[str, Any]:
        """Public wrapper for FFmpeg validation"""
        return self._validate_ffmpeg()
    
    def validate_api_keys(self) -> Dict[str, Any]:  
        """Public wrapper for API keys validation"""
        return self._validate_api_keys()
    
    def validate_dependencies(self) -> Dict[str, Any]:
        """Public wrapper for dependencies validation"""
        return self._validate_dependencies()

def run_preflight_check() -> Dict[str, Any]:
    """Run complete preflight validation"""
    validator = PreflightValidator()
    return validator.validate_all_systems()

if __name__ == "__main__":
    print("🚀 Running Preflight Validation...")
    result = run_preflight_check()
    
    if not result['safe_to_proceed']:
        print(f"\n💥 VALIDATION FAILED - Exit code 1")
        sys.exit(1)
    else:
        print(f"\n✅ VALIDATION PASSED - Safe to proceed")
        sys.exit(0)