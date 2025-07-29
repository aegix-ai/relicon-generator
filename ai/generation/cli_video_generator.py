#!/usr/bin/env python3
"""
    Command Line Interface for Relicon Video Generator
    Allows the clean system to be called from the old frontend
"""

import sys
import argparse
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from ai.generation.video_generator import VideoGenerator


def main():
    """CLI entry point for video generation"""
    
    # Check if called with JSON input (old interface)
    if not sys.argv[1:]:
        # Read JSON from stdin (old interface)
        try:
            input_data = sys.stdin.read()
            brand_info = json.loads(input_data)
            job_id = brand_info.get('job_id', 'cli_job')
            output_path = f"outputs/{job_id}.mp4"
        except:
            print("ERROR: Invalid JSON input", file=sys.stderr)
            sys.exit(1)
    else:
        # Parse command line arguments (new interface)
        parser = argparse.ArgumentParser(description='Generate AI video ads')
        parser.add_argument('--brand-name', required=True, help='Brand name')
        parser.add_argument('--brand-description', required=True, help='Brand description')
        parser.add_argument('--target-audience', default='general audience', help='Target audience')
        parser.add_argument('--tone', default='friendly', help='Tone of voice')
        parser.add_argument('--duration', type=int, default=30, help='Video duration in seconds')
        parser.add_argument('--call-to-action', default='Take action now', help='Call to action')
        parser.add_argument('--job-id', required=True, help='Job ID')
        parser.add_argument('--output-path', required=True, help='Output video path')
        
        args = parser.parse_args()
        
        brand_info = {
            'brand_name': args.brand_name,
            'brand_description': args.brand_description,
            'target_audience': args.target_audience,
            'tone': args.tone,
            'duration': args.duration,
            'call_to_action': args.call_to_action
        }
        job_id = args.job_id
        output_path = args.output_path
    
    try:
        # Create absolute path
        if not output_path.startswith('/'):
            output_path = str(project_root / output_path)
        
        # Generate video using clean system
        generator = VideoGenerator()
        result_path = generator.generate_video(brand_info, output_path, job_id)
        
        print(f"SUCCESS: Enhanced video generated at {result_path}")
        
    except Exception as e:
        print(f"ERROR: Video generation failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
