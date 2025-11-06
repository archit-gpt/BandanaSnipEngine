"""
Video export functionality using ffmpeg for bandana_snip_engine.
Handles snippet extraction, cropping, and format conversion.
"""

import subprocess
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available on PATH."""
    try:
        subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_video_info(video_path: str) -> Dict[str, Any]:
    """Get video metadata using ffprobe."""
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
        
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        raise ValueError(f"Could not get video info for {video_path}: {e}")


def build_ffmpeg_command(
    input_path: str,
    output_path: str,
    start_sec: float,
    duration_sec: int,
    height: int,
    crop_mode: str,
    fps: int,
    crf: int,
    audio_bitrate: str,
    fade_out_sec: float
) -> List[str]:
    """Build ffmpeg command for snippet export."""
    
    # Base command
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output
        '-ss', str(start_sec),  # Start time
        '-i', input_path,       # Input file
        '-t', str(duration_sec) # Duration
    ]
    
    # Video filters
    filters = []
    
    # Scale to target height, maintain aspect ratio
    filters.append(f'scale=-2:{height}:flags=lanczos')
    
    # Crop based on mode
    if crop_mode == "9x16_full":
        # 9:16 aspect ratio (1080x1920 for height=1920)
        crop_width = int(height * 9 / 16)
        filters.append(f'crop={crop_width}:{height}:(in_w-{crop_width})/2:(in_h-{height})/2')
    elif crop_mode == "4x5_center":
        # 4:5 aspect ratio (1080x1350 for height=1920 -> 1536x1920, but we want 1080x1350)
        crop_width = int(height * 4 / 5)
        crop_height = int(crop_width * 5 / 4)
        filters.append(f'crop={crop_width}:{crop_height}:(in_w-{crop_width})/2:(in_h-{crop_height})/2')
    
    # Frame rate
    filters.append(f'fps={fps}')
    
    # Fade out
    fade_start = duration_sec - fade_out_sec
    filters.append(f'fade=t=out:st={fade_start}:d={fade_out_sec}')
    
    # Add video filter chain
    if filters:
        cmd.extend(['-vf', ','.join(filters)])
    
    # Video codec and quality - ensure maximum compatibility
    cmd.extend([
        '-c:v', 'libx264',
        '-preset', 'slow',  # Better compression, more compatible
        '-profile:v', 'high',  # High profile for better compatibility
        '-level', '4.0',  # Level 4.0 for wide compatibility
        '-crf', str(crf),
        '-pix_fmt', 'yuv420p',  # Essential for QuickTime compatibility
        '-movflags', '+faststart'  # Enable streaming/progressive download
    ])
    
    # Audio codec and bitrate - AAC-LC for maximum compatibility
    cmd.extend([
        '-c:a', 'aac',
        '-profile:a', 'aac_low',  # AAC-LC profile
        '-b:a', audio_bitrate,
        '-ar', '44100'  # Standard sample rate
    ])
    
    # Output file
    cmd.append(output_path)
    
    return cmd


def export_snippet(
    src_path: str,
    output_path: str,
    start_sec: float,
    duration_sec: int,
    height: int = 1920,
    crop_mode: str = "9x16_full",
    fps: int = 30,
    crf: int = 17,
    audio_bitrate: str = "192k",
    fade_out_sec: float = 0.6
) -> None:
    """
    Export a single video snippet using ffmpeg.
    
    Args:
        src_path: Source video file path
        output_path: Output snippet file path
        start_sec: Start time in seconds
        duration_sec: Duration in seconds
        height: Target video height
        crop_mode: Crop mode ("9x16_full" or "4x5_center")
        fps: Target frame rate
        crf: Constant Rate Factor for video quality
        audio_bitrate: Audio bitrate (e.g., "192k")
        fade_out_sec: Fade out duration in seconds
    """
    if not check_ffmpeg():
        raise RuntimeError(
            "ffmpeg not found. Please install ffmpeg and ensure it's on your PATH.\n"
            "Visit https://ffmpeg.org/download.html for installation instructions."
        )
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Build and execute ffmpeg command
    cmd = build_ffmpeg_command(
        src_path, output_path, start_sec, duration_sec,
        height, crop_mode, fps, crf, audio_bitrate, fade_out_sec
    )
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        error_msg = f"ffmpeg failed with return code {e.returncode}\n"
        error_msg += f"Command: {' '.join(cmd)}\n"
        error_msg += f"stderr: {e.stderr}"
        raise RuntimeError(error_msg)


def export_all_snippets(
    src_path: str,
    segments: List[Tuple[float, int]],
    output_dir: str,
    config: Dict[str, Any]
) -> List[str]:
    """
    Export all snippets for a single video.
    
    Args:
        src_path: Source video file path
        segments: List of (start_time, duration) tuples
        output_dir: Output directory
        config: Export configuration
        
    Returns:
        List of generated output file paths
    """
    input_stem = Path(src_path).stem
    output_paths = []
    
    # Get config values
    height = config.get('video_height', 1920)
    crop_modes = config.get('crops', ["9x16_full"])
    fps = config.get('target_fps', 30)
    crf = config.get('crf', 17)
    audio_bitrate = config.get('audio_bitrate', "192k")
    fade_out_sec = config.get('fade_out_sec', 0.6)
    
    for i, (start_time, duration) in enumerate(segments, 1):
        for crop_mode in crop_modes:
            # Generate output filename
            crop_suffix = crop_mode.replace('x', '_').replace('_full', '').replace('_center', '')
            output_filename = f"{input_stem}_snippet{i}_{duration}s_{crop_suffix}.mp4"
            output_path = os.path.join(output_dir, output_filename)
            
            # Export snippet
            export_snippet(
                src_path=src_path,
                output_path=output_path,
                start_sec=start_time,
                duration_sec=duration,
                height=height,
                crop_mode=crop_mode,
                fps=fps,
                crf=crf,
                audio_bitrate=audio_bitrate,
                fade_out_sec=fade_out_sec
            )
            
            output_paths.append(output_path)
    
    return output_paths


def write_manifest(
    manifest_data: Dict[str, List[Tuple[float, int]]],
    output_dir: str
) -> str:
    """
    Write manifest.json with all processed segments.
    
    Args:
        manifest_data: Dict mapping input filename to list of (start, duration) tuples
        output_dir: Output directory
        
    Returns:
        Path to manifest file
    """
    manifest_path = os.path.join(output_dir, "manifest.json")
    
    # Convert tuples to dicts for JSON serialization
    json_data = {}
    for filename, segments in manifest_data.items():
        json_data[filename] = [
            {"start": start, "duration": duration}
            for start, duration in segments
        ]
    
    with open(manifest_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    return manifest_path