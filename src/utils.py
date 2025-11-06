"""
Utility functions for bandana_snip_engine.
"""

import yaml
import os
from typing import Dict, Any
from pathlib import Path


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, looks for config.yaml in project root.
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Look for config.yaml in project root
        current_dir = Path(__file__).parent
        config_path = current_dir.parent / "config.yaml"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config.get('defaults', {})


def validate_video_file(file_path: str) -> bool:
    """
    Check if file is a supported video format.
    
    Args:
        file_path: Path to video file
        
    Returns:
        True if supported video format
    """
    if not os.path.exists(file_path):
        return False
    
    supported_extensions = {'.mp4', '.mov', '.mkv', '.avi', '.m4v'}
    return Path(file_path).suffix.lower() in supported_extensions


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to MM:SS format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in megabytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in MB
    """
    if not os.path.exists(file_path):
        return 0.0
    
    size_bytes = os.path.getsize(file_path)
    return size_bytes / (1024 * 1024)


def ensure_output_dir(output_dir: str) -> str:
    """
    Ensure output directory exists and return absolute path.
    
    Args:
        output_dir: Output directory path
        
    Returns:
        Absolute path to output directory
    """
    abs_path = os.path.abspath(output_dir)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename.strip()


def get_available_durations() -> list:
    """Get list of available snippet durations."""
    return [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20]


def validate_segments(segments: list, video_duration: float) -> bool:
    """
    Validate that segments are reasonable.
    
    Args:
        segments: List of (start, duration) tuples
        video_duration: Total video duration in seconds
        
    Returns:
        True if segments are valid
    """
    if len(segments) != 5:
        return False
    
    for start, duration in segments:
        if start < 0 or duration <= 0:
            return False
        if start + duration > video_duration + 1:  # Allow 1s tolerance
            return False
    
    return True