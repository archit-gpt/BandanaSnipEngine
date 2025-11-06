"""
Clip Studio - Clean, minimalistic video snippet creator.
"""

import streamlit as st
import os
import sys
import json
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from snip_engine import make_manual_anchor_variants, detect_beats
from exporter import export_snippet, check_ffmpeg
from utils import load_config, ensure_output_dir
import librosa

# Page config
st.set_page_config(
    page_title="Clip Studio",
    page_icon="◾",
    layout="centered"
)

# Initialize session state
if 'config' not in st.session_state:
    try:
        st.session_state.config = load_config()
    except Exception as e:
        st.error(f"Failed to load config: {e}")
        st.stop()


def create_manual_variants(anchor_time: float, tmp_path: str, durations: List[int]) -> List[Dict[str, Any]]:
    """Create variants starting at the anchor time with custom durations."""
    
    # Load audio for beat detection (fallback if needed)
    try:
        y, sr = librosa.load(tmp_path, sr=None)
        beats = detect_beats(y, sr)
    except:
        beats = np.array([])
    
    variants = make_manual_anchor_variants(
        anchor_time, durations, beats, st.session_state.config
    )
    
    return variants


def export_variants(tmp_path: str, filename: str, anchor_time: float, durations: List[int]):
    """Export all variants to Downloads in a folder named after the video."""
    try:
        # Create output directory in Downloads with video_name_start_time format
        downloads_dir = os.path.expanduser("~/Downloads")
        file_stem = Path(filename).stem
        folder_name = f"{file_stem}_{anchor_time}"
        output_dir = os.path.join(downloads_dir, folder_name)
        output_dir = ensure_output_dir(output_dir)
        
        # Create variants
        variants = create_manual_variants(anchor_time, tmp_path, durations)
        
        # Export each variant
        exported_files = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, variant in enumerate(variants):
            output_filename = f"{file_stem}_clip-v{variant['index']}_{variant['dur']}s.mp4"
            output_path = os.path.join(output_dir, output_filename)
            
            status_text.text(f"Exporting clip {variant['index']} of {len(variants)}...")
            
            export_snippet(
                src_path=tmp_path,
                output_path=output_path,
                start_sec=variant['start'],
                duration_sec=variant['dur'],
                height=st.session_state.config.get('video_height', 1920),
                crop_mode="9x16_full",
                fps=st.session_state.config.get('target_fps', 30),
                crf=st.session_state.config.get('crf', 17),
                audio_bitrate=st.session_state.config.get('audio_bitrate', "192k"),
                fade_out_sec=st.session_state.config.get('fade_out_sec', 0.6)
            )
            
            exported_files.append(output_filename)
            progress_bar.progress((i + 1) / len(variants))
        
        # Create manifest
        manifest_data = {
            "anchor_time_sec": anchor_time,
            "anchor_source": "manual",
            "durations": durations,
            "variants": variants
        }
        
        manifest_path = os.path.join(output_dir, f"{file_stem}_manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest_data, f, indent=2)
        
        status_text.empty()
        progress_bar.empty()
        st.success(f"✓ Exported {len(exported_files)} clips to ~/Downloads/{folder_name}/")
        st.info(f"📁 Folder: {output_dir}")
        
    except Exception as e:
        st.error(f"Export failed: {e}")


def main():
    st.title("Clip Studio")
    
    # Check ffmpeg first
    if not check_ffmpeg():
        st.error("⚠️ **ffmpeg not found!** Please install ffmpeg and ensure it's on your PATH.")
        st.stop()
    
    # Settings section
    with st.expander("⚙️ Clip Settings", expanded=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Number of clips
            num_clips = st.slider(
                "Number of clips",
                min_value=1,
                max_value=10,
                value=10,
                help="How many clips to generate"
            )
        
        with col2:
            # Preset duration options
            preset_options = {
                "Preferred (12,8,14,9,15,7,13,10,8,16)": [12, 8, 14, 9, 15, 7, 13, 10, 8, 16],
                "Default (12,15,9,13,8)": [12, 15, 9, 13, 8],
                "Short (6,8,10,12,15)": [6, 8, 10, 12, 15],
                "Long (15,20,25,30,35)": [15, 20, 25, 30, 35],
                "Custom": "custom"
            }
            
            preset_choice = st.selectbox(
                "Duration preset",
                options=list(preset_options.keys()),
                help="Choose a preset or select Custom"
            )
        
        # Duration inputs
        if preset_choice == "Custom":
            st.write("**Custom Durations (seconds):**")
            durations = []
            cols = st.columns(min(num_clips, 5))  # Max 5 columns for layout
            
            # User's preferred duration sequence: [12, 8, 14, 9, 15, 7, 13, 10, 8, 16]
            preferred_durations = [12.0, 8.0, 14.0, 9.0, 15.0, 7.0, 13.0, 10.0, 8.0, 16.0]
            
            for i in range(num_clips):
                col_idx = i % 5
                with cols[col_idx]:
                    # Use preferred durations if available, otherwise use default progression
                    default_value = preferred_durations[i] if i < len(preferred_durations) else 12 + (i * 2)
                    duration = st.number_input(
                        f"Clip {i+1}",
                        min_value=0.1,
                        max_value=120.0,
                        value=float(default_value),
                        step=0.1,
                        format="%.1f",
                        key=f"duration_{i}"
                    )
                    durations.append(duration)
        else:
            # Use preset, but adapt to number of clips
            base_durations = preset_options[preset_choice]
            durations = []
            for i in range(num_clips):
                durations.append(base_durations[i % len(base_durations)])
            
            # Show the durations being used
            duration_text = ", ".join([f"{d}s" for d in durations])
            st.write(f"**Using durations:** {duration_text}")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload video file",
        type=['mp4', 'mov', 'mkv', 'avi', 'm4v']
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        # Display video
        st.video(tmp_path)
        
        # Anchor time input
        anchor_time = st.number_input(
            "Start Time (seconds)",
            min_value=0.0,
            max_value=300.0,
            value=30.0,
            step=0.1,
            format="%.1f",
            help="Enter the exact start time for all clips"
        )
        
        # Export button
        export_text = f"Export {num_clips} Clip{'s' if num_clips != 1 else ''}"
        if st.button(export_text, type="primary"):
            export_variants(tmp_path, uploaded_file.name, anchor_time, durations)


if __name__ == "__main__":
    main()