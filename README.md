# 🎬 Bandana Snip Engine

A production-ready desktop application with **Human-in-the-Loop Anchor Selection** that analyzes music videos to help creators find and export viral-worthy moments for TikTok, Reels, and Shorts.

## What It Does

Bandana Snip Engine uses **Manual Anchor Mode** - putting the human creator in control:

1. **Upload** your music video (mp4/mov/mkv, up to 4 minutes)
2. **Get AI Suggestions** - the engine analyzes and proposes up to 5 anchor candidates
3. **Choose Your Anchor** - either select an AI suggestion OR manually set the perfect moment
4. **Generate 5 Variants** - automatically creates clips with different durations around your chosen anchor
5. **Export** TikTok/Reels-ready snippets in 9:16 or 4:5 aspect ratios

**Why Human-in-the-Loop?** You know your content better than any algorithm. The AI provides smart suggestions, but you make the final creative decision about where the hook truly is.

## Key Features

### 🎯 Manual + AI-Assisted Workflow
- **Video Player**: Scrub through your video to find the perfect moment
- **AI Suggestions**: Get up to 5 algorithmically-detected anchor candidates
- **Manual Override**: Set anchor time precisely with keyboard shortcuts and nudge buttons
- **Instant Preview**: See exactly how your 5 variants will be positioned

### 🎬 Smart Variant Generation
- **5 Different Durations**: Default [12, 15, 9, 13, 8] seconds (fully customizable)
- **Intelligent Positioning**: Anchor occurs 25%-45% into each clip for maximum impact
- **Beat Snapping**: Automatically aligns clip starts to musical beats (±350ms window)
- **Micro Jitter**: Variants 2-5 get ±150ms randomization for variety
- **Professional Export**: H.264 video (CRF 17), AAC audio (192k), smooth fade-out

### 📊 Advanced Analysis (For Suggestions)
- **Audio Features (70%)**: Onset strength, RMS energy, repetition detection
- **Visual Features (30%)**: Motion analysis and scene change detection
- **Viral Moment Detection**: Novelty bias (build-up → payoff) and chant detection
- **Beat Detection**: Musical timing analysis for perfect clip alignment

## Installation

### Prerequisites
- Python 3.7+
- ffmpeg (must be on PATH)

### Install ffmpeg
```bash
# macOS
brew install ffmpeg

# Windows 
# Download from https://ffmpeg.org/download.html and add to PATH

# Linux
sudo apt install ffmpeg  # Ubuntu/Debian
sudo yum install ffmpeg  # CentOS/RHEL
```

### Install Python Dependencies
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python scripts/verify_install.py
```

## Usage

### Launch the GUI
```bash
streamlit run app/gui_app.py
```

This opens a web interface at `http://localhost:8501`

### Basic Workflow

#### 1. Upload & Configure
- **Upload Video**: Drag and drop your video file
- **Select Durations**: Choose exactly 5 durations for your variants (default: [12, 15, 9, 13, 8]s)
- **Choose Crop Mode**: 9:16 (TikTok/Reels) or 4:5 (Instagram)
- **Set Output Directory**: Where your exported files will be saved

#### 2. Find Your Anchor
**Option A: AI Suggestions**
- Click "🔍 Analyze & Suggest Anchors"
- View 5 AI-suggested anchor points with confidence scores
- Click any suggestion to select it as your anchor

**Option B: Manual Selection**
- Use the video player to scrub to your desired moment
- Enter the exact timestamp OR use +/-1s and +/-0.1s adjustment buttons
- Click "📍 Set Anchor (Manual)" to confirm

#### 3. Preview & Export
- **Preview Variants**: See exactly where each of your 5 clips will start and end
- **Export Options**: Export individual variants or all 5 at once
- **Files Created**:
  - `{video_name}_anchor-v1_{duration}s.mp4` through `v5`
  - `{video_name}_manifest.json` with all timing data

### Output Format
- **Naming**: `song_anchor-v{1-5}_{duration}s.mp4`
- **Quality**: H.264 video (CRF 17), AAC audio (192k)
- **Aspect Ratios**: 1080x1920 (9:16) or 1080x1350 (4:5)
- **Special Effects**: 0.6s fade-out for smooth looping
- **Manifest**: JSON file with anchor time, source (manual/suggested), and all variant details

### File Size Limits
- **Default**: 200MB per file (Streamlit default)
- **Increased**: 1GB per file (using `.streamlit/config.toml`)
- **Custom**: Edit `.streamlit/config.toml` to set `maxUploadSize` (in MB)

## Configuration

Edit `config.yaml` to customize defaults:

```yaml
defaults:
  durations_sec: [12, 15, 9, 13, 8]    # 5 variant durations
  target_fps: 30                       # Output frame rate
  video_height: 1920                   # Output resolution height
  crops: ["9x16_full", "4x5_center"]   # Available crop modes
  fade_out_sec: 0.6                    # Fade out duration
  crf: 17                              # Video quality (lower = better)
  audio_bitrate: "192k"                # Audio quality
  fusion:
    audio_weight: 0.7                  # Audio vs visual analysis weight
    visual_weight: 0.3
  anchor:
    beat_snap_ms: 350                  # Beat snapping window (±ms)
    start_offsets_factor: [0.35, 0.45, 0.25, 0.35, 0.30]  # Where anchor lands in each clip
    jitter_ms: 150                     # Micro-randomization for variants 2-5
  suggestions:
    enabled: true                      # Enable AI suggestions
    max_candidates: 5                  # Max number of suggestions
    min_spacing_sec: 2.0               # Minimum spacing between suggestions
```

## Troubleshooting

### "File must be 200.0MB or smaller"
- Video file exceeds Streamlit's default upload limit
- **Solution 1**: Use the included `.streamlit/config.toml` (supports up to 1GB)
- **Solution 2**: Compress video: `ffmpeg -i large_video.mp4 -crf 28 compressed_video.mp4`

### "ffmpeg not found"
- Ensure ffmpeg is installed and on your PATH
- Test with: `ffmpeg -version`
- Restart terminal after installation

### "Could not load audio"
- Video file may be corrupted or unsupported format
- Try converting to MP4 first: `ffmpeg -i input.mov output.mp4`

### AI suggestions seem wrong
- Remember: suggestions are starting points - trust your creative judgment
- The algorithm works best with music videos that have clear rhythmic patterns
- For dialogue or ambient content, manual selection is often better

### Export failing
- Check output directory has write permissions
- Ensure enough disk space (exports can be large)
- Verify ffmpeg can write to the target location

## Performance & Limitations

### Performance
- **Target**: <30 seconds analysis per 3-minute video on modern laptop
- **Optimized for**: Videos up to 4 minutes, 1080p+
- **Memory Usage**: Modest - most videos process without issues

### Current Limitations
- **Single Video Processing**: One video at a time (by design for precision)
- **Audio Dependency**: Requires clear audio track for beat detection
- **Visual Analysis**: Basic motion detection only
- **Suggestion Quality**: Varies by content type (music videos work best)

## Development

### Project Structure
```
bandana_snip_engine/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config.yaml              # Configuration
├── .streamlit/
│   └── config.toml          # Streamlit settings (file upload limits)
├── app/
│   └── gui_app.py           # Human-in-the-loop Streamlit interface
├── src/
│   ├── __init__.py
│   ├── snip_engine.py       # Core analysis algorithms + suggestion engine
│   ├── exporter.py          # ffmpeg integration
│   └── utils.py             # Helper functions
├── scripts/
│   └── verify_install.py    # Installation checker
└── tests/
    └── test_engine_smoke.py # Basic functionality tests
```

### Running Tests
```bash
python tests/test_engine_smoke.py
```

### Key Technical Components
- **AI Analysis**: librosa for audio analysis, OpenCV for video analysis
- **Beat Detection**: Temporal pattern recognition and onset detection
- **Human Interface**: Streamlit with video player and waveform visualization
- **Export Pipeline**: ffmpeg subprocess calls with quality optimization

## Manifest Schema

Each export creates a manifest JSON file with this structure:

```json
{
  "anchor_time_sec": 45.67,
  "anchor_source": "manual",
  "durations": [12, 15, 9, 13, 8],
  "variants": [
    {
      "index": 1,
      "start": 30.45,
      "dur": 12,
      "beat_snapped": true,
      "jitter_ms": 0
    }
  ]
}
```

## License

Production-ready code with robust error handling and user experience optimizations.

---

*Built for content creators who want full creative control with intelligent assistance.*