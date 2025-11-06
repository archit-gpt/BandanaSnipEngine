#!/bin/bash
set -euo pipefail

# Bandana Snip Engine - Quick Run Script
# Usage: ./run_local.sh (runs app without reinstalling dependencies)

# Auto-detect script directory and navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

echo "▶ Launching Bandana Snip Engine..."
echo "================================="

cd "$PROJECT_DIR"

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Run ./start_bandana.sh first to set up."
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Quick verification
echo "🔍 Quick environment check..."
python -c "import cv2, streamlit; print(f'✅ OpenCV {cv2.__version__} | Streamlit {streamlit.__version__}')" || {
    echo "❌ Dependencies missing or broken. Run ./start_bandana.sh to reinstall."
    exit 1
}

# Start the application
echo "🚀 Starting Streamlit app..."
echo "App will open at: http://localhost:8501"
echo "Press Ctrl+C to stop the application"
echo "======================================================="

exec streamlit run app/gui_app.py