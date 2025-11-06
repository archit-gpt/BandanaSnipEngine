#!/bin/bash
set -euo pipefail

# Bandana Snip Engine Startup Script (Human-in-the-Loop Mode)
# Usage: ./start_bandana.sh

# Configuration
APP_CMD="streamlit run app/gui_app.py"
REQUIREMENTS_FILE="requirements.txt"

# Auto-detect script directory and navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

echo "◾ Starting Clip Studio..."
echo "========================="
echo "🔧 Project directory: $PROJECT_DIR"

cd "$PROJECT_DIR"

# Python detection strategy
detect_python() {
    if command -v python3.12 >/dev/null 2>&1; then
        echo "python3.12"
    elif command -v python3.11 >/dev/null 2>&1; then
        echo "python3.11"
    elif command -v python3 >/dev/null 2>&1; then
        echo "python3"
    else
        echo "❌ No suitable Python found. Install Python 3.8+ and try again."
        exit 1
    fi
}

PYTHON_CMD=$(detect_python)
PYTHON_VERSION=$($PYTHON_CMD --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "🐍 Using Python: $PYTHON_CMD ($PYTHON_VERSION)"

# Check if venv needs recreation
VENV_NEEDS_RECREATE=false

if [ ! -d ".venv" ]; then
    echo "📁 Virtual environment not found. Creating new one..."
    VENV_NEEDS_RECREATE=true
elif [ ! -f ".venv/pyvenv.cfg" ]; then
    echo "🔄 Broken venv detected (missing pyvenv.cfg). Recreating..."
    VENV_NEEDS_RECREATE=true
else
    # Check if venv python matches our chosen python
    VENV_PYTHON_VERSION=$(grep "version = " .venv/pyvenv.cfg | cut -d' ' -f3 | cut -d'.' -f1,2 || echo "unknown")
    if [ "$VENV_PYTHON_VERSION" != "$PYTHON_VERSION" ]; then
        echo "🔄 Python version mismatch (venv: $VENV_PYTHON_VERSION, system: $PYTHON_VERSION). Recreating..."
        VENV_NEEDS_RECREATE=true
    fi
fi

# Recreate venv if needed
if [ "$VENV_NEEDS_RECREATE" = true ]; then
    [ -d ".venv" ] && rm -rf .venv
    echo "🔨 Creating virtual environment with $PYTHON_CMD..."
    $PYTHON_CMD -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Verify activation (fail fast if pip defaults to user installation)
echo "🔍 Verifying venv activation..."
PIP_OUTPUT=$(pip --version 2>&1)
if echo "$PIP_OUTPUT" | grep -q "Defaulting to user installation"; then
    echo "❌ Virtual environment activation failed. pip is using user installation."
    echo "❌ Debug info: $PIP_OUTPUT"
    exit 1
fi

# Upgrade pip, setuptools, wheel
echo "📦 Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

# Install dependencies with wheel-only strategy for opencv
echo "📦 Installing dependencies..."

# Set wheel-only for opencv to prevent source compilation
if [[ "$PYTHON_VERSION" == "3.8" ]]; then
    echo "🔧 Python 3.8 detected - using wheel-only strategy for opencv"
    PIP_ONLY_BINARY="opencv-python,opencv-python-headless" pip install -r "$REQUIREMENTS_FILE"
else
    pip install -r "$REQUIREMENTS_FILE"
fi

# Post-install sanity check
echo "🔍 Performing post-install sanity check..."
python - <<'PY'
import sys, cv2, platform
print(f"✅ OK Python {sys.version.split()[0]} | OpenCV {cv2.__version__} | macOS {platform.mac_ver()[0]}")
PY

# Verify installation script if it exists
if [ -f "scripts/verify_install.py" ]; then
    echo "🔍 Running verification script..."
    python scripts/verify_install.py
    if [ $? -ne 0 ]; then
        echo "❌ Dependencies verification failed."
        exit 1
    fi
fi

# Start the application
echo "▶ Launching Clip Studio interface..."
echo "App will open at: http://localhost:8501"
echo "Press Ctrl+C to stop the application"
echo "======================================================="

exec $APP_CMD