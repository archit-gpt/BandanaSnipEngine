#!/usr/bin/env python3
"""
Installation verification script for bandana_snip_engine.
Checks dependencies and system requirements.
"""

import sys
import subprocess
import importlib
from pathlib import Path


def check_python_version():
    """Check Python version is 3.7+"""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 7):
        print(f"❌ Python 3.7+ required, found {sys.version}")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} OK")
    return True


def check_dependencies():
    """Check all required Python packages are installed."""
    print("\n📦 Checking Python dependencies...")
    
    required_packages = [
        'numpy',
        'scipy', 
        'librosa',
        'soundfile',
        'cv2',  # opencv-python
        'moviepy',
        'tqdm',
        'streamlit',
        'yaml',  # pyyaml
        'matplotlib',
        'watchdog'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
                print(f"✅ opencv-python ({cv2.__version__}) OK")
            elif package == 'yaml':
                import yaml
                print(f"✅ pyyaml OK")
            else:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
                print(f"✅ {package} ({version}) OK")
                
        except ImportError:
            print(f"❌ {package} not found")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True


def check_ffmpeg():
    """Check ffmpeg is available and working."""
    print("\n🎬 Checking ffmpeg...")
    
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Extract version from output
        first_line = result.stdout.split('\n')[0]
        print(f"✅ {first_line}")
        
        return True
        
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print("❌ ffmpeg not found or not working")
        print("\nInstallation instructions:")
        print("• macOS: brew install ffmpeg")
        print("• Windows: Download from https://ffmpeg.org/download.html")
        print("• Linux: sudo apt install ffmpeg (Ubuntu/Debian)")
        print("         sudo yum install ffmpeg (CentOS/RHEL)")
        print("\nEnsure ffmpeg is in your PATH after installation.")
        return False


def check_project_structure():
    """Check project files are in place."""
    print("\n📁 Checking project structure...")
    
    current_dir = Path(__file__).parent.parent
    required_files = [
        'config.yaml',
        'requirements.txt',
        'src/__init__.py',
        'src/snip_engine.py',
        'src/exporter.py', 
        'src/utils.py',
        'app/gui_app.py'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        full_path = current_dir / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} missing")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ Missing files: {', '.join(missing_files)}")
        return False
    
    return True


def test_basic_functionality():
    """Test basic imports and functionality."""
    print("\n⚙️  Testing basic functionality...")
    
    try:
        # Add src to path
        src_path = Path(__file__).parent.parent / "src"
        sys.path.insert(0, str(src_path))
        
        # Test imports
        from utils import load_config
        from exporter import check_ffmpeg
        
        # Test config loading
        config = load_config()
        print("✅ Config loading works")
        
        # Test ffmpeg check
        ffmpeg_ok = check_ffmpeg()
        if ffmpeg_ok:
            print("✅ ffmpeg integration works")
        else:
            print("❌ ffmpeg integration failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False


def main():
    """Run all verification checks."""
    print("🔍 Bandana Snip Engine - Installation Verification")
    print("=" * 50)
    
    checks = [
        check_python_version,
        check_dependencies, 
        check_ffmpeg,
        check_project_structure,
        test_basic_functionality
    ]
    
    all_passed = True
    
    for check in checks:
        if not check():
            all_passed = False
    
    print("\n" + "=" * 50)
    
    if all_passed:
        print("🎉 All checks passed! Installation is ready.")
        print("\nTo run the application:")
        print("  streamlit run app/gui_app.py")
        sys.exit(0)
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()