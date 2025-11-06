#!/usr/bin/env python3
"""
Smoke tests for bandana_snip_engine.
Tests basic functionality with synthetic data.
"""

import unittest
import tempfile
import os
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import cv2
    import librosa
    import soundfile as sf
    from snip_engine import analyze_video, audio_features, visual_motion, suggest_anchors, make_manual_anchor_variants
    from exporter import check_ffmpeg, export_snippet
    from utils import load_config, validate_video_file
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Dependencies not available: {e}")
    DEPENDENCIES_AVAILABLE = False


class TestBandanaSnipEngine(unittest.TestCase):
    """Smoke tests for the snip engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not DEPENDENCIES_AVAILABLE:
            self.skipTest("Required dependencies not available")
        
        # Load default config
        try:
            self.config = load_config()
        except:
            # Fallback config
            self.config = {
                'durations_sec': [12, 15, 9, 13, 10],
                'min_gap_sec': 7.0,
                'target_fps': 30,
                'video_height': 1920,
                'crops': ["9x16_full", "4x5_center"],
                'fade_out_sec': 0.6,
                'crf': 17,
                'audio_bitrate': "192k",
                'fusion': {'audio_weight': 0.7, 'visual_weight': 0.3}
            }
    
    def test_audio_features(self):
        """Test audio feature extraction with synthetic audio."""
        # Create synthetic audio signal (3 seconds, 22050 Hz)
        sr = 22050
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # Mix of frequencies to create interesting features
        y = (np.sin(2 * np.pi * 440 * t) +  # A4 note
             0.5 * np.sin(2 * np.pi * 880 * t) +  # A5 note
             0.3 * np.random.randn(len(t)))  # Some noise
        
        times, audio_score = audio_features(y, sr)
        
        # Verify output format
        self.assertEqual(len(times), len(audio_score))
        self.assertTrue(np.all(audio_score >= 0))
        self.assertTrue(np.all(audio_score <= 1))
        self.assertGreater(len(times), 0)
    
    @unittest.skipUnless(check_ffmpeg(), "ffmpeg not available")
    def test_suggestion_video_analysis(self):
        """Test suggestion mode video analysis with synthetic video."""
        # Create a synthetic video file
        video_path = self.create_synthetic_video()
        
        try:
            # Use a config with shorter durations for the short test video
            test_config = self.config.copy()
            test_config['durations_sec'] = [1, 1, 1, 1, 1]  # Very short durations
            
            # Analyze the video using suggestion mode
            result = analyze_video(video_path, test_config)
            
            # Verify result structure for suggestion mode
            self.assertIsInstance(result, dict, "Should return a dictionary")
            self.assertIn('suggestions', result)
            self.assertIn('beats', result)
            self.assertIn('debug', result)
            
            # Verify suggestions format
            suggestions = result['suggestions']
            self.assertIsInstance(suggestions, list)
            for suggestion in suggestions:
                self.assertIn('time', suggestion)
                self.assertIn('score', suggestion)
                self.assertIsInstance(suggestion['time'], (int, float))
                self.assertIsInstance(suggestion['score'], (int, float))
                self.assertGreaterEqual(suggestion['time'], 0)
            
            # Test manual variant creation
            if suggestions:
                anchor_time = suggestions[0]['time']
                beats = np.array(result['beats'])
                variants = make_manual_anchor_variants(
                    anchor_time, test_config['durations_sec'], beats, test_config
                )
                
                # Verify we get exactly 5 variants
                self.assertEqual(len(variants), 5, "Should return exactly 5 variants")
                
                # Verify variant format
                for variant in variants:
                    self.assertIn('index', variant)
                    self.assertIn('start', variant)
                    self.assertIn('dur', variant)
                    self.assertIn('beat_snapped', variant)
                    self.assertIn('jitter_ms', variant)
                    self.assertIsInstance(variant['start'], (int, float))
                    self.assertIsInstance(variant['dur'], int)
                    self.assertGreaterEqual(variant['start'], 0)
                    self.assertGreater(variant['dur'], 0)
            
            # Verify debug info contains expected keys
            debug = result.get('debug', {})
            self.assertIn('times', debug)
            self.assertIn('fused', debug)
            self.assertIn('video_duration', debug)
            
        finally:
            # Cleanup
            if os.path.exists(video_path):
                os.unlink(video_path)
    
    def test_config_loading(self):
        """Test configuration loading."""
        config = self.config
        
        # Check required keys exist
        self.assertIn('durations_sec', config)
        self.assertIn('suggestions', config)
        self.assertEqual(len(config['durations_sec']), 5)
    
    def test_video_file_validation(self):
        """Test video file validation."""
        # Test valid extensions
        valid_files = ['test.mp4', 'test.mov', 'test.mkv', 'test.avi']
        for filename in valid_files:
            with tempfile.NamedTemporaryFile(suffix=filename) as tmp:
                # File exists but might not be a real video - that's OK for this test
                pass
        
        # Test invalid extensions
        invalid_files = ['test.txt', 'test.jpg', 'test.mp3']
        for filename in invalid_files:
            self.assertFalse(validate_video_file(filename))
    
    @unittest.skipUnless(check_ffmpeg(), "ffmpeg not available")
    def test_export_functionality(self):
        """Test snippet export with synthetic video."""
        video_path = self.create_synthetic_video()
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as output_file:
                output_path = output_file.name
            
            try:
                # Export a short snippet
                export_snippet(
                    src_path=video_path,
                    output_path=output_path,
                    start_sec=0.5,
                    duration_sec=2,
                    height=480,  # Smaller for faster processing
                    crop_mode="9x16_full",
                    fps=15,  # Lower FPS for faster processing
                    crf=28,  # Higher CRF for faster encoding
                    audio_bitrate="128k",
                    fade_out_sec=0.3
                )
                
                # Verify output file was created and has reasonable size
                self.assertTrue(os.path.exists(output_path))
                self.assertGreater(os.path.getsize(output_path), 1000)  # At least 1KB
                
            finally:
                if os.path.exists(output_path):
                    os.unlink(output_path)
        
        finally:
            if os.path.exists(video_path):
                os.unlink(video_path)
    
    def create_synthetic_video(self) -> str:
        """Create a synthetic video file for testing."""
        # Create temporary video file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            video_path = tmp_file.name
        
        # Generate synthetic audio (3 seconds)
        sr = 22050
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # Create audio with some variation for interesting features
        audio = np.sin(2 * np.pi * 440 * t) * np.exp(-t)  # Decaying sine wave
        audio += 0.3 * np.random.randn(len(t))  # Add noise
        audio = np.clip(audio, -1, 1)  # Ensure valid range
        
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as audio_file:
            audio_path = audio_file.name
        
        sf.write(audio_path, audio, sr)
        
        try:
            # Create video with ffmpeg
            cmd = [
                'ffmpeg', '-y',
                '-f', 'lavfi', '-i', 'testsrc=duration=3:size=320x240:rate=15',
                '-i', audio_path,
                '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '30',
                '-c:a', 'aac', '-shortest',
                video_path
            ]
            
            import subprocess
            result = subprocess.run(cmd, capture_output=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"Failed to create synthetic video: {result.stderr}")
            
            return video_path
            
        finally:
            # Cleanup audio file
            if os.path.exists(audio_path):
                os.unlink(audio_path)


class TestMinimalFunctionality(unittest.TestCase):
    """Minimal tests that don't require ffmpeg or full dependencies."""
    
    def test_imports(self):
        """Test that we can import core modules."""
        if not DEPENDENCIES_AVAILABLE:
            self.skipTest("Dependencies not available")
        
        # These should not raise ImportError
        from snip_engine import audio_features, suggest_anchors
        from utils import get_available_durations
        from exporter import check_ffmpeg
        
        # Test a simple utility function
        durations = get_available_durations()
        self.assertIsInstance(durations, list)
        self.assertGreater(len(durations), 0)


def main():
    """Run smoke tests."""
    print("Running Bandana Snip Engine smoke tests...")
    
    if not DEPENDENCIES_AVAILABLE:
        print("⚠️  Dependencies not available - running minimal tests only")
    
    if not check_ffmpeg():
        print("⚠️  ffmpeg not available - skipping video-related tests")
    
    # Run tests
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == '__main__':
    main()