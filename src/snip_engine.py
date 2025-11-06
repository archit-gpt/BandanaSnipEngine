"""
Core analysis engine for bandana_snip_engine.
Analyzes audio and visual features to find viral-worthy moments.
"""

import numpy as np
import librosa
import cv2
import random
from typing import List, Tuple, Dict, Any, Optional
from scipy import ndimage, signal
from scipy.signal import find_peaks, butter, sosfilt
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def audio_features(y: np.ndarray, sr: int, hop_length: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract audio features for viral moment detection.
    
    Args:
        y: Audio time series
        sr: Sample rate
        hop_length: FFT hop length
        
    Returns:
        Tuple of (times, audio_score) where audio_score is normalized [0,1]
    """
    # Onset strength (spectral flux / impact)
    onset_envelope = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=hop_length, aggregate=np.median
    )
    
    # RMS energy
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    
    # Chroma CENS for repetition/chorus detection
    chroma = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop_length)
    
    # Compute self-similarity for repetition detection
    # Create stacked chroma features manually for recurrence analysis
    if chroma.shape[1] > 3:
        chroma_stack = np.concatenate([
            chroma[:, :-2],
            chroma[:, 1:-1], 
            chroma[:, 2:]
        ], axis=0)
    else:
        chroma_stack = chroma
        
    rec_matrix = librosa.segment.recurrence_matrix(
        chroma_stack, metric='cosine', mode='affinity'
    )
    
    # Sum along diagonal bands to find repetitive patterns
    repetition_score = np.zeros(chroma.shape[1])
    for lag in range(1, min(20, rec_matrix.shape[0])):
        if lag < rec_matrix.shape[0] and lag < rec_matrix.shape[1]:
            diag = np.diagonal(rec_matrix, offset=lag)
            if len(diag) > 0:
                repetition_score[:len(diag)] += diag
    
    # Ensure all arrays have same length
    min_len = min(len(onset_envelope), len(rms), len(repetition_score))
    onset_envelope = onset_envelope[:min_len]
    rms = rms[:min_len]
    repetition_score = repetition_score[:min_len]
    
    # Normalize features to [0,1]
    def normalize(arr):
        if np.max(arr) > np.min(arr):
            return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        return np.zeros_like(arr)
    
    onset_norm = normalize(onset_envelope)
    rms_norm = normalize(rms)
    rep_norm = normalize(repetition_score)
    
    # Weighted combination
    audio_score = 0.45 * onset_norm + 0.35 * rms_norm + 0.20 * rep_norm
    
    # Time axis
    times = librosa.frames_to_time(
        np.arange(len(audio_score)), sr=sr, hop_length=hop_length
    )
    
    return times, audio_score


def visual_motion(video_path: str, fps_sample: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract visual motion features from video.
    
    Args:
        video_path: Path to video file
        fps_sample: Target FPS for sampling frames
        
    Returns:
        Tuple of (times, motion_score) normalized [0,1]
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    # Calculate frame skip to achieve target sampling rate
    frame_skip = max(1, int(fps / fps_sample))
    
    motion_scores = []
    times = []
    prev_gray = None
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % frame_skip == 0:
            # Convert to grayscale and resize for efficiency
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (320, 240))
            
            if prev_gray is not None:
                # Calculate frame difference
                diff = cv2.absdiff(gray, prev_gray)
                motion_score = np.mean(diff)
                motion_scores.append(motion_score)
                times.append(frame_idx / fps)
            
            prev_gray = gray
            
        frame_idx += 1
    
    cap.release()
    
    if not motion_scores:
        return np.array([0]), np.array([0])
    
    motion_scores = np.array(motion_scores)
    times = np.array(times)
    
    # Normalize to [0,1]
    if np.max(motion_scores) > np.min(motion_scores):
        motion_scores = (motion_scores - np.min(motion_scores)) / (
            np.max(motion_scores) - np.min(motion_scores)
        )
    else:
        motion_scores = np.zeros_like(motion_scores)
    
    return times, motion_scores


def fuse_scores(
    audio_times: np.ndarray,
    audio_scores: np.ndarray,
    visual_times: np.ndarray,
    visual_scores: np.ndarray,
    audio_weight: float = 0.7,
    visual_weight: float = 0.3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fuse audio and visual scores onto common timebase.
    
    Args:
        audio_times: Audio feature timestamps
        audio_scores: Audio feature scores [0,1]
        visual_times: Visual feature timestamps  
        visual_scores: Visual feature scores [0,1]
        audio_weight: Weight for audio features
        visual_weight: Weight for visual features
        
    Returns:
        Tuple of (times, fused_scores)
    """
    # Use audio timebase as reference
    times = audio_times
    
    # Interpolate visual scores onto audio timebase
    if len(visual_times) > 1:
        visual_interp = np.interp(times, visual_times, visual_scores)
    else:
        visual_interp = np.zeros_like(times)
    
    # Weighted fusion
    fused_scores = audio_weight * audio_scores + visual_weight * visual_interp
    
    return times, fused_scores


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize array to [0,1] range."""
    if np.max(arr) > np.min(arr):
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    return np.zeros_like(arr)


def detect_beats(y: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
    """
    Detect beat times using librosa.
    
    Args:
        y: Audio time series
        sr: Sample rate
        hop_length: FFT hop length
        
    Returns:
        Array of beat times in seconds
    """
    try:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
        beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
        return beat_times
    except:
        return np.array([])


def periodicity_score(envelope: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
    """
    Compute periodicity score for chant/repetition detection.
    Analyzes RMS envelope in 1.5-4.5 Hz range (syllabic rate).
    
    Args:
        envelope: RMS envelope of audio
        sr: Sample rate
        hop_length: FFT hop length
        
    Returns:
        Periodicity score array [0,1]
    """
    if len(envelope) < 10:
        return np.zeros_like(envelope)
    
    # Frame rate for the envelope
    frame_rate = sr / hop_length
    
    # Design bandpass filter for 1.5-4.5 Hz
    nyquist = frame_rate / 2
    if nyquist <= 4.5:
        # If frame rate too low, skip filtering
        return np.zeros_like(envelope)
    
    low = 1.5 / nyquist
    high = min(4.5 / nyquist, 0.99)
    
    try:
        sos = butter(2, [low, high], btype='band', output='sos')
        filtered = sosfilt(sos, envelope)
        
        # Compute local standard deviation as periodicity proxy
        window_size = int(frame_rate)  # 1 second window
        periodicity = np.zeros_like(envelope)
        
        for i in range(len(envelope)):
            start = max(0, i - window_size // 2)
            end = min(len(envelope), i + window_size // 2)
            window = filtered[start:end]
            periodicity[i] = np.std(window)
        
        return _normalize(periodicity)
    except:
        return np.zeros_like(envelope)


def compute_novelty(fused_scores: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
    """
    Compute novelty curve to favor peaks preceded by valleys.
    
    Args:
        fused_scores: Fused audio/visual scores
        sr: Sample rate
        hop_length: FFT hop length
        
    Returns:
        Novelty scores [0,1]
    """
    # Create a median-filtered version (local background)
    kernel_size = int(2.0 * sr / hop_length)  # 2 second kernel
    kernel_size = max(3, kernel_size)
    
    # Pad and apply median filter
    padded = np.pad(fused_scores, pad_width=kernel_size//2, mode='edge')
    filtered = ndimage.median_filter(padded, size=kernel_size)
    filtered = filtered[kernel_size//2:-kernel_size//2]
    
    # Novelty = difference from local background
    novelty = np.maximum(0, fused_scores - filtered)
    return _normalize(novelty)


def beat_snap(start_time: float, beats: np.ndarray, window_ms: float = 350.0) -> float:
    """
    Snap start time to nearest beat within window.
    
    Args:
        start_time: Original start time
        beats: Array of beat times
        window_ms: Snap window in milliseconds
        
    Returns:
        Snapped start time
    """
    if len(beats) == 0:
        return start_time
    
    window_sec = window_ms / 1000.0
    
    # Find beats within window
    close_beats = beats[np.abs(beats - start_time) <= window_sec]
    
    if len(close_beats) == 0:
        return start_time
    
    # Return closest beat
    closest_idx = np.argmin(np.abs(close_beats - start_time))
    return float(close_beats[closest_idx])


def detect_anchor(
    times: np.ndarray,
    audio_scores: np.ndarray, 
    visual_times: np.ndarray,
    visual_scores: np.ndarray,
    y: np.ndarray,
    sr: int,
    config: Dict[str, Any]
) -> Tuple[float, float, np.ndarray]:
    """
    Detect the dominant anchor moment using audio+visual analysis.
    
    Args:
        times: Audio feature timestamps
        audio_scores: Audio feature scores [0,1]
        visual_times: Visual feature timestamps
        visual_scores: Visual feature scores [0,1]
        y: Raw audio signal
        sr: Sample rate
        config: Configuration dictionary
        
    Returns:
        Tuple of (anchor_time, confidence, fused_scores)
    """
    fusion_config = config.get('fusion', {})
    audio_weight = fusion_config.get('audio_weight', 0.7)
    visual_weight = fusion_config.get('visual_weight', 0.3)
    
    # Fuse audio and visual scores
    fused_times, fused_scores = fuse_scores(
        times, audio_scores, visual_times, visual_scores,
        audio_weight, visual_weight
    )
    
    # Compute novelty bias (prefer peaks after valleys)
    novelty = compute_novelty(fused_scores, sr)
    
    # Compute periodicity bias for chant detection
    hop_length = 512
    rms_envelope = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    periodicity = periodicity_score(rms_envelope, sr, hop_length)
    
    # Interpolate periodicity onto fused timebase if needed
    if len(periodicity) != len(fused_scores):
        rms_times = librosa.frames_to_time(np.arange(len(periodicity)), sr=sr, hop_length=hop_length)
        periodicity = np.interp(fused_times, rms_times, periodicity)
    
    # Combined peak strength
    peak_strength = fused_scores * (1 + 0.25 * periodicity) * (0.6 + 0.4 * novelty)
    
    # Find peaks with minimum separation
    min_gap_sec = config.get('min_gap_sec', 7.0)
    min_distance = max(1, int(min_gap_sec / (fused_times[1] - fused_times[0])))
    
    peaks, _ = find_peaks(peak_strength, distance=min_distance)
    
    if len(peaks) == 0:
        # Fallback to global maximum
        anchor_idx = int(np.argmax(peak_strength))
        anchor_time = float(fused_times[anchor_idx])
        confidence = 0.5
    else:
        # Best peak
        peak_values = peak_strength[peaks]
        best_peak_idx = peaks[np.argmax(peak_values)]
        anchor_time = float(fused_times[best_peak_idx])
        
        # Compute confidence
        best_value = peak_strength[best_peak_idx]
        neighborhood = peak_strength[peaks]
        
        if len(neighborhood) > 1:
            median_val = np.median(neighborhood)
            iqr = np.percentile(neighborhood, 75) - np.percentile(neighborhood, 25)
            confidence = (best_value - median_val) / (iqr + 1e-9)
            confidence = float(np.clip(confidence, 0, 1))
        else:
            confidence = 0.8  # Single peak case
    
    return anchor_time, confidence, peak_strength


def make_anchor_variants(
    anchor_time: float,
    durations: List[float],
    beats: np.ndarray,
    config: Dict[str, Any],
    secondary_anchor: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Generate 5 variants around the anchor time.
    
    Args:
        anchor_time: Primary anchor time
        durations: List of 5 durations
        beats: Beat times array
        config: Configuration dictionary
        secondary_anchor: Optional secondary anchor for fallback
        
    Returns:
        List of variant dictionaries
    """
    anchor_config = config.get('anchor', {})
    start_offsets = anchor_config.get('start_offsets_factor', [0.35, 0.45, 0.25, 0.35, 0.30])
    jitter_ms = anchor_config.get('jitter_ms', 150)
    beat_snap_ms = anchor_config.get('beat_snap_ms', 350)
    
    variants = []
    
    # If we have secondary anchor, allocate 3+2 split
    if secondary_anchor is not None:
        anchors = [anchor_time] * 3 + [secondary_anchor] * 2
    else:
        anchors = [anchor_time] * 5
    
    for i, (duration, anchor) in enumerate(zip(durations, anchors), start=1):
        # Calculate start time
        offset_factor = start_offsets[i-1] if i-1 < len(start_offsets) else 0.35
        start = max(0.0, anchor - offset_factor * duration)
        
        # Beat snap
        start = beat_snap(start, beats, beat_snap_ms)
        
        # Apply jitter (except for first variant)
        jitter_applied = 0
        if i > 1:
            jitter_applied = random.uniform(-jitter_ms, jitter_ms)
            start = max(0.0, start + jitter_applied / 1000.0)
        
        # Visual hint for variant 5
        visual_hint = "max_motion_near_anchor" if i == 5 else None
        
        variant = {
            "index": i,
            "start": float(start),
            "dur": float(duration),
            "beat_snapped": True,
            "jitter_ms": int(abs(jitter_applied)) if i > 1 else 0,
            "visual_hint": visual_hint
        }
        
        variants.append(variant)
    
    return variants


def pick_segments(
    times: np.ndarray,
    scores: np.ndarray,
    durations: List[float],
    min_gap_sec: float = 7.0
) -> List[Tuple[float, int]]:
    """
    Pick 5 segments based on peak scores with minimum gap constraint.
    
    Args:
        times: Time array
        scores: Fused score array
        durations: List of 5 segment durations
        min_gap_sec: Minimum gap between segments
        
    Returns:
        List of (start_time, duration) tuples
    """
    if len(durations) != 5:
        raise ValueError("Must provide exactly 5 durations")
    
    # Smooth scores slightly to avoid noise peaks
    smoothed_scores = ndimage.gaussian_filter1d(scores, sigma=1.0)
    
    # Find initial peaks
    peaks, properties = find_peaks(
        smoothed_scores,
        height=np.percentile(smoothed_scores, 60),  # Above 60th percentile
        distance=int(min_gap_sec / (times[1] - times[0]))  # Convert to samples
    )
    
    if len(peaks) == 0:
        # Fallback: use highest scoring points
        peaks = np.argsort(smoothed_scores)[-10:]  # Top 10 candidates
    
    # Select top 5 peaks by score
    peak_scores = smoothed_scores[peaks]
    top_peak_indices = np.argsort(peak_scores)[-5:]
    selected_peaks = peaks[top_peak_indices]
    
    # Sort by time
    selected_peaks = np.sort(selected_peaks)
    
    # Ensure minimum gap between peaks
    final_peaks = [selected_peaks[0]]
    for peak in selected_peaks[1:]:
        if times[peak] - times[final_peaks[-1]] >= min_gap_sec:
            final_peaks.append(peak)
        elif len(final_peaks) < 5:
            # If we need more peaks, find the next best one
            continue
            
    # If we don't have enough peaks, fill with evenly spaced ones
    while len(final_peaks) < 5:
        # Find a good position that maintains minimum gap
        video_duration = times[-1]
        for i in range(len(times)):
            candidate_time = times[i]
            valid = True
            for existing_peak in final_peaks:
                if abs(candidate_time - times[existing_peak]) < min_gap_sec:
                    valid = False
                    break
            if valid and candidate_time > min_gap_sec and candidate_time < video_duration - max(durations):
                final_peaks.append(i)
                break
        
        if len(final_peaks) == len(selected_peaks):  # No progress made
            break
    
    # Limit to 5 peaks
    final_peaks = final_peaks[:5]
    
    # Generate segments
    segments = []
    for i, peak_idx in enumerate(final_peaks):
        duration = durations[i % len(durations)]
        peak_time = times[peak_idx]
        
        # Position start time so peak occurs at 35% into the segment
        start_time = max(0, peak_time - 0.35 * duration)
        
        # Ensure segment doesn't exceed video bounds
        video_duration = times[-1]
        if start_time + duration > video_duration:
            start_time = max(0, video_duration - duration)
        
        segments.append((start_time, duration))
    
    return segments


def suggest_anchors(
    video_path: str, 
    config: Dict[str, Any], 
    max_candidates: int = 5, 
    min_spacing_sec: float = 2.0
) -> List[Dict[str, float]]:
    """
    Generate anchor suggestions for human selection.
    
    Args:
        video_path: Path to video file
        config: Configuration dictionary
        max_candidates: Maximum number of suggestions
        min_spacing_sec: Minimum spacing between suggestions
        
    Returns:
        List of {"time": float, "score": float} sorted by score (descending)
    """
    # Load audio
    try:
        y, sr = librosa.load(video_path, sr=None)
    except Exception as e:
        raise ValueError(f"Could not load audio from {video_path}: {e}")
    
    # Extract features
    audio_times, audio_scores = audio_features(y, sr)
    visual_times, visual_scores = visual_motion(video_path)
    
    # Get peak strength using existing logic
    _, _, peak_strength = detect_anchor(
        audio_times, audio_scores, visual_times, visual_scores,
        y, sr, config
    )
    
    # Find peaks with minimum spacing
    min_distance = max(1, int(min_spacing_sec / (audio_times[1] - audio_times[0])))
    peaks, _ = find_peaks(peak_strength, distance=min_distance)
    
    if len(peaks) == 0:
        # Fallback to highest points
        peak_indices = np.argsort(peak_strength)[-max_candidates:]
        peaks = peak_indices[peak_indices > 0]  # Ensure valid indices
    
    # Get peak values and sort by strength
    peak_values = peak_strength[peaks]
    peak_times = audio_times[peaks]
    
    # Create suggestions sorted by score (descending)
    suggestions = []
    for time, score in zip(peak_times, peak_values):
        suggestions.append({"time": float(time), "score": float(score)})
    
    # Sort by score and limit
    suggestions.sort(key=lambda x: x["score"], reverse=True)
    return suggestions[:max_candidates]


def make_manual_anchor_variants(
    anchor_time: float,
    durations: List[float],
    beats: np.ndarray,
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Generate 5 variants starting at the manually selected anchor time.
    
    Args:
        anchor_time: Manually selected anchor time (this becomes the start time)
        durations: List of 5 durations
        beats: Beat times array (unused - anchor is exact)
        config: Configuration dictionary
        
    Returns:
        List of variant dictionaries
    """
    variants = []
    
    for i, duration in enumerate(durations, start=1):
        # Anchor time IS the start time - that's the point of human selection
        start = max(0.0, anchor_time)
        
        variant = {
            "index": i,
            "start": float(start),
            "dur": float(duration),
            "beat_snapped": False,  # Human chose exact time
            "jitter_ms": 0  # No jitter needed - human is precise
        }
        
        variants.append(variant)
    
    return variants


def analyze_video(video_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main analysis function for suggestion mode.
    Returns beat info and anchor suggestions for human selection.
    
    Args:
        video_path: Path to video file
        config: Configuration dictionary
        
    Returns:
        Dictionary with suggestions and beat data
    """
    # Load audio
    try:
        y, sr = librosa.load(video_path, sr=None)
    except Exception as e:
        raise ValueError(f"Could not load audio from {video_path}: {e}")
    
    # Extract features for debug visualization
    audio_times, audio_scores = audio_features(y, sr)
    visual_times, visual_scores = visual_motion(video_path)
    
    # Detect beats for snapping
    beats = detect_beats(y, sr)
    
    # Get suggestion configuration
    suggestions_config = config.get('suggestions', {})
    enabled = suggestions_config.get('enabled', True)
    max_candidates = suggestions_config.get('max_candidates', 5)
    min_spacing = suggestions_config.get('min_spacing_sec', 2.0)
    
    # Generate suggestions if enabled
    suggestions = []
    if enabled:
        suggestions = suggest_anchors(video_path, config, max_candidates, min_spacing)
    
    # Get fused scores for debug visualization
    fusion_config = config.get('fusion', {})
    audio_weight = fusion_config.get('audio_weight', 0.7)
    visual_weight = fusion_config.get('visual_weight', 0.3)
    
    fused_times, fused_scores = fuse_scores(
        audio_times, audio_scores, visual_times, visual_scores,
        audio_weight, visual_weight
    )
    
    # Build result
    result = {
        "suggestions": suggestions,
        "beats": beats.tolist() if len(beats) > 0 else [],
        "debug": {
            "times": fused_times.tolist() if len(fused_times) > 0 else [],
            "fused": fused_scores.tolist() if len(fused_scores) > 0 else [],
            "video_duration": float(fused_times[-1]) if len(fused_times) > 0 else 0.0
        }
    }
    
    return result