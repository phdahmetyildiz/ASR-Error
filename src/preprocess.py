"""
Audio Preprocessing Module for ASR Demo

This module handles audio file loading, preprocessing, and feature extraction
for the ASR pipeline. It supports WAV and MP3 formats and prepares audio
for transcription by resampling, normalizing, and converting to mono.

References:
- LibriSpeech Dataset: https://www.openslr.org/12/
- Common Voice: https://commonvoice.mozilla.org/
- TORGO Database (dysarthric speech): http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html
"""

import io
import os
import tempfile
from dataclasses import dataclass
from typing import Optional, Tuple, Union, BinaryIO

import librosa
import numpy as np
import soundfile as sf


# Constants
TARGET_SAMPLE_RATE = 16000  # Standard for ASR models
MAX_FILE_SIZE_MB = 10
SILENCE_THRESHOLD_DB = -40  # dB threshold for silence detection
MIN_SILENCE_DURATION = 0.1  # Minimum silence duration in seconds


@dataclass
class AudioMetadata:
    """Metadata extracted from processed audio."""
    duration_seconds: float
    sample_rate: int
    num_samples: int
    is_mono: bool
    original_sample_rate: Optional[int] = None
    file_size_mb: Optional[float] = None


@dataclass
class PauseInfo:
    """Information about detected pauses in audio."""
    num_pauses: int
    total_pause_duration: float
    average_pause_duration: float
    pause_intervals: list  # List of (start, end) tuples in seconds


class AudioPreprocessingError(Exception):
    """Custom exception for audio preprocessing errors."""
    pass


def validate_audio_file(file_path: str) -> None:
    """
    Validate that the audio file exists and is within size limits.
    
    Args:
        file_path: Path to the audio file
        
    Raises:
        AudioPreprocessingError: If file is invalid or too large
    """
    if not os.path.exists(file_path):
        raise AudioPreprocessingError(f"Audio file not found: {file_path}")
    
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise AudioPreprocessingError(
            f"File size ({file_size_mb:.2f}MB) exceeds maximum allowed ({MAX_FILE_SIZE_MB}MB)"
        )


def load_audio(
    file_input: Union[str, BinaryIO],
    target_sr: int = TARGET_SAMPLE_RATE
) -> Tuple[np.ndarray, int, Optional[int]]:
    """
    Load audio from file path or file-like object.
    
    Args:
        file_input: Path to audio file or file-like object (e.g., Streamlit UploadedFile)
        target_sr: Target sample rate for resampling
        
    Returns:
        Tuple of (audio_array, target_sample_rate, original_sample_rate)
        
    Raises:
        AudioPreprocessingError: If audio cannot be loaded
    """
    try:
        if isinstance(file_input, str):
            # File path provided
            validate_audio_file(file_input)
            audio, original_sr = librosa.load(file_input, sr=None, mono=False)
        else:
            # File-like object (e.g., from Streamlit upload)
            file_input.seek(0)
            audio, original_sr = sf.read(file_input)
            audio = audio.T  # Transpose to match librosa format (channels, samples)
            if audio.ndim == 1:
                audio = audio.reshape(1, -1)
        
        return audio, target_sr, original_sr
        
    except Exception as e:
        raise AudioPreprocessingError(f"Failed to load audio: {str(e)}")


def convert_to_mono(audio: np.ndarray) -> np.ndarray:
    """
    Convert stereo audio to mono by averaging channels.
    
    Args:
        audio: Audio array, shape (channels, samples) or (samples,)
        
    Returns:
        Mono audio array, shape (samples,)
    """
    if audio.ndim == 1:
        return audio
    elif audio.ndim == 2:
        if audio.shape[0] <= 2:  # (channels, samples) format
            return np.mean(audio, axis=0)
        else:  # Might be (samples, channels) format
            return np.mean(audio, axis=1)
    else:
        raise AudioPreprocessingError(f"Unexpected audio shape: {audio.shape}")


def resample_audio(
    audio: np.ndarray,
    original_sr: int,
    target_sr: int = TARGET_SAMPLE_RATE
) -> np.ndarray:
    """
    Resample audio to target sample rate.
    
    Args:
        audio: Audio array
        original_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        Resampled audio array
    """
    if original_sr == target_sr:
        return audio
    
    return librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio to have maximum amplitude of 1.0.
    
    Args:
        audio: Audio array
        
    Returns:
        Normalized audio array
    """
    return librosa.util.normalize(audio)


def detect_pauses(
    audio: np.ndarray,
    sr: int = TARGET_SAMPLE_RATE,
    threshold_db: float = SILENCE_THRESHOLD_DB,
    min_silence_duration: float = MIN_SILENCE_DURATION
) -> PauseInfo:
    """
    Detect pauses (silence) in audio using energy-based detection.
    
    This is useful for analyzing speech patterns in disordered speech,
    where abnormal pause patterns may indicate specific conditions.
    
    Args:
        audio: Audio array
        sr: Sample rate
        threshold_db: Silence threshold in dB (default: -40dB)
        min_silence_duration: Minimum silence duration in seconds
        
    Returns:
        PauseInfo object with pause statistics
    """
    # Calculate RMS energy in small frames
    frame_length = int(0.025 * sr)  # 25ms frames
    hop_length = int(0.010 * sr)    # 10ms hop
    
    # Compute RMS energy
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Convert to dB
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    
    # Find silent frames
    is_silent = rms_db < threshold_db
    
    # Convert to time
    frame_times = librosa.frames_to_time(np.arange(len(rms_db)), sr=sr, hop_length=hop_length)
    
    # Find pause intervals
    pause_intervals = []
    in_pause = False
    pause_start = 0
    
    for i, silent in enumerate(is_silent):
        if silent and not in_pause:
            in_pause = True
            pause_start = frame_times[i]
        elif not silent and in_pause:
            in_pause = False
            pause_end = frame_times[i]
            duration = pause_end - pause_start
            if duration >= min_silence_duration:
                pause_intervals.append((pause_start, pause_end))
    
    # Handle case where audio ends in silence
    if in_pause:
        pause_end = frame_times[-1]
        duration = pause_end - pause_start
        if duration >= min_silence_duration:
            pause_intervals.append((pause_start, pause_end))
    
    # Calculate statistics
    num_pauses = len(pause_intervals)
    if num_pauses > 0:
        durations = [end - start for start, end in pause_intervals]
        total_pause_duration = sum(durations)
        average_pause_duration = total_pause_duration / num_pauses
    else:
        total_pause_duration = 0.0
        average_pause_duration = 0.0
    
    return PauseInfo(
        num_pauses=num_pauses,
        total_pause_duration=total_pause_duration,
        average_pause_duration=average_pause_duration,
        pause_intervals=pause_intervals
    )


def preprocess_audio(
    file_input: Union[str, BinaryIO],
    target_sr: int = TARGET_SAMPLE_RATE,
    save_processed: bool = False,
    output_path: Optional[str] = None
) -> Tuple[np.ndarray, AudioMetadata]:
    """
    Complete audio preprocessing pipeline.
    
    Performs the following steps:
    1. Load audio file
    2. Convert to mono
    3. Resample to target sample rate (16kHz)
    4. Normalize volume
    
    Args:
        file_input: Path to audio file or file-like object
        target_sr: Target sample rate (default: 16000)
        save_processed: Whether to save the processed audio
        output_path: Path for saving processed audio (optional)
        
    Returns:
        Tuple of (processed_audio_array, AudioMetadata)
    """
    # Load audio
    audio, target_sr, original_sr = load_audio(file_input, target_sr)
    
    # Get original channel info
    is_originally_mono = audio.ndim == 1 or (audio.ndim == 2 and audio.shape[0] == 1)
    
    # Convert to mono
    audio = convert_to_mono(audio)
    
    # Resample
    audio = resample_audio(audio, original_sr, target_sr)
    
    # Normalize
    audio = normalize_audio(audio)
    
    # Calculate file size if path provided
    file_size_mb = None
    if isinstance(file_input, str):
        file_size_mb = os.path.getsize(file_input) / (1024 * 1024)
    
    # Create metadata
    metadata = AudioMetadata(
        duration_seconds=len(audio) / target_sr,
        sample_rate=target_sr,
        num_samples=len(audio),
        is_mono=True,  # Always mono after processing
        original_sample_rate=original_sr,
        file_size_mb=file_size_mb
    )
    
    # Save if requested
    if save_processed:
        if output_path is None:
            output_path = tempfile.mktemp(suffix=".wav", prefix="processed_")
        sf.write(output_path, audio, target_sr)
    
    return audio, metadata


def save_audio_to_tempfile(audio: np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> str:
    """
    Save audio array to a temporary WAV file.
    
    Args:
        audio: Audio array
        sr: Sample rate
        
    Returns:
        Path to temporary file
    """
    temp_path = tempfile.mktemp(suffix=".wav", prefix="temp_audio_")
    sf.write(temp_path, audio, sr)
    return temp_path


def get_audio_duration(file_input: Union[str, BinaryIO]) -> float:
    """
    Get duration of audio file without full preprocessing.
    
    Args:
        file_input: Path to audio file or file-like object
        
    Returns:
        Duration in seconds
    """
    try:
        if isinstance(file_input, str):
            return librosa.get_duration(path=file_input)
        else:
            file_input.seek(0)
            audio, sr = sf.read(file_input)
            return len(audio) / sr
    except Exception as e:
        raise AudioPreprocessingError(f"Failed to get audio duration: {str(e)}")
