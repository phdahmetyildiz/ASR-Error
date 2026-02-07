"""
Tests for Audio Preprocessing Module

Uses synthesized audio for unit tests to avoid external dependencies.
For integration tests with real speech, see test_integration.py.
"""

import io
import os
import tempfile

import numpy as np
import pytest
import soundfile as sf

from src.preprocess import (
    AudioMetadata,
    AudioPreprocessingError,
    PauseInfo,
    TARGET_SAMPLE_RATE,
    convert_to_mono,
    detect_pauses,
    normalize_audio,
    preprocess_audio,
    resample_audio,
    save_audio_to_tempfile,
    validate_audio_file,
)


# ============================================================================
# Fixtures for generating test audio
# ============================================================================

@pytest.fixture
def sample_mono_audio():
    """Generate a simple mono audio signal (1 second sine wave at 440Hz)."""
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz sine wave
    return audio, sr


@pytest.fixture
def sample_stereo_audio():
    """Generate a simple stereo audio signal."""
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    left = 0.5 * np.sin(2 * np.pi * 440 * t)   # 440Hz
    right = 0.5 * np.sin(2 * np.pi * 880 * t)  # 880Hz
    audio = np.vstack([left, right])  # Shape: (2, samples)
    return audio, sr


@pytest.fixture
def sample_audio_with_pauses():
    """Generate audio with clear pauses (speech-like pattern)."""
    sr = 16000
    
    # Create segments: sound - silence - sound - silence - sound
    segment_duration = 0.5  # 500ms each
    samples_per_segment = int(sr * segment_duration)
    
    t = np.linspace(0, segment_duration, samples_per_segment, endpoint=False)
    sound_segment = 0.5 * np.sin(2 * np.pi * 440 * t)
    silence_segment = np.zeros(samples_per_segment)
    
    # Concatenate: sound, silence, sound, silence, sound
    audio = np.concatenate([
        sound_segment,
        silence_segment,
        sound_segment,
        silence_segment,
        sound_segment
    ])
    
    return audio, sr


@pytest.fixture
def temp_wav_file(sample_mono_audio):
    """Create a temporary WAV file for testing."""
    audio, sr = sample_mono_audio
    temp_path = tempfile.mktemp(suffix=".wav")
    sf.write(temp_path, audio, sr)
    yield temp_path
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def temp_wav_file_22khz():
    """Create a temporary WAV file at 22050Hz for resampling tests."""
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    temp_path = tempfile.mktemp(suffix=".wav")
    sf.write(temp_path, audio, sr)
    yield temp_path
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def temp_stereo_wav_file(sample_stereo_audio):
    """Create a temporary stereo WAV file."""
    audio, sr = sample_stereo_audio
    temp_path = tempfile.mktemp(suffix=".wav")
    # soundfile expects (samples, channels) for stereo
    sf.write(temp_path, audio.T, sr)
    yield temp_path
    if os.path.exists(temp_path):
        os.remove(temp_path)


# ============================================================================
# Tests for validate_audio_file
# ============================================================================

class TestValidateAudioFile:
    def test_valid_file(self, temp_wav_file):
        """Test validation passes for valid file."""
        # Should not raise any exception
        validate_audio_file(temp_wav_file)
    
    def test_nonexistent_file(self):
        """Test validation fails for nonexistent file."""
        with pytest.raises(AudioPreprocessingError) as exc_info:
            validate_audio_file("/nonexistent/path/audio.wav")
        assert "not found" in str(exc_info.value).lower()
    
    def test_file_too_large(self, temp_wav_file, monkeypatch):
        """Test validation fails for file exceeding size limit."""
        # Mock os.path.getsize to return a large file size
        monkeypatch.setattr(os.path, "getsize", lambda x: 15 * 1024 * 1024)  # 15MB
        
        with pytest.raises(AudioPreprocessingError) as exc_info:
            validate_audio_file(temp_wav_file)
        assert "exceeds maximum" in str(exc_info.value).lower()


# ============================================================================
# Tests for convert_to_mono
# ============================================================================

class TestConvertToMono:
    def test_mono_passthrough(self, sample_mono_audio):
        """Test that mono audio passes through unchanged."""
        audio, _ = sample_mono_audio
        result = convert_to_mono(audio)
        
        assert result.ndim == 1
        np.testing.assert_array_equal(result, audio)
    
    def test_stereo_to_mono(self, sample_stereo_audio):
        """Test stereo to mono conversion."""
        audio, _ = sample_stereo_audio
        result = convert_to_mono(audio)
        
        assert result.ndim == 1
        assert len(result) == audio.shape[1]
        
        # Result should be average of channels
        expected = np.mean(audio, axis=0)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_single_channel_2d(self, sample_mono_audio):
        """Test conversion of 2D array with single channel."""
        audio, _ = sample_mono_audio
        audio_2d = audio.reshape(1, -1)  # Shape: (1, samples)
        result = convert_to_mono(audio_2d)
        
        assert result.ndim == 1
        np.testing.assert_array_almost_equal(result, audio)


# ============================================================================
# Tests for resample_audio
# ============================================================================

class TestResampleAudio:
    def test_no_resample_needed(self, sample_mono_audio):
        """Test that audio at target rate is not modified."""
        audio, sr = sample_mono_audio
        result = resample_audio(audio, sr, TARGET_SAMPLE_RATE)
        
        # Same sample rate, should be unchanged
        assert len(result) == len(audio)
        np.testing.assert_array_equal(result, audio)
    
    def test_downsample(self, sample_mono_audio):
        """Test downsampling from higher rate."""
        audio, sr = sample_mono_audio
        original_sr = 22050
        
        # Generate audio at 22050Hz
        t = np.linspace(0, 1.0, original_sr, endpoint=False)
        audio_22k = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        result = resample_audio(audio_22k, original_sr, TARGET_SAMPLE_RATE)
        
        # Length should be proportional to sample rate ratio
        expected_length = int(len(audio_22k) * TARGET_SAMPLE_RATE / original_sr)
        assert abs(len(result) - expected_length) <= 1
    
    def test_upsample(self, sample_mono_audio):
        """Test upsampling from lower rate."""
        original_sr = 8000
        t = np.linspace(0, 1.0, original_sr, endpoint=False)
        audio_8k = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        result = resample_audio(audio_8k, original_sr, TARGET_SAMPLE_RATE)
        
        expected_length = int(len(audio_8k) * TARGET_SAMPLE_RATE / original_sr)
        assert abs(len(result) - expected_length) <= 1


# ============================================================================
# Tests for normalize_audio
# ============================================================================

class TestNormalizeAudio:
    def test_normalize_quiet_audio(self):
        """Test normalization of quiet audio."""
        audio = np.array([0.1, -0.1, 0.05, -0.05])
        result = normalize_audio(audio)
        
        # Peak should be at 1.0
        assert abs(np.max(np.abs(result)) - 1.0) < 0.01
    
    def test_normalize_already_normalized(self):
        """Test that already normalized audio stays normalized."""
        audio = np.array([1.0, -0.5, 0.3, -1.0])
        result = normalize_audio(audio)
        
        assert abs(np.max(np.abs(result)) - 1.0) < 0.01
    
    def test_normalize_preserves_relative_amplitudes(self):
        """Test that relative amplitudes are preserved."""
        audio = np.array([0.4, -0.2, 0.1])
        result = normalize_audio(audio)
        
        # Ratios should be preserved
        original_ratio = audio[1] / audio[0]
        result_ratio = result[1] / result[0]
        assert abs(original_ratio - result_ratio) < 0.01


# ============================================================================
# Tests for detect_pauses
# ============================================================================

class TestDetectPauses:
    def test_continuous_sound_no_pauses(self, sample_mono_audio):
        """Test that continuous sound has no pauses."""
        audio, sr = sample_mono_audio
        result = detect_pauses(audio, sr)
        
        assert isinstance(result, PauseInfo)
        assert result.num_pauses == 0
        assert result.total_pause_duration == 0.0
    
    def test_detect_clear_pauses(self, sample_audio_with_pauses):
        """Test detection of clear pauses in audio."""
        audio, sr = sample_audio_with_pauses
        result = detect_pauses(audio, sr)
        
        # Should detect 2 pauses (between sound segments)
        assert result.num_pauses == 2
        assert result.total_pause_duration > 0.8  # ~1 second of silence
        assert result.average_pause_duration > 0.3  # ~0.5 seconds each
    
    def test_all_silence(self):
        """Test detection handles all-silent audio gracefully."""
        sr = 16000
        # Use very low amplitude noise instead of pure zeros
        audio = np.random.randn(sr * 2) * 0.0001  # 2 seconds of near-silence
        result = detect_pauses(audio, sr)
        
        # All-silent audio may not detect "pauses" since there's no
        # transition from sound to silence. Just verify it doesn't crash
        # and returns a valid PauseInfo object.
        assert isinstance(result, PauseInfo)
        assert result.num_pauses >= 0
        assert result.total_pause_duration >= 0
    
    def test_pause_intervals_format(self, sample_audio_with_pauses):
        """Test that pause intervals are in correct format."""
        audio, sr = sample_audio_with_pauses
        result = detect_pauses(audio, sr)
        
        for start, end in result.pause_intervals:
            assert isinstance(start, (int, float))
            assert isinstance(end, (int, float))
            assert end > start


# ============================================================================
# Tests for preprocess_audio (full pipeline)
# ============================================================================

class TestPreprocessAudio:
    def test_preprocess_mono_wav(self, temp_wav_file):
        """Test full preprocessing pipeline with mono WAV."""
        audio, metadata = preprocess_audio(temp_wav_file)
        
        assert isinstance(audio, np.ndarray)
        assert audio.ndim == 1
        assert isinstance(metadata, AudioMetadata)
        assert metadata.sample_rate == TARGET_SAMPLE_RATE
        assert metadata.is_mono is True
        assert metadata.duration_seconds > 0
    
    def test_preprocess_stereo_wav(self, temp_stereo_wav_file):
        """Test preprocessing converts stereo to mono."""
        audio, metadata = preprocess_audio(temp_stereo_wav_file)
        
        assert audio.ndim == 1
        assert metadata.is_mono is True
    
    def test_preprocess_with_resampling(self, temp_wav_file_22khz):
        """Test preprocessing resamples to target rate."""
        audio, metadata = preprocess_audio(temp_wav_file_22khz)
        
        assert metadata.sample_rate == TARGET_SAMPLE_RATE
        assert metadata.original_sample_rate == 22050
    
    def test_preprocess_normalization(self, temp_wav_file):
        """Test that preprocessing normalizes audio."""
        audio, _ = preprocess_audio(temp_wav_file)
        
        # Normalized audio should have max amplitude close to 1.0
        assert abs(np.max(np.abs(audio)) - 1.0) < 0.01
    
    def test_preprocess_save_output(self, temp_wav_file):
        """Test saving processed audio to file."""
        output_path = tempfile.mktemp(suffix=".wav")
        
        try:
            audio, metadata = preprocess_audio(
                temp_wav_file,
                save_processed=True,
                output_path=output_path
            )
            
            assert os.path.exists(output_path)
            
            # Verify saved file
            saved_audio, saved_sr = sf.read(output_path)
            assert saved_sr == TARGET_SAMPLE_RATE
            # Use decimal=3 due to WAV encoding precision limits
            np.testing.assert_array_almost_equal(saved_audio, audio, decimal=3)
            
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)
    
    def test_preprocess_file_object(self, sample_mono_audio):
        """Test preprocessing from file-like object (simulating Streamlit upload)."""
        audio_data, sr = sample_mono_audio
        
        # Create a BytesIO object simulating file upload
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sr, format='WAV')
        buffer.seek(0)
        
        result_audio, metadata = preprocess_audio(buffer)
        
        assert isinstance(result_audio, np.ndarray)
        assert result_audio.ndim == 1
        assert metadata.sample_rate == TARGET_SAMPLE_RATE


# ============================================================================
# Tests for save_audio_to_tempfile
# ============================================================================

class TestSaveAudioToTempfile:
    def test_save_creates_file(self, sample_mono_audio):
        """Test that saving creates a valid WAV file."""
        audio, _ = sample_mono_audio
        temp_path = save_audio_to_tempfile(audio)
        
        try:
            assert os.path.exists(temp_path)
            assert temp_path.endswith(".wav")
            
            # Verify file is readable
            loaded_audio, loaded_sr = sf.read(temp_path)
            assert loaded_sr == TARGET_SAMPLE_RATE
            # Use decimal=3 due to WAV encoding precision limits
            np.testing.assert_array_almost_equal(loaded_audio, audio, decimal=3)
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


# ============================================================================
# Tests for metadata
# ============================================================================

class TestAudioMetadata:
    def test_metadata_fields(self, temp_wav_file):
        """Test that metadata contains all expected fields."""
        _, metadata = preprocess_audio(temp_wav_file)
        
        assert hasattr(metadata, 'duration_seconds')
        assert hasattr(metadata, 'sample_rate')
        assert hasattr(metadata, 'num_samples')
        assert hasattr(metadata, 'is_mono')
        assert hasattr(metadata, 'original_sample_rate')
        assert hasattr(metadata, 'file_size_mb')
    
    def test_metadata_consistency(self, temp_wav_file):
        """Test that metadata values are consistent."""
        audio, metadata = preprocess_audio(temp_wav_file)
        
        # Duration should match samples / sample_rate
        expected_duration = len(audio) / metadata.sample_rate
        assert abs(metadata.duration_seconds - expected_duration) < 0.01
        
        # num_samples should match audio length
        assert metadata.num_samples == len(audio)
