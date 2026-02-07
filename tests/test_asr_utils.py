"""
Tests for ASR Transcription Module

Note: These tests require downloading the ASR model on first run,
which may take several minutes depending on internet connection.
The model is cached for subsequent runs.
"""

import io
import os
import tempfile

import numpy as np
import pytest
import soundfile as sf

from src.asr_utils import (
    ASRError,
    ASRTranscriber,
    DEFAULT_MODEL,
    SUPPORTED_MODELS,
    TranscriptionResult,
    get_available_models,
    transcribe_audio,
)
from src.preprocess import TARGET_SAMPLE_RATE


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_speech_audio():
    """
    Generate synthetic audio that resembles speech patterns.
    
    This creates a simple audio signal with varying frequencies
    to simulate speech-like patterns.
    """
    sr = TARGET_SAMPLE_RATE
    duration = 2.0  # 2 seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Create speech-like signal with multiple frequency components
    # and amplitude modulation
    envelope = np.exp(-((t - 1.0) ** 2) / 0.5)  # Gaussian envelope
    frequencies = [200, 400, 800, 1200]  # Formant-like frequencies
    
    audio = np.zeros_like(t)
    for f in frequencies:
        audio += 0.25 * np.sin(2 * np.pi * f * t)
    
    audio = audio * envelope * 0.8  # Apply envelope
    
    return audio, sr


@pytest.fixture
def temp_speech_wav(sample_speech_audio):
    """Create a temporary WAV file with speech-like audio."""
    audio, sr = sample_speech_audio
    temp_path = tempfile.mktemp(suffix=".wav")
    sf.write(temp_path, audio, sr)
    yield temp_path
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def transcriber():
    """Create a transcriber instance with default model."""
    return ASRTranscriber(model_name=DEFAULT_MODEL)


# ============================================================================
# Tests for ASRTranscriber initialization
# ============================================================================

class TestASRTranscriberInit:
    def test_default_initialization(self):
        """Test transcriber initializes with default settings."""
        transcriber = ASRTranscriber()
        
        assert transcriber.model_name == DEFAULT_MODEL
        assert transcriber.device in ["cpu", "cuda"]
        assert not transcriber.is_loaded
    
    def test_custom_model(self):
        """Test transcriber accepts supported custom models."""
        for model_name in SUPPORTED_MODELS.keys():
            transcriber = ASRTranscriber(model_name=model_name)
            assert transcriber.model_name == model_name
    
    def test_unsupported_model_raises_error(self):
        """Test that unsupported model raises ASRError."""
        with pytest.raises(ASRError) as exc_info:
            ASRTranscriber(model_name="nonexistent/model")
        assert "unsupported model" in str(exc_info.value).lower()
    
    def test_explicit_cpu_device(self):
        """Test transcriber accepts explicit CPU device."""
        transcriber = ASRTranscriber(device="cpu")
        assert transcriber.device == "cpu"


# ============================================================================
# Tests for transcription
# ============================================================================

class TestTranscription:
    """
    Tests for actual transcription functionality.
    
    Note: These tests may be slow on first run due to model download.
    """
    
    @pytest.mark.slow
    def test_transcribe_numpy_array(self, transcriber, sample_speech_audio):
        """Test transcription from numpy array."""
        audio, sr = sample_speech_audio
        
        result = transcriber.transcribe(audio)
        
        assert isinstance(result, TranscriptionResult)
        assert isinstance(result.text, str)
        assert result.model_name == DEFAULT_MODEL
        assert result.processing_time_seconds > 0
    
    @pytest.mark.slow
    def test_transcribe_wav_file(self, transcriber, temp_speech_wav):
        """Test transcription from WAV file path."""
        result = transcriber.transcribe(temp_speech_wav)
        
        assert isinstance(result, TranscriptionResult)
        assert isinstance(result.text, str)
    
    @pytest.mark.slow
    def test_transcribe_file_object(self, transcriber, sample_speech_audio):
        """Test transcription from file-like object."""
        audio, sr = sample_speech_audio
        
        buffer = io.BytesIO()
        sf.write(buffer, audio, sr, format='WAV')
        buffer.seek(0)
        
        result = transcriber.transcribe(buffer)
        
        assert isinstance(result, TranscriptionResult)
        assert isinstance(result.text, str)
    
    @pytest.mark.slow
    def test_transcription_result_fields(self, transcriber, sample_speech_audio):
        """Test that TranscriptionResult has all expected fields."""
        audio, _ = sample_speech_audio
        
        result = transcriber.transcribe(audio)
        
        assert hasattr(result, 'text')
        assert hasattr(result, 'model_name')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'processing_time_seconds')
        assert hasattr(result, 'word_timestamps')
    
    @pytest.mark.slow
    def test_transcribe_with_preprocessing(self, transcriber, temp_speech_wav):
        """Test transcription with automatic preprocessing."""
        result = transcriber.transcribe_with_preprocessing(temp_speech_wav)
        
        assert isinstance(result, TranscriptionResult)
        assert isinstance(result.text, str)


# ============================================================================
# Tests for simple transcribe function
# ============================================================================

class TestTranscribeFunction:
    @pytest.mark.slow
    def test_simple_transcribe(self, temp_speech_wav):
        """Test simple transcribe_audio function."""
        text = transcribe_audio(temp_speech_wav)
        
        assert isinstance(text, str)
    
    @pytest.mark.slow
    def test_simple_transcribe_with_model(self, temp_speech_wav):
        """Test simple transcribe_audio with specific model."""
        text = transcribe_audio(temp_speech_wav, model_name=DEFAULT_MODEL)
        
        assert isinstance(text, str)


# ============================================================================
# Tests for utility functions
# ============================================================================

class TestUtilityFunctions:
    def test_get_available_models(self):
        """Test getting list of available models."""
        models = get_available_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
        assert DEFAULT_MODEL in models
    
    def test_supported_models_includes_wav2vec2(self):
        """Test that Wav2Vec2 models are supported."""
        models = get_available_models()
        wav2vec2_models = [m for m in models if "wav2vec2" in m.lower()]
        assert len(wav2vec2_models) > 0
    
    def test_supported_models_includes_whisper(self):
        """Test that Whisper models are supported."""
        models = get_available_models()
        whisper_models = [m for m in models if "whisper" in m.lower()]
        assert len(whisper_models) > 0


# ============================================================================
# Tests for error handling
# ============================================================================

class TestErrorHandling:
    def test_transcribe_nonexistent_file(self, transcriber):
        """Test error handling for nonexistent file."""
        with pytest.raises(ASRError):
            transcriber.transcribe("/nonexistent/path/audio.wav")
    
    def test_model_lazy_loading(self):
        """Test that model is not loaded until transcription."""
        transcriber = ASRTranscriber()
        
        # Model should not be loaded yet
        assert not transcriber.is_loaded
        
        # Note: We don't trigger transcription here to avoid
        # slow test. The actual lazy loading is tested in
        # transcription tests.


# ============================================================================
# Tests for model state
# ============================================================================

class TestModelState:
    @pytest.mark.slow
    def test_model_loaded_after_transcription(self, transcriber, sample_speech_audio):
        """Test that model is marked as loaded after transcription."""
        audio, _ = sample_speech_audio
        
        assert not transcriber.is_loaded
        
        transcriber.transcribe(audio)
        
        assert transcriber.is_loaded
    
    @pytest.mark.slow
    def test_model_reused_across_transcriptions(self, transcriber, sample_speech_audio):
        """Test that model is reused for multiple transcriptions."""
        audio, _ = sample_speech_audio
        
        # First transcription (loads model)
        result1 = transcriber.transcribe(audio)
        
        # Second transcription (reuses model)
        result2 = transcriber.transcribe(audio)
        
        # Both should return valid results
        assert isinstance(result1.text, str)
        assert isinstance(result2.text, str)
        
        # Model should still be loaded
        assert transcriber.is_loaded
