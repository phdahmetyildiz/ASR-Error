"""
ASR Transcription Module for Disordered Speech Demo

This module provides automatic speech recognition functionality using
pre-trained models from Hugging Face. It supports transcription of
preprocessed audio files.

Supported Models:
- facebook/wav2vec2-base-960h (default): Trained on LibriSpeech 960h
- openai/whisper-tiny: Smaller, multilingual model
- openai/whisper-base: Balanced size/accuracy

References:
- Wav2Vec 2.0: https://arxiv.org/abs/2006.11477
- Whisper: https://arxiv.org/abs/2212.04356
- Hugging Face Transformers: https://huggingface.co/docs/transformers
"""

import os
import tempfile
import time
from dataclasses import dataclass
from typing import Optional, Union, BinaryIO, Dict, Any

import numpy as np
import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    WhisperProcessor,
    WhisperForConditionalGeneration,
)

from src.preprocess import (
    preprocess_audio,
    TARGET_SAMPLE_RATE,
)


# Default model for ASR
DEFAULT_MODEL = "facebook/wav2vec2-base-960h"

# Supported models with their types
SUPPORTED_MODELS = {
    "facebook/wav2vec2-base-960h": "wav2vec2",
    "facebook/wav2vec2-large-960h": "wav2vec2",
    "openai/whisper-tiny": "whisper",
    "openai/whisper-base": "whisper",
    "openai/whisper-small": "whisper",
}


@dataclass
class TranscriptionResult:
    """Result of ASR transcription."""
    text: str
    model_name: str
    confidence: Optional[float] = None
    processing_time_seconds: Optional[float] = None
    word_timestamps: Optional[list] = None


class ASRError(Exception):
    """Custom exception for ASR-related errors."""
    pass


class ASRTranscriber:
    """
    Automatic Speech Recognition transcriber.
    
    Supports multiple models from Hugging Face and provides
    a unified interface for transcription.
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the ASR transcriber.
        
        Args:
            model_name: Name of the Hugging Face model to use
            device: Device to run inference on ('cpu', 'cuda', or None for auto)
            cache_dir: Directory to cache downloaded models
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Validate model
        if model_name not in SUPPORTED_MODELS:
            raise ASRError(
                f"Unsupported model: {model_name}. "
                f"Supported models: {list(SUPPORTED_MODELS.keys())}"
            )
        
        self.model_type = SUPPORTED_MODELS[model_name]
        self._model = None
        self._processor = None
    
    def _load_model(self):
        """Lazy load the model and processor."""
        if self._model is not None:
            return
        
        try:
            if self.model_type == "wav2vec2":
                self._load_wav2vec2()
            elif self.model_type == "whisper":
                self._load_whisper()
        except Exception as e:
            raise ASRError(f"Failed to load model {self.model_name}: {str(e)}")
    
    def _load_wav2vec2(self):
        """Load Wav2Vec2 model."""
        self._processor = Wav2Vec2Processor.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )
        self._model = Wav2Vec2ForCTC.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        ).to(self.device)
        self._model.eval()
    
    def _load_whisper(self):
        """Load Whisper model."""
        self._processor = WhisperProcessor.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )
        self._model = WhisperForConditionalGeneration.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        ).to(self.device)
        self._model.eval()
    
    def _transcribe_wav2vec2(self, audio: np.ndarray) -> str:
        """Transcribe using Wav2Vec2 model."""
        # Prepare input
        inputs = self._processor(
            audio,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        )
        
        input_values = inputs.input_values.to(self.device)
        
        # Run inference
        with torch.no_grad():
            logits = self._model(input_values).logits
        
        # Decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self._processor.batch_decode(predicted_ids)[0]
        
        return transcription
    
    def _transcribe_whisper(self, audio: np.ndarray) -> str:
        """Transcribe using Whisper model."""
        # Prepare input
        inputs = self._processor(
            audio,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt"
        )
        
        input_features = inputs.input_features.to(self.device)
        
        # Run inference
        with torch.no_grad():
            generated_ids = self._model.generate(input_features)
        
        # Decode
        transcription = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        
        return transcription
    
    def transcribe(
        self,
        audio_input: Union[str, np.ndarray, BinaryIO],
        return_timestamps: bool = False
    ) -> TranscriptionResult:
        """
        Transcribe audio to text.
        
        Args:
            audio_input: Path to audio file, numpy array, or file-like object
            return_timestamps: Whether to return word-level timestamps (Whisper only)
            
        Returns:
            TranscriptionResult with transcription and metadata
        """
        start_time = time.time()
        
        # Ensure model is loaded
        self._load_model()
        
        try:
            # Handle different input types
            if isinstance(audio_input, np.ndarray):
                # Already preprocessed numpy array
                audio = audio_input.astype(np.float32)
            elif isinstance(audio_input, str):
                # File path - preprocess
                audio, metadata = preprocess_audio(audio_input)
                audio = audio.astype(np.float32)
            else:
                # File-like object (e.g., Streamlit upload)
                audio, metadata = preprocess_audio(audio_input)
                audio = audio.astype(np.float32)
            
            # Perform transcription based on model type
            if self.model_type == "wav2vec2":
                text = self._transcribe_wav2vec2(audio)
            elif self.model_type == "whisper":
                text = self._transcribe_whisper(audio)
            else:
                raise ASRError(f"Unknown model type: {self.model_type}")
            
            processing_time = time.time() - start_time
            
            return TranscriptionResult(
                text=text.strip(),
                model_name=self.model_name,
                processing_time_seconds=processing_time,
                word_timestamps=None  # TODO: Implement timestamp extraction
            )
            
        except ASRError:
            raise
        except Exception as e:
            raise ASRError(f"Transcription failed: {str(e)}")
    
    def transcribe_with_preprocessing(
        self,
        audio_input: Union[str, BinaryIO]
    ) -> TranscriptionResult:
        """
        Transcribe audio with automatic preprocessing.
        
        This is the recommended method for raw audio files that
        may need resampling, normalization, etc.
        
        Args:
            audio_input: Path to audio file or file-like object
            
        Returns:
            TranscriptionResult with transcription and metadata
        """
        # Preprocess audio
        audio_array, metadata = preprocess_audio(audio_input)
        
        # Transcribe
        return self.transcribe(audio_array)
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None


def transcribe_audio(
    audio_input: Union[str, np.ndarray, BinaryIO],
    model_name: str = DEFAULT_MODEL
) -> str:
    """
    Simple function to transcribe audio to text.
    
    This is a convenience function that creates a transcriber,
    transcribes the audio, and returns just the text.
    
    Args:
        audio_input: Path to audio file, numpy array, or file-like object
        model_name: Name of the model to use
        
    Returns:
        Transcribed text
    """
    transcriber = ASRTranscriber(model_name=model_name)
    result = transcriber.transcribe(audio_input)
    return result.text


def get_available_models() -> list:
    """Return list of supported model names."""
    return list(SUPPORTED_MODELS.keys())


def download_model(model_name: str, cache_dir: Optional[str] = None) -> None:
    """
    Pre-download a model for offline use.
    
    Args:
        model_name: Name of the model to download
        cache_dir: Directory to save the model
    """
    if model_name not in SUPPORTED_MODELS:
        raise ASRError(f"Unsupported model: {model_name}")
    
    print(f"Downloading model: {model_name}")
    
    model_type = SUPPORTED_MODELS[model_name]
    
    if model_type == "wav2vec2":
        Wav2Vec2Processor.from_pretrained(model_name, cache_dir=cache_dir)
        Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir=cache_dir)
    elif model_type == "whisper":
        WhisperProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        WhisperForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir)
    
    print(f"Model {model_name} downloaded successfully")
