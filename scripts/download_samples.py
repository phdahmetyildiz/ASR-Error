#!/usr/bin/env python
"""
Download Sample Audio Files for Testing

This script downloads sample audio files from publicly available datasets
for testing the ASR demo application.

Data Sources:
- LibriSpeech: Open Speech Recognition dataset (clean English speech)
  https://www.openslr.org/12/
  
- Common Voice: Mozilla's multilingual speech dataset
  https://commonvoice.mozilla.org/

Note: For disordered speech testing, you would need to obtain samples from:
- TORGO Database: http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html
  (Requires registration for academic use)
- UASpeech: http://www.isle.illinois.edu/sst/data/UASpeech/
  (Requires license agreement)

Usage:
    python scripts/download_samples.py
"""

import os
import urllib.request
import zipfile
import tempfile
from pathlib import Path


# Sample audio files from LibriSpeech (via OpenSLR mirrors)
SAMPLE_FILES = [
    {
        "name": "librispeech_sample_1.wav",
        "url": "https://www.openslr.org/resources/12/test-clean.tar.gz",
        "description": "LibriSpeech test-clean sample",
        "ground_truth": "Sample transcription - actual transcription varies by file"
    }
]

# For demo purposes, we'll create synthetic test files
# In a real scenario, you would download actual speech samples


def create_sample_directory():
    """Create the samples directory if it doesn't exist."""
    samples_dir = Path(__file__).parent.parent / "data" / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    return samples_dir


def create_synthetic_sample(samples_dir: Path):
    """
    Create a synthetic audio sample for testing.
    
    This creates a simple tone-based audio file for basic functionality testing.
    For actual speech recognition testing, use real speech samples.
    """
    import numpy as np
    import soundfile as sf
    
    print("Creating synthetic test audio...")
    
    # Parameters
    sample_rate = 16000
    duration = 3.0  # seconds
    
    # Create a speech-like signal (multiple harmonics with modulation)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Fundamental frequency with vibrato
    f0 = 150 + 10 * np.sin(2 * np.pi * 5 * t)  # ~150Hz with 5Hz vibrato
    
    # Generate harmonics (simulating vowel formants)
    signal = np.zeros_like(t)
    for harmonic in [1, 2, 3, 4, 5]:
        amplitude = 1.0 / harmonic
        signal += amplitude * np.sin(2 * np.pi * f0 * harmonic * t)
    
    # Apply amplitude envelope (speech-like bursts)
    envelope = np.ones_like(t)
    # Create syllable-like pattern
    for i in range(6):
        start = int(i * len(t) / 6)
        end = int((i + 0.7) * len(t) / 6)
        if end < len(t):
            envelope[start:end] *= 0.8 + 0.2 * np.sin(np.linspace(0, np.pi, end - start))
    
    signal = signal * envelope
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8
    
    # Save
    output_path = samples_dir / "synthetic_test.wav"
    sf.write(str(output_path), signal, sample_rate)
    
    print(f"Created: {output_path}")
    
    # Create accompanying ground truth file
    gt_path = samples_dir / "synthetic_test_ground_truth.txt"
    with open(gt_path, 'w') as f:
        f.write("This is a synthetic test audio file for basic functionality testing.\n")
        f.write("For accurate speech recognition testing, please use real speech samples.\n")
    
    print(f"Created: {gt_path}")
    
    return output_path


def create_readme(samples_dir: Path):
    """Create a README file explaining the sample data."""
    readme_content = """# Sample Audio Data

This directory contains sample audio files for testing the ASR demo application.

## Synthetic Test Files

- `synthetic_test.wav` - A synthetic audio file for basic functionality testing
- `synthetic_test_ground_truth.txt` - Ground truth text for the synthetic file

## Using Real Speech Samples

For accurate speech recognition testing, you should use real speech samples.
Recommended datasets:

### Clean Speech
- **LibriSpeech**: https://www.openslr.org/12/
  - Large-scale corpus of read English speech
  - Free for research and commercial use

- **Common Voice**: https://commonvoice.mozilla.org/
  - Multilingual speech dataset from Mozilla
  - Creative Commons license

### Disordered Speech
- **TORGO Database**: http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html
  - Dysarthric speakers with cerebral palsy or ALS
  - Requires registration for academic use

- **UASpeech**: http://www.isle.illinois.edu/sst/data/UASpeech/
  - Dysarthric speech from speakers with cerebral palsy
  - Requires license agreement

## File Format Requirements

- Format: WAV or MP3
- Sample rate: Will be resampled to 16kHz
- Channels: Will be converted to mono
- Max size: 10MB

## Adding Your Own Samples

1. Place your audio file (`.wav` or `.mp3`) in this directory
2. Create a corresponding `_ground_truth.txt` file with the reference transcription
3. Use the ASR demo to analyze the file
"""
    
    readme_path = samples_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"Created: {readme_path}")


def main():
    """Main function to download/create sample files."""
    print("=" * 60)
    print("ASR Demo - Sample Data Setup")
    print("=" * 60)
    
    # Create samples directory
    samples_dir = create_sample_directory()
    print(f"\nSamples directory: {samples_dir}")
    
    # Try to create synthetic sample
    try:
        create_synthetic_sample(samples_dir)
    except ImportError as e:
        print(f"Warning: Could not create synthetic sample: {e}")
        print("Install numpy and soundfile to create synthetic samples.")
    
    # Create README
    create_readme(samples_dir)
    
    print("\n" + "=" * 60)
    print("Sample data setup complete!")
    print("=" * 60)
    print("\nFor real speech testing, please download samples from:")
    print("- LibriSpeech: https://www.openslr.org/12/")
    print("- Common Voice: https://commonvoice.mozilla.org/")
    print("\nFor disordered speech samples (requires registration):")
    print("- TORGO: http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html")


if __name__ == "__main__":
    main()
