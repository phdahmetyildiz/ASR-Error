# 🎙️ ASR Demo for Disordered Speech with Error Analysis

A web-based application that demonstrates Automatic Speech Recognition (ASR) capabilities tailored for disordered speech analysis. The tool allows users to upload audio files, transcribe them using pre-trained models, compare against ground truth text, and perform comprehensive error analysis.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

- **🎵 Audio Upload & Processing**: Support for WAV and MP3 files with automatic preprocessing (resampling, normalization, mono conversion)
- **📝 ASR Transcription**: Powered by Hugging Face Transformers (Wav2Vec2, Whisper)
- **📊 Error Analysis**: 
  - Word Error Rate (WER) and Character Error Rate (CER)
  - Detailed error breakdown (substitutions, insertions, deletions)
  - Color-coded diff visualization
- **🔍 Speech Metrics**:
  - Speech rate (words per minute)
  - Pause detection and analysis
- **🧬 Phonetic Pattern Detection**: Rule-based identification of error patterns common in disordered speech
- **📥 Export**: Download results as JSON for further analysis
- **🐳 Docker Support**: Containerized deployment with pre-downloaded model

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/asr-disordered-speech.git
cd asr-disordered-speech

# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t asr-disordered-speech .
docker run -p 8501:8501 asr-disordered-speech
```

Open your browser at `http://localhost:8501`

### Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/asr-disordered-speech.git
cd asr-disordered-speech

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements_python.txt

# Run the application
streamlit run app.py
```

## 📖 Usage

1. **Upload Audio**: Click "Choose an audio file" to upload a WAV or MP3 file (max 10MB)
2. **Enter Ground Truth**: Type or paste the correct transcription in the text area
3. **Analyze**: Click the "🚀 Analyze Speech" button
4. **Review Results**:
   - View transcription comparison
   - Check error metrics and visualizations
   - Examine phonetic patterns (if detected)
5. **Export**: Download results as JSON for further analysis

## 📁 Project Structure

```
asr-disordered-speech/
├── app.py                    # Main Streamlit application
├── src/
│   ├── __init__.py
│   ├── preprocess.py         # Audio preprocessing module
│   ├── asr_utils.py          # ASR transcription module
│   └── analysis_utils.py     # Error analysis module
├── tests/
│   ├── test_preprocess.py
│   ├── test_asr_utils.py
│   └── test_analysis_utils.py
├── scripts/
│   └── download_samples.py   # Sample data download script
├── data/
│   └── samples/              # Sample audio files
├── Dockerfile
├── docker-compose.yml
├── requirements_python.txt
└── README.md
```

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run specific test module
pytest tests/test_preprocess.py -v

# Run with coverage
pytest --cov=src --cov-report=html
```

## 🔧 Configuration

### Supported ASR Models

| Model | Type | Description |
|-------|------|-------------|
| `facebook/wav2vec2-base-960h` | Wav2Vec2 | Default model, trained on LibriSpeech 960h |
| `facebook/wav2vec2-large-960h` | Wav2Vec2 | Larger, more accurate version |
| `openai/whisper-tiny` | Whisper | Small, fast, multilingual |
| `openai/whisper-base` | Whisper | Balanced size/accuracy |
| `openai/whisper-small` | Whisper | Better accuracy, larger |

### Audio Requirements

- **Formats**: WAV, MP3
- **Sample Rate**: Automatically resampled to 16kHz
- **Channels**: Automatically converted to mono
- **Max Size**: 10MB

## 📊 Understanding the Metrics

### Word Error Rate (WER)

WER measures the edit distance between the reference and hypothesis at the word level:

```
WER = (S + D + I) / N
```

Where:
- S = Substitutions (wrong words)
- D = Deletions (missing words)
- I = Insertions (extra words)
- N = Total words in reference

### Phonetic Patterns

The tool identifies common error patterns that may indicate speech disorders:

- **Fricative Substitution**: Errors in sounds like 'f', 'v', 's', 'z' (common in Dysarthria)
- **Final Consonant Deletion**: Missing consonants at word endings
- **Stop Consonant Errors**: Issues with 'p', 'b', 't', 'd', 'k', 'g'
- **Word Length Effect**: Increased errors in longer words (may indicate apraxia)

> ⚠️ **Disclaimer**: These patterns are identified through simple rule-based analysis and should not be used for clinical diagnosis. Always consult qualified speech-language pathologists for assessment.

## 📚 Sample Datasets

For testing with real speech, consider these datasets:

### Clean Speech
- [LibriSpeech](https://www.openslr.org/12/) - Large English speech corpus
- [Common Voice](https://commonvoice.mozilla.org/) - Multilingual dataset

### Disordered Speech (requires registration)
- [TORGO Database](http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html) - Dysarthric speech
- [UASpeech](http://www.isle.illinois.edu/sst/data/UASpeech/) - Dysarthric speech

## 🛠️ Development

### Adding New Models

1. Add model to `SUPPORTED_MODELS` in `src/asr_utils.py`
2. Implement loading logic in `_load_<model_type>()` method
3. Add transcription logic in `_transcribe_<model_type>()` method

### Extending Analysis

Add new phonetic patterns in `src/analysis_utils.py`:

```python
# In identify_phonetic_patterns()
if <condition>:
    patterns.append(PhoneticPattern(
        pattern_name="New Pattern",
        description="Description of the pattern",
        affected_words=affected_words,
        frequency=len(affected_words),
        clinical_significance="Clinical notes"
    ))
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for the Transformers library
- [Streamlit](https://streamlit.io/) for the web framework
- [JiWER](https://github.com/jitsi/jiwer) for WER calculation
- [Librosa](https://librosa.org/) for audio processing

## 📬 Contact

For questions or contributions, please open an issue on GitHub.

---

**Note**: This tool is intended for research and educational purposes. It should not be used for clinical diagnosis of speech disorders. Always consult qualified healthcare professionals for medical assessments.
