# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Voice Recording Summary is a Vietnamese speech-to-text transcription application built with Streamlit, using PhoWhisper-large model optimized with Faster-Whisper (CTranslate2) for real-time audio transcription.

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Install PhoWhisper if needed
pip install git+https://github.com/VinAIResearch/PhoWhisper.git

# Run the application
streamlit run app.py

# Python formatting (if configured)
python -m black .
python -m flake8 .
```

## Architecture

The application consists of two main modules:

1. **app.py**: Streamlit UI application that handles:
   - Audio input via microphone recording (audio-recorder-streamlit) or file upload
   - Model loading with caching (@st.cache_resource)
   - Progress tracking during transcription
   - Results display with timestamps and download options

2. **audio_transcriber.py**: Core transcription module containing `FasterWhisperTranscriber` class that:
   - Loads PhoWhisper-large-ct2 model from HuggingFace (kiendt/PhoWhisper-large-ct2)
   - Auto-detects CUDA availability for GPU acceleration
   - Implements VAD (Voice Activity Detection) for better accuracy
   - Provides methods for transcription with/without timestamps and language detection

## Key Implementation Details

- **Model**: Uses PhoWhisper-large-ct2, a Vietnamese-optimized Whisper model converted to CTranslate2 format
- **Device Selection**: Automatically uses CUDA if available, falls back to CPU with optimized compute types
- **Audio Processing**: Handles multiple formats (WAV, MP3, M4A, AAC, OGG) with format detection
- **Progress Tracking**: Real-time progress updates during transcription with time estimation
- **Session State**: Uses Streamlit session state to persist transcription results across interactions

## Critical Paths

- Model loading: `load_model()` → `FasterWhisperTranscriber.__init__()` → `WhisperModel()`
- Transcription flow: Audio input → `save_audio_file()` → `transcriber.transcribe()` → Display results
- Audio formats detected via magic bytes in `save_audio_file()`

## Dependencies

Core dependencies from requirements.txt:
- streamlit: Web UI framework
- audio-recorder-streamlit: Browser microphone recording
- faster-whisper: Optimized Whisper inference
- librosa: Audio duration extraction
- soundfile, scipy, numpy: Audio processing