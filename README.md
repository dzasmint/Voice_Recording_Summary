# Live Audio Transcription with PhoWhisper

A Streamlit application for real-time audio recording and Vietnamese speech-to-text transcription using PhoWhisper and CTranslate2.

## Features

- 🎤 Live audio recording through web browser
- 🇻🇳 Vietnamese speech-to-text optimized with PhoWhisper
- ⚡ Fast inference with CTranslate2 optimization
- 💾 Download transcriptions as text files
- 🖥️ Clean and intuitive Streamlit interface

## Prerequisites

- Python 3.8 or higher
- FFmpeg (for audio processing)
- CUDA (optional, for GPU acceleration)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Voice_Recording_Summary
```

2. Install FFmpeg (if not already installed):
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Install PhoWhisper (if not included in pip):
```bash
pip install git+https://github.com/VinAIResearch/PhoWhisper.git
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Click the "🎤 Start Recording" button to begin recording

4. Speak clearly in Vietnamese into your microphone

5. Click "⏹️ Stop Recording" when finished

6. Click "📝 Transcribe" to convert your speech to text

7. View the transcription and download if needed

## Project Structure

```
Voice_Recording_Summary/
├── app.py                    # Main Streamlit application
├── audio_transcriber.py      # PhoWhisper transcription module
├── requirements.txt          # Python dependencies
├── README.md                # This file
└── CLAUDE.md               # AI assistant documentation
```

## Configuration

The application automatically detects available hardware:
- Uses GPU (CUDA) if available for faster processing
- Falls back to CPU with optimized settings

## Models

- **PhoWhisper-base**: Vietnamese-optimized Whisper model by VinAI Research
- **CTranslate2**: Inference optimization for faster transcription
- **Fallback**: Standard OpenAI Whisper if PhoWhisper is unavailable

## Troubleshooting

### Audio Recording Issues
- Ensure your browser has microphone permissions
- Check that no other application is using the microphone

### Model Loading Issues
- The first run may take longer as models are downloaded
- Ensure you have sufficient disk space (~1-2GB for models)

### Performance
- GPU acceleration requires CUDA-compatible NVIDIA GPU
- CPU inference is slower but works on all systems

## License

This project uses open-source models and libraries. Please refer to their respective licenses:
- PhoWhisper: Apache 2.0 License
- OpenAI Whisper: MIT License
- Streamlit: Apache 2.0 License