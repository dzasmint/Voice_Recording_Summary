# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Voice Recording Summary is a new project in the Dragon Capital AI ecosystem for transcribing and summarizing audio recordings. This project is currently in initial setup phase.

## Project Context

This project is part of a larger AI/ML ecosystem at Dragon Capital that includes:
- Company_Dashboard: Streamlit-based financial analysis dashboard
- Real_Estate_Financial_Model_Forecast: Real estate forecasting application

## Development Commands

Since this is a new project, typical Python/Streamlit commands should be used:

```bash
# Install dependencies (once requirements.txt is created)
pip install -r requirements.txt

# Run the application (once app.py is created)
streamlit run app.py

# Python linting (if configured)
python -m flake8 .
python -m black .

# Run tests (if implemented)
pytest tests/
```

## Architecture Guidelines

When implementing features in this voice recording summary application:

1. **Follow Modular Structure**: Separate concerns into distinct modules:
   - `core/audio_processor.py` for audio file handling
   - `core/transcription.py` for speech-to-text
   - `core/summarization.py` for text summarization
   - `utils/` for helper functions

2. **Technology Stack**: Use Python with Streamlit for consistency with other Dragon Capital projects. Consider:
   - OpenAI Whisper or similar for transcription
   - LLMs (OpenAI GPT, Claude) for summarization
   - Audio libraries like pydub or librosa for processing

3. **Configuration Management**: Store sensitive data (API keys) in environment variables or config files (never commit these)

4. **UI Patterns**: Follow Streamlit patterns from Company_Dashboard project in the parent directory

5. **Error Handling**: Implement robust error handling for:
   - File upload validation
   - API rate limits
   - Audio format compatibility
   - Network failures

## Key Implementation Considerations

- **Audio Format Support**: Support common formats (mp3, wav, m4a, etc.)
- **File Size Limits**: Implement appropriate limits for audio uploads
- **Progress Indicators**: Show progress for long-running operations (transcription, summarization)
- **Export Options**: Allow users to export summaries in various formats
- **Session Management**: Use Streamlit session state for user data persistence

## Integration Points

When integrating with existing Dragon Capital systems:
- Follow the configuration patterns from Company_Dashboard
- Use similar utility function structures
- Maintain consistent error handling and logging approaches