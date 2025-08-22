import streamlit as st
import tempfile
import os
from audio_recorder_streamlit import audio_recorder
from audio_transcriber import FasterWhisperTranscriber

st.set_page_config(
    page_title="Live Audio Transcription - PhoWhisper",
    page_icon="🎤",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load PhoWhisper-large model with CTranslate2 optimization"""
    try:
        # Using PhoWhisper-large-ct2 from HuggingFace
        transcriber = FasterWhisperTranscriber(
            model_size="kiendt/PhoWhisper-large-ct2",
            device="auto",
            compute_type="default"
        )
        return transcriber
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def save_audio_file(audio_bytes, file_ext=".wav"):
    """Save audio bytes to a temporary audio file"""
    # Try to detect format from magic bytes
    if audio_bytes[:4] == b'RIFF':
        file_ext = ".wav"
    elif audio_bytes[:3] == b'ID3' or audio_bytes[:2] == b'\xff\xfb':
        file_ext = ".mp3"
    elif b'ftyp' in audio_bytes[:12]:
        file_ext = ".m4a"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        tmp_file.write(audio_bytes)
        return tmp_file.name

def main():
    st.title("🎤 Live Audio Transcription")
    st.markdown("### Vietnamese Speech-to-Text using PhoWhisper-large & Faster-Whisper")
    
    transcriber = load_model()
    
    if transcriber is None:
        st.error("Failed to load model. Please check your installation.")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Audio Input")
        
        # Choose input method
        input_method = st.radio(
            "Choose input method:",
            ["🎤 Record from Microphone", "📱 Upload Audio File"],
            horizontal=True
        )
        
        audio_bytes = None
        
        if input_method == "🎤 Record from Microphone":
            st.info("Click the microphone button to start/stop recording")
            audio_bytes = audio_recorder(text="Click to record", icon_size="2x")
        else:
            st.info("Upload audio file (supports MP3, WAV, M4A, AAC, OGG)")
            uploaded_file = st.file_uploader(
                "Choose an audio file",
                type=["mp3", "wav", "m4a", "aac", "ogg", "flac", "mp4"],
                label_visibility="collapsed"
            )
            if uploaded_file is not None:
                audio_bytes = uploaded_file.read()
                st.success(f"Uploaded: {uploaded_file.name}")
        
        if audio_bytes:
            if input_method == "🎤 Record from Microphone":
                st.success("Recording captured!")
            st.audio(audio_bytes)
            
            # Language selection
            language = st.selectbox(
                "Select Language",
                ["vi", "en", "auto"],
                index=0,
                format_func=lambda x: {"vi": "Vietnamese", "en": "English", "auto": "Auto-detect"}[x]
            )
            
            # Advanced options
            with st.expander("Advanced Options"):
                beam_size = st.slider("Beam Size", 1, 10, 5)
                temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
                show_timestamps = st.checkbox("Show timestamps", False)
            
            if st.button("📝 Transcribe", type="primary", use_container_width=True):
                temp_audio_path = save_audio_file(audio_bytes)
                
                try:
                    # Create progress container
                    progress_container = st.container()
                    with progress_container:
                        progress_bar = st.progress(0, text="Initializing transcription...")
                        status_text = st.empty()
                    
                    # Step 1: Prepare audio
                    progress_bar.progress(10, text="Preparing audio file...")
                    status_text.text("📁 Loading audio data...")
                    
                    if language == "auto":
                        # Step 2: Detect language
                        progress_bar.progress(20, text="Detecting language...")
                        status_text.text("🔍 Analyzing audio language...")
                        lang_info = transcriber.detect_language(temp_audio_path)
                        detected_lang = lang_info["language"]
                        status_text.text(f"✅ Detected language: {detected_lang} (confidence: {lang_info['probability']:.2%})")
                        language = detected_lang
                        progress_bar.progress(30, text=f"Language detected: {detected_lang}")
                    else:
                        progress_bar.progress(30, text="Language set...")
                    
                    # Step 3: Start transcription
                    progress_bar.progress(40, text="Starting transcription...")
                    status_text.text("🎯 Processing audio with PhoWhisper-large...")
                    
                    # Get audio duration for progress estimation
                    import librosa
                    audio_duration = librosa.get_duration(path=temp_audio_path)
                    
                    # Estimate processing time (roughly 1/4 to 1/3 of audio duration on CPU)
                    device_factor = 0.1 if transcriber.device == "cuda" else 0.3
                    estimated_time = audio_duration * device_factor
                    
                    status_text.text(f"⏱️ Audio duration: {audio_duration:.1f} seconds | Estimated processing: ~{estimated_time:.0f} seconds")
                    
                    # Step 4: Transcribe
                    progress_bar.progress(50, text=f"Transcribing {audio_duration:.0f}s audio... (~{estimated_time:.0f}s remaining)")
                    
                    # Transcribe with or without timestamps
                    if show_timestamps:
                        status_text.text("📝 Generating transcription with timestamps...")
                        result = transcriber.transcribe_with_timestamps(
                            temp_audio_path,
                            language=language
                        )
                    else:
                        status_text.text("📝 Generating transcription...")
                        result = transcriber.transcribe(
                            temp_audio_path,
                            language=language,
                            beam_size=beam_size,
                            temperature=temperature
                        )
                    
                    # Step 5: Process results
                    progress_bar.progress(90, text="Processing results...")
                    status_text.text("🔄 Formatting output...")
                    
                    if result.get("text"):
                        st.session_state.transcription = result["text"]
                        st.session_state.segments = result.get("segments", [])
                        st.session_state.duration = result.get("duration", 0)
                        
                        # Step 6: Complete
                        progress_bar.progress(100, text="Transcription complete!")
                        status_text.text("✅ Successfully transcribed!")
                        st.success("Transcription completed successfully!")
                        
                        # Clear progress indicators after 2 seconds
                        import time
                        time.sleep(2)
                        progress_container.empty()
                    else:
                        progress_bar.empty()
                        status_text.empty()
                        st.warning("No speech detected in the audio.")
                        
                except Exception as e:
                    if 'progress_bar' in locals():
                        progress_bar.empty()
                    if 'status_text' in locals():
                        status_text.empty()
                    st.error(f"Transcription error: {str(e)}")
                finally:
                    if os.path.exists(temp_audio_path):
                        os.remove(temp_audio_path)
    
    with col2:
        st.markdown("#### Transcription Result")
        
        if "transcription" in st.session_state and st.session_state.transcription:
            st.text_area(
                "Transcribed Text:",
                value=st.session_state.transcription,
                height=200,
                key="transcription_display"
            )
            
            # Show segments with timestamps if available
            if st.session_state.get("segments"):
                with st.expander("View segments with timestamps"):
                    for segment in st.session_state.segments:
                        st.write(f"**[{segment['start']:.2f}s - {segment['end']:.2f}s]** {segment['text']}")
            
            # Show duration
            if st.session_state.get("duration"):
                st.info(f"Audio duration: {st.session_state.duration:.2f} seconds")
            
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                st.download_button(
                    label="📥 Download as Text",
                    data=st.session_state.transcription,
                    file_name="transcription.txt",
                    mime="text/plain"
                )
            with col2_2:
                if st.button("🗑️ Clear", use_container_width=True):
                    for key in ["transcription", "segments", "duration"]:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
        else:
            st.info("Record audio and click 'Transcribe' to see the text here.")
    
    with st.sidebar:
        st.markdown("### ℹ️ Information")
        st.markdown(f"**Model:** PhoWhisper-large-ct2")
        st.markdown(f"**Backend:** Faster-Whisper")
        st.markdown(f"**Device:** {'GPU' if transcriber.device == 'cuda' else 'CPU'}")
        st.markdown(f"**Compute Type:** {transcriber.compute_type}")
        
        st.markdown("---")
        st.markdown("### 📋 Instructions")
        st.markdown("""
        **Option 1: Record from Microphone**
        1. Select "Record from Microphone"
        2. Click the microphone button
        3. Speak clearly and click again to stop
        
        **Option 2: Upload Audio File**
        1. Select "Upload Audio File"
        2. Upload audio from iPhone/Android/PC
        3. Supports MP3, WAV, M4A, AAC, OGG
        
        **Then:**
        4. Select language (Vietnamese by default)
        5. Click 'Transcribe' to convert speech to text
        6. View and download the transcription
        """)
        
        st.markdown("---")
        st.markdown("### 🚀 Features")
        st.markdown("""
        - **PhoWhisper-large**: State-of-the-art Vietnamese ASR
        - **Faster-Whisper**: Optimized inference with CTranslate2
        - **VAD Filter**: Voice Activity Detection for better accuracy
        - **Multi-language**: Supports Vietnamese and English
        - **Timestamps**: Optional word-level timestamps
        """)

if __name__ == "__main__":
    main()