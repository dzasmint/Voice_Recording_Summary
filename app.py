import streamlit as st
import tempfile
import os
from audio_recorder_streamlit import audio_recorder
from audio_transcriber import FasterWhisperTranscriber
from live_transcription_cloud import create_live_transcription_wrapper

st.set_page_config(
    page_title="Live Audio Transcription - PhoWhisper",
    page_icon="🎤",
    layout="wide"
)

@st.cache_resource
def load_model(model_name="PhoWhisper-small", device="auto", compute_type="default"):
    """Load PhoWhisper model with CTranslate2 optimization"""
    try:
        from model_config import get_model_info
        model_info = get_model_info(model_name)
        
        transcriber = FasterWhisperTranscriber(
            model_name=model_name,
            device=device,
            compute_type=compute_type
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
    st.markdown("### Vietnamese Speech-to-Text using PhoWhisper Models & Faster-Whisper")
    
    # Add tabs for different modes
    tab1, tab2 = st.tabs(["📱 Standard Transcription", "🎙️ Live Transcription"])
    
    # Model and device selection in sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Model selection
        st.markdown("#### 🤖 Model Selection")
        model_option = st.selectbox(
            "Choose Model",
            ["PhoWhisper-small", "PhoWhisper-medium", "PhoWhisper-large-ct2"],
            index=0,  # Default to small model
            format_func=lambda x: {
                "PhoWhisper-small": "Small (Fast, 39M params)",
                "PhoWhisper-medium": "Medium (Balanced, 769M params)",
                "PhoWhisper-large-ct2": "Large (Accurate, 1.5B params)"
            }[x],
            help="Small: 5-10x faster with good accuracy. Medium: Balanced speed and accuracy. Large: Highest accuracy but slower."
        )
        
        # Show model details
        from model_config import get_model_info
        model_info = get_model_info(model_option)
        with st.expander("Model Details", expanded=False):
            st.markdown(f"**Repository:** {model_info['repo_id']}")
            st.markdown(f"**Size:** {model_info['size']}")
            st.markdown(f"**Performance:** {model_info['performance']}")
            st.markdown(f"**Best for:** {model_info['recommended_for']}")
        
        # Device selection
        st.markdown("#### 💻 Device Selection")
        device_option = st.selectbox(
            "Select Device",
            ["auto", "cpu", "cuda", "mps"],
            index=0,
            format_func=lambda x: {
                "auto": "Auto-detect",
                "cpu": "CPU",
                "cuda": "GPU (CUDA)",
                "mps": "GPU (Metal)"
            }[x],
            help="Select the device for model inference. Auto-detect will choose the best available option."
        )
        
        if device_option == "mps":
            st.info("Metal Performance Shaders will be used for acceleration on Apple Silicon")
        
        st.markdown("---")
    
    transcriber = load_model(model_name=model_option, device=device_option)
    
    if transcriber is None:
        st.error("Failed to load model. Please check your installation.")
        return
    
    # Tab 1: Standard Transcription
    with tab1:
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
                # Recording stability options
                with st.expander("⚙️ Recording Settings", expanded=False):
                    recording_mode = st.radio(
                        "Recording Mode",
                        ["Standard", "Enhanced (Better Stability)"],
                        index=0,
                        help="Enhanced mode provides better stability with automatic reconnection"
                    )
                    
                    # Additional settings for stability
                    auto_retry = st.checkbox("Auto-retry on failure", value=True, 
                                            help="Automatically retry recording if connection is lost")
                    max_duration = st.slider("Max recording duration (seconds)", 10, 300, 60,
                                            help="Automatically stop recording after this duration to prevent data loss")
                
                if recording_mode == "Enhanced (Better Stability)":
                    try:
                        from audio_recorder_enhanced import create_audio_recorder_ui
                        st.info("📍 Enhanced Mode: Click button to start/stop recording")
                        with st.container():
                            audio_bytes = create_audio_recorder_ui()
                            
                            # Add connection status indicator
                            if 'audio_recorder' in st.session_state:
                                status = st.session_state.audio_recorder.get_recording_status()
                                if status['status'] == 'recording':
                                    st.success(f"🔴 Recording... ({status['duration']:.1f}s)")
                                elif status['status'] == 'reconnecting':
                                    st.warning("🔄 Reconnecting audio...")
                                    
                    except ImportError:
                        st.warning("Enhanced recorder not available. Using standard recorder.")
                        st.info("💡 Tip: For better stability, try shorter recording sessions")
                        
                        # Add retry mechanism for standard recorder
                        if auto_retry and 'retry_count' not in st.session_state:
                            st.session_state.retry_count = 0
                        
                        try:
                            audio_bytes = audio_recorder(
                                text="Click to record", 
                                icon_size="2x",
                                key=f"audio_recorder_{st.session_state.get('retry_count', 0)}"
                            )
                        except Exception as e:
                            st.error(f"Recording failed: {e}")
                            if auto_retry and st.session_state.retry_count < 3:
                                st.session_state.retry_count += 1
                                st.info("Retrying... Please click the record button again")
                                st.rerun()
                else:
                    st.info("🎤 Standard Mode: Click the microphone button to record")
                    
                    # Tips for stable recording
                    with st.expander("💡 Tips for stable recording"):
                        st.markdown("""
                        - Keep recording sessions under 60 seconds for best stability
                        - Ensure stable internet connection
                        - Allow microphone permissions when prompted
                        - Try refreshing the page if recording stops working
                        - Use Chrome or Firefox for best compatibility
                        """)
                    
                    # Use stable recorder with better error handling
                    try:
                        from stable_recorder import stable_audio_recorder
                        audio_bytes = stable_audio_recorder()
                    except ImportError:
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
                        model_display_name = {
                            "PhoWhisper-small": "PhoWhisper-small",
                            "PhoWhisper-medium": "PhoWhisper-medium",
                            "PhoWhisper-large-ct2": "PhoWhisper-large"
                        }.get(model_option, model_option)
                        status_text.text(f"🎯 Processing audio with {model_display_name}...")
                        
                        # Get audio duration for progress estimation
                        import librosa
                        audio_duration = librosa.get_duration(path=temp_audio_path)
                        
                        # Estimate processing time based on device and model
                        # Model speed factors: Small is fastest, medium is balanced, large is slowest
                        model_factors = {
                            "PhoWhisper-small": 0.2,
                            "PhoWhisper-medium": 0.5,
                            "PhoWhisper-large-ct2": 1.0
                        }
                        model_factor = model_factors.get(model_option, 1.0)
                        
                        if transcriber.device == "cuda":
                            device_factor = 0.1
                        elif hasattr(transcriber, 'use_mps_tensors') and transcriber.use_mps_tensors:
                            device_factor = 0.2  # MPS acceleration via CPU
                        else:
                            device_factor = 0.3
                        
                        estimated_time = audio_duration * device_factor * model_factor
                        
                        status_text.text(f"⏱️ Audio duration: {audio_duration:.1f} seconds | Estimated processing: ~{estimated_time:.0f} seconds")
                        
                        # Step 4: Transcribe
                        progress_bar.progress(50, text=f"Transcribing {audio_duration:.0f}s audio... (~{estimated_time:.0f}s remaining)")
                        
                        # Transcribe with or without timestamps
                        if show_timestamps:
                            status_text.text(f"📝 Generating transcription with timestamps ({model_display_name})...")
                            result = transcriber.transcribe_with_timestamps(
                                temp_audio_path,
                                language=language
                            )
                        else:
                            status_text.text(f"📝 Generating transcription ({model_display_name})...")
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
    
    # Tab 2: Live Transcription
    with tab2:
        # Language selection for live mode
        language = st.selectbox(
            "Select Language for Live Transcription",
            ["vi", "en", "auto"],
            index=0,
            format_func=lambda x: {"vi": "Vietnamese", "en": "English", "auto": "Auto-detect"}[x],
            key="live_language"
        )
        
        # Create live transcription UI (automatically detects cloud vs local)
        create_live_transcription_wrapper(transcriber, language=language)
    
    with st.sidebar:
        st.markdown("### ℹ️ Information")
        st.markdown(f"**Model:** {model_option}")
        st.markdown(f"**Backend:** Faster-Whisper")
        
        # Display device info properly
        if hasattr(transcriber, 'use_mps_tensors') and transcriber.use_mps_tensors:
            device_display = 'CPU + MPS Acceleration'
        else:
            device_display = {
                'cuda': 'GPU (CUDA)',
                'cpu': 'CPU'
            }.get(transcriber.device, transcriber.device)
        
        st.markdown(f"**Device:** {device_display}")
        st.markdown(f"**Compute Type:** {transcriber.compute_type}")
        
        st.markdown("---")
        st.markdown("### 📋 Instructions")
        st.markdown("""
        **Standard Transcription:**
        1. Select "Standard Transcription" tab
        2. Record or upload audio
        3. Click 'Transcribe' for full transcription
        
        **Live Transcription:**
        1. Select "Live Transcription" tab
        2. Click 'Start Live Transcription'
        3. Speak - text appears in real-time
        4. Click 'Stop' when done
        """)
        
        st.markdown("---")
        st.markdown("### 🚀 Features")
        st.markdown("""
        - **Multiple Models**: Choose between Small (fast), Medium (balanced), or Large (accurate)
        - **Live Transcription**: Real-time speech-to-text as you speak
        - **PhoWhisper**: State-of-the-art Vietnamese ASR models
        - **Faster-Whisper**: Optimized inference with CTranslate2
        - **Metal Support**: Accelerated inference on Apple Silicon
        - **VAD Filter**: Voice Activity Detection for better accuracy
        - **Multi-language**: Supports Vietnamese and English
        - **Timestamps**: Optional word-level timestamps
        """)

if __name__ == "__main__":
    main()