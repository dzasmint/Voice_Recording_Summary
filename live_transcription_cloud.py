"""Cloud-compatible live transcription module using browser audio recording"""

import streamlit as st
import tempfile
import time
import os
from audio_recorder_streamlit import audio_recorder
import numpy as np

def create_live_transcription_ui_cloud(transcriber, language="vi"):
    """
    Create Streamlit UI for live transcription that works on Streamlit Cloud.
    Uses browser-based audio recording instead of server-side audio capture.
    """
    
    # Initialize session state
    if "live_recordings" not in st.session_state:
        st.session_state.live_recordings = []
    if "live_transcript_cloud" not in st.session_state:
        st.session_state.live_transcript_cloud = []
    if "is_recording_cloud" not in st.session_state:
        st.session_state.is_recording_cloud = False
    if "recording_counter" not in st.session_state:
        st.session_state.recording_counter = 0
    
    st.markdown("### 🎙️ Live Transcription")
    st.info("Cloud version: Record short audio segments for near real-time transcription")
    
    # Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        recording_duration = st.slider(
            "Recording segment duration",
            min_value=5,
            max_value=30,
            value=10,
            step=5,
            help="Duration of each recording segment in seconds"
        )
    
    with col2:
        auto_mode = st.checkbox(
            "Auto-continue recording",
            value=False,
            help="Automatically start next recording segment after transcription"
        )
    
    # Control section
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🔴 Start Session", disabled=st.session_state.is_recording_cloud):
            st.session_state.is_recording_cloud = True
            st.session_state.live_transcript_cloud = []
            st.session_state.recording_counter = 0
            st.rerun()
    
    with col2:
        if st.button("⏹️ End Session", disabled=not st.session_state.is_recording_cloud):
            st.session_state.is_recording_cloud = False
            st.rerun()
    
    with col3:
        if st.button("🗑️ Clear Transcript"):
            st.session_state.live_transcript_cloud = []
            st.session_state.recording_counter = 0
            st.rerun()
    
    # Recording interface
    if st.session_state.is_recording_cloud:
        st.markdown("#### 🔴 Recording Session Active")
        
        # Create unique key for each recording
        recording_key = f"audio_recorder_cloud_{st.session_state.recording_counter}"
        
        # Instructions
        st.markdown(f"**Segment {st.session_state.recording_counter + 1}**: Click the microphone and speak for up to {recording_duration} seconds")
        
        # Audio recorder
        audio_bytes = audio_recorder(
            text="Click to record segment",
            recording_color="#e74c3c",
            neutral_color="#3498db",
            icon_size="2x",
            key=recording_key
        )
        
        if audio_bytes:
            # Process the recorded audio
            with st.spinner("Transcribing audio segment..."):
                # Save audio to temporary file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_file.write(audio_bytes)
                    tmp_path = tmp_file.name
                
                try:
                    # Transcribe with optimized settings for speed
                    result = transcriber.transcribe(
                        tmp_path,
                        language=language,
                        beam_size=1,
                        temperature=0.0
                    )
                    
                    if result.get("text"):
                        # Add to transcript
                        st.session_state.live_transcript_cloud.append({
                            "segment": st.session_state.recording_counter + 1,
                            "text": result["text"].strip(),
                            "timestamp": time.strftime("%H:%M:%S")
                        })
                        
                        st.success(f"✅ Segment {st.session_state.recording_counter + 1} transcribed!")
                        
                        # Increment counter for next recording
                        st.session_state.recording_counter += 1
                        
                        # Auto-continue if enabled
                        if auto_mode:
                            time.sleep(1)
                            st.rerun()
                    
                except Exception as e:
                    st.error(f"Transcription error: {str(e)}")
                
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            
            # Show option to continue
            if not auto_mode:
                if st.button("➡️ Record Next Segment", key="next_segment"):
                    st.rerun()
    
    # Display transcript
    st.markdown("#### 📝 Live Transcript")
    
    if st.session_state.live_transcript_cloud:
        # Combine all segments
        full_text = " ".join([item["text"] for item in st.session_state.live_transcript_cloud])
        
        # Display in text area
        st.text_area(
            "Full Transcript:",
            value=full_text,
            height=200,
            key="cloud_transcript_display"
        )
        
        # Show segments with timestamps
        with st.expander("View segments with timestamps"):
            for item in st.session_state.live_transcript_cloud:
                st.write(f"**[{item['timestamp']} - Segment {item['segment']}]** {item['text']}")
        
        # Download button
        st.download_button(
            label="📥 Download Transcript",
            data=full_text,
            file_name="live_transcript.txt",
            mime="text/plain"
        )
    else:
        st.info("Start a recording session to begin live transcription")

def is_running_on_cloud():
    """Check if the app is running on Streamlit Cloud"""
    # Streamlit Cloud sets specific environment variables
    return (
        os.environ.get("STREAMLIT_SHARING") is not None or
        os.environ.get("STREAMLIT_RUNTIME_ENV") == "cloud" or
        os.path.exists("/mount/src") or  # Streamlit Cloud uses this path
        os.environ.get("HOME") == "/home/adminuser"  # Common in Streamlit Cloud
    )

def create_live_transcription_wrapper(transcriber, language="vi"):
    """
    Wrapper function that chooses the appropriate live transcription UI
    based on the deployment environment
    """
    if is_running_on_cloud():
        # Use cloud-compatible version
        st.warning("⚠️ Running on Streamlit Cloud - Using browser-based recording for live transcription")
        create_live_transcription_ui_cloud(transcriber, language)
    else:
        # Try to use local version with sounddevice
        try:
            import sounddevice as sd
            # Test if audio devices are available
            sd.query_devices()
            
            # Import and use the original live transcription
            from live_transcription import create_live_transcription_ui
            create_live_transcription_ui(transcriber, language)
            
        except (ImportError, sd.PortAudioError, OSError) as e:
            # Fallback to cloud version if sounddevice doesn't work
            st.warning("⚠️ Local audio devices not available - Using browser-based recording")
            create_live_transcription_ui_cloud(transcriber, language)