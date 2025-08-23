"""Stable audio recording with session management and auto-recovery"""

import streamlit as st
from audio_recorder_streamlit import audio_recorder
import time
import hashlib

def get_session_id():
    """Generate a unique session ID for recording stability"""
    # Use session state to maintain consistency
    if 'session_id' not in st.session_state:
        st.session_state.session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
    return st.session_state.session_id

def stable_audio_recorder():
    """
    Stable audio recorder with automatic recovery and session management
    """
    
    # Initialize recording state
    if 'recording_attempts' not in st.session_state:
        st.session_state.recording_attempts = 0
    
    if 'last_audio' not in st.session_state:
        st.session_state.last_audio = None
    
    if 'recording_status' not in st.session_state:
        st.session_state.recording_status = "ready"
    
    # Create unique key for each recording attempt
    recorder_key = f"recorder_{get_session_id()}_{st.session_state.recording_attempts}"
    
    try:
        # Record audio with unique key to prevent caching issues
        audio_bytes = audio_recorder(
            text="🎤 Click to record",
            icon_size="2x",
            key=recorder_key,
            pause_threshold=2.0,  # Auto-stop after 2 seconds of silence
            sample_rate=16000,    # Lower sample rate for stability
            auto_start=False,      # Manual start for better control
            energy_threshold=0.01  # Adjust sensitivity
        )
        
        # Handle successful recording
        if audio_bytes:
            st.session_state.last_audio = audio_bytes
            st.session_state.recording_status = "success"
            st.session_state.recording_attempts = 0  # Reset attempts on success
            return audio_bytes
            
    except Exception as e:
        # Handle recording errors
        st.session_state.recording_status = "error"
        st.session_state.recording_attempts += 1
        
        if st.session_state.recording_attempts < 3:
            st.warning(f"Recording interrupted (Attempt {st.session_state.recording_attempts}/3)")
            st.info("Click the record button again to retry")
            
            # Force refresh the recorder
            time.sleep(0.5)
            if st.button("🔄 Reset Recorder"):
                st.session_state.recording_attempts = 0
                st.rerun()
        else:
            st.error("Recording failed after 3 attempts. Please refresh the page.")
            if st.button("🔄 Refresh Page"):
                st.session_state.clear()
                st.rerun()
    
    # Return last successful recording if available
    return st.session_state.last_audio

def chunked_audio_recorder(chunk_duration=30):
    """
    Record audio in chunks to prevent timeout issues
    
    Args:
        chunk_duration: Maximum duration for each chunk in seconds
    """
    
    if 'audio_chunks' not in st.session_state:
        st.session_state.audio_chunks = []
    
    if 'is_recording_chunk' not in st.session_state:
        st.session_state.is_recording_chunk = False
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if not st.session_state.is_recording_chunk:
            if st.button("🎤 Start Recording", type="primary", use_container_width=True):
                st.session_state.is_recording_chunk = True
                st.session_state.chunk_start_time = time.time()
                st.rerun()
        else:
            if st.button("⏹️ Stop Recording", type="secondary", use_container_width=True):
                st.session_state.is_recording_chunk = False
                st.rerun()
    
    with col2:
        if st.session_state.is_recording_chunk:
            elapsed = time.time() - st.session_state.get('chunk_start_time', time.time())
            remaining = max(0, chunk_duration - elapsed)
            st.metric("Time Remaining", f"{remaining:.0f}s")
            
            # Auto-stop after chunk duration
            if elapsed >= chunk_duration:
                st.session_state.is_recording_chunk = False
                st.warning(f"Auto-stopped after {chunk_duration}s")
                st.rerun()
    
    with col3:
        chunks_count = len(st.session_state.audio_chunks)
        if chunks_count > 0:
            st.metric("Chunks", chunks_count)
    
    # Record current chunk
    if st.session_state.is_recording_chunk:
        chunk_key = f"chunk_{len(st.session_state.audio_chunks)}"
        audio_chunk = audio_recorder(
            text="Recording...",
            icon_size="1x",
            key=chunk_key,
            recording_color="#ff4b4b",
            neutral_color="#0ea5e9"
        )
        
        if audio_chunk:
            st.session_state.audio_chunks.append(audio_chunk)
            st.success(f"Chunk {len(st.session_state.audio_chunks)} saved")
    
    # Combine and return all chunks
    if st.session_state.audio_chunks and not st.session_state.is_recording_chunk:
        if st.button("📦 Process Recording"):
            # Combine all audio chunks
            combined_audio = b''.join(st.session_state.audio_chunks)
            st.session_state.audio_chunks = []  # Clear chunks
            return combined_audio
    
    return None