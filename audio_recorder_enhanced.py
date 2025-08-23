"""Enhanced audio recorder with improved stability and error handling"""

import streamlit as st
import numpy as np
import sounddevice as sd
import queue
import threading
import time
import io
import wave
from datetime import datetime

class EnhancedAudioRecorder:
    def __init__(self, sample_rate=16000, channels=1, chunk_duration=0.1):
        """
        Initialize enhanced audio recorder with buffering
        
        Args:
            sample_rate: Audio sample rate (default 16kHz for speech)
            channels: Number of audio channels (1 for mono)
            chunk_duration: Duration of each audio chunk in seconds
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        
        # Audio buffer and recording state
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.recording_thread = None
        self.audio_buffer = []
        self.error_count = 0
        self.max_errors = 5
        
        # Connection monitoring
        self.last_data_time = None
        self.connection_timeout = 2.0  # seconds
        
    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream - handles incoming audio data"""
        if status:
            self.error_count += 1
            if self.error_count > self.max_errors:
                st.warning(f"Audio input error: {status}")
        else:
            self.error_count = 0
            
        if self.is_recording:
            # Add timestamp to track data flow
            self.last_data_time = time.time()
            # Copy data to avoid reference issues
            self.audio_queue.put(indata.copy())
    
    def start_recording(self):
        """Start audio recording with error handling"""
        if self.is_recording:
            return False
            
        try:
            self.audio_buffer = []
            self.is_recording = True
            self.error_count = 0
            self.last_data_time = time.time()
            
            # Start audio stream with callback
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=self.audio_callback,
                blocksize=self.chunk_size,
                dtype=np.float32
            )
            self.stream.start()
            
            # Start processing thread
            self.recording_thread = threading.Thread(target=self._process_audio_queue)
            self.recording_thread.daemon = True
            self.recording_thread.start()
            
            return True
            
        except Exception as e:
            st.error(f"Failed to start recording: {e}")
            self.is_recording = False
            return False
    
    def _process_audio_queue(self):
        """Process audio queue in background thread"""
        while self.is_recording:
            try:
                # Check for connection timeout
                if self.last_data_time and (time.time() - self.last_data_time) > self.connection_timeout:
                    st.warning("Audio connection lost - attempting to reconnect...")
                    self._reconnect_stream()
                
                # Process audio data from queue
                while not self.audio_queue.empty():
                    try:
                        audio_chunk = self.audio_queue.get(timeout=0.1)
                        self.audio_buffer.append(audio_chunk)
                    except queue.Empty:
                        break
                        
                time.sleep(0.01)  # Small delay to prevent CPU overuse
                
            except Exception as e:
                st.error(f"Audio processing error: {e}")
                time.sleep(0.1)
    
    def _reconnect_stream(self):
        """Attempt to reconnect audio stream"""
        try:
            if hasattr(self, 'stream'):
                self.stream.stop()
                self.stream.close()
            
            time.sleep(0.5)
            
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=self.audio_callback,
                blocksize=self.chunk_size,
                dtype=np.float32
            )
            self.stream.start()
            self.last_data_time = time.time()
            st.success("Audio reconnected successfully")
            
        except Exception as e:
            st.error(f"Reconnection failed: {e}")
    
    def stop_recording(self):
        """Stop recording and return audio data"""
        if not self.is_recording:
            return None
            
        self.is_recording = False
        
        # Wait for processing thread to finish
        if self.recording_thread:
            self.recording_thread.join(timeout=1.0)
        
        # Stop and close stream
        try:
            if hasattr(self, 'stream'):
                self.stream.stop()
                self.stream.close()
        except Exception as e:
            st.warning(f"Error closing stream: {e}")
        
        # Process remaining queue items
        while not self.audio_queue.empty():
            try:
                audio_chunk = self.audio_queue.get_nowait()
                self.audio_buffer.append(audio_chunk)
            except queue.Empty:
                break
        
        # Combine audio chunks
        if self.audio_buffer:
            audio_data = np.concatenate(self.audio_buffer, axis=0)
            return self._create_wav_bytes(audio_data)
        
        return None
    
    def _create_wav_bytes(self, audio_data):
        """Convert numpy array to WAV bytes"""
        # Normalize audio to 16-bit PCM
        audio_normalized = np.int16(audio_data * 32767)
        
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_normalized.tobytes())
        
        wav_buffer.seek(0)
        return wav_buffer.read()
    
    def get_recording_status(self):
        """Get current recording status and statistics"""
        if not self.is_recording:
            return {"status": "stopped", "duration": 0, "buffer_size": 0}
        
        duration = len(self.audio_buffer) * self.chunk_duration if self.audio_buffer else 0
        buffer_size = len(self.audio_buffer)
        
        # Check connection health
        connection_healthy = True
        if self.last_data_time:
            time_since_data = time.time() - self.last_data_time
            connection_healthy = time_since_data < 0.5
        
        return {
            "status": "recording" if connection_healthy else "reconnecting",
            "duration": duration,
            "buffer_size": buffer_size,
            "error_count": self.error_count,
            "connection_healthy": connection_healthy
        }

def create_audio_recorder_ui():
    """Create enhanced audio recorder UI component"""
    
    # Initialize recorder in session state
    if 'audio_recorder' not in st.session_state:
        st.session_state.audio_recorder = EnhancedAudioRecorder()
    
    if 'recording_state' not in st.session_state:
        st.session_state.recording_state = False
    
    recorder = st.session_state.audio_recorder
    
    # Recording controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.session_state.recording_state:
            if st.button("⏹️ Stop Recording", type="primary", use_container_width=True):
                audio_bytes = recorder.stop_recording()
                st.session_state.recording_state = False
                if audio_bytes:
                    st.session_state.last_recording = audio_bytes
                    return audio_bytes
        else:
            if st.button("🎤 Start Recording", type="primary", use_container_width=True):
                if recorder.start_recording():
                    st.session_state.recording_state = True
                    st.session_state.last_recording = None
    
    with col2:
        if st.session_state.recording_state:
            status = recorder.get_recording_status()
            if status['connection_healthy']:
                st.success(f"Recording: {status['duration']:.1f}s")
            else:
                st.warning("Reconnecting...")
    
    with col3:
        if st.session_state.recording_state:
            status = recorder.get_recording_status()
            if status['error_count'] > 0:
                st.warning(f"Errors: {status['error_count']}")
    
    # Show last recording if available
    if hasattr(st.session_state, 'last_recording') and st.session_state.last_recording:
        return st.session_state.last_recording
    
    return None