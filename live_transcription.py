"""Live transcription module for real-time speech-to-text"""

import numpy as np
import threading
import queue
import time
from collections import deque
import sounddevice as sd
import tempfile
import os
from faster_whisper import WhisperModel
import streamlit as st

class LiveTranscriber:
    def __init__(self, transcriber, language="vi", sample_rate=16000, 
                 chunk_duration=3.0, overlap_duration=0.5):
        """
        Initialize live transcription handler
        
        Args:
            transcriber: FasterWhisperTranscriber instance
            language: Language code for transcription
            sample_rate: Audio sample rate (16kHz for Whisper)
            chunk_duration: Duration of audio chunks to process (seconds)
            overlap_duration: Overlap between chunks for better continuity
        """
        self.transcriber = transcriber
        self.language = language
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        
        # Calculate chunk sizes
        self.chunk_size = int(sample_rate * chunk_duration)
        self.overlap_size = int(sample_rate * overlap_duration)
        
        # Audio buffer and processing queue
        self.audio_buffer = deque(maxlen=self.chunk_size + self.overlap_size)
        self.processing_queue = queue.Queue()
        self.transcription_queue = queue.Queue()
        
        # Thread control
        self.is_recording = False
        self.processing_thread = None
        
        # Transcription state
        self.full_transcript = []
        self.last_processed_text = ""
        
        # Performance tracking
        self.last_process_time = 0
        self.processing_times = deque(maxlen=10)
        
    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream - called for each audio chunk from microphone"""
        if status:
            print(f"Audio callback status: {status}")
        
        if self.is_recording:
            # Add audio data to buffer
            audio_data = indata[:, 0] if len(indata.shape) > 1 else indata
            self.audio_buffer.extend(audio_data.flatten())
            
            # Check if we have enough data to process
            if len(self.audio_buffer) >= self.chunk_size:
                # Extract chunk for processing
                chunk = np.array(list(self.audio_buffer)[:self.chunk_size])
                
                # Add to processing queue if not too backed up
                if self.processing_queue.qsize() < 3:  # Limit queue size to prevent lag
                    self.processing_queue.put(chunk.copy())
                    
                # Remove processed audio, keeping overlap
                for _ in range(self.chunk_size - self.overlap_size):
                    if self.audio_buffer:
                        self.audio_buffer.popleft()
    
    def process_audio_chunks(self):
        """Background thread for processing audio chunks"""
        while self.is_recording or not self.processing_queue.empty():
            try:
                # Get audio chunk from queue
                chunk = self.processing_queue.get(timeout=0.1)
                
                # Skip if chunk is too quiet (silence detection)
                if np.max(np.abs(chunk)) < 0.01:  # Silence threshold
                    continue
                
                # Save chunk to temporary file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    import soundfile as sf
                    sf.write(tmp_file.name, chunk, self.sample_rate)
                    tmp_path = tmp_file.name
                
                try:
                    # Measure processing time
                    start_time = time.time()
                    
                    # Transcribe the chunk
                    result = self.transcriber.transcribe(
                        tmp_path,
                        language=self.language,
                        task="transcribe",
                        beam_size=1,  # Faster for real-time
                        best_of=1,    # Faster for real-time
                        temperature=0.0
                    )
                    
                    # Track processing time
                    process_time = time.time() - start_time
                    self.processing_times.append(process_time)
                    
                    # Extract transcribed text
                    text = result.get("text", "").strip()
                    
                    if text and text != self.last_processed_text:
                        # Avoid duplicate text from overlapping chunks
                        self.last_processed_text = text
                        self.transcription_queue.put({
                            "text": text,
                            "timestamp": time.time(),
                            "processing_time": process_time,
                            "is_partial": True
                        })
                        
                finally:
                    # Clean up temporary file
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing audio chunk: {e}")
    
    def start_recording(self):
        """Start live transcription"""
        if not self.is_recording:
            self.is_recording = True
            self.audio_buffer.clear()
            self.full_transcript = []
            self.last_processed_text = ""
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self.process_audio_chunks)
            self.processing_thread.start()
            
            # Start audio stream
            self.stream = sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=int(self.sample_rate * 0.1),  # 100ms blocks
                dtype=np.float32
            )
            self.stream.start()
            
            return True
        return False
    
    def stop_recording(self):
        """Stop live transcription"""
        if self.is_recording:
            self.is_recording = False
            
            # Stop audio stream
            if hasattr(self, 'stream'):
                self.stream.stop()
                self.stream.close()
            
            # Wait for processing to complete
            if self.processing_thread:
                self.processing_thread.join(timeout=5)
            
            # Process any remaining audio
            if len(self.audio_buffer) > self.sample_rate * 0.5:  # At least 0.5 seconds
                chunk = np.array(list(self.audio_buffer))
                self.processing_queue.put(chunk)
                
                # Give it time to process
                time.sleep(1)
            
            return True
        return False
    
    def get_transcription(self):
        """Get the latest transcription results"""
        results = []
        
        # Collect all available transcriptions
        while not self.transcription_queue.empty():
            try:
                result = self.transcription_queue.get_nowait()
                results.append(result)
                
                # Add to full transcript
                if result["text"]:
                    self.full_transcript.append(result["text"])
                    
            except queue.Empty:
                break
        
        return results
    
    def get_full_transcript(self):
        """Get the complete transcript"""
        return " ".join(self.full_transcript)
    
    def get_performance_stats(self):
        """Get performance statistics"""
        if self.processing_times:
            avg_time = np.mean(self.processing_times)
            max_time = np.max(self.processing_times)
            
            return {
                "avg_processing_time": avg_time,
                "max_processing_time": max_time,
                "queue_size": self.processing_queue.qsize(),
                "buffer_size": len(self.audio_buffer),
                "rtf": avg_time / self.chunk_duration  # Real-time factor
            }
        return None

def create_live_transcription_ui(transcriber, language="vi"):
    """Create Streamlit UI for live transcription"""
    
    # Initialize session state
    if "live_transcriber" not in st.session_state:
        st.session_state.live_transcriber = None
    if "live_transcript" not in st.session_state:
        st.session_state.live_transcript = []
    if "is_live_recording" not in st.session_state:
        st.session_state.is_live_recording = False
    
    st.markdown("### 🎙️ Live Transcription")
    st.info("Real-time transcription as you speak - results appear instantly!")
    
    # Configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        chunk_duration = st.slider(
            "Chunk duration (seconds)",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.5,
            help="Shorter chunks = faster response, longer chunks = better accuracy"
        )
    
    with col2:
        overlap_duration = st.slider(
            "Overlap (seconds)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Overlap helps maintain context between chunks"
        )
    
    with col3:
        show_stats = st.checkbox("Show performance stats", value=False)
    
    # Control buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🔴 Start Live Transcription", disabled=st.session_state.is_live_recording):
            # Create live transcriber
            st.session_state.live_transcriber = LiveTranscriber(
                transcriber=transcriber,
                language=language,
                chunk_duration=chunk_duration,
                overlap_duration=overlap_duration
            )
            
            # Start recording
            if st.session_state.live_transcriber.start_recording():
                st.session_state.is_live_recording = True
                st.session_state.live_transcript = []
                st.rerun()
    
    with col2:
        if st.button("⏹️ Stop", disabled=not st.session_state.is_live_recording):
            if st.session_state.live_transcriber:
                st.session_state.live_transcriber.stop_recording()
                st.session_state.is_live_recording = False
                
                # Get final transcript
                final_transcript = st.session_state.live_transcriber.get_full_transcript()
                if final_transcript:
                    st.session_state.live_transcript.append({
                        "text": final_transcript,
                        "is_final": True
                    })
                st.rerun()
    
    with col3:
        if st.button("🗑️ Clear Transcript"):
            st.session_state.live_transcript = []
            st.rerun()
    
    # Live transcription display
    if st.session_state.is_live_recording:
        st.markdown("#### 🔴 Recording... Speak now!")
        
        # Create placeholders for live updates
        transcript_placeholder = st.empty()
        stats_placeholder = st.empty()
        
        # Continuous update loop
        if st.session_state.live_transcriber:
            # Get new transcriptions
            new_results = st.session_state.live_transcriber.get_transcription()
            
            for result in new_results:
                st.session_state.live_transcript.append(result)
            
            # Display current transcript
            with transcript_placeholder.container():
                if st.session_state.live_transcript:
                    # Show recent transcriptions
                    recent_text = []
                    for item in st.session_state.live_transcript[-5:]:  # Show last 5 chunks
                        if item.get("is_partial", False):
                            recent_text.append(f"🔄 {item['text']}")
                        else:
                            recent_text.append(item['text'])
                    
                    st.text_area(
                        "Live Transcript:",
                        value="\n".join(recent_text),
                        height=200,
                        key="live_display"
                    )
            
            # Show performance stats
            if show_stats and st.session_state.live_transcriber:
                stats = st.session_state.live_transcriber.get_performance_stats()
                if stats:
                    with stats_placeholder.container():
                        st.markdown("##### Performance Stats")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Avg Processing", f"{stats['avg_processing_time']:.2f}s")
                        with col2:
                            st.metric("Queue Size", stats['queue_size'])
                        with col3:
                            rtf_color = "🟢" if stats['rtf'] < 0.5 else "🟡" if stats['rtf'] < 1.0 else "🔴"
                            st.metric("Real-time Factor", f"{rtf_color} {stats['rtf']:.2f}x")
            
            # Auto-refresh for live updates
            time.sleep(0.5)
            st.rerun()
    
    else:
        # Show final transcript
        if st.session_state.live_transcript:
            st.markdown("#### 📝 Transcript")
            
            # Combine all text
            full_text = ""
            for item in st.session_state.live_transcript:
                if isinstance(item, dict):
                    text = item.get("text", "")
                else:
                    text = str(item)
                if text:
                    full_text += text + " "
            
            if full_text:
                st.text_area(
                    "Final Transcript:",
                    value=full_text.strip(),
                    height=200,
                    key="final_transcript_display"
                )
                
                # Download button
                st.download_button(
                    label="📥 Download Transcript",
                    data=full_text.strip(),
                    file_name="live_transcript.txt",
                    mime="text/plain"
                )
        else:
            st.info("Click 'Start Live Transcription' to begin real-time speech-to-text")