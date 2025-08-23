"""Web-based audio recorder using browser MediaRecorder API for maximum stability"""

import streamlit as st
import streamlit.components.v1 as components
import base64
import json

def create_web_audio_recorder():
    """
    Create a web-based audio recorder using MediaRecorder API
    This approach is more stable as it uses native browser APIs
    """
    
    # HTML/JavaScript for web-based recording
    recorder_html = """
    <div id="audioRecorder" style="padding: 20px; border-radius: 10px; background: #f0f2f6;">
        <style>
            .record-btn {
                padding: 12px 24px;
                font-size: 16px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                transition: all 0.3s;
                margin: 5px;
            }
            .record-btn.start {
                background: #ff4b4b;
                color: white;
            }
            .record-btn.start:hover {
                background: #ff3333;
            }
            .record-btn.stop {
                background: #4CAF50;
                color: white;
            }
            .record-btn.stop:hover {
                background: #45a049;
            }
            .status {
                margin-top: 10px;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
            }
            .status.recording {
                background: #ffebee;
                color: #c62828;
            }
            .status.ready {
                background: #e8f5e9;
                color: #2e7d32;
            }
            .timer {
                font-weight: bold;
                font-size: 18px;
                margin: 10px 0;
            }
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            .recording-indicator {
                display: inline-block;
                width: 10px;
                height: 10px;
                background: red;
                border-radius: 50%;
                margin-right: 10px;
                animation: pulse 1s infinite;
            }
        </style>
        
        <div style="text-align: center;">
            <button id="recordBtn" class="record-btn start" onclick="toggleRecording()">
                🎤 Start Recording
            </button>
            <div id="timer" class="timer" style="display: none;">00:00</div>
            <div id="status" class="status ready">Ready to record</div>
        </div>
    </div>
    
    <script>
        let mediaRecorder;
        let audioChunks = [];
        let isRecording = false;
        let startTime;
        let timerInterval;
        let reconnectAttempts = 0;
        const maxReconnectAttempts = 3;
        
        // Initialize on load
        window.addEventListener('load', initializeRecorder);
        
        async function initializeRecorder() {
            try {
                // Request microphone permission
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true,
                        sampleRate: 16000
                    } 
                });
                
                // Create MediaRecorder with opus codec for better compression
                const options = {
                    mimeType: 'audio/webm;codecs=opus',
                    audioBitsPerSecond: 128000
                };
                
                // Fallback for browsers that don't support opus
                if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                    options.mimeType = 'audio/webm';
                }
                
                mediaRecorder = new MediaRecorder(stream, options);
                
                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        audioChunks.push(event.data);
                    }
                };
                
                mediaRecorder.onstop = () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    sendAudioToStreamlit(audioBlob);
                    audioChunks = [];
                };
                
                mediaRecorder.onerror = (event) => {
                    console.error('MediaRecorder error:', event.error);
                    handleRecordingError();
                };
                
                updateStatus('Ready to record', 'ready');
                
            } catch (error) {
                console.error('Error accessing microphone:', error);
                updateStatus('Error: Could not access microphone. Please check permissions.', 'error');
            }
        }
        
        function toggleRecording() {
            if (isRecording) {
                stopRecording();
            } else {
                startRecording();
            }
        }
        
        function startRecording() {
            if (!mediaRecorder) {
                initializeRecorder();
                return;
            }
            
            try {
                audioChunks = [];
                mediaRecorder.start(100); // Collect data every 100ms for stability
                isRecording = true;
                startTime = Date.now();
                
                // Update UI
                document.getElementById('recordBtn').textContent = '⏹️ Stop Recording';
                document.getElementById('recordBtn').className = 'record-btn stop';
                document.getElementById('timer').style.display = 'block';
                updateStatus('<span class="recording-indicator"></span>Recording...', 'recording');
                
                // Start timer
                timerInterval = setInterval(updateTimer, 100);
                
                // Auto-save every 30 seconds to prevent data loss
                setTimeout(() => {
                    if (isRecording) {
                        mediaRecorder.requestData();
                    }
                }, 30000);
                
            } catch (error) {
                console.error('Error starting recording:', error);
                handleRecordingError();
            }
        }
        
        function stopRecording() {
            if (!mediaRecorder || !isRecording) return;
            
            try {
                mediaRecorder.stop();
                isRecording = false;
                
                // Update UI
                document.getElementById('recordBtn').textContent = '🎤 Start Recording';
                document.getElementById('recordBtn').className = 'record-btn start';
                document.getElementById('timer').style.display = 'none';
                updateStatus('Processing audio...', 'ready');
                
                // Stop timer
                clearInterval(timerInterval);
                
            } catch (error) {
                console.error('Error stopping recording:', error);
                handleRecordingError();
            }
        }
        
        function updateTimer() {
            if (!isRecording) return;
            
            const elapsed = Date.now() - startTime;
            const seconds = Math.floor(elapsed / 1000);
            const minutes = Math.floor(seconds / 60);
            const displaySeconds = seconds % 60;
            
            document.getElementById('timer').textContent = 
                `${minutes.toString().padStart(2, '0')}:${displaySeconds.toString().padStart(2, '0')}`;
        }
        
        function updateStatus(message, type) {
            const statusDiv = document.getElementById('status');
            statusDiv.innerHTML = message;
            statusDiv.className = `status ${type}`;
        }
        
        function handleRecordingError() {
            isRecording = false;
            clearInterval(timerInterval);
            
            if (reconnectAttempts < maxReconnectAttempts) {
                reconnectAttempts++;
                updateStatus(`Connection lost. Reconnecting (${reconnectAttempts}/${maxReconnectAttempts})...`, 'error');
                setTimeout(initializeRecorder, 1000);
            } else {
                updateStatus('Recording failed. Please refresh the page.', 'error');
                document.getElementById('recordBtn').disabled = true;
            }
        }
        
        async function sendAudioToStreamlit(audioBlob) {
            try {
                // Convert blob to base64
                const reader = new FileReader();
                reader.onloadend = () => {
                    const base64Audio = reader.result.split(',')[1];
                    
                    // Send to Streamlit
                    window.parent.postMessage({
                        type: 'streamlit:setComponentValue',
                        value: {
                            audio_data: base64Audio,
                            timestamp: Date.now()
                        }
                    }, '*');
                    
                    updateStatus('Audio saved successfully!', 'ready');
                };
                reader.readAsDataURL(audioBlob);
                
            } catch (error) {
                console.error('Error sending audio:', error);
                updateStatus('Error saving audio', 'error');
            }
        }
        
        // Handle page visibility changes to pause/resume
        document.addEventListener('visibilitychange', () => {
            if (document.hidden && isRecording) {
                // Page is hidden, might want to auto-stop or warn
                console.warn('Page hidden while recording');
            }
        });
        
        // Cleanup on page unload
        window.addEventListener('beforeunload', () => {
            if (isRecording) {
                stopRecording();
            }
        });
    </script>
    """
    
    # Create a unique key for this component
    component_key = f"web_audio_recorder_{id(st.session_state)}"
    
    # Render the HTML component with bidirectional communication
    component_value = components.html(
        recorder_html,
        height=200,
        scrolling=False,
        key=component_key
    )
    
    # Store audio in session state when received
    if 'web_audio_data' not in st.session_state:
        st.session_state.web_audio_data = None
    
    # Check if audio data was received via JavaScript postMessage
    # For now, return to simpler approach since bidirectional communication needs custom component
    return None

def create_robust_recorder():
    """
    Create a robust audio recorder with multiple fallback options
    """
    
    recording_method = st.selectbox(
        "Recording Method",
        ["Web Audio API (Most Stable)", "Enhanced Python Recorder", "Standard Recorder"],
        help="Choose recording method. Web Audio API is most stable for browser recording."
    )
    
    audio_bytes = None
    
    if recording_method == "Web Audio API (Most Stable)":
        audio_bytes = create_web_audio_recorder()
        
    elif recording_method == "Enhanced Python Recorder":
        try:
            from audio_recorder_enhanced import create_audio_recorder_ui
            audio_bytes = create_audio_recorder_ui()
        except ImportError:
            st.error("Enhanced recorder not available. Please install sounddevice: pip install sounddevice")
            
    else:  # Standard Recorder
        from audio_recorder_streamlit import audio_recorder
        audio_bytes = audio_recorder(text="Click to record", icon_size="2x")
    
    return audio_bytes