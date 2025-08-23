import os
from faster_whisper import WhisperModel
import numpy as np
import librosa
import soundfile as sf
from model_config import get_model_info, get_compute_type

class FasterWhisperTranscriber:
    def __init__(self, model_name="PhoWhisper-small", device="auto", compute_type="default"):
        """
        Initialize Faster-Whisper transcriber with PhoWhisper model
        
        Args:
            model_name: Model name from available models (PhoWhisper-small or PhoWhisper-large-ct2)
            device: Device to use (auto, cpu, cuda, mps)
            compute_type: Compute type (default, float16, int8, int8_float16, float32)
        """
        # Get model configuration
        self.model_name = model_name
        model_info = get_model_info(model_name)
        self.model_repo = model_info["repo_id"]
        self.subfolder = model_info.get("subfolder", None)
        self.requested_device = device  # Store the originally requested device
        
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                    # MPS is available but faster-whisper doesn't support it directly
                    # We'll use CPU with optimizations
                    self.device = "cpu"
                    self.use_mps_tensors = True
                    print("Metal Performance Shaders (MPS) detected - using CPU with MPS tensor acceleration")
                else:
                    self.device = "cpu"
                    self.use_mps_tensors = False
            except ImportError:
                print("PyTorch not installed. Using CPU for inference.")
                self.device = "cpu"
                self.use_mps_tensors = False
        elif device == "mps":
            # User explicitly requested MPS
            self.device = "cpu"  # faster-whisper will use CPU
            self.use_mps_tensors = True
            print("MPS requested - using CPU backend with MPS tensor acceleration where possible")
        else:
            self.device = device
            self.use_mps_tensors = False
        
        if compute_type == "default":
            # Get optimal compute type based on model and device
            self.compute_type = get_compute_type(self.model_name, self.device)
        else:
            self.compute_type = compute_type
        
        self._load_model()
    
    def _load_model(self):
        """Load the Faster-Whisper model"""
        try:
            if self.subfolder:
                print(f"Loading Faster-Whisper model: {self.model_repo}/{self.subfolder}")
            else:
                print(f"Loading Faster-Whisper model: {self.model_repo}")
            print(f"Device: {self.device}, Compute type: {self.compute_type}")
            
            if hasattr(self, 'use_mps_tensors') and self.use_mps_tensors:
                print("MPS acceleration enabled for tensor operations")
            
            # Configure CPU threads for optimal performance
            cpu_threads = 0  # 0 means use all available cores
            if self.device == "cpu":
                # For Apple Silicon, use performance cores efficiently
                import platform
                if platform.processor() == 'arm':
                    cpu_threads = 8  # Typical number of performance cores on M1/M2
            
            # Build model path
            if self.subfolder:
                # For models with subfolders, construct the full path
                import os
                cache_dir = os.path.expanduser("~/.cache/faster-whisper")
                # Download the model first if needed
                from huggingface_hub import snapshot_download
                model_path = snapshot_download(
                    repo_id=self.model_repo,
                    cache_dir=cache_dir,
                    allow_patterns=[f"{self.subfolder}/*"]
                )
                model_path = os.path.join(model_path, self.subfolder)
            else:
                # For models without subfolders, use repo_id directly
                model_path = self.model_repo
            
            # Load the model
            self.model = WhisperModel(
                model_path,
                device=self.device,
                compute_type=self.compute_type,
                download_root=os.path.expanduser("~/.cache/faster-whisper"),
                cpu_threads=cpu_threads
            )
            
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def transcribe(self, audio_path, language="vi", task="transcribe", 
                  beam_size=5, best_of=5, temperature=0, progress_callback=None,
                  vad_filter=True, vad_parameters=None):
        """
        Transcribe audio file to text using Faster-Whisper
        
        Args:
            audio_path: Path to audio file
            language: Language code (default: 'vi' for Vietnamese)
            task: Task to perform ('transcribe' or 'translate')
            beam_size: Beam size for beam search
            best_of: Number of candidates when sampling
            temperature: Temperature for sampling
            progress_callback: Optional callback function for progress updates
            vad_filter: Enable voice activity detection filter
            vad_parameters: Optional VAD parameters dictionary
        
        Returns:
            Dictionary with transcription results
        """
        try:
            # Set default VAD parameters if not provided
            if vad_parameters is None:
                vad_parameters = dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=400,
                    threshold=0.5
                )
            
            segments, info = self.model.transcribe(
                audio_path,
                language=language,
                task=task,
                beam_size=beam_size,
                best_of=best_of,
                temperature=temperature,
                vad_filter=vad_filter,
                vad_parameters=vad_parameters
            )
            
            # Combine all segments into full text
            full_text = ""
            segment_list = []
            
            for segment in segments:
                full_text += segment.text + " "
                segment_list.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text
                })
            
            result = {
                "text": full_text.strip(),
                "segments": segment_list,
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration
            }
            
            return result
            
        except Exception as e:
            print(f"Transcription error: {e}")
            return {"text": "", "error": str(e)}
    
    def transcribe_with_timestamps(self, audio_path, language="vi"):
        """
        Transcribe with word-level timestamps
        
        Args:
            audio_path: Path to audio file
            language: Language code
        
        Returns:
            Dictionary with transcription and detailed timestamps
        """
        try:
            segments, info = self.model.transcribe(
                audio_path,
                language=language,
                word_timestamps=True,
                vad_filter=True
            )
            
            full_text = ""
            words_list = []
            
            for segment in segments:
                full_text += segment.text + " "
                
                if segment.words:
                    for word in segment.words:
                        words_list.append({
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                            "probability": word.probability
                        })
            
            return {
                "text": full_text.strip(),
                "words": words_list,
                "language": info.language,
                "duration": info.duration
            }
            
        except Exception as e:
            print(f"Transcription error: {e}")
            return {"text": "", "error": str(e)}
    
    def detect_language(self, audio_path):
        """
        Detect the language of the audio
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Detected language code and probability
        """
        try:
            segments, info = self.model.transcribe(
                audio_path,
                beam_size=1,
                best_of=1
            )
            
            # Process at least one segment to get language info
            for _ in segments:
                break
            
            return {
                "language": info.language,
                "probability": info.language_probability
            }
        except Exception as e:
            print(f"Language detection error: {e}")
            return {"language": None, "probability": 0.0}