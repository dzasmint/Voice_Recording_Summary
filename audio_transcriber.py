import os
from faster_whisper import WhisperModel
import numpy as np
import librosa
import soundfile as sf

class FasterWhisperTranscriber:
    def __init__(self, model_size="kiendt/PhoWhisper-large-ct2", device="auto", compute_type="default"):
        """
        Initialize Faster-Whisper transcriber with PhoWhisper model
        
        Args:
            model_size: Model name or HuggingFace repo (default: kiendt/PhoWhisper-large-ct2)
            device: Device to use (auto, cpu, cuda)
            compute_type: Compute type (default, float16, int8, int8_float16)
        """
        self.model_size = model_size
        
        if device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        if compute_type == "default":
            self.compute_type = "float16" if self.device == "cuda" else "int8"
        else:
            self.compute_type = compute_type
        
        self._load_model()
    
    def _load_model(self):
        """Load the Faster-Whisper model"""
        try:
            print(f"Loading Faster-Whisper model: {self.model_size}")
            print(f"Device: {self.device}, Compute type: {self.compute_type}")
            
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=os.path.expanduser("~/.cache/faster-whisper")
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def transcribe(self, audio_path, language="vi", task="transcribe", 
                  beam_size=5, best_of=5, temperature=0, progress_callback=None):
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
        
        Returns:
            Dictionary with transcription results
        """
        try:
            segments, info = self.model.transcribe(
                audio_path,
                language=language,
                task=task,
                beam_size=beam_size,
                best_of=best_of,
                temperature=temperature,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=400,
                    threshold=0.5
                )
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