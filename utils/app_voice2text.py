import sounddevice as sd
import soundfile as sf
import numpy as np
import os
from messageHandler import logger
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import librosa
from messageHandler import logger
import noisereduce as nr

class AppSpeechRecognition:
    def __init__(self):
        self.model_id = 'Pardner/whisper-small-fa'
        try:
            logger.info(f"Loading model {self.model_id}")
            self.processor = WhisperProcessor.from_pretrained(self.model_id)
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_id)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
        self.create_output_path("tmp/output_audio.wav")
        self.recording = None
        self.fs = 16000  # Sample rate
        self.is_recording = False

    def create_output_path(self, path):
        """Create output directory if it doesn't exist"""
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        self.out_put_audio_path = path
    
        
    def start_recording(self):
        """Start recording from microphone with improved quality"""
        try:
            logger.info("Starting high-quality microphone recording")
            self.recording = []
            self.is_recording = True
            
            # Audio quality parameters
            self.fs = 44100  # Higher sample rate (CD quality)
            self.blocksize = 2048  # Optimal buffer size
            self.device = self.get_best_input_device()  # Select best available microphone
            
            def callback(indata, frames, time, status):
                if status:
                    logger.warning(f"Recording status: {status}")
                if self.is_recording:
                    # Apply simple noise reduction by clipping very quiet samples
                    indata = np.clip(indata, -0.1, 0.1)  # Adjust threshold as needed
                    self.recording.append(indata.copy())
            
            # Start high-quality recording
            self.stream = sd.InputStream(
                samplerate=self.fs,
                channels=1,
                dtype='float32',
                blocksize=self.blocksize,
                device=self.device,
                callback=callback,
                latency='high'  # Better for recording quality
            )
            self.stream.start()
            return True
        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            return False

    def get_best_input_device(self):
        """Select the best available input device"""
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        
        # Prefer devices with high sample rates and low latency
        best_device = None
        best_score = -1
        
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:  # Is input device
                score = 0
                # Prefer devices that support our target sample rate
                if self.fs <= device['default_samplerate']:
                    score += 2
                # Prefer ASIO/WASAPI drivers on Windows
                if hostapis[device['hostapi']]['name'] in ['ASIO', 'WASAPI']:
                    score += 1
                # Prefer devices with lower latency
                if device['default_low_input_latency'] < 0.1:
                    score += 1
                
                if score > best_score:
                    best_score = score
                    best_device = i
        
        return best_device if best_device is not None else sd.default.device[0]

    def stop_recording(self):
        """Stop recording and save with quality enhancements"""
        try:
            logger.info("Stopping recording and saving with quality processing")
            self.is_recording = False
            
            if hasattr(self, 'stream'):
                self.stream.stop()
                self.stream.close()
            
            if self.recording:
                # Concatenate all recorded chunks
                audio_data = np.concatenate(self.recording, axis=0)
                
                # Normalize audio to -3dB to prevent clipping
                peak = np.max(np.abs(audio_data))
                if peak > 0:
                    audio_data = audio_data * (0.7 / peak)  # -3dB headroom
                
                # Create directory if needed
                os.makedirs(os.path.dirname(self.out_put_audio_path), exist_ok=True)
                
                # Save as high-quality WAV
                sf.write(
                    self.out_put_audio_path,
                    audio_data,
                    self.fs,
                    subtype='PCM_24'  # Higher bit depth
                )
                logger.info(f"High-quality recording saved to {self.out_put_audio_path}")
                
                # Resample to 16kHz for Whisper if needed
                if self.fs != 16000:
                    audio_data = librosa.resample(
                        audio_data.T,
                        orig_sr=self.fs,
                        target_sr=16000
                    ).T
                    self.fs = 16000
                # Apply noise reduction
                if audio_data.ndim > 1:
                    audio_data = audio_data[:, 0]  # convert to mono

                audio_data = nr.reduce_noise(y=audio_data, sr=self.fs)
                return True
            return False
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            return False
    