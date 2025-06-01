from transformers import WhisperForConditionalGeneration, WhisperProcessor
import librosa
from messageHandler import logger

class SpeechRecognition:
    def __init__(self):
        self.model_id = 'Pardner/whisper-small-fa'
        try:
            logger.info(f"Loading model {self.model_id}")
            self.processor = WhisperProcessor.from_pretrained(self.model_id)
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_id)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
        self.out_put_audio_path = "tmp/output_audio.wav"
    
    

    
    
    def speech_recognition(self,audio_path):
        audio, sr = librosa.load(audio_path, sr=16000)
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        predicted_ids = self.model.generate(inputs["input_features"])
        pred_transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        a = 0
    

if __name__ == "__main__":
    speech_rec = SpeechRecognition()
    speech_rec.speech_recognition(audio_path="tmp/output_audio.wav")

