from transformers import WhisperForConditionalGeneration, WhisperProcessor
import librosa
import subprocess
import re
from Levenshtein import distance
from messageHandler import logger

class SpeechRecognition:
    def __init__(self):
        self.dict_rep = self.create_replace_dict()
        self.normalized_sampel_txt = self.create_sampel_text()
        self.model_id = 'Pardner/whisper-small-fa'
        try:
            logger.info(f"Loading model {self.model_id}")
            self.processor = WhisperProcessor.from_pretrained(self.model_id)
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_id)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
        self.out_put_audio_path = "tmp/output_audio.wav"
    
    @staticmethod
    def create_replace_dict():
        dict_rep = {
            1648: '',
            8203: '',
            1612: '',
            1615: '',
            1617: '',
            1616: '',
            1614: '',
            1618: '',
            8206: '',
            8207: '',
            173: '',
            1620: '',
            1600: '',
            1569: '',
            1611: '',
            8205: '',
            1644: '',
            37: 1642,
            1643: 47,
            1609: 1740,
            1610: 1740,
            1574: 1740,
            1603: 1705,
            1729: 1607,
            1730: 1607,
            1728: 1607,
            1749: 1607,
            1726: 1607,
            8204: 38,
            1571: 1575,
            1572: 1608,
            1573: 1575,
            1577: 1607,
            8211: 45,
            95: 45,
            39: '',  # new
            34: '',  # new
            # ص, ث → س
            1589: 1587,  # ص → س
            1579: 1587,  # ث → س
            # ط → ت
            1591: 1578,  # ط → ت
            # ض, ظ, ذ → ز
            1590: 1586,  # ض → ز
            1592: 1586,  # ظ → ز
            1584: 1586,  # ذ → ز
            # غ → ق
            1594: 1602,  # غ → ق
        }
        # farsi number 2 en number
        dict_rep.update({1632+i: 48+i for i in range(10)})
        # farsi number 2 en number
        dict_rep.update({1776+i: 48+i for i in range(10)})
        return dict_rep

    def create_sampel_text(self):
        normalized_text = []
        with open('sample.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            line = self.normalize_txt(line)
            normalized_text.append(line)
        return normalized_text
    
    def text_similarity(self,gt_norm, pred_norm):
        try:
            logger.info(f"Calculating similarity")
            lev_distance = distance(gt_norm, pred_norm)
            max_len = max(len(gt_norm), len(pred_norm))
            similarity_ratio = 1 - (lev_distance / max_len) if max_len != 0 else 1.0
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            similarity_ratio = 0.0
            lev_distance = 0
        
        return {
            'similarity_ratio': similarity_ratio,
            'levenshtein_distance': lev_distance,
            'normalized_truth': gt_norm,
            'normalized_predicted': pred_norm
        }

    
    def extract_audio(self, video_path):
        try:
            logger.info(f"Extracting audio from {video_path} to {self.out_put_audio_path}")
            command = [
                "ffmpeg",
                "-i", video_path,
                "-vn",              # No video
                "-acodec", "pcm_s16le",  # PCM codec for WAV (better for speech recognition)
                "-ar", "16000",     # Sample rate (16kHz is often sufficient for speech)
                "-ac", "1",         # Mono audio
                "-y",              # Overwrite output
                "-loglevel", "error",  # Only show errors
                self.out_put_audio_path
            ]
            subprocess.run(command, check=True, stderr=subprocess.PIPE)
            logger.info(f"Audio extracted successfully to {self.out_put_audio_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed with error:\n{e.stderr.decode()}")
            return False
        return True

    def normalize_txt(self,txt):
        # Remove punctuation and whitespace
        txt = re.sub(r'[^\w\s]', '', txt)  # Remove punctuation
        txt = re.sub(r'\s+', '', txt)      # Remove all whitespace
        out_txt = ''
        txt = txt.strip()  # remove space from start and end
        txt = re.sub(r'\s+', ' ', txt)  # Remove multiple spaces
        txt = txt.lower()
        for char in txt:
            if ord(char) in self.dict_rep: # replace char
                if self.dict_rep[ord(char)]:
                    out_txt += chr(self.dict_rep[ord(char)])
            else:  # remove multi space
                out_txt += char
        return out_txt
    
    def speech_recognition(self,video_path,user_txr):
        # Load audio
        try:
            logger.info(f"Starting speech recognition for video: {video_path}")
            self.extract_audio(video_path=video_path)
            audio, sr = librosa.load(self.out_put_audio_path, sr=16000)
            inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
            predicted_ids = self.model.generate(inputs["input_features"])
            pred_transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            normalized_pred_transcription = self.normalize_txt(pred_transcription)
            self.normalized_ground_truth = self.normalize_txt(user_txr)
            similarity = self.text_similarity(
                gt_norm=self.normalized_ground_truth,
                pred_norm=normalized_pred_transcription
            )
            logger.info("End of speech recognition")
        except Exception as e:
            logger.error(f"Error during speech recognition: {e}")
            similarity = {
                'similarity_ratio': 0.0,
                'levenshtein_distance': 0,
                'normalized_truth': '',
                'normalized_predicted': ''
            }
        # logger.info(f"Similarity result: {similarity}")
        print(f"similarity is {similarity['similarity_ratio']}")
        if similarity['similarity_ratio'] > 0.55:
           return "Accept"
        else: 
           return "Reject"
        #return similarity

if __name__ == "__main__":
    speech_rec = SpeechRecognition()
    speech_rec.speech_recognition(video_path="video_test/video_10318151.mp4", user_txr=speech_rec.normalized_sampel_txt[0])

