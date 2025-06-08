# 🗣️ Persian Speech Recognition

[![License](https://img.shields.io/github/license/IranTechNest/PersianSpeechRecognition)](LICENSE)
[![Stars](https://img.shields.io/github/stars/IranTechNest/PersianSpeechRecognition)](https://github.com/IranTechNest/PersianSpeechRecognition/stargazers)
[![Issues](https://img.shields.io/github/issues/IranTechNest/PersianSpeechRecognition)](https://github.com/IranTechNest/PersianSpeechRecognition/issues)

[![Contributors](https://img.shields.io/github/contributors/IranTechNest/PersianSpeechRecognition)](https://github.com/IranTechNest/PersianSpeechRecognition/graphs/contributors)

A Persian (Farsi) speech recognition system designed to convert spoken Persian language into text. This project leverages deep learning models to achieve high-accuracy transcription for Persian speech.

![alt text](data/ezgif-72c76af537cad5.gif)
## 🪄 Features

- **Pre-trained models** for Persian speech recognition
- **Dataset support** for training custom models

### 🛠️ Prerequisites
- Python 3.7+
- pip
- (Optional) CUDA for GPU acceleration

## ⚙️ 2 Environment Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/IranTechNest/PersianSpeechRecognition.git 
   ```
    then 

   ```bash
   cd PersianSpeechRecognition
    ```
2. Install dependencies:

    ```bash
    pip install -r requirements.txt 
    ```

## 🚀 Usage
1. Run app:

```bash
 python app.py
```
2. Speech Recognition
You can choose from the following supported models for speech recognition:

- `small`
- `small-v2`
- `medium`
- `large`
 ``` bash
from src.voice2txt import SpeechRecognition

speech = SpeechRecognition(model='medium')
pred_transcription = speech.speech_recognition("output_audio.wav") 

pred_transcription = [سلام من اومدم یک دو سه چهار]
```

## 🤝 Contributing [![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/IranTechNest/PersianSpeechRecognition/issues)


We love contributions from the community! Here's how you can help improve PersianSpeechRecognition:

- 🐛 **Report bugs** by [opening an issue](https://github.com/IranTechNest/PersianSpeechRecognition/issues)
- 💡 **Suggest new features** through GitHub issues
- 📝 **Improve documentation** (fix typos, add examples)
- 💻 **Contribute code** - see our [Contribution Guide](CONTRIBUTING.md)

First time contributing? Look for issues labeled [`good first issue`](https://github.com/IranTechNest/PersianSpeechRecognition/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22


## 🙏 Acknowledgments
Thanks to Mozilla Common Voice for open-source datasets

Thank of https://huggingface.co/masoudkaviani for funtuing the models

Inspired by DeepSpeech and HuggingFace Transformers

## 📬 Contact
For questions or support, please open an issue or contact the maintainers at IranTechNest.

## ℹ️ About PersianSpeechRecognition

[![Telegram](https://img.shields.io/badge/Join-Telegram%20Group-blue?logo=telegram)](https://t.me/+Cfc8CgrtIfRmNzRk)

A Persian speech recognition toolkit for developers and researchers...
## 📌 Citation
``` 
@misc{raminram_persianspeech,
  title        = {PersianSpeechRecognition},
  author       = {Ramin Rahimi},
  howpublished = {\url{https://github.com/IranTechNest/PersianSpeechRecognition}},
  year         = {2025},
  note         = {Accessed: 2025-06-02},
  email        = {raminram6970@gmail.com}
}

