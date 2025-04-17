# Clip‑GPT‑Captioning

A Python toolkit for generating natural‑language captions on images and videos by combining OpenAI’s CLIP for visual embeddings with a GPT‑style language model.

---

## 🚀 Features

- **Single‑image captioning** via `image2caption.py`  
- **Video captioning** (uniform frame sampling) via `v2c.py` or `video2caption.py`  
- **Customizable**  
  - Number of frames, model size, temperature, checkpoint name  
- **Multiprocessing** for faster inference on videos  
- **Outputs**  
  - SRT subtitle files (`.srt`)  
  - JSON transcripts (`.json`)

---

## 🔧 Installation

1. **Clone the repo**  
   ```bash
   git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
   cd VideoCaptionerWithClip/src

