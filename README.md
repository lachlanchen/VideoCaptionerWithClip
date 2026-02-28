[English](README.md) · [العربية](i18n/README.ar.md) · [Español](i18n/README.es.md) · [Français](i18n/README.fr.md) · [日本語](i18n/README.ja.md) · [한국어](i18n/README.ko.md) · [Tiếng Việt](i18n/README.vi.md) · [中文 (简体)](i18n/README.zh-Hans.md) · [中文（繁體）](i18n/README.zh-Hant.md) · [Deutsch](i18n/README.de.md) · [Русский](i18n/README.ru.md)


# Clip-GPT-Captioning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Status](https://img.shields.io/badge/README-Expanded-success)
![Repo Layout](https://img.shields.io/badge/Layout-Root%20Scripts-informational)
![Legacy Scripts](https://img.shields.io/badge/Legacy%20Scripts-Present-orange)
![i18n](https://img.shields.io/badge/i18n-Enabled-brightgreen)
![Maintained Path](https://img.shields.io/badge/Video-v2c.py-2ea44f)

A Python toolkit for generating natural-language captions on images and videos by combining OpenAI CLIP visual embeddings with a GPT-style language model.

## ✨ Overview

This repository provides:

- Inference scripts for image captioning and video subtitle generation.
- A training pipeline that learns a mapping from CLIP visual embeddings to GPT-2 token embeddings.
- Dataset generation utilities for Flickr30k-style data.
- Automatic checkpoint download for supported model sizes when weights are missing.
- Multilingual README variants under `i18n/` (see language bar above).

The current implementation includes both newer and legacy scripts. Some legacy files are kept for reference and are documented below.

## 🚀 Features

- Single-image captioning via `image2caption.py`.
- Video captioning (uniform frame sampling) via `v2c.py` or `video2caption.py`.
- Customizable runtime options:
  - Number of frames.
  - Model size.
  - Sampling temperature.
  - Checkpoint name.
- Multiprocessing/threaded captioning for faster video inference.
- Output artifacts:
  - SRT subtitle files (`.srt`).
  - JSON transcripts (`.json`) in `v2c.py`.
- Training and evaluation entry points for CLIP+GPT2 mapping experiments.

### At a glance

| Area | Primary script(s) | Notes |
|---|---|---|
| Image captioning | `image2caption.py`, `i2c.py`, `predict.py` | CLI + reusable class |
| Video captioning | `v2c.py` | Recommended maintained path |
| Legacy video flow | `video2caption.py`, `video2caption_v1.1.py` | Contains machine-specific assumptions |
| Dataset build | `dataset_generation.py` | Produces `data/processed/dataset.pkl` |
| Train / eval | `training.py`, `evaluate.py` | Uses CLIP+GPT2 mapping |

## 🧱 Architecture (High Level)

The core model in `model/model.py` has three parts:

1. `ImageEncoder`: extracts CLIP image embedding.
2. `Mapping`: projects CLIP embedding into a GPT prefix embedding sequence.
3. `TextDecoder`: GPT-2 language model head that autoregressively generates caption tokens.

Training (`Net.train_forward`) uses precomputed CLIP image embeddings + tokenized captions.
Inference (`Net.forward`) uses a PIL image and decodes tokens until EOS or `max_len`.

### Data flow

1. Prepare dataset: `dataset_generation.py` reads `data/raw/results.csv` and images in `data/raw/flickr30k_images/`, writes `data/processed/dataset.pkl`.
2. Train: `training.py` loads pickled tuples `(image_name, image_embedding, caption)` and trains mapper/decoder layers.
3. Evaluate: `evaluate.py` renders generated captions over held-out test images.
4. Serve inference:
   - image: `image2caption.py` / `predict.py` / `i2c.py`.
   - video: `v2c.py` (recommended), `video2caption.py` (legacy).

## 🗂️ Project Structure

```text
VideoCaptionerWithClip/
├── README.md
├── image2caption.py               # Single-image caption CLI
├── predict.py                     # Alternate single-image caption CLI
├── i2c.py                         # Reusable ImageCaptioner class + CLI
├── v2c.py                         # Video -> SRT + JSON (threaded frame captioning)
├── video2caption.py               # Alternate video -> SRT implementation (legacy constraints)
├── video2caption_v1.1.py          # Older variant
├── video2caption_v1.0_not_work.py # Explicitly marked non-working legacy file
├── training.py                    # Model training entrypoint
├── evaluate.py                    # Test-split evaluation and rendered outputs
├── dataset_generation.py          # Builds data/processed/dataset.pkl
├── data/
│   ├── __init__.py
│   └── dataset.py                 # Dataset + DataLoader helpers
├── model/
│   ├── __init__.py
│   ├── model.py                   # CLIP encoder + mapping + GPT2 decoder
│   └── trainer.py                 # Training/validation/test utility class
├── utils/
│   ├── __init__.py
│   ├── config.py                  # ConfigS / ConfigL defaults
│   ├── downloads.py               # Google Drive checkpoint downloader
│   └── lr_warmup.py               # LR warmup schedule
├── i18n/                          # Multilingual README variants
└── .auto-readme-work/             # Auto-README pipeline artifacts
```

## 📋 Prerequisites

- Python `3.10+` recommended.
- CUDA-capable GPU is optional but strongly recommended for training and large-model inference.
- `ffmpeg` is not directly required by current scripts (OpenCV is used for frame extraction).
- Internet access is needed the first time models/checkpoints are downloaded from Hugging Face / Google Drive.

No lockfile is currently present (`requirements.txt` / `pyproject.toml` missing), so dependencies are inferred from imports.

## 🛠️ Installation

### Canonical setup from current repository layout

```bash
git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers pillow matplotlib numpy tqdm opencv-python pandas wandb gdown
```

### Original README installation snippet (preserved)

The previous README ended mid-block. The original commands are preserved below exactly as source-of-truth historical content:

```bash
git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip/src
```

Note: the current repository snapshot places scripts at repo root, not under `src/`.

## ▶️ Quick Start

### Image captioning (quick run)

```bash
python image2caption.py -I /path/to/image.jpg -S L -C model.pt
```

### Video captioning (recommended path)

```bash
python v2c.py -V /path/to/video.mp4 -N 10
```

## 🎯 Usage

### 1. Image captioning (`image2caption.py`)

```bash
python image2caption.py \
  -I /path/to/image.jpg \
  -S L \
  -C model.pt \
  -R ./data/result/prediction \
  -T 1.0
```

Arguments:

- `-I, --img-path`: input image path.
- `-S, --size`: model size (`S` or `L`).
- `-C, --checkpoint-name`: checkpoint filename in `weights/{small|large}`.
- `-R, --res-path`: output directory for rendered captioned image.
- `-T, --temperature`: sampling temperature.

### 2. Alternate image CLI (`predict.py`)

```bash
python predict.py \
  -I /path/to/image.jpg \
  -S L \
  -C model.pt \
  -R ./data/result/prediction \
  -T 1.0
```

`predict.py` is functionally similar to `image2caption.py`; output text formatting differs slightly.

### 3. Image captioning class API (`i2c.py`)

```bash
python i2c.py -I /path/to/image.jpg -S L -C model.pt -R ./data/result/prediction -T 1.0
```

Or import in your own script:

```python
from i2c import ImageCaptioner

captioner = ImageCaptioner(model_size="L", checkpoint_name="model.pt")
captioner.set_image_path("/path/to/image.jpg")
caption = captioner.generate_caption(save_image=True)
print(caption)
```

### 4. Video to subtitles + JSON (`v2c.py`)

```bash
python v2c.py -V /path/to/video.mp4 -N 10
```

Outputs next to the input video:

- `<video_basename>_caption.srt`
- `<video_basename>_caption.json`
- `<video_basename>_captioning_frames/`

### 5. Alternate video pipeline (`video2caption.py`)

```bash
python video2caption.py -V /path/to/video.mp4 -N 10
```

Important: this script currently contains machine-specific hardcoded paths:

- Python path default: `/home/lachlan/miniconda3/envs/caption/bin/python`
- Caption script path: `/home/lachlan/Projects/image_captioning/clip-gpt-captioning/src/image2caption.py`

Use `v2c.py` unless you intentionally maintain these paths.

### 6. Legacy variant (`video2caption_v1.1.py`)

This script is retained for historical reference. Prefer `v2c.py` for active use.

### 7. Dataset generation

```bash
python dataset_generation.py
```

Expected raw inputs:

- `data/raw/results.csv` (pipe-separated captions table).
- `data/raw/flickr30k_images/` (image files referenced by CSV).

Output:

- `data/processed/dataset.pkl`

### 8. Training

```bash
python training.py -S L -C model.pt
```

Training uses Weights & Biases (`wandb`) logging by default.

### 9. Evaluation

```bash
python evaluate.py \
  -I ./data/raw/flickr30k_images \
  -R ./data/result/eval \
  -S L \
  -C model.pt \
  -T 1.0
```

Evaluation renders predicted captions onto test images and saves them under:

- `<res-path>/<checkpoint_name_without_ext>_<SIZE>/`

## ⚙️ Configuration

Model configurations are defined in `utils/config.py`:

| Config | CLIP backbone | GPT model | Weights dir |
|---|---|---|---|
| `ConfigS` | `openai/clip-vit-base-patch32` | `gpt2` | `weights/small` |
| `ConfigL` | `openai/clip-vit-large-patch14` | `gpt2-medium` | `weights/large` |

Key defaults from config classes:

| Field | `ConfigS` | `ConfigL` |
|---|---:|---:|
| `epochs` | 150 | 120 |
| `lr` | 3e-3 | 5e-3 |
| `batch_size_exp` | 6 | 5 |
| `ep_len` | 4 | 4 |
| `max_len` | 40 | 40 |

Checkpoint auto-download IDs are in `utils/downloads.py`:

| Size | Google Drive ID |
|---|---|
| `L` | `1Gh32arzhW06C1ZJyzcJSSfdJDi3RgWoG` |
| `S` | `1pSQruQyg8KJq6VmzhMLFbT_VaHJMdlWF` |

## 📦 Output Files

### Image inference

- Saved image with overlaid/generated title at `--res-path`.
- Filename pattern: `<input_stem>-R<SIZE>.jpg`.

### Video inference (`v2c.py`)

- SRT: `<video_stem>_caption.srt`
- JSON: `<video_stem>_caption.json`
- Frame images: `<video_stem>_captioning_frames/`

Example JSON element:

```json
{
  "start": "00:00:03,200",
  "end": "00:00:03,700",
  "lang": "en",
  "text": "A dog running through a field."
}
```

## 🧪 Examples

### Quick image caption example

```bash
python image2caption.py -I ./examples/dog.jpg -S S -C model.pt
```

Expected behavior:

- If `weights/small/model.pt` is missing, it is downloaded.
- A captioned image is written to `./data/result/prediction` by default.
- Caption text is printed to stdout.

### Quick video caption example

```bash
python v2c.py -V ./examples/demo.mp4 -N 8
```

Expected behavior:

- 8 uniformly sampled frames are captioned.
- `.srt` and `.json` files are generated alongside the input video.

### End-to-end training/evaluation sequence

```bash
python dataset_generation.py
python training.py -S L -C model.pt
python evaluate.py -I ./data/raw/flickr30k_images -R ./data/result/eval -S L -C model.pt -T 1.0
```

## 🧭 Development Notes

- Legacy overlap exists across `v2c.py`, `video2caption.py`, and `video2caption_v1.*`.
- `video2caption_v1.0_not_work.py` is intentionally retained as non-working legacy code.
- `training.py` currently selects `ConfigL()` via `config = ConfigL() if args.size.upper() else ConfigS()`, which always resolves to `ConfigL` for non-empty `--size` values.
- `model/trainer.py` uses `self.dataset` in `test_step`, while initializer assigns `self.test_dataset`; this can break sampling in training runs unless adjusted.
- `video2caption_v1.1.py` references `self.config.transform`, but `ConfigS`/`ConfigL` do not define `transform`.
- No CI/test suite is currently defined in this repository snapshot.
- i18n note: language links are present at the top of this README; translated files may be added under `i18n/`.
- Current state note: the language bar links `i18n/README.ru.md`, but that file is not present in this snapshot.

## 🩺 Troubleshooting

- `AssertionError: Image does not exist`
  - Confirm `-I/--img-path` points to a valid file.
- `Dataset file not found. Downloading...`
  - `MiniFlickrDataset` raises this when `data/processed/dataset.pkl` is missing; run `python dataset_generation.py` first.
- `Path to the test image folder does not exist`
  - Confirm `evaluate.py -I` points to an existing folder.
- Slow or failing first run
  - Initial run downloads Hugging Face models and may download checkpoints from Google Drive.
- `video2caption.py` returns empty captions
  - Verify hardcoded script path and Python executable path, or switch to `v2c.py`.
- `wandb` prompts for login during training
  - Run `wandb login` or disable logging manually in `training.py` if needed.

## 🛣️ Roadmap

- Add dependency lockfiles (`requirements.txt` or `pyproject.toml`) for reproducible installs.
- Unify duplicate video pipelines into one maintained implementation.
- Remove hardcoded machine paths from legacy scripts.
- Fix known training/evaluation edge-case bugs in `training.py` and `model/trainer.py`.
- Add automated tests and CI.
- Populate `i18n/` with translated README files referenced in the language bar.

## 🤝 Contributing

Contributions are welcome. Suggested workflow:

```bash
# 1) Fork and clone
git clone git@github.com:<your-user>/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip

# 2) Create a feature branch
git checkout -b feat/your-change

# 3) Make changes and commit
git add .
git commit -m "feat: describe your change"

# 4) Push and open a PR
git push origin feat/your-change
```

If you change model behavior, include:

- Reproducible command(s).
- Before/after sample outputs.
- Notes on checkpoint or dataset assumptions.

## 🙌 Support

No explicit donation/sponsorship configuration is present in the current repository snapshot.

If sponsorship links are added later, they should be preserved in this section.

## 📄 License

No license file is present in the current repository snapshot.

Assumption note: until a `LICENSE` file is added, reuse/distribution terms are undefined.
