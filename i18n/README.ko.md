[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)



[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# Clip-GPT-Captioning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Status](https://img.shields.io/badge/README-Expanded-success)
![Repo Layout](https://img.shields.io/badge/Layout-Root%20Scripts-informational)
![Legacy Scripts](https://img.shields.io/badge/Legacy%20Scripts-Present-orange)
![i18n](https://img.shields.io/badge/i18n-Enabled-brightgreen)
![Maintained Path](https://img.shields.io/badge/Video-v2c.py-2ea44f)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-2ea44f?style=flat-square)
![Issues](https://img.shields.io/github/issues-raw/lachlanchen/VideoCaptionerWithClip?style=flat-square)
![Last Commit](https://img.shields.io/github/last-commit/lachlanchen/VideoCaptionerWithClip?style=flat-square)

---

## 🧭 빠른 네비게이션

| 섹션 | 용도 |
|---|---|
| Snapshot | 저장소 범위와 현재 스크립트 구성을 확인 |
| Overview | 목표와 기능을 확인 |
| Usage | 정확한 CLI/API 사용 흐름을 따름 |
| Troubleshooting | 일반적인 런타임 이슈를 빠르게 해결 |
| Roadmap | 정리/개선 예정 항목을 확인 |

---

OpenAI CLIP의 시각 임베딩과 GPT 계열 언어 모델을 결합해 이미지 및 비디오의 자연어 캡션을 생성하는 Python 도구 모음입니다.

## 🧭 Snapshot

| 범주 | 내용 |
|---|---|
| 작업 범위 | 이미지 및 비디오 캡셔닝 |
| 핵심 산출물 | SRT 자막, JSON 자막, 캡션이 붙은 이미지 |
| 주요 스크립트 | `i2c.py`, `v2c.py`, `image2caption.py` |
| 레거시 경로 | `video2caption.py` 및 버전별 동생 파일들 (기록용으로 유지) |
| 데이터셋 흐름 | `data/raw/results.csv` + `data/raw/flickr30k_images/` |

## ✨ 개요

이 저장소는 다음 기능을 제공합니다:

- 이미지 캡션 생성 및 비디오 자막 생성을 위한 추론 스크립트.
- CLIP 시각 임베딩을 GPT-2 토큰 임베딩으로 매핑하는 학습 파이프라인.
- Flickr30k 스타일 데이터셋 생성을 위한 유틸리티.
- 필요한 모델 가중치가 없을 때 지원되는 크기의 체크포인트를 자동으로 다운로드.
- `i18n/` 하위의 다국어 README 버전(위의 언어 바 참조).

현재 구현에는 신규 스크립트와 레거시 스크립트가 모두 포함되어 있습니다. 일부 레거시 파일은 참조용으로 보존되어 있으며 아래에 정리되어 있습니다.

## 🚀 특징

- `image2caption.py`를 통한 단일 이미지 캡션 생성.
- `v2c.py` 또는 `video2caption.py`를 통한 비디오 캡션 생성(균등 프레임 샘플링).
- 실행 옵션 커스터마이즈:
  - 프레임 수.
  - 모델 크기.
  - 샘플링 온도.
  - 체크포인트 이름.
- 멀티프로세싱/스레드 기반 캡션 처리로 비디오 추론 속도 향상.
- 출력 산출물:
  - SRT 자막 파일 (`.srt`).
  - `v2c.py`에서 생성되는 JSON 전사본 (`.json`).
- CLIP+GPT2 매핑 실험을 위한 학습 및 평가 진입점.

### 한눈에 보기

| 영역 | 주요 스크립트 | 비고 |
|---|---|---|
| 이미지 캡셔닝 | `image2caption.py`, `i2c.py`, `predict.py` | CLI + 재사용 가능한 클래스 |
| 비디오 캡셔닝 | `v2c.py` | 유지보수 권장 경로 |
| 레거시 비디오 플로우 | `video2caption.py`, `video2caption_v1.1.py` | 장비 특화 가정값 포함 |
| 데이터셋 구축 | `dataset_generation.py` | `data/processed/dataset.pkl` 생성 |
| 학습/평가 | `training.py`, `evaluate.py` | CLIP+GPT2 매핑 사용 |

## 🧱 아키텍처 (개요)

`model/model.py`의 핵심 모델은 세 부분으로 구성됩니다:

1. `ImageEncoder`: CLIP 이미지 임베딩 추출.
2. `Mapping`: CLIP 임베딩을 GPT 접두사 임베딩 시퀀스로 투영.
3. `TextDecoder`: GPT-2 언어 모델 헤드가 자기회귀 방식으로 캡션 토큰 생성.

학습(`Net.train_forward`)은 미리 계산된 CLIP 이미지 임베딩 + 토큰화된 캡션을 사용합니다.
추론(`Net.forward`)은 PIL 이미지를 받아 EOS 또는 `max_len`에 도달할 때까지 토큰을 디코딩합니다.

### 데이터 흐름

1. 데이터셋 준비: `dataset_generation.py`가 `data/raw/results.csv`와 `data/raw/flickr30k_images/`의 이미지를 읽어 `data/processed/dataset.pkl`을 작성.
2. 학습: `training.py`가 피클 튜플 `(image_name, image_embedding, caption)`을 로드하고 매퍼/디코더 계층을 학습.
3. 평가: `evaluate.py`가 보류된 테스트 이미지에 대한 예측 캡션을 출력.
4. 추론 실행:
   - 이미지: `image2caption.py` / `predict.py` / `i2c.py`.
   - 비디오: `v2c.py`(권장), `video2caption.py`(레거시).

## 🗂️ 프로젝트 구조

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

## 📋 선행 조건

- Python `3.10+` 권장.
- CUDA 지원 GPU는 선택이지만, 학습과 대형 모델 추론에서는 강력 권장됩니다.
- 현재 스크립트에서는 `ffmpeg`가 직접 요구되지 않습니다(OpenCV가 프레임 추출에 사용됨).
- Hugging Face / Google Drive에서 모델/체크포인트를 처음 내려받을 때 인터넷 연결이 필요합니다.

현재 `requirements.txt` 또는 `pyproject.toml` 같은 잠금 파일은 없으므로 의존성은 임포트 항목을 기준으로 추론됩니다.

## 🛠️ 설치

### 현재 저장소 레이아웃 기준 정식 설치

```bash

git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers pillow matplotlib numpy tqdm opencv-python pandas wandb gdown
```

### 원본 README 설치 스니펫(원문 보존)

이전 README는 중간에 멈춘 상태였습니다. 원본 역사적 내용은 다음과 같이 그대로 보존됩니다:

```bash
git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip/src
```

참고: 현재 저장소 구조에서는 스크립트가 루트에 있으며 `src/` 하위가 아닙니다.

## ▶️ 빠른 시작

| 목표 | 명령 |
|---|---|
| 이미지 캡션 생성 | `python image2caption.py -I /path/to/image.jpg -S L -C model.pt` |
| 비디오 캡션 생성 | `python v2c.py -V /path/to/video.mp4 -N 10` |
| 데이터셋 빌드 | `python dataset_generation.py` |

### 이미지 캡셔닝(빠른 실행)

```bash
python image2caption.py -I /path/to/image.jpg -S L -C model.pt
```

### 비디오 캡셔닝(권장 경로)

```bash
python v2c.py -V /path/to/video.mp4 -N 10
```

## 🎯 사용법

### 1. 이미지 캡셔닝 (`image2caption.py`)

```bash
python image2caption.py \
  -I /path/to/image.jpg \
  -S L \
  -C model.pt \
  -R ./data/result/prediction \
  -T 1.0
```

인자:

- `-I, --img-path`: 입력 이미지 경로.
- `-S, --size`: 모델 크기 (`S` 또는 `L`).
- `-C, --checkpoint-name`: `weights/{small|large}` 내 체크포인트 파일명.
- `-R, --res-path`: 결과 캡션 이미지가 저장될 디렉터리.
- `-T, --temperature`: 샘플링 온도.

### 2. 대체 이미지 CLI (`predict.py`)

```bash
python predict.py \
  -I /path/to/image.jpg \
  -S L \
  -C model.pt \
  -R ./data/result/prediction \
  -T 1.0
```

`predict.py`는 `image2caption.py`와 기능적으로 유사하지만, 출력 텍스트 포맷이 약간 다릅니다.

### 3. 이미지 캡셔닝 클래스 API (`i2c.py`)

```bash
python i2c.py -I /path/to/image.jpg -S L -C model.pt -R ./data/result/prediction -T 1.0
```

또는 직접 스크립트에서 임포트:

```python
from i2c import ImageCaptioner

captioner = ImageCaptioner(model_size="L", checkpoint_name="model.pt")
captioner.set_image_path("/path/to/image.jpg")
caption = captioner.generate_caption(save_image=True)
print(caption)
```

### 4. 비디오 자막 + JSON (`v2c.py`)

```bash
python v2c.py -V /path/to/video.mp4 -N 10
```

입력 비디오와 동일 폴더에 출력:

- `<video_basename>_caption.srt`
- `<video_basename>_caption.json`
- `<video_basename>_captioning_frames/`

### 5. 대체 비디오 파이프라인 (`video2caption.py`)

```bash
python video2caption.py -V /path/to/video.mp4 -N 10
```

중요: 이 스크립트는 현재 장비 특화 하드코딩 경로를 포함합니다.

- Python 기본 경로: `/home/lachlan/miniconda3/envs/caption/bin/python`
- 캡션 스크립트 경로: `/home/lachlan/Projects/image_captioning/clip-gpt-captioning/src/image2caption.py`

이 경로를 의도적으로 유지하지 않는 이상 `v2c.py` 사용을 권장합니다.

### 6. 레거시 변형 (`video2caption_v1.1.py`)

이 스크립트는 기록 보존 목적으로만 남겨둡니다. 실제 사용은 `v2c.py`를 우선하세요.

### 7. 데이터셋 생성

```bash
python dataset_generation.py
```

필수 입력:

- `data/raw/results.csv` (파이프(`|`) 구분 캡션 테이블)
- `data/raw/flickr30k_images/` (CSV에서 참조하는 이미지 파일들)

출력:

- `data/processed/dataset.pkl`

### 8. 학습

```bash
python training.py -S L -C model.pt
```

학습은 기본적으로 Weights & Biases(`wandb`) 로깅을 사용합니다.

### 9. 평가

```bash
python evaluate.py \
  -I ./data/raw/flickr30k_images \
  -R ./data/result/eval \
  -S L \
  -C model.pt \
  -T 1.0
```

평가 시 예측 캡션이 테스트 이미지에 렌더링되어 다음 위치에 저장됩니다.

- `<res-path>/<checkpoint_name_without_ext>_<SIZE>/`

## ⚙️ 설정

모델 설정은 `utils/config.py`에서 정의됩니다.

| Config | CLIP 백본 | GPT 모델 | 가중치 폴더 |
|---|---|---|---|
| `ConfigS` | `openai/clip-vit-base-patch32` | `gpt2` | `weights/small` |
| `ConfigL` | `openai/clip-vit-large-patch14` | `gpt2-medium` | `weights/large` |

설정 클래스의 주요 기본값:

| 필드 | `ConfigS` | `ConfigL` |
|---|---:|---:|
| `epochs` | 150 | 120 |
| `lr` | 3e-3 | 5e-3 |
| `batch_size_exp` | 6 | 5 |
| `ep_len` | 4 | 4 |
| `max_len` | 40 | 40 |

체크포인트 자동 다운로드 ID는 `utils/downloads.py`에 있습니다.

| 크기 | Google Drive ID |
|---|---|
| `L` | `1Gh32arzhW06C1ZJyzcJSSfdJDi3RgWoG` |
| `S` | `1pSQruQyg8KJq6VmzhMLFbT_VaHJMdlWF` |

## 📦 출력 파일

### 이미지 추론

- `--res-path` 위치에 캡션 텍스트가 오버레이된 이미지가 저장됩니다.
- 파일명 규칙: `<input_stem>-R<SIZE>.jpg`.

### 비디오 추론 (`v2c.py`)

- SRT: `<video_stem>_caption.srt`
- JSON: `<video_stem>_caption.json`
- 프레임 이미지: `<video_stem>_captioning_frames/`

JSON 예시:

```json
{
  "start": "00:00:03,200",
  "end": "00:00:03,700",
  "lang": "en",
  "text": "A dog running through a field."
}
```

## 🧪 사용 예시

### 빠른 이미지 캡션 예시

```bash
python image2caption.py -I ./examples/dog.jpg -S S -C model.pt
```

예상 동작:

- `weights/small/model.pt`가 없으면 자동 다운로드됩니다.
- 캡션이 들어간 이미지가 기본적으로 `./data/result/prediction`에 저장됩니다.
- 캡션 텍스트가 stdout으로 출력됩니다.

### 빠른 비디오 캡션 예시

```bash
python v2c.py -V ./examples/demo.mp4 -N 8
```

예상 동작:

- 8개의 균일 샘플링 프레임이 캡션됩니다.
- 입력 비디오 옆에 `.srt`와 `.json` 파일이 생성됩니다.

### 엔드투엔드 학습/평가 순서

```bash
python dataset_generation.py
python training.py -S L -C model.pt
python evaluate.py -I ./data/raw/flickr30k_images -R ./data/result/eval -S L -C model.pt -T 1.0
```

## 🧭 개발 노트

- `v2c.py`, `video2caption.py`, `video2caption_v1.*` 간에 레거시 중첩이 존재합니다.
- `video2caption_v1.0_not_work.py`는 의도적으로 동작하지 않는 레거시 코드로 보존됩니다.
- `training.py`는 현재 `config = ConfigL() if args.size.upper() else ConfigS()`를 사용해, 빈 값이 아닌 `--size`가 들어가면 항상 `ConfigL`로 해석됩니다.
- `model/trainer.py`는 `test_step`에서 `self.dataset`을 사용하지만, 초기화는 `self.test_dataset`에 할당되어 있어 학습 실행 시 샘플링이 깨질 수 있습니다.
- `video2caption_v1.1.py`는 `self.config.transform`을 참조하지만, `ConfigS`/`ConfigL`에는 `transform`이 정의되어 있지 않습니다.
- 이 저장소 스냅샷에는 현재 CI/테스트 스위트가 정의되어 있지 않습니다.
- i18n 참고: 상단의 언어 링크가 존재하며 번역 파일은 `i18n/`에 추가될 수 있습니다.
- 현재 상태: 언어 바에는 `i18n/README.ru.md` 링크가 있으나, 현재 스냅샷에는 해당 파일이 없습니다.

## 🩺 트러블슈팅

- `AssertionError: Image does not exist`
  - `-I/--img-path`가 실제 파일을 가리키는지 확인하세요.
- `Dataset file not found. Downloading...`
  - `MiniFlickrDataset`는 `data/processed/dataset.pkl`이 없을 때 해당 메시지를 띄웁니다. 먼저 `python dataset_generation.py` 실행.
- `Path to the test image folder does not exist`
  - `evaluate.py -I`가 기존 폴더를 가리키는지 확인.
- 초기 실행이 느리거나 실패
  - 첫 실행에서 Hugging Face 모델을 다운로드하고 Google Drive 체크포인트를 가져올 수 있습니다.
- `video2caption.py`가 빈 캡션을 반환
  - 하드코딩된 스크립트 경로와 Python 실행 경로를 확인하거나 `v2c.py`로 전환.
- `wandb`가 학습 중 로그인 요청
  - `wandb login` 실행 또는 필요 시 `training.py`에서 로깅 비활성화.

## 🛣️ 로드맵

- 재현 가능한 설치를 위한 의존성 잠금 파일(`requirements.txt` 또는 `pyproject.toml`) 추가.
- 중복 비디오 파이프라인을 하나의 유지보수 경로로 통합.
- 레거시 스크립트의 하드코딩 경로 제거.
- `training.py`와 `model/trainer.py`의 알려진 학습/평가 엣지 케이스 버그 수정.
- 자동화 테스트 및 CI 추가.
- 언어 바에서 참조되는 번역 README 파일을 `i18n/`에 실제로 채움.

## 🤝 기여

기여를 환영합니다. 권장 작업 흐름:

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

모델 동작을 변경한다면 다음을 포함하세요:

- 재현 가능한 실행 명령.
- 변경 전후 샘플 출력.
- 체크포인트 또는 데이터셋 가정에 대한 노트.

---

## 📄 라이선스

현재 저장소 스냅샷에는 라이선스 파일이 없습니다.

참고: `LICENSE` 파일이 추가될 때까지 재사용 및 배포 조건이 정해져 있지 않습니다.


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
