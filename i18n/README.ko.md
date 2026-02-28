[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


# Clip-GPT-Captioning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Status](https://img.shields.io/badge/README-Expanded-success)
![Repo Layout](https://img.shields.io/badge/Layout-Root%20Scripts-informational)
![Legacy Scripts](https://img.shields.io/badge/Legacy%20Scripts-Present-orange)
![i18n](https://img.shields.io/badge/i18n-Enabled-brightgreen)
![Maintained Path](https://img.shields.io/badge/Video-v2c.py-2ea44f)

OpenAI CLIP 시각 임베딩과 GPT 스타일 언어 모델을 결합해 이미지와 비디오에 자연어 캡션을 생성하는 Python 툴킷입니다.

## ✨ 개요

이 저장소는 다음을 제공합니다.

- 이미지 캡셔닝 및 비디오 자막 생성을 위한 추론 스크립트.
- CLIP 시각 임베딩에서 GPT-2 토큰 임베딩으로의 매핑을 학습하는 학습 파이프라인.
- Flickr30k 스타일 데이터를 위한 데이터셋 생성 유틸리티.
- 가중치가 없을 때 지원되는 모델 크기에 대해 체크포인트 자동 다운로드.
- `i18n/` 하위의 다국어 README 변형(상단 언어 바 참조).

현재 구현에는 최신 스크립트와 레거시 스크립트가 모두 포함되어 있습니다. 일부 레거시 파일은 참고용으로 유지되며 아래에 문서화되어 있습니다.

## 🚀 기능

- `image2caption.py`를 통한 단일 이미지 캡셔닝.
- `v2c.py` 또는 `video2caption.py`를 통한 비디오 캡셔닝(균일 프레임 샘플링).
- 사용자 지정 가능한 런타임 옵션:
  - 프레임 수.
  - 모델 크기.
  - 샘플링 temperature.
  - 체크포인트 이름.
- 더 빠른 비디오 추론을 위한 멀티프로세싱/스레드 캡셔닝.
- 출력 산출물:
  - SRT 자막 파일(`.srt`).
  - `v2c.py`의 JSON 전사(`.json`).
- CLIP+GPT2 매핑 실험을 위한 학습 및 평가 진입점.

### 한눈에 보기

| 영역 | 주요 스크립트 | 비고 |
|---|---|---|
| 이미지 캡셔닝 | `image2caption.py`, `i2c.py`, `predict.py` | CLI + 재사용 가능한 클래스 |
| 비디오 캡셔닝 | `v2c.py` | 권장되는 유지보수 경로 |
| 레거시 비디오 흐름 | `video2caption.py`, `video2caption_v1.1.py` | 머신 종속 가정 포함 |
| 데이터셋 구축 | `dataset_generation.py` | `data/processed/dataset.pkl` 생성 |
| 학습 / 평가 | `training.py`, `evaluate.py` | CLIP+GPT2 매핑 사용 |

## 🧱 아키텍처 (상위 수준)

`model/model.py`의 핵심 모델은 세 부분으로 구성됩니다.

1. `ImageEncoder`: CLIP 이미지 임베딩 추출.
2. `Mapping`: CLIP 임베딩을 GPT prefix 임베딩 시퀀스로 투영.
3. `TextDecoder`: 캡션 토큰을 자기회귀적으로 생성하는 GPT-2 언어 모델 헤드.

학습(`Net.train_forward`)은 사전 계산된 CLIP 이미지 임베딩 + 토크나이즈된 캡션을 사용합니다.
추론(`Net.forward`)은 PIL 이미지를 사용해 EOS 또는 `max_len`까지 토큰을 디코딩합니다.

### 데이터 흐름

1. 데이터셋 준비: `dataset_generation.py`가 `data/raw/results.csv` 및 `data/raw/flickr30k_images/`의 이미지를 읽고 `data/processed/dataset.pkl`을 작성합니다.
2. 학습: `training.py`가 피클된 튜플 `(image_name, image_embedding, caption)`을 로드해 mapper/decoder 레이어를 학습합니다.
3. 평가: `evaluate.py`가 홀드아웃 테스트 이미지에 생성 캡션을 렌더링합니다.
4. 추론 제공:
   - 이미지: `image2caption.py` / `predict.py` / `i2c.py`.
   - 비디오: `v2c.py`(권장), `video2caption.py`(레거시).

## 🗂️ 프로젝트 구조

```text
VideoCaptionerWithClip/
├── README.md
├── image2caption.py               # 단일 이미지 캡션 CLI
├── predict.py                     # 대체 단일 이미지 캡션 CLI
├── i2c.py                         # 재사용 가능한 ImageCaptioner 클래스 + CLI
├── v2c.py                         # 비디오 -> SRT + JSON (스레드 기반 프레임 캡셔닝)
├── video2caption.py               # 대체 비디오 -> SRT 구현 (레거시 제약)
├── video2caption_v1.1.py          # 구버전 변형
├── video2caption_v1.0_not_work.py # 동작하지 않는 레거시 파일로 명시
├── training.py                    # 모델 학습 진입점
├── evaluate.py                    # 테스트 분할 평가 및 렌더링 출력
├── dataset_generation.py          # data/processed/dataset.pkl 생성
├── data/
│   ├── __init__.py
│   └── dataset.py                 # Dataset + DataLoader 도우미
├── model/
│   ├── __init__.py
│   ├── model.py                   # CLIP 인코더 + 매핑 + GPT2 디코더
│   └── trainer.py                 # 학습/검증/테스트 유틸리티 클래스
├── utils/
│   ├── __init__.py
│   ├── config.py                  # ConfigS / ConfigL 기본값
│   ├── downloads.py               # Google Drive 체크포인트 다운로더
│   └── lr_warmup.py               # LR 워밍업 스케줄
├── i18n/                          # 다국어 README 변형
└── .auto-readme-work/             # Auto-README 파이프라인 산출물
```

## 📋 사전 요구사항

- Python `3.10+` 권장.
- CUDA 지원 GPU는 선택 사항이지만 학습 및 대형 모델 추론에서 강력히 권장.
- 현재 스크립트에서는 `ffmpeg`가 직접 필요하지 않습니다(OpenCV를 프레임 추출에 사용).
- Hugging Face / Google Drive에서 모델/체크포인트를 처음 다운로드할 때 인터넷 연결이 필요합니다.

현재 락파일이 없습니다(`requirements.txt` / `pyproject.toml` 누락). 따라서 의존성은 import 문을 기준으로 추론합니다.

## 🛠️ 설치

### 현재 저장소 레이아웃 기준 권장 설정

```bash
git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers pillow matplotlib numpy tqdm opencv-python pandas wandb gdown
```

### 원본 README 설치 스니펫 (보존)

이전 README는 코드 블록 중간에서 끝났습니다. 원본 명령은 소스 기준의 과거 기록으로 아래에 그대로 보존합니다.

```bash
git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip/src
```

참고: 현재 저장소 스냅샷에서는 스크립트가 `src/`가 아니라 저장소 루트에 배치되어 있습니다.

## ▶️ 빠른 시작

### 이미지 캡셔닝 (빠른 실행)

```bash
python image2caption.py -I /path/to/image.jpg -S L -C model.pt
```

### 비디오 캡셔닝 (권장 경로)

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
- `-S, --size`: 모델 크기(`S` 또는 `L`).
- `-C, --checkpoint-name`: `weights/{small|large}` 내 체크포인트 파일명.
- `-R, --res-path`: 캡션이 렌더링된 출력 이미지 저장 디렉터리.
- `-T, --temperature`: 샘플링 temperature.

### 2. 대체 이미지 CLI (`predict.py`)

```bash
python predict.py \
  -I /path/to/image.jpg \
  -S L \
  -C model.pt \
  -R ./data/result/prediction \
  -T 1.0
```

`predict.py`는 `image2caption.py`와 기능적으로 유사하지만 출력 텍스트 포맷이 약간 다릅니다.

### 3. 이미지 캡셔닝 클래스 API (`i2c.py`)

```bash
python i2c.py -I /path/to/image.jpg -S L -C model.pt -R ./data/result/prediction -T 1.0
```

또는 자체 스크립트에서 import:

```python
from i2c import ImageCaptioner

captioner = ImageCaptioner(model_size="L", checkpoint_name="model.pt")
captioner.set_image_path("/path/to/image.jpg")
caption = captioner.generate_caption(save_image=True)
print(caption)
```

### 4. 비디오 -> 자막 + JSON (`v2c.py`)

```bash
python v2c.py -V /path/to/video.mp4 -N 10
```

입력 비디오 옆에 다음이 출력됩니다.

- `<video_basename>_caption.srt`
- `<video_basename>_caption.json`
- `<video_basename>_captioning_frames/`

### 5. 대체 비디오 파이프라인 (`video2caption.py`)

```bash
python video2caption.py -V /path/to/video.mp4 -N 10
```

중요: 이 스크립트에는 현재 머신 종속 하드코딩 경로가 있습니다.

- Python 경로 기본값: `/home/lachlan/miniconda3/envs/caption/bin/python`
- 캡션 스크립트 경로: `/home/lachlan/Projects/image_captioning/clip-gpt-captioning/src/image2caption.py`

이 경로를 의도적으로 유지보수하지 않는다면 `v2c.py` 사용을 권장합니다.

### 6. 레거시 변형 (`video2caption_v1.1.py`)

이 스크립트는 과거 참고용으로 유지됩니다. 실제 사용은 `v2c.py`를 권장합니다.

### 7. 데이터셋 생성

```bash
python dataset_generation.py
```

예상 원본 입력:

- `data/raw/results.csv` (파이프 구분 캡션 테이블).
- `data/raw/flickr30k_images/` (CSV가 참조하는 이미지 파일).

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

평가는 테스트 이미지에 예측 캡션을 렌더링하고 다음 위치에 저장합니다.

- `<res-path>/<checkpoint_name_without_ext>_<SIZE>/`

## ⚙️ 설정

모델 설정은 `utils/config.py`에 정의되어 있습니다.

| Config | CLIP 백본 | GPT 모델 | 가중치 디렉터리 |
|---|---|---|---|
| `ConfigS` | `openai/clip-vit-base-patch32` | `gpt2` | `weights/small` |
| `ConfigL` | `openai/clip-vit-large-patch14` | `gpt2-medium` | `weights/large` |

설정 클래스의 주요 기본값:

| Field | `ConfigS` | `ConfigL` |
|---|---:|---:|
| `epochs` | 150 | 120 |
| `lr` | 3e-3 | 5e-3 |
| `batch_size_exp` | 6 | 5 |
| `ep_len` | 4 | 4 |
| `max_len` | 40 | 40 |

체크포인트 자동 다운로드 ID는 `utils/downloads.py`에 있습니다.

| Size | Google Drive ID |
|---|---|
| `L` | `1Gh32arzhW06C1ZJyzcJSSfdJDi3RgWoG` |
| `S` | `1pSQruQyg8KJq6VmzhMLFbT_VaHJMdlWF` |

## 📦 출력 파일

### 이미지 추론

- `--res-path`에 생성 캡션/타이틀이 오버레이된 저장 이미지.
- 파일명 패턴: `<input_stem>-R<SIZE>.jpg`.

### 비디오 추론 (`v2c.py`)

- SRT: `<video_stem>_caption.srt`
- JSON: `<video_stem>_caption.json`
- 프레임 이미지: `<video_stem>_captioning_frames/`

JSON 요소 예시:

```json
{
  "start": "00:00:03,200",
  "end": "00:00:03,700",
  "lang": "en",
  "text": "A dog running through a field."
}
```

## 🧪 예시

### 빠른 이미지 캡셔닝 예시

```bash
python image2caption.py -I ./examples/dog.jpg -S S -C model.pt
```

예상 동작:

- `weights/small/model.pt`가 없으면 다운로드합니다.
- 기본적으로 캡션 이미지가 `./data/result/prediction`에 저장됩니다.
- 캡션 텍스트가 stdout에 출력됩니다.

### 빠른 비디오 캡셔닝 예시

```bash
python v2c.py -V ./examples/demo.mp4 -N 8
```

예상 동작:

- 균일 샘플링된 8개 프레임에 대해 캡셔닝을 수행합니다.
- 입력 비디오와 같은 위치에 `.srt`와 `.json` 파일을 생성합니다.

### 엔드투엔드 학습/평가 시퀀스

```bash
python dataset_generation.py
python training.py -S L -C model.pt
python evaluate.py -I ./data/raw/flickr30k_images -R ./data/result/eval -S L -C model.pt -T 1.0
```

## 🧭 개발 노트

- `v2c.py`, `video2caption.py`, `video2caption_v1.*` 사이에 레거시 중복이 존재합니다.
- `video2caption_v1.0_not_work.py`는 의도적으로 동작하지 않는 레거시 코드로 유지됩니다.
- `training.py`는 현재 `config = ConfigL() if args.size.upper() else ConfigS()`를 사용하므로, `--size` 값이 비어 있지 않다면 항상 `ConfigL`이 선택됩니다.
- `model/trainer.py`는 `test_step`에서 `self.dataset`을 사용하지만 초기화에서는 `self.test_dataset`을 할당하므로, 수정하지 않으면 학습 실행 중 샘플링이 깨질 수 있습니다.
- `video2caption_v1.1.py`는 `self.config.transform`을 참조하지만 `ConfigS`/`ConfigL`에는 `transform`이 정의되어 있지 않습니다.
- 현재 저장소 스냅샷에는 CI/테스트 스위트가 정의되어 있지 않습니다.
- i18n 참고: 이 README 상단에 언어 링크가 있으며, 번역 파일은 `i18n/` 아래에 추가될 수 있습니다.
- 현재 상태 참고: 언어 바는 `i18n/README.ru.md`를 링크하지만 이 파일은 현재 스냅샷에 없습니다.

## 🩺 문제 해결

- `AssertionError: Image does not exist`
  - `-I/--img-path`가 유효한 파일을 가리키는지 확인하세요.
- `Dataset file not found. Downloading...`
  - `MiniFlickrDataset`는 `data/processed/dataset.pkl`이 없을 때 이 오류를 발생시킵니다. 먼저 `python dataset_generation.py`를 실행하세요.
- `Path to the test image folder does not exist`
  - `evaluate.py -I`가 존재하는 폴더를 가리키는지 확인하세요.
- 첫 실행이 느리거나 실패함
  - 최초 실행 시 Hugging Face 모델을 다운로드하고 Google Drive에서 체크포인트를 다운로드할 수 있습니다.
- `video2caption.py`가 빈 캡션을 반환함
  - 하드코딩된 스크립트/파이썬 경로를 확인하거나 `v2c.py`로 전환하세요.
- 학습 중 `wandb` 로그인 프롬프트가 뜸
  - `wandb login`을 실행하거나 필요 시 `training.py`에서 로깅을 수동 비활성화하세요.

## 🛣️ 로드맵

- 재현 가능한 설치를 위해 의존성 락파일(`requirements.txt` 또는 `pyproject.toml`) 추가.
- 중복된 비디오 파이프라인을 단일 유지보수 구현으로 통합.
- 레거시 스크립트의 하드코딩 머신 경로 제거.
- `training.py` 및 `model/trainer.py`의 알려진 학습/평가 엣지 케이스 버그 수정.
- 자동화 테스트 및 CI 추가.
- 언어 바에서 참조되는 번역 README 파일로 `i18n/` 채우기.

## 🤝 기여

기여를 환영합니다. 권장 워크플로:

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

모델 동작을 변경했다면 다음을 포함해 주세요.

- 재현 가능한 명령어.
- 변경 전/후 샘플 출력.
- 체크포인트 또는 데이터셋 가정 관련 메모.

## 🙌 지원

현재 저장소 스냅샷에는 명시적인 후원/스폰서 설정이 없습니다.

추후 후원 링크가 추가되면 이 섹션에 보존되어야 합니다.

## 📄 라이선스

현재 저장소 스냅샷에는 라이선스 파일이 없습니다.

가정 참고: `LICENSE` 파일이 추가되기 전까지는 재사용/배포 조건이 정의되어 있지 않습니다.
