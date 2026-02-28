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

這是一個 Python 工具包，透過結合 OpenAI CLIP 視覺嵌入與 GPT 風格語言模型，為圖片與影片產生自然語言描述。

## ✨ 概覽

此儲存庫提供：

- 用於圖片描述與影片字幕生成的推論腳本。
- 透過 CLIP 視覺嵌入對映到 GPT-2 token 嵌入的訓練流程。
- Flickr30k 風格資料集的生成工具。
- 當權重缺失時，支援模型尺寸的自動 checkpoint 下載。
- 位於 `i18n/` 的多語 README 版本（見上方語言列）。

目前實作同時包含較新的腳本與舊版腳本。部分舊版檔案為參考用途而保留，並已在下文說明。

## 🚀 功能

- 透過 `image2caption.py` 進行單張圖片描述。
- 透過 `v2c.py` 或 `video2caption.py` 進行影片描述（均勻抽幀）。
- 可自訂執行選項：
  - 影格數量。
  - 模型尺寸。
  - 取樣溫度。
  - Checkpoint 名稱。
- 多程序/多執行緒描述流程以加速影片推論。
- 輸出產物：
  - SRT 字幕檔（`.srt`）。
  - `v2c.py` 產生的 JSON 逐字稿（`.json`）。
- 用於 CLIP+GPT2 對映實驗的訓練與評估入口。

### 一覽

| 區域 | 主要腳本 | 備註 |
|---|---|---|
| 圖片描述 | `image2caption.py`, `i2c.py`, `predict.py` | CLI + 可重用類別 |
| 影片描述 | `v2c.py` | 建議的維護路徑 |
| 舊版影片流程 | `video2caption.py`, `video2caption_v1.1.py` | 含機器特定假設 |
| 資料集建置 | `dataset_generation.py` | 產生 `data/processed/dataset.pkl` |
| 訓練 / 評估 | `training.py`, `evaluate.py` | 使用 CLIP+GPT2 對映 |

## 🧱 架構（高層）

`model/model.py` 中的核心模型包含三個部分：

1. `ImageEncoder`：萃取 CLIP 圖像嵌入。
2. `Mapping`：將 CLIP 嵌入投影為 GPT 前綴嵌入序列。
3. `TextDecoder`：GPT-2 語言模型頭，以自回歸方式生成描述 token。

訓練（`Net.train_forward`）使用預先計算的 CLIP 圖像嵌入與分詞後描述。
推論（`Net.forward`）使用 PIL 圖片，解碼 token 直到 EOS 或 `max_len`。

### 資料流程

1. 準備資料集：`dataset_generation.py` 讀取 `data/raw/results.csv` 與 `data/raw/flickr30k_images/` 中圖片，寫入 `data/processed/dataset.pkl`。
2. 訓練：`training.py` 載入 pickled tuple `(image_name, image_embedding, caption)` 並訓練 mapper/decoder 層。
3. 評估：`evaluate.py` 在保留測試圖片上渲染生成描述。
4. 提供推論：
   - 圖片：`image2caption.py` / `predict.py` / `i2c.py`。
   - 影片：`v2c.py`（建議）、`video2caption.py`（舊版）。

## 🗂️ 專案結構

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

## 📋 先決條件

- 建議使用 Python `3.10+`。
- 可使用 CUDA 的 GPU 為可選，但強烈建議用於訓練與大型模型推論。
- 目前腳本不直接需要 `ffmpeg`（影格擷取使用 OpenCV）。
- 首次執行下載 Hugging Face / Google Drive 的模型與 checkpoint 時需要網路連線。

目前沒有 lockfile（缺少 `requirements.txt` / `pyproject.toml`），因此依賴套件由 import 內容推斷。

## 🛠️ 安裝

### 依目前儲存庫結構的標準安裝

```bash
git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers pillow matplotlib numpy tqdm opencv-python pandas wandb gdown
```

### 原始 README 安裝片段（保留）

先前 README 在程式碼區塊中途結束。下列原始指令完整保留，作為具權威性的歷史內容：

```bash
git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip/src
```

注意：目前儲存庫快照將腳本放在 repo 根目錄，而非 `src/`。

## ▶️ 快速開始

### 圖片描述（快速執行）

```bash
python image2caption.py -I /path/to/image.jpg -S L -C model.pt
```

### 影片描述（建議路徑）

```bash
python v2c.py -V /path/to/video.mp4 -N 10
```

## 🎯 使用方式

### 1. 圖片描述（`image2caption.py`）

```bash
python image2caption.py \
  -I /path/to/image.jpg \
  -S L \
  -C model.pt \
  -R ./data/result/prediction \
  -T 1.0
```

參數：

- `-I, --img-path`：輸入圖片路徑。
- `-S, --size`：模型尺寸（`S` 或 `L`）。
- `-C, --checkpoint-name`：`weights/{small|large}` 中的 checkpoint 檔名。
- `-R, --res-path`：輸出加上描述文字圖片的目錄。
- `-T, --temperature`：取樣溫度。

### 2. 替代圖片 CLI（`predict.py`）

```bash
python predict.py \
  -I /path/to/image.jpg \
  -S L \
  -C model.pt \
  -R ./data/result/prediction \
  -T 1.0
```

`predict.py` 功能與 `image2caption.py` 類似；輸出文字格式略有差異。

### 3. 圖片描述類別 API（`i2c.py`）

```bash
python i2c.py -I /path/to/image.jpg -S L -C model.pt -R ./data/result/prediction -T 1.0
```

或在你的腳本中匯入：

```python
from i2c import ImageCaptioner

captioner = ImageCaptioner(model_size="L", checkpoint_name="model.pt")
captioner.set_image_path("/path/to/image.jpg")
caption = captioner.generate_caption(save_image=True)
print(caption)
```

### 4. 影片轉字幕 + JSON（`v2c.py`）

```bash
python v2c.py -V /path/to/video.mp4 -N 10
```

輸出於輸入影片旁：

- `<video_basename>_caption.srt`
- `<video_basename>_caption.json`
- `<video_basename>_captioning_frames/`

### 5. 替代影片流程（`video2caption.py`）

```bash
python video2caption.py -V /path/to/video.mp4 -N 10
```

重要：此腳本目前包含機器特定硬編碼路徑：

- Python 路徑預設：`/home/lachlan/miniconda3/envs/caption/bin/python`
- 描述腳本路徑：`/home/lachlan/Projects/image_captioning/clip-gpt-captioning/src/image2caption.py`

除非你打算維護這些路徑，否則請使用 `v2c.py`。

### 6. 舊版變體（`video2caption_v1.1.py`）

此腳本保留作為歷史參考。實際使用請優先選擇 `v2c.py`。

### 7. 生成資料集

```bash
python dataset_generation.py
```

預期原始輸入：

- `data/raw/results.csv`（以 pipe 分隔的描述表格）。
- `data/raw/flickr30k_images/`（CSV 引用的圖片檔）。

輸出：

- `data/processed/dataset.pkl`

### 8. 訓練

```bash
python training.py -S L -C model.pt
```

訓練預設啟用 Weights & Biases（`wandb`）記錄。

### 9. 評估

```bash
python evaluate.py \
  -I ./data/raw/flickr30k_images \
  -R ./data/result/eval \
  -S L \
  -C model.pt \
  -T 1.0
```

評估會將預測描述渲染到測試圖片上，並儲存於：

- `<res-path>/<checkpoint_name_without_ext>_<SIZE>/`

## ⚙️ 設定

模型設定定義於 `utils/config.py`：

| 設定 | CLIP backbone | GPT model | 權重目錄 |
|---|---|---|---|
| `ConfigS` | `openai/clip-vit-base-patch32` | `gpt2` | `weights/small` |
| `ConfigL` | `openai/clip-vit-large-patch14` | `gpt2-medium` | `weights/large` |

設定類別的關鍵預設值：

| 欄位 | `ConfigS` | `ConfigL` |
|---|---:|---:|
| `epochs` | 150 | 120 |
| `lr` | 3e-3 | 5e-3 |
| `batch_size_exp` | 6 | 5 |
| `ep_len` | 4 | 4 |
| `max_len` | 40 | 40 |

Checkpoint 自動下載 ID 位於 `utils/downloads.py`：

| 尺寸 | Google Drive ID |
|---|---|
| `L` | `1Gh32arzhW06C1ZJyzcJSSfdJDi3RgWoG` |
| `S` | `1pSQruQyg8KJq6VmzhMLFbT_VaHJMdlWF` |

## 📦 輸出檔案

### 圖片推論

- 儲存帶有覆蓋/生成標題的圖片至 `--res-path`。
- 檔名格式：`<input_stem>-R<SIZE>.jpg`。

### 影片推論（`v2c.py`）

- SRT：`<video_stem>_caption.srt`
- JSON：`<video_stem>_caption.json`
- 影格圖片：`<video_stem>_captioning_frames/`

JSON 元素範例：

```json
{
  "start": "00:00:03,200",
  "end": "00:00:03,700",
  "lang": "en",
  "text": "A dog running through a field."
}
```

## 🧪 範例

### 快速圖片描述範例

```bash
python image2caption.py -I ./examples/dog.jpg -S S -C model.pt
```

預期行為：

- 若缺少 `weights/small/model.pt`，會自動下載。
- 預設會將描述圖片輸出到 `./data/result/prediction`。
- 描述文字會輸出到 stdout。

### 快速影片描述範例

```bash
python v2c.py -V ./examples/demo.mp4 -N 8
```

預期行為：

- 會對 8 個均勻抽樣影格產生描述。
- `.srt` 與 `.json` 檔會在輸入影片旁生成。

### 端到端訓練/評估流程

```bash
python dataset_generation.py
python training.py -S L -C model.pt
python evaluate.py -I ./data/raw/flickr30k_images -R ./data/result/eval -S L -C model.pt -T 1.0
```

## 🧭 開發說明

- `v2c.py`、`video2caption.py` 與 `video2caption_v1.*` 之間存在舊版功能重疊。
- `video2caption_v1.0_not_work.py` 刻意保留為不可運作的舊版程式碼。
- `training.py` 目前透過 `config = ConfigL() if args.size.upper() else ConfigS()` 選擇 `ConfigL()`，對非空 `--size` 值都會解析為 `ConfigL`。
- `model/trainer.py` 在 `test_step` 使用 `self.dataset`，但初始化時指定的是 `self.test_dataset`；若不調整，訓練流程中的抽樣可能失敗。
- `video2caption_v1.1.py` 參考了 `self.config.transform`，但 `ConfigS`/`ConfigL` 並未定義 `transform`。
- 目前儲存庫快照未定義 CI/測試套件。
- i18n 說明：本 README 頂部已有語言連結；翻譯檔可新增於 `i18n/`。
- 目前狀態說明：語言列連到 `i18n/README.ru.md`，但此快照中該檔案不存在。

## 🩺 疑難排解

- `AssertionError: Image does not exist`
  - 確認 `-I/--img-path` 指向有效檔案。
- `Dataset file not found. Downloading...`
  - 當 `data/processed/dataset.pkl` 缺失時，`MiniFlickrDataset` 會拋出此訊息；請先執行 `python dataset_generation.py`。
- `Path to the test image folder does not exist`
  - 確認 `evaluate.py -I` 指向現有資料夾。
- 首次執行過慢或失敗
  - 初次執行會下載 Hugging Face 模型，也可能從 Google Drive 下載 checkpoint。
- `video2caption.py` 回傳空白描述
  - 請檢查硬編碼腳本路徑與 Python 執行檔路徑，或改用 `v2c.py`。
- 訓練時 `wandb` 要求登入
  - 執行 `wandb login`，或視需求在 `training.py` 手動關閉記錄。

## 🛣️ 路線圖

- 新增依賴 lockfile（`requirements.txt` 或 `pyproject.toml`）以便可重現安裝。
- 將重複的影片流程整合為單一維護實作。
- 移除舊版腳本中的機器硬編碼路徑。
- 修正 `training.py` 與 `model/trainer.py` 已知的訓練/評估邊界案例錯誤。
- 新增自動化測試與 CI。
- 補齊 `i18n/` 中語言列所引用的 README 翻譯檔。

## 🤝 貢獻

歡迎貢獻。建議流程：

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

若你變更了模型行為，請附上：

- 可重現的指令。
- 變更前/後的範例輸出。
- 關於 checkpoint 或資料集假設的說明。

## 🙌 支援

目前儲存庫快照未包含明確的捐助/贊助設定。

若日後新增贊助連結，應保留於本節。

## 📄 授權

目前儲存庫快照未包含授權檔案。

假設說明：在新增 `LICENSE` 檔之前，重用/散佈條款均未定義。
