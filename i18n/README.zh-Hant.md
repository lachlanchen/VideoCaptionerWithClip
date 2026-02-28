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

這是一個 Python 工具包，透過結合 OpenAI CLIP 視覺嵌入與 GPT 風格語言模型，為圖片與影片產生自然語言字幕。

## 🧭 Snapshot

| 維度 | 詳細 |
|---|---|
| 任務覆蓋 | 圖片與影片字幕生成 |
| 核心輸出 | SRT 字幕、JSON 逐字稿、標註後的圖片 |
| 主要腳本 | `i2c.py`、`v2c.py`、`image2caption.py` |
| 舊版路徑 | `video2caption.py` 與其版本分支（保留供參考） |
| 資料流程 | `data/raw/results.csv` + `data/raw/flickr30k_images/` |

## ✨ 概覽

此儲存庫提供：

- 圖片描述與影片字幕生成的推論腳本。
- 一套訓練流程，學習將 CLIP 視覺嵌入映射到 GPT-2 token 嵌入。
- 適用於 Flickr30k 風格資料的資料集生成工具。
- 當缺少權重時，自動下載支援的模型尺寸。
- `i18n/` 下的多語 README 版本（見語言列）。

目前實作包含新舊兩套腳本。部分舊版檔案仍保留以供參考，並在下方說明。

## 🚀 功能

- 透過 `image2caption.py` 進行單張圖片字幕輸出。
- 透過 `v2c.py` 或 `video2caption.py` 進行影片字幕（均勻取樣影格）。
- 可調整的執行參數：
  - 影格數。
  - 模型尺寸。
  - 採樣溫度。
  - Checkpoint 名稱。
- 多核心/多執行緒的影片推論，提升處理速度。
- 輸出成果：
  - SRT 字幕檔（`.srt`）。
  - `v2c.py` 產生的 JSON 逐字稿（`.json`）。
- CLIP+GPT2 映射實驗的訓練與評估入口。

### 一眼看懂

| 區域 | 主要腳本 | 說明 |
|---|---|---|
| 圖片描述 | `image2caption.py`、`i2c.py`、`predict.py` | CLI 與可重用類別 |
| 影片描述 | `v2c.py` | 建議使用的維護路徑 |
| 舊版影片流程 | `video2caption.py`、`video2caption_v1.1.py` | 含機器特定假設 |
| 資料集建置 | `dataset_generation.py` | 產生 `data/processed/dataset.pkl` |
| 訓練 / 評估 | `training.py`、`evaluate.py` | 使用 CLIP+GPT2 映射 |

## 🧱 架構（高階）

`model/model.py` 的核心模型包含三個部份：

1. `ImageEncoder`：萃取 CLIP 圖像嵌入。
2. `Mapping`：將 CLIP 嵌入投影到 GPT 前綴嵌入序列。
3. `TextDecoder`：GPT-2 語言模型頭，透過自回歸方式逐字生成字幕 token。

訓練流程（`Net.train_forward`）使用預先計算好的 CLIP 圖像嵌入與分詞後字幕。
推論流程（`Net.forward`）使用 PIL 圖片，並持續解碼 token 至 EOS 或 `max_len`。

### 資料流程

1. 準備資料集：`dataset_generation.py` 讀取 `data/raw/results.csv` 與 `data/raw/flickr30k_images/`，並寫入 `data/processed/dataset.pkl`。
2. 訓練：`training.py` 載入 pickled tuple `(image_name, image_embedding, caption)`，並訓練映射／解碼層。
3. 評估：`evaluate.py` 在保留測試影像上輸出預測字幕。
4. 推論入口：
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
- 可用 CUDA 的 GPU 非必要但強烈建議用於訓練與大型模型推論。
- 目前腳本不直接依賴 `ffmpeg`（影格擷取使用 OpenCV）。
- 首次從 Hugging Face / Google Drive 下載模型與 checkpoint 時需要網路連線。

目前未提供鎖檔（缺少 `requirements.txt` / `pyproject.toml`），故可推測依賴自匯入模組。

## 🛠️ 安裝

### 依目前儲存庫佈局的標準安裝方式

```bash
git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers pillow matplotlib numpy tqdm opencv-python pandas wandb gdown
```

### 保留原始 README 的安裝片段

舊版 README 在程式碼區段中途中斷，保留原始指令如下，作為歷史真值內容：

```bash
git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip/src
```

注意：目前儲存庫快照將腳本放在 repo 根目錄，而不是 `src/` 下。

## ▶️ 快速開始

| 目標 | 指令 |
|---|---|
| 對圖片進行字幕 | `python image2caption.py -I /path/to/image.jpg -S L -C model.pt` |
| 對影片進行字幕 | `python v2c.py -V /path/to/video.mp4 -N 10` |
| 建立資料集 | `python dataset_generation.py` |

### 圖片字幕（快速執行）

```bash
python image2caption.py -I /path/to/image.jpg -S L -C model.pt
```

### 影片字幕（建議路徑）

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
- `-C, --checkpoint-name`：`weights/{small|large}` 下的 checkpoint 檔名。
- `-R, --res-path`：輸出加註字幕圖片的資料夾。
- `-T, --temperature`：採樣溫度。

### 2. 替代圖片 CLI（`predict.py`）

```bash
python predict.py \
  -I /path/to/image.jpg \
  -S L \
  -C model.pt \
  -R ./data/result/prediction \
  -T 1.0
```

`predict.py` 功能上與 `image2caption.py` 類似，輸出文字格式略有差異。

### 3. 圖片描述類別 API（`i2c.py`）

```bash
python i2c.py -I /path/to/image.jpg -S L -C model.pt -R ./data/result/prediction -T 1.0
```

或在自訂腳本中引入：

```python
from i2c import ImageCaptioner

captioner = ImageCaptioner(model_size="L", checkpoint_name="model.pt")
captioner.set_image_path("/path/to/image.jpg")
caption = captioner.generate_caption(save_image=True)
print(caption)
```

### 4. 影片轉字幕與 JSON（`v2c.py`）

```bash
python v2c.py -V /path/to/video.mp4 -N 10
```

輸出與輸入影片同目錄：

- `<video_basename>_caption.srt`
- `<video_basename>_caption.json`
- `<video_basename>_captioning_frames/`

### 5. 替代影片流程（`video2caption.py`）

```bash
python video2caption.py -V /path/to/video.mp4 -N 10
```

重要：此腳本目前仍保留機器特定的硬編碼路徑：

- Python 預設路徑：`/home/lachlan/miniconda3/envs/caption/bin/python`
- 字幕腳本路徑：`/home/lachlan/Projects/image_captioning/clip-gpt-captioning/src/image2caption.py`

除非你刻意維護這些路徑，否則請改用 `v2c.py`。

### 6. 舊版變體（`video2caption_v1.1.py`）

此腳本保留為歷史參考；實際使用請優先採用 `v2c.py`。

### 7. 生成資料集

```bash
python dataset_generation.py
```

預期輸入：

- `data/raw/results.csv`（以管線符號 `|` 分隔的描述表）。
- `data/raw/flickr30k_images/`（CSV 中參照的圖片檔）。

輸出：

- `data/processed/dataset.pkl`

### 8. 訓練

```bash
python training.py -S L -C model.pt
```

訓練預設使用 Weights & Biases（`wandb`）記錄。

### 9. 評估

```bash
python evaluate.py \
  -I ./data/raw/flickr30k_images \
  -R ./data/result/eval \
  -S L \
  -C model.pt \
  -T 1.0
```

評估會將預測結果疊加到測試影像並儲存至：

- `<res-path>/<checkpoint_name_without_ext>_<SIZE>/`

## ⚙️ 設定

模型設定定義於 `utils/config.py`：

| 設定 | CLIP backbone | GPT model | 權重目錄 |
|---|---|---|---|
| `ConfigS` | `openai/clip-vit-base-patch32` | `gpt2` | `weights/small` |
| `ConfigL` | `openai/clip-vit-large-patch14` | `gpt2-medium` | `weights/large` |

關鍵預設值：

| 欄位 | `ConfigS` | `ConfigL` |
|---|---:|---:|
| `epochs` | 150 | 120 |
| `lr` | 3e-3 | 5e-3 |
| `batch_size_exp` | 6 | 5 |
| `ep_len` | 4 | 4 |
| `max_len` | 40 | 40 |

Checkpoint 自動下載 ID 記錄於 `utils/downloads.py`：

| 尺寸 | Google Drive ID |
|---|---|
| `L` | `1Gh32arzhW06C1ZJyzcJSSfdJDi3RgWoG` |
| `S` | `1pSQruQyg8KJq6VmzhMLFbT_VaHJMdlWF` |

## 📦 輸出檔案

### 圖片推論

- 儲存疊加字幕的圖片到 `--res-path`。
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

### 快速圖片字幕範例

```bash
python image2caption.py -I ./examples/dog.jpg -S S -C model.pt
```

預期行為：

- 如果 `weights/small/model.pt` 不存在，將會下載。
- 預設會輸出字幕圖片至 `./data/result/prediction`。
- 字幕文字將輸出至標準輸出（stdout）。

### 快速影片字幕範例

```bash
python v2c.py -V ./examples/demo.mp4 -N 8
```

預期行為：

- 會對 8 個均勻抽樣的影格產生字幕。
- `.srt` 與 `.json` 會與輸入影片一併生成。

### 端到端訓練與評估流程

```bash
python dataset_generation.py
python training.py -S L -C model.pt
python evaluate.py -I ./data/raw/flickr30k_images -R ./data/result/eval -S L -C model.pt -T 1.0
```

## 🧭 開發說明

- `v2c.py`、`video2caption.py` 與 `video2caption_v1.*` 間有舊版流程重疊。
- `video2caption_v1.0_not_work.py` 有意保留作為不能運作的舊版程式碼。
- `training.py` 目前透過 `config = ConfigL() if args.size.upper() else ConfigS()` 選取 `ConfigL()`，對非空 `--size` 都會解析為 `ConfigL`。
- `model/trainer.py` 在 `test_step` 使用 `self.dataset`，但初始化時是 `self.test_dataset`，可能導致訓練取樣失敗，需視情況修正。
- `video2caption_v1.1.py` 使用 `self.config.transform`，但 `ConfigS`／`ConfigL` 並未定義 `transform`。
- 目前本儲存庫快照未建立 CI / 測試套件。
- i18n 備註：README 頂部已有語言連結，可在 `i18n/` 中新增翻譯。
- 現況備註：語言列鏈接 `i18n/README.ru.md`，但本快照中該檔案不存在。

## 🩺 疑難排解

- `AssertionError: Image does not exist`
  - 確認 `-I/--img-path` 指向有效檔案。
- `Dataset file not found. Downloading...`
  - 當缺少 `data/processed/dataset.pkl` 時，`MiniFlickrDataset` 會拋出此訊息；請先執行 `python dataset_generation.py`。
- `Path to the test image folder does not exist`
  - 確認 `evaluate.py -I` 指向現有資料夾。
- 首次執行緩慢或失敗
  - 首次執行會下載 Hugging Face 模型，並可能從 Google Drive 下載 checkpoint。
- `video2caption.py` 回傳空白字幕
  - 請檢查硬編碼腳本路徑與 Python 可執行檔，或改用 `v2c.py`。
- `wandb` 提示訓練時登入
  - 執行 `wandb login`，或視需求在 `training.py` 中手動停用紀錄。

## 🛣️ 里程碑

- 新增依賴鎖定檔（`requirements.txt` 或 `pyproject.toml`）以提升可重現性。
- 將重複的影片流程整併為單一路徑維護。
- 移除舊版腳本中的機器硬編碼路徑。
- 修正 `training.py` 與 `model/trainer.py` 已知的訓練／評估邊界案例錯誤。
- 新增自動化測試與 CI。
- 補齊語言列中引用到的 `i18n/` README 翻譯檔。

## 🤝 貢獻

歡迎提交貢獻。建議流程如下：

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

如果你變更模型行為，請一併提供：

- 可重現的指令。
- 變更前後的輸出範例。
- Checkpoint 與資料集假設備註。

## ❤️ Support

| Donate | PayPal | Stripe |
|---|---|---|
| [![Donate](https://img.shields.io/badge/Donate-LazyingArt-0EA5E9?style=for-the-badge&logo=ko-fi&logoColor=white)](https://chat.lazying.art/donate) | [![PayPal](https://img.shields.io/badge/PayPal-RongzhouChen-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/RongzhouChen) | [![Stripe](https://img.shields.io/badge/Stripe-Donate-635BFF?style=for-the-badge&logo=stripe&logoColor=white)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## 📄 授權

本儲存庫快照目前未提供授權檔。

假設說明：在加入 `LICENSE` 檔前，重製與散佈條款仍未定義。
