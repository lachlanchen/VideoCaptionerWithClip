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

一个 Python 工具包，通过结合 OpenAI CLIP 视觉特征与 GPT 风格的语言模型，为图像和视频生成自然语言标题。

## 🧭 Snapshot

| 维度 | 详细信息 |
|---|---|
| 任务范围 | 图像与视频字幕生成 |
| 核心输出 | SRT 字幕、JSON 转录、带字幕图像 |
| 主要脚本 | `i2c.py`、`v2c.py`、`image2caption.py` |
| 历史路径 | `video2caption.py` 及其版本（保留用于历史参考） |
| 数据集流程 | `data/raw/results.csv` + `data/raw/flickr30k_images/` |

## ✨ 概览

该仓库提供：

- 图像描述与视频字幕生成推理脚本。
- 训练流水线：学习 CLIP 视觉嵌入到 GPT-2 token 嵌入的映射。
- 用于 Flickr30k 风格数据集的生成工具。
- 当缺少权重时自动下载支持的模型尺寸 checkpoint。
- `i18n/` 下的多语言 README 版本（见上方语言栏）。

当前实现包含较新脚本与历史遗留脚本。部分遗留文件保留用于参考，并在下文说明。

## 🚀 特性

- 通过 `image2caption.py` 实现单张图像描述。
- 通过 `v2c.py` 或 `video2caption.py` 实现视频描述（均匀采样帧）。
- 可自定义运行选项：
  - 帧数
  - 模型尺寸
  - 采样温度
  - Checkpoint 名称
- 多进程/多线程推理以加速视频字幕生成。
- 输出产物：
  - SRT 字幕文件（`.srt`）
  - `v2c.py` 的 JSON 转录文件（`.json`）
- CLIP+GPT2 映射实验的训练与评估入口。

### 一览

| 模块 | 主要脚本 | 说明 |
|---|---|---|
| 图像描述 | `image2caption.py`, `i2c.py`, `predict.py` | CLI + 可复用类 |
| 视频描述 | `v2c.py` | 推荐的维护路径 |
| 遗留视频流程 | `video2caption.py`, `video2caption_v1.1.py` | 包含特定机器假设 |
| 数据集构建 | `dataset_generation.py` | 生成 `data/processed/dataset.pkl` |
| 训练 / 评估 | `training.py`, `evaluate.py` | 使用 CLIP+GPT2 映射 |

## 🧱 架构（高层）

`model/model.py` 中的核心模型由三部分组成：

1. `ImageEncoder`：提取 CLIP 图像嵌入。
2. `Mapping`：将 CLIP 嵌入投影为 GPT 前缀嵌入序列。
3. `TextDecoder`：GPT-2 语言模型头，自回归生成标题 token。

训练（`Net.train_forward`）使用预计算的 CLIP 图像嵌入与分词后的 captions。
推理（`Net.forward`）使用 PIL 图像并持续解码 token，直到 EOS 或 `max_len`。

### 数据流

1. 准备数据集：`dataset_generation.py` 读取 `data/raw/results.csv` 与 `data/raw/flickr30k_images/`，写入 `data/processed/dataset.pkl`。
2. 训练：`training.py` 加载 pickle 元组 `(image_name, image_embedding, caption)` 并训练映射/解码层。
3. 评估：`evaluate.py` 在留出的测试图像上渲染生成的描述。
4. 提供推理：
   - 图像：`image2caption.py` / `predict.py` / `i2c.py`
   - 视频：`v2c.py`（推荐）、`video2caption.py`（遗留）

## 🗂️ 项目结构

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

## 📋 前置条件

- 推荐 Python `3.10+`。
- 具备 CUDA 的 GPU 非必需，但强烈建议用于训练和大模型推理。
- 目前脚本未直接依赖 `ffmpeg`（使用 OpenCV 进行抽帧）。
- 首次从 Hugging Face / Google Drive 下载模型和 checkpoint 时需要联网。

当前仓库快照中未提供 lockfile（缺少 `requirements.txt` / `pyproject.toml`），因此依赖项需从导入中推断。

## 🛠️ 安装

### 基于当前仓库结构的标准安装

```bash
git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers pillow matplotlib numpy tqdm opencv-python pandas wandb gdown
```

### 原 README 的安装片段（保留）

原始 README 在中途中断。按历史原文保留如下命令，作为源级参考：

```bash
git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip/src
```

注意：当前仓库快照中的脚本位于仓库根目录，不在 `src/` 下。

## ▶️ 快速开始

| 目标 | 命令 |
|---|---|
| 生成图像标题 | `python image2caption.py -I /path/to/image.jpg -S L -C model.pt` |
| 生成视频标题 | `python v2c.py -V /path/to/video.mp4 -N 10` |
| 构建数据集 | `python dataset_generation.py` |

### 图像描述（快速运行）

```bash
python image2caption.py -I /path/to/image.jpg -S L -C model.pt
```

### 视频描述（推荐路径）

```bash
python v2c.py -V /path/to/video.mp4 -N 10
```

## 🎯 用法

### 1. 图像描述（`image2caption.py`）

```bash
python image2caption.py \
  -I /path/to/image.jpg \
  -S L \
  -C model.pt \
  -R ./data/result/prediction \
  -T 1.0
```

参数：

- `-I, --img-path`：输入图像路径。
- `-S, --size`：模型尺寸（`S` 或 `L`）。
- `-C, --checkpoint-name`：`weights/{small|large}` 中的 checkpoint 文件名。
- `-R, --res-path`：输出渲染后图像的目录。
- `-T, --temperature`：采样温度。

### 2. 备用图像 CLI（`predict.py`）

```bash
python predict.py \
  -I /path/to/image.jpg \
  -S L \
  -C model.pt \
  -R ./data/result/prediction \
  -T 1.0
```

`predict.py` 在功能上与 `image2caption.py` 相同，输出文本格式略有差异。

### 3. 图像描述类 API（`i2c.py`）

```bash
python i2c.py -I /path/to/image.jpg -S L -C model.pt -R ./data/result/prediction -T 1.0
```

或在你自己的脚本中导入使用：

```python
from i2c import ImageCaptioner

captioner = ImageCaptioner(model_size="L", checkpoint_name="model.pt")
captioner.set_image_path("/path/to/image.jpg")
caption = captioner.generate_caption(save_image=True)
print(caption)
```

### 4. 视频转字幕 + JSON（`v2c.py`）

```bash
python v2c.py -V /path/to/video.mp4 -N 10
```

输出位于输入视频同级目录：

- `<video_basename>_caption.srt`
- `<video_basename>_caption.json`
- `<video_basename>_captioning_frames/`

### 5. 备用视频流程（`video2caption.py`）

```bash
python video2caption.py -V /path/to/video.mp4 -N 10
```

注意：该脚本当前包含机器特定硬编码路径：

- Python 默认路径：`/home/lachlan/miniconda3/envs/caption/bin/python`
- 描述脚本路径：`/home/lachlan/Projects/image_captioning/clip-gpt-captioning/src/image2caption.py`

除非你有意维护这些路径，否则请优先使用 `v2c.py`。

### 6. 遗留变体（`video2caption_v1.1.py`）

该脚本保留为历史参考。实际使用请优先使用 `v2c.py`。

### 7. 数据集生成

```bash
python dataset_generation.py
```

期望输入：

- `data/raw/results.csv`（用管道分隔的 captions 表）。
- `data/raw/flickr30k_images/`（CSV 中引用的图像文件）。

输出：

- `data/processed/dataset.pkl`

### 8. 训练

```bash
python training.py -S L -C model.pt
```

训练默认使用 Weights & Biases（`wandb`）日志。

### 9. 评估

```bash
python evaluate.py \
  -I ./data/raw/flickr30k_images \
  -R ./data/result/eval \
  -S L \
  -C model.pt \
  -T 1.0
```

评估会将预测描述渲染到测试图像并保存到：

- `<res-path>/<checkpoint_name_without_ext>_<SIZE>/`

## ⚙️ 配置

模型配置定义在 `utils/config.py` 中：

| 配置 | CLIP 主干 | GPT 模型 | Weights 目录 |
|---|---|---|---|
| `ConfigS` | `openai/clip-vit-base-patch32` | `gpt2` | `weights/small` |
| `ConfigL` | `openai/clip-vit-large-patch14` | `gpt2-medium` | `weights/large` |

配置类关键默认值：

| 字段 | `ConfigS` | `ConfigL` |
|---|---:|---:|
| `epochs` | 150 | 120 |
| `lr` | 3e-3 | 5e-3 |
| `batch_size_exp` | 6 | 5 |
| `ep_len` | 4 | 4 |
| `max_len` | 40 | 40 |

Checkpoint 自动下载 ID 定义于 `utils/downloads.py`：

| 尺寸 | Google Drive ID |
|---|---|
| `L` | `1Gh32arzhW06C1ZJyzcJSSfdJDi3RgWoG` |
| `S` | `1pSQruQyg8KJq6VmzhMLFbT_VaHJMdlWF` |

## 📦 输出文件

### 图像推理

- 在 `--res-path` 保存带叠加/生成字幕的图像。
- 文件名格式：`<input_stem>-R<SIZE>.jpg`。

### 视频推理（`v2c.py`）

- SRT：`<video_stem>_caption.srt`
- JSON：`<video_stem>_caption.json`
- 帧图像：`<video_stem>_captioning_frames/`

JSON 示例：

```json
{
  "start": "00:00:03,200",
  "end": "00:00:03,700",
  "lang": "en",
  "text": "A dog running through a field."
}
```

## 🧪 示例

### 快速图像描述示例

```bash
python image2caption.py -I ./examples/dog.jpg -S S -C model.pt
```

预期行为：

- 若缺少 `weights/small/model.pt`，会自动下载。
- 默认将带描述的图像写入 `./data/result/prediction`。
- 标题文本将输出到 stdout。

### 快速视频描述示例

```bash
python v2c.py -V ./examples/demo.mp4 -N 8
```

预期行为：

- 将对 8 帧均匀采样帧生成描述。
- 在输入视频同目录生成 `.srt` 与 `.json` 文件。

### 端到端训练/评估流程

```bash
python dataset_generation.py
python training.py -S L -C model.pt
python evaluate.py -I ./data/raw/flickr30k_images -R ./data/result/eval -S L -C model.pt -T 1.0
```

## 🧭 开发说明

- `v2c.py`、`video2caption.py` 与 `video2caption_v1.*` 之间存在遗留重叠。
- `video2caption_v1.0_not_work.py` 被有意保留为不可用的遗留代码。
- `training.py` 当前通过 `config = ConfigL() if args.size.upper() else ConfigS()` 选择 `ConfigL()`；对于非空 `--size`，始终解析到 `ConfigL`。
- `model/trainer.py` 在 `test_step` 中使用 `self.dataset`，但初始化器赋值的是 `self.test_dataset`；若未修正可能导致训练运行采样异常。
- `video2caption_v1.1.py` 引用了 `self.config.transform`，但 `ConfigS`/`ConfigL` 未定义 `transform`。
- 当前仓库快照尚未定义 CI/test 套件。
- i18n 说明：语言链接位于本 README 顶部，翻译文件可补充到 `i18n/` 下。
- 当前状态说明：语言栏链接了 `i18n/README.ru.md`，但该文件在当前快照中不存在。

## 🩺 故障排查

- `AssertionError: Image does not exist`
  - 确认 `-I/--img-path` 指向有效文件。
- `Dataset file not found. Downloading...`
  - 当缺少 `data/processed/dataset.pkl` 时，`MiniFlickrDataset` 会触发；请先运行 `python dataset_generation.py`。
- `Path to the test image folder does not exist`
  - 确认 `evaluate.py -I` 指向一个真实存在的文件夹。
- 首次运行较慢或失败
  - 首次运行会下载 Hugging Face 模型，并可能从 Google Drive 下载 checkpoint。
- `video2caption.py` 返回空描述
  - 检查硬编码脚本路径和 Python 可执行路径，或切换到 `v2c.py`。
- 训练过程中 `wandb` 要求登录
  - 运行 `wandb login`，或按需在 `training.py` 中手动停用日志。

## 🛣️ 路线图

- 增加依赖 lockfile（`requirements.txt` 或 `pyproject.toml`）以实现可复现安装。
- 将重复的视频流水线合并为单一维护实现。
- 移除遗留脚本中的机器硬编码路径。
- 修复 `training.py` 和 `model/trainer.py` 中已知的训练/评估边界问题。
- 增加自动化测试与 CI。
- 完善 `i18n/` 中语言栏中列出的翻译 README。

## 🤝 贡献

欢迎贡献。建议工作流如下：

```bash
# 1) Fork 并 clone
git clone git@github.com:<your-user>/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip

# 2) 创建分支
git checkout -b feat/your-change

# 3) 修改并提交
git add .
git commit -m "feat: describe your change"

# 4) 推送并提交 PR
git push origin feat/your-change
```

如果你修改模型行为，请包含：

- 可复现的命令。
- 修改前后的示例输出。
- checkpoint 或数据集假设说明。

## ❤️ Support

| Donate | PayPal | Stripe |
|---|---|---|
| [![Donate](https://img.shields.io/badge/Donate-LazyingArt-0EA5E9?style=for-the-badge&logo=ko-fi&logoColor=white)](https://chat.lazying.art/donate) | [![PayPal](https://img.shields.io/badge/PayPal-RongzhouChen-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/RongzhouChen) | [![Stripe](https://img.shields.io/badge/Stripe-Donate-635BFF?style=for-the-badge&logo=stripe&logoColor=white)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## 📄 许可证

当前仓库快照中不存在许可证文件。

假设说明：在添加 `LICENSE` 文件之前，复用与分发条款尚未定义。
