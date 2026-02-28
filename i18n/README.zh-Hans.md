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

## 🧭 快速导航

| Section | What to use it for |
|---|---|
| Snapshot | 查看仓库范围和当前脚本清单 |
| Overview | 阅读目标和功能 |
| Usage | 按照精确的 CLI/API 流程使用 |
| Troubleshooting | 快速排查常见运行问题 |
| Roadmap | 跟进已知清理和改进目标 |

---

一个将 OpenAI CLIP 图像特征与 GPT 风格语言模型结合，用于生成图像与视频自然语言字幕的 Python 工具包。

## 🧭 快照

| Dimension | Details |
|---|---|
| 任务覆盖范围 | 图像与视频字幕生成 |
| 核心产物 | SRT 字幕、JSON 转录文本、带字幕的图像 |
| 主要脚本 | `i2c.py`、`v2c.py`、`image2caption.py` |
| 旧路径 | `video2caption.py` 及其版本分支（保留用于历史参考） |
| 数据集流程 | `data/raw/results.csv` + `data/raw/flickr30k_images/` |

## ✨ 概览

该仓库提供以下内容：

- 图像字幕与视频字幕生成推理脚本。
- 学习 CLIP 图像嵌入到 GPT-2 token 嵌入映射的训练流水线。
- 用于 Flickr30k 风格数据的 数据集生成工具。
- 在权重缺失时自动下载所支持模型尺寸的检查点。
- `i18n/` 下的多语言 README 版本（见上方语言栏）。

当前实现同时保留了较新脚本与历史遗留脚本。部分旧文件仅保留用于参考，在下方有说明。

## 🚀 功能

- 通过 `image2caption.py` 支持单张图像字幕生成。
- 通过 `v2c.py` 或 `video2caption.py` 支持视频字幕（均匀抽帧）。
- 可自定义运行参数：
  - 帧数
  - 模型大小
  - 采样温度
  - 检查点名称
- 多进程 / 多线程加速视频推理。
- 输出文件：
  - SRT 字幕文件（`.srt`）
  - `v2c.py` 输出的 JSON 转录文本（`.json`）
- CLIP+GPT2 映射实验的训练与评估入口。

### 一览

| Area | Primary script(s) | Notes |
|---|---|---|
| 图像字幕 | `image2caption.py`、`i2c.py`、`predict.py` | CLI 与可复用类 |
| 视频字幕 | `v2c.py` | 推荐的主维护路径 |
| 旧版视频流程 | `video2caption.py`、`video2caption_v1.1.py` | 包含机器相关的硬编码假设 |
| 数据集构建 | `dataset_generation.py` | 生成 `data/processed/dataset.pkl` |
| 训练 / 评估 | `training.py`、`evaluate.py` | 使用 CLIP+GPT2 映射 |

## 🧱 架构（高层）

`model/model.py` 中的核心模型包含三部分：

1. `ImageEncoder`：提取 CLIP 图像嵌入。
2. `Mapping`：将 CLIP 嵌入映射到 GPT 前缀嵌入序列。
3. `TextDecoder`：GPT-2 解码头，按自回归方式生成字幕 token。

训练阶段（`Net.train_forward`）使用预计算的 CLIP 图像嵌入与分词后的字幕。
推理阶段（`Net.forward`）使用 PIL 图像并持续解码 token，直到 EOS 或 `max_len`。

### 数据流

1. 准备数据集：`dataset_generation.py` 读取 `data/raw/results.csv` 与 `data/raw/flickr30k_images/` 中的图像，写入 `data/processed/dataset.pkl`。
2. 训练：`training.py` 载入 pickled 元组 `(image_name, image_embedding, caption)` 并训练映射层与解码层。
3. 评估：`evaluate.py` 在留出测试图像上渲染生成字幕。
4. 提供推理入口：
   - 图像：`image2caption.py` / `predict.py` / `i2c.py`
   - 视频：`v2c.py`（推荐）、`video2caption.py`（历史版本）

## 🗂️ 项目结构

```text
VideoCaptionerWithClip/
├── README.md
├── image2caption.py               # 单张图像字幕 CLI
├── predict.py                     # 替代的单张图像字幕 CLI
├── i2c.py                         # 可复用的 ImageCaptioner 类 + CLI
├── v2c.py                         # 视频 -> SRT + JSON（多线程逐帧字幕）
├── video2caption.py               # 替代的视频 -> SRT 实现（遗留限制）
├── video2caption_v1.1.py          # 更早版本
├── video2caption_v1.0_not_work.py # 明确标注为不再可用的遗留文件
├── training.py                    # 模型训练入口
├── evaluate.py                    # 测试集评估与结果渲染
├── dataset_generation.py          # 构建 data/processed/dataset.pkl
├── data/
│   ├── __init__.py
│   └── dataset.py                 # Dataset + DataLoader 辅助
├── model/
│   ├── __init__.py
│   ├── model.py                   # CLIP 编码器 + 映射 + GPT2 解码器
│   └── trainer.py                 # 训练/验证/测试辅助类
├── utils/
│   ├── __init__.py
│   ├── config.py                  # ConfigS / ConfigL 默认配置
│   ├── downloads.py               # Google Drive 检查点下载工具
│   └── lr_warmup.py               # 学习率热身调度
├── i18n/                          # 多语言 README 版本
└── .auto-readme-work/             # 自动 README 流水线产物
```

## 📋 前置条件

- 推荐 Python `3.10+`。
- 训练与大模型推理建议具备 CUDA GPU；非必须。
- 当前脚本不直接依赖 `ffmpeg`（帧抽取使用 OpenCV）。
- 首次从 Hugging Face / Google Drive 下载模型或检查点时需要联网。

当前仓库暂未提供锁文件（缺少 `requirements.txt` / `pyproject.toml`），因此依赖以 import 引用为准。

## 🛠️ 安装

### 按当前仓库布局进行标准安装

```bash

git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers pillow matplotlib numpy tqdm opencv-python pandas wandb gdown
```

### 保留历史 README 的安装片段

原始 README 在中间处中断。为保留历史内容，以下命令按原样保留：

```bash
git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip/src
```

注意：当前仓库快照将脚本放在仓库根目录，而非 `src/`。

## ▶️ 快速开始

| Goal | Command |
|---|---|
| 生成图像字幕 | `python image2caption.py -I /path/to/image.jpg -S L -C model.pt` |
| 生成视频字幕 | `python v2c.py -V /path/to/video.mp4 -N 10` |
| 构建数据集 | `python dataset_generation.py` |

### 图像字幕（快速运行）

```bash
python image2caption.py -I /path/to/image.jpg -S L -C model.pt
```

### 视频字幕（推荐路径）

```bash
python v2c.py -V /path/to/video.mp4 -N 10
```

## 🎯 用法

### 1. 图像字幕（`image2caption.py`）

```bash
python image2caption.py \
  -I /path/to/image.jpg \
  -S L \
  -C model.pt \
  -R ./data/result/prediction \
  -T 1.0
```

参数说明：

- `-I, --img-path`：输入图片路径。
- `-S, --size`：模型大小（`S` 或 `L`）。
- `-C, --checkpoint-name`：`weights/{small|large}` 下的检查点文件名。
- `-R, --res-path`：渲染后带字幕图像的输出目录。
- `-T, --temperature`：采样温度。

### 2. 替代图像 CLI（`predict.py`）

```bash
python predict.py \
  -I /path/to/image.jpg \
  -S L \
  -C model.pt \
  -R ./data/result/prediction \
  -T 1.0
```

`predict.py` 与 `image2caption.py` 功能一致；仅输出文本格式略有差异。

### 3. 图像字幕类 API（`i2c.py`）

```bash
python i2c.py -I /path/to/image.jpg -S L -C model.pt -R ./data/result/prediction -T 1.0
```

或在你自己的脚本中导入：

```python
from i2c import ImageCaptioner

captioner = ImageCaptioner(model_size="L", checkpoint_name="model.pt")
captioner.set_image_path("/path/to/image.jpg")
caption = captioner.generate_caption(save_image=True)
print(caption)
```

### 4. 视频字幕 + JSON（`v2c.py`）

```bash
python v2c.py -V /path/to/video.mp4 -N 10
```

输出文件位于输入视频同目录：

- `<video_basename>_caption.srt`
- `<video_basename>_caption.json`
- `<video_basename>_captioning_frames/`

### 5. 替代视频流程（`video2caption.py`）

```bash
python video2caption.py -V /path/to/video.mp4 -N 10
```

重要提示：该脚本目前包含机器相关的硬编码路径：

- Python 默认路径：`/home/lachlan/miniconda3/envs/caption/bin/python`
- 字幕脚本路径：`/home/lachlan/Projects/image_captioning/clip-gpt-captioning/src/image2caption.py`

除非你有意维护这些路径，否则请使用 `v2c.py`。

### 6. 历史版本（`video2caption_v1.1.py`）

该脚本仅保留用于历史参考。日常使用请优先选用 `v2c.py`。

### 7. 数据集生成

```bash
python dataset_generation.py
```

期望输入：

- `data/raw/results.csv`（制表符分隔的字幕表）
- `data/raw/flickr30k_images/`（CSV 中引用的图像文件）

输出：

- `data/processed/dataset.pkl`

### 8. 训练

```bash
python training.py -S L -C model.pt
```

训练默认启用 Weights & Biases（`wandb`）日志。

### 9. 评估

```bash
python evaluate.py \
  -I ./data/raw/flickr30k_images \
  -R ./data/result/eval \
  -S L \
  -C model.pt \
  -T 1.0
```

评估会将预测字幕渲染到测试图像上，并保存在：

- `<res-path>/<checkpoint_name_without_ext>_<SIZE>/`

## ⚙️ 配置

模型配置定义在 `utils/config.py`：

| Config | CLIP backbone | GPT model | Weights dir |
|---|---|---|---|
| `ConfigS` | `openai/clip-vit-base-patch32` | `gpt2` | `weights/small` |
| `ConfigL` | `openai/clip-vit-large-patch14` | `gpt2-medium` | `weights/large` |

配置类关键默认值：

| Field | `ConfigS` | `ConfigL` |
|---|---:|---:|
| `epochs` | 150 | 120 |
| `lr` | 3e-3 | 5e-3 |
| `batch_size_exp` | 6 | 5 |
| `ep_len` | 4 | 4 |
| `max_len` | 40 | 40 |

检查点自动下载 ID 在 `utils/downloads.py` 中：

| Size | Google Drive ID |
|---|---|
| `L` | `1Gh32arzhW06C1ZJyzcJSSfdJDi3RgWoG` |
| `S` | `1pSQruQyg8KJq6VmzhMLFbT_VaHJMdlWF` |

## 📦 输出文件

### 图像推理

- 在 `--res-path` 下保存带有叠加/生成标题的图像。
- 文件名格式：`<input_stem>-R<SIZE>.jpg`。

### 视频推理（`v2c.py`）

- SRT：`<video_stem>_caption.srt`
- JSON：`<video_stem>_caption.json`
- 帧图像：`<video_stem>_captioning_frames/`

示例 JSON 元素：

```json
{
  "start": "00:00:03,200",
  "end": "00:00:03,700",
  "lang": "en",
  "text": "A dog running through a field."
}
```

## 🧪 示例

### 快速图像字幕示例

```bash
python image2caption.py -I ./examples/dog.jpg -S S -C model.pt
```

预期表现：

- 若 `weights/small/model.pt` 缺失会自动下载。
- 默认会将带字幕图像写入 `./data/result/prediction`。
- 字幕文本会打印到标准输出。

### 快速视频字幕示例

```bash
python v2c.py -V ./examples/demo.mp4 -N 8
```

预期表现：

- 会对 8 帧均匀采样图像生成字幕。
- 同时在输入视频旁边生成 `.srt` 与 `.json` 文件。

### 端到端训练/评估流程

```bash
python dataset_generation.py
python training.py -S L -C model.pt
python evaluate.py -I ./data/raw/flickr30k_images -R ./data/result/eval -S L -C model.pt -T 1.0
```

## 🧭 开发说明

- `v2c.py`、`video2caption.py` 与 `video2caption_v1.*` 之间存在遗留功能重叠。
- `video2caption_v1.0_not_work.py` 故意保留为不可用的历史遗留代码。
- `training.py` 当前使用 `config = ConfigL() if args.size.upper() else ConfigS()` 选择配置，非空 `--size` 会始终解析到 `ConfigL`。
- `model/trainer.py` 在 `test_step` 中使用 `self.dataset`，但初始化时赋值的是 `self.test_dataset`；这会在训练运行时导致采样问题，需修正后再使用。
- `video2caption_v1.1.py` 引用了 `self.config.transform`，但 `ConfigS`/`ConfigL` 并未定义该字段。
- 本仓库当前未定义 CI / 测试套件。
- i18n 说明：语言栏位于本 README 顶部，翻译文件可在 `i18n/` 下新增。
- 当前状态说明：语言栏已指向 `i18n/README.ru.md`，但该文件在此快照中不存在。

## 🩺 故障排查

- `AssertionError: Image does not exist`
  - 确认 `-I/--img-path` 指向有效文件。
- `Dataset file not found. Downloading...`
  - `MiniFlickrDataset` 在 `data/processed/dataset.pkl` 缺失时抛出；先运行 `python dataset_generation.py`。
- `Path to the test image folder does not exist`
  - 确认 `evaluate.py -I` 指向存在的文件夹。
- 首次运行缓慢或失败
  - 首次运行会下载 Hugging Face 模型，并可能从 Google Drive 拉取检查点。
- `video2caption.py` 返回空字幕
  - 检查硬编码脚本路径与 Python 执行路径，或切换到 `v2c.py`。
- 训练中 `wandb` 要求登录
  - 运行 `wandb login`，或如有需要在 `training.py` 中手动禁用日志。

## 🛣️ 路线图

- 增加依赖锁文件（`requirements.txt` 或 `pyproject.toml`）以便复现安装。
- 将重复的视频流水线整合为一个主维护实现。
- 从遗留脚本中移除硬编码机器路径。
- 修复 `training.py` 与 `model/trainer.py` 中已知训练/评估边界问题。
- 增加自动化测试与 CI。
- 在语言栏列出的目标文件下补充 `i18n/` 的 README 翻译。

## 🤝 贡献

欢迎贡献。建议流程：

```bash
# 1) Fork 并 clone
git clone git@github.com:<your-user>/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip

# 2) 创建特性分支
git checkout -b feat/your-change

# 3) 修改并提交
git add .
git commit -m "feat: describe your change"

# 4) 推送并提 PR
git push origin feat/your-change
```

如果你修改了模型行为，请一并补充：

- 可复现的命令
- 修改前/后的样例输出
- 检查点与数据集假设说明

---

## 📄 许可证

当前仓库快照中没有许可证文件。

说明：在添加 `LICENSE` 文件前，重用/分发条款尚未定义。


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
