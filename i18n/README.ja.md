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

## 🧭 Quick Navigation

| セクション | 用途 |
|---|---|
| Snapshot | リポジトリの範囲と現行スクリプト構成を把握 |
| Overview | 目的と機能を確認 |
| Usage | CLI/API の操作手順を確認 |
| Troubleshooting | よくある実行エラーを迅速に解消 |
| Roadmap | 整理・改善の対象を把握 |

---

OpenAI の CLIP 視覚埋め込みと GPT 系言語モデルを組み合わせ、画像と動画の自然言語キャプションを生成する Python ツールキットです。

## 🧭 Snapshot

| 項目 | 内容 |
|---|---|
| 対応タスク | 画像と動画のキャプション生成 |
| 主な出力 | SRT 字幕、JSON トランスクリプト、キャプション付き画像 |
| 主要スクリプト | `i2c.py`、`v2c.py`、`image2caption.py` |
| レガシーパス | `video2caption.py` および派生版（履歴保持のため保持） |
| データセットの流れ | `data/raw/results.csv` + `data/raw/flickr30k_images/` |

## ✨ Overview

このリポジトリは以下を提供します。

- 画像キャプションと動画字幕生成の推論スクリプト。
- CLIP 視覚埋め込みを GPT-2 トークン埋め込みへ写像する学習パイプライン。
- Flickr30k 形式データ向けのデータセット生成ユーティリティ。
- 重みが存在しない場合、対応サイズのチェックポイントを自動ダウンロード。
- `i18n/` 配下の多言語 README（上部の言語バーを参照）。

現在の実装では、新規とレガシーのスクリプトが混在しています。いくつかのレガシーなファイルは参照用として残っており、以下で補足します。

## 🚀 Features

- `image2caption.py` による単画像キャプション。
- `v2c.py` か `video2caption.py` による動画キャプション（均一フレームサンプリング）。
- 実行時オプションをカスタマイズ可能:
  - フレーム数
  - モデルサイズ
  - サンプリング温度
  - チェックポイント名
- マルチプロセス/スレッド対応で動画推論を高速化。
- 出力成果物:
  - SRT 字幕ファイル（`.srt`）
  - `v2c.py` の JSON トランスクリプト（`.json`）
- CLIP + GPT2 マッピング実験用の学習・評価エントリポイント。

### At a glance

| 領域 | 主なスクリプト | 備考 |
|---|---|---|
| 画像キャプション | `image2caption.py`、`i2c.py`、`predict.py` | CLI と再利用可能なクラス |
| 動画キャプション | `v2c.py` | 推奨の保守対象 |
| レガシー動画フロー | `video2caption.py`、`video2caption_v1.1.py` | 機材依存の前提を含む |
| データセット構築 | `dataset_generation.py` | `data/processed/dataset.pkl` を生成 |
| 学習・評価 | `training.py`、`evaluate.py` | CLIP + GPT2 の写像を使用 |

## 🧱 Architecture (High Level)

`model/model.py` のコアモデルは 3 つの構成要素からなります。

1. `ImageEncoder`: CLIP 画像埋め込みを抽出。
2. `Mapping`: CLIP 埋め込みを GPT プレフィックス埋め込み列へ投影。
3. `TextDecoder`: GPT-2 言語モデルヘッドとして、自己回帰でキャプショントークンを生成。

学習（`Net.train_forward`）は、事前計算済みの CLIP 画像埋め込みとトークン化済みキャプションを使います。
推論（`Net.forward`）は PIL 画像を受け取り、EOS または `max_len` に達するまでトークンをデコードします。

### Data flow

1. データセット準備: `dataset_generation.py` が `data/raw/results.csv` と `data/raw/flickr30k_images/` の画像を読み取り、`data/processed/dataset.pkl` を生成します。
2. 学習: `training.py` が pickle 化されたタプル `(image_name, image_embedding, caption)` を読み込み、マッパー/デコーダ層を学習します。
3. 評価: `evaluate.py` がホールドアウト済みテスト画像に対して生成キャプションをレンダリングします。
4. 推論実行:
   - 画像: `image2caption.py` / `predict.py` / `i2c.py`
   - 動画: `v2c.py`（推奨）、`video2caption.py`（レガシー）

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

## 📋 前提条件

- Python `3.10+` を推奨。
- 学習と大規模モデル推論では CUDA 対応 GPU が望ましいですが、必須ではありません。
- 現行スクリプトでは `ffmpeg` は直接必要ありません（フレーム抽出は OpenCV）。
- Hugging Face / Google Drive から初回モデル・チェックポイントを取得する際に、ネットワーク接続が必要です。

現在、ロックファイルは存在しません（`requirements.txt` / `pyproject.toml` が未配置）ため、依存関係は import から推定されます。

## 🛠️ インストール

### 現行リポジトリ構成に沿った公式手順

```bash
git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers pillow matplotlib numpy tqdm opencv-python pandas wandb gdown
```

### 元 README のインストール断片（保管）

以前の README は途中で途切れていたため、ソース・オブ・トゥルースとして履歴のコマンドをそのまま残しています。

```bash
git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip/src
```

注記: 現在のスナップショットではスクリプトがリポジトリ直下にあり、`src/` 配下にはありません。

## ▶️ Quick Start

| 目的 | コマンド |
|---|---|
| 画像をキャプションする | `python image2caption.py -I /path/to/image.jpg -S L -C model.pt` |
| 動画をキャプションする | `python v2c.py -V /path/to/video.mp4 -N 10` |
| データセットを構築する | `python dataset_generation.py` |

### 画像キャプション（簡易）

```bash
python image2caption.py -I /path/to/image.jpg -S L -C model.pt
```

### 動画キャプション（推奨）

```bash
python v2c.py -V /path/to/video.mp4 -N 10
```

## 🎯 利用方法

### 1. 画像キャプション (`image2caption.py`)

```bash
python image2caption.py \
  -I /path/to/image.jpg \
  -S L \
  -C model.pt \
  -R ./data/result/prediction \
  -T 1.0
```

引数:

- `-I, --img-path`: 入力画像のパス。
- `-S, --size`: モデルサイズ（`S` または `L`）。
- `-C, --checkpoint-name`: `weights/{small|large}` 内のチェックポイント名。
- `-R, --res-path`: 生成画像の保存先ディレクトリ。
- `-T, --temperature`: サンプリング温度。

### 2. 代替画像 CLI (`predict.py`)

```bash
python predict.py \
  -I /path/to/image.jpg \
  -S L \
  -C model.pt \
  -R ./data/result/prediction \
  -T 1.0
```

`predict.py` は `image2caption.py` と機能的にはほぼ同等ですが、テキストの整形形式がわずかに異なります。

### 3. 画像キャプション用クラス API (`i2c.py`)

```bash
python i2c.py -I /path/to/image.jpg -S L -C model.pt -R ./data/result/prediction -T 1.0
```

または独自スクリプトでインポートします。

```python
from i2c import ImageCaptioner

captioner = ImageCaptioner(model_size="L", checkpoint_name="model.pt")
captioner.set_image_path("/path/to/image.jpg")
caption = captioner.generate_caption(save_image=True)
print(caption)
```

### 4. 動画→字幕 + JSON (`v2c.py`)

```bash
python v2c.py -V /path/to/video.mp4 -N 10
```

入力動画の隣に以下を出力します。

- `<video_basename>_caption.srt`
- `<video_basename>_caption.json`
- `<video_basename>_captioning_frames/`

### 5. 代替動画パイプライン (`video2caption.py`)

```bash
python video2caption.py -V /path/to/video.mp4 -N 10
```

重要: このスクリプトには現在、マシン固有のハードコードパスが含まれます。

- Python デフォルトパス: `/home/lachlan/miniconda3/envs/caption/bin/python`
- キャプション実行スクリプトパス: `/home/lachlan/Projects/image_captioning/clip-gpt-captioning/src/image2caption.py`

これらのパスを維持したまま使用する場合を除き、`v2c.py` を使ってください。

### 6. レガシー版 (`video2caption_v1.1.py`)

このスクリプトは履歴参照用に保持されています。実運用では `v2c.py` を推奨します。

### 7. データセット生成

```bash
python dataset_generation.py
```

想定入力:

- `data/raw/results.csv`（`|` 区切りのキャプション表）
- `data/raw/flickr30k_images/`（CSV が参照する画像ファイル）

出力:

- `data/processed/dataset.pkl`

### 8. 学習

```bash
python training.py -S L -C model.pt
```

学習時にはデフォルトで Weights & Biases（`wandb`）のロギングが有効です。

### 9. 評価

```bash
python evaluate.py \
  -I ./data/raw/flickr30k_images \
  -R ./data/result/eval \
  -S L \
  -C model.pt \
  -T 1.0
```

評価ではテスト画像に対して予測キャプションをレンダリングし、以下へ保存します。

- `<res-path>/<checkpoint_name_without_ext>_<SIZE>/`

## ⚙️ 設定

モデル設定は `utils/config.py` に定義されています。

| 設定 | CLIP バックボーン | GPT モデル | 重みディレクトリ |
|---|---|---|---|
| `ConfigS` | `openai/clip-vit-base-patch32` | `gpt2` | `weights/small` |
| `ConfigL` | `openai/clip-vit-large-patch14` | `gpt2-medium` | `weights/large` |

設定クラスの主要デフォルト:

| 項目 | `ConfigS` | `ConfigL` |
|---|---:|---:|
| `epochs` | 150 | 120 |
| `lr` | 3e-3 | 5e-3 |
| `batch_size_exp` | 6 | 5 |
| `ep_len` | 4 | 4 |
| `max_len` | 40 | 40 |

チェックポイント自動ダウンロード ID は `utils/downloads.py` にあります。

| サイズ | Google Drive ID |
|---|---|
| `L` | `1Gh32arzhW06C1ZJyzcJSSfdJDi3RgWoG` |
| `S` | `1pSQruQyg8KJq6VmzhMLFbT_VaHJMdlWF` |

## 📦 出力ファイル

### 画像推論

- `--res-path` に、上に文字が重畳された生成画像を保存。
- ファイル名: `<input_stem>-R<SIZE>.jpg`

### 動画推論 (`v2c.py`)

- SRT: `<video_stem>_caption.srt`
- JSON: `<video_stem>_caption.json`
- フレーム画像: `<video_stem>_captioning_frames/`

JSON 要素例:

```json
{
  "start": "00:00:03,200",
  "end": "00:00:03,700",
  "lang": "en",
  "text": "A dog running through a field."
}
```

## 🧪 使用例

### 画像キャプション（簡易）

```bash
python image2caption.py -I ./examples/dog.jpg -S S -C model.pt
```

想定される挙動:

- `weights/small/model.pt` が見つからない場合、自動でダウンロードされます。
- キャプション画像は既定で `./data/result/prediction` に保存されます。
- キャプション文は標準出力へ表示されます。

### 動画キャプション（簡易）

```bash
python v2c.py -V ./examples/demo.mp4 -N 8
```

想定される挙動:

- 8 枚の均一サンプリングフレームに対してキャプションを生成。
- 入力動画の隣に `.srt` と `.json` が生成されます。

### 学習・評価のエンドツーエンドシーケンス

```bash
python dataset_generation.py
python training.py -S L -C model.pt
python evaluate.py -I ./data/raw/flickr30k_images -R ./data/result/eval -S L -C model.pt -T 1.0
```

## 🧭 開発ノート

- `v2c.py`、`video2caption.py`、`video2caption_v1.*` の間で重複実装が残っています。
- `video2caption_v1.0_not_work.py` は意図的に非動作のレガシーコードとして保持。
- `training.py` は `config = ConfigL() if args.size.upper() else ConfigS()` で `--size` が空でない限り常に `ConfigL` を選びます。
- `model/trainer.py` は `test_step` で `self.dataset` を参照しますが、初期化では `self.test_dataset` を代入しているため、調整しないと学習時のサンプリングが崩れる可能性があります。
- `video2caption_v1.1.py` は `self.config.transform` を参照しますが、`ConfigS` / `ConfigL` は `transform` を定義していません。
- このリポジトリの現在のスナップショットでは CI/test suite が未定義です。
- i18n メモ: 言語リンクはこの README 冒頭にあります。翻訳ファイルは `i18n/` 配下に追加できます。
- 現在の状態メモ: 言語バーが `i18n/README.ru.md` を参照していますが、このスナップショットには該当ファイルがありません。

## 🩺 トラブルシューティング

- `AssertionError: Image does not exist`
  - `-I/--img-path` が有効なファイルを指しているか確認してください。
- `Dataset file not found. Downloading...`
  - `MiniFlickrDataset` は `data/processed/dataset.pkl` がない場合にこのメッセージを出し、`python dataset_generation.py` を先に実行する必要があります。
- `Path to the test image folder does not exist`
  - `evaluate.py -I` が既存フォルダを指しているか確認してください。
- 初回起動が遅い／失敗する
  - 初回は Hugging Face のモデル取得と Google Drive からのチェックポイント取得が発生するためです。
- `video2caption.py` が空のキャプションを返す
  - ハードコードパスや Python 実行パスを確認するか、`v2c.py` に切り替えてください。
- `wandb` が学習時にログインを要求する
  - `wandb login` を実行するか、必要に応じて `training.py` でロギングを手動で無効化してください。

## 🛣️ Roadmap

- 再現性を高めるため依存ロックファイル（`requirements.txt` または `pyproject.toml`）を追加。
- 重複した動画パイプラインを 1 つに統合し、維持対象を一本化。
- レガシースクリプトの機材依存ハードコードパスを除去。
- `training.py` と `model/trainer.py` の既知の学習・評価エッジケース不具合を修正。
- 自動テストと CI を追加。
- 言語バーで参照される `i18n/` 配下の README を翻訳版で揃える。

## 🤝 Contributing

コントリビューションは歓迎します。推奨ワークフロー:

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

モデル挙動を変更する場合は以下を含めてください。

- 再現可能なコマンド。
- 変更前後のサンプル出力。
- チェックポイントまたはデータセットの前提条件メモ。

---

## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## 📄 License

現在のリポジトリスナップショットにはライセンスファイルがありません。

注記: `LICENSE` ファイルが追加されるまでは、再利用・再配布の条件は未定義です。
