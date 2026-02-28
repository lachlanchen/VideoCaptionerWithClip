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

OpenAI CLIP の視覚埋め込みと GPT 系言語モデルを組み合わせ、画像と動画に対する自然言語キャプションを生成する Python ツールキットです。

## 🧭 スナップショット

| 項目 | 内容 |
|---|---|
| 対応タスク | 画像と動画のキャプション生成 |
| 主な出力 | SRT 字幕、JSON トランスクリプト、キャプション付き画像 |
| 主要スクリプト | `i2c.py`、`v2c.py`、`image2caption.py` |
| レガシーパス | `video2caption.py` とそのバージョン群（履歴保持のため） |
| データセットフロー | `data/raw/results.csv` + `data/raw/flickr30k_images/` |

## ✨ 概要

このリポジトリは次を提供します。

- 画像キャプションと動画字幕生成の推論スクリプト。
- CLIP 視覚埋め込みを GPT-2 トークン埋め込みに写像する学習パイプライン。
- Flickr30k 形式データ向けのデータセット生成ユーティリティ。
- 重みが存在しない場合、対応サイズのチェックポイントを自動ダウンロード。
- `i18n/` 配下の多言語 README（上部の言語バー参照）。

現在の実装には新旧のスクリプトが共存しています。いくつかのレガシースクリプトは参照用として残っており、後述で説明します。

## 🚀 機能

- `image2caption.py` による単画像キャプション。
- `v2c.py` または `video2caption.py` による動画キャプション（均一フレームサンプリング）。
- 実行時設定をカスタマイズ可能:
  - フレーム数
  - モデルサイズ
  - サンプリング温度
  - チェックポイント名
- マルチプロセス/スレッド化による高速動画推論。
- 出力成果物:
  - SRT 字幕ファイル（`.srt`）
  - `v2c.py` の JSON トランスクリプト（`.json`）
- CLIP + GPT2 の写像実験向けトレーニング/評価エントリポイント。

### 概観

| 領域 | 主なスクリプト | 備考 |
|---|---|---|
| 画像キャプション | `image2caption.py`, `i2c.py`, `predict.py` | CLI + 再利用可能なクラス |
| 動画キャプション | `v2c.py` | 推奨で維持される経路 |
| レガシー動画フロー | `video2caption.py`, `video2caption_v1.1.py` | マシン依存前提を含む |
| データセット生成 | `dataset_generation.py` | `data/processed/dataset.pkl` を生成 |
| 学習/評価 | `training.py`, `evaluate.py` | CLIP+GPT2 写像を利用 |

## 🧱 アーキテクチャ（高レベル）

`model/model.py` の中核モデルは 3 つのパートで構成されます。

1. `ImageEncoder`: CLIP 画像埋め込みを抽出。
2. `Mapping`: CLIP 埋め込みを GPT プレフィックス埋め込み列へ射影。
3. `TextDecoder`: GPT-2 言語モデルヘッドとして、自己回帰的にキャプショントークンを生成。

学習（`Net.train_forward`）は事前計算済みの CLIP 画像埋め込みとトークン化済みキャプションを使います。
推論（`Net.forward`）では PIL 画像を受け取り、EOS または `max_len` に到達するまでトークンをデコードします。

### データフロー

1. データセット準備: `dataset_generation.py` が `data/raw/results.csv` と `data/raw/flickr30k_images/` の画像を読み取り、`data/processed/dataset.pkl` を作成。
2. 学習: `training.py` がピクル化されたタプル `(image_name, image_embedding, caption)` を読み込み、マッパー/デコーダ層を学習。
3. 評価: `evaluate.py` がホールドアウトしたテスト画像に対して生成キャプションをレンダリング。
4. 推論の実行:
   - 画像: `image2caption.py` / `predict.py` / `i2c.py`
   - 動画: `v2c.py`（推奨）, `video2caption.py`（レガシー）

## 🗂️ プロジェクト構成

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
- 学習と大きめモデルの推論には CUDA 対応 GPU が望ましい（必須ではありません）。
- 現在のスクリプトでは `ffmpeg` は直接必要ありません（フレーム抽出は OpenCV を使用）。
- Hugging Face / Google Drive から初回モデル/チェックポイントを取得するためにはネットワーク接続が必要です。

現在、ロックファイルは存在しません（`requirements.txt` / `pyproject.toml` が未配置）ので、依存関係は import から推定されます。

## 🛠️ インストール

### 現在のリポジトリ構成に合わせた正規セットアップ

```bash
git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers pillow matplotlib numpy tqdm opencv-python pandas wandb gdown
```

### 元 README 由来のインストール断片（保持）

以前の README は途中で途切れていたため、ソース・オブ・トゥルースとして歴史的コマンドをそのまま残します。

```bash
git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip/src
```

注記: 現在のリポジトリではスクリプトはルート配下にあり、`src/` 配下にはありません。

## ▶️ クイックスタート

| 目的 | コマンド |
|---|---|
| 画像のキャプション生成 | `python image2caption.py -I /path/to/image.jpg -S L -C model.pt` |
| 動画のキャプション生成 | `python v2c.py -V /path/to/video.mp4 -N 10` |
| データセット生成 | `python dataset_generation.py` |

### 画像キャプションの簡易実行

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
- `-C, --checkpoint-name`: `weights/{small|large}` 配下のチェックポイント名。
- `-R, --res-path`: キャプション画像の保存先ディレクトリ。
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

`predict.py` は `image2caption.py` と機能的にほぼ同等で、出力テキストの整形がわずかに異なります。

### 3. 画像キャプションクラス API (`i2c.py`)

```bash
python i2c.py -I /path/to/image.jpg -S L -C model.pt -R ./data/result/prediction -T 1.0
```

または独自スクリプトからインポートできます。

```python
from i2c import ImageCaptioner

captioner = ImageCaptioner(model_size="L", checkpoint_name="model.pt")
captioner.set_image_path("/path/to/image.jpg")
caption = captioner.generate_caption(save_image=True)
print(caption)
```

### 4. 動画 -> 字幕＋JSON (`v2c.py`)

```bash
python v2c.py -V /path/to/video.mp4 -N 10
```

入力動画の隣に以下が出力されます。

- `<video_basename>_caption.srt`
- `<video_basename>_caption.json`
- `<video_basename>_captioning_frames/`

### 5. 代替動画パイプライン (`video2caption.py`)

```bash
python video2caption.py -V /path/to/video.mp4 -N 10
```

重要: このスクリプトは現在、マシン依存のハードコードパスを含みます。

- Python デフォルトパス: `/home/lachlan/miniconda3/envs/caption/bin/python`
- キャプション実行スクリプトパス: `/home/lachlan/Projects/image_captioning/clip-gpt-captioning/src/image2caption.py`

これらのパスを意図的に維持する場合を除き、`v2c.py` を使用してください。

### 6. レガシー版 (`video2caption_v1.1.py`)

このスクリプトは履歴参照用に保持されています。実運用では `v2c.py` を推奨します。

### 7. データセット生成

```bash
python dataset_generation.py
```

想定される入力:

- `data/raw/results.csv`（`|` 区切りのキャプション表）
- `data/raw/flickr30k_images/`（CSV で参照される画像）

出力:

- `data/processed/dataset.pkl`

### 8. 学習

```bash
python training.py -S L -C model.pt
```

学習は既定で Weights & Biases（`wandb`）ロギングを使用します。

### 9. 評価

```bash
python evaluate.py \
  -I ./data/raw/flickr30k_images \
  -R ./data/result/eval \
  -S L \
  -C model.pt \
  -T 1.0
```

評価ではホールドアウトしたテスト画像上に予測キャプションを描画し、次の場所へ保存します。

- `<res-path>/<checkpoint_name_without_ext>_<SIZE>/`

## ⚙️ 設定

モデル設定は `utils/config.py` に定義されています。

| 設定名 | CLIP バックボーン | GPT モデル | 重みディレクトリ |
|---|---|---|---|
| `ConfigS` | `openai/clip-vit-base-patch32` | `gpt2` | `weights/small` |
| `ConfigL` | `openai/clip-vit-large-patch14` | `gpt2-medium` | `weights/large` |

各設定クラスの主要デフォルト:

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

- `--res-path` へ、上にテキストを重畳して保存された画像を出力。
- ファイル名パターン: `<input_stem>-R<SIZE>.jpg`。

### 動画推論 (`v2c.py`)

- SRT: `<video_stem>_caption.srt`
- JSON: `<video_stem>_caption.json`
- フレーム画像: `<video_stem>_captioning_frames/`

JSON の要素例:

```json
{
  "start": "00:00:03,200",
  "end": "00:00:03,700",
  "lang": "en",
  "text": "A dog running through a field."
}
```

## 🧪 使用例

### 画像キャプションの簡易例

```bash
python image2caption.py -I ./examples/dog.jpg -S S -C model.pt
```

想定される動作:

- `weights/small/model.pt` が存在しない場合は自動でダウンロードされます。
- キャプション付き画像は既定で `./data/result/prediction` に保存されます。
- キャプション本文は標準出力へ出力されます。

### 動画キャプションの簡易例

```bash
python v2c.py -V ./examples/demo.mp4 -N 8
```

想定される動作:

- 8 つの均一サンプリングフレームがキャプション化されます。
- 入力動画の隣に `.srt` と `.json` が生成されます。

### エンドツーエンドの学習・評価シーケンス

```bash
python dataset_generation.py
python training.py -S L -C model.pt
python evaluate.py -I ./data/raw/flickr30k_images -R ./data/result/eval -S L -C model.pt -T 1.0
```

## 🧭 開発ノート

- `v2c.py`、`video2caption.py`、`video2caption_v1.*` 間に重複が残っています。
- `video2caption_v1.0_not_work.py` は意図的に非動作のレガシーコードとして保持。
- `training.py` は `config = ConfigL() if args.size.upper() else ConfigS()` で `config` を選択しており、`--size` が空でない場合は常に `ConfigL` になります。
- `model/trainer.py` の `test_step` では `self.dataset` を参照しますが、初期化側は `self.test_dataset` を代入しているため、調整しないと学習実行時にサンプリングが壊れる可能性があります。
- `video2caption_v1.1.py` は `self.config.transform` を参照しますが、`ConfigS` / `ConfigL` は `transform` を定義していません。
- このスナップショット時点で CI/テストスイートは定義されていません。
- i18n の注: 言語リンクはこの README の冒頭にあります。翻訳ファイルは `i18n/` に追加可能です。
- 現状ノート: 言語バーは `i18n/README.ru.md` を指していますが、このスナップショットには当該ファイルがありません。

## 🩺 トラブルシューティング

- `AssertionError: Image does not exist`
  - `-I/--img-path` が有効なファイルを指しているか確認します。
- `Dataset file not found. Downloading...`
  - `MiniFlickrDataset` は `data/processed/dataset.pkl` が欠落しているとこのメッセージを出し、`python dataset_generation.py` の先行実行を要求します。
- `Path to the test image folder does not exist`
  - `evaluate.py -I` が既存のフォルダを指しているか確認します。
- 初回実行が遅い／失敗する
  - 初回は Hugging Face モデルのダウンロードや Google Drive からのチェックポイント取得が発生するためです。
- `video2caption.py` が空のキャプションを返す
  - ハードコードされたスクリプトパスや Python 実行パスを確認するか、`v2c.py` に切り替えます。
- 学習中に `wandb` がログインを要求する
  - `wandb login` を実行するか、必要に応じて `training.py` で手動でロギングを無効化してください。

## 🛣️ ロードマップ

- 再現性のあるインストールのため、依存ロックファイル（`requirements.txt` または `pyproject.toml`）を追加。
- 重複した動画パイプラインを 1 つに統一し、単一の維持対象実装へ集約。
- レガシースクリプトからマシン依存パスを除去。
- `training.py` と `model/trainer.py` の既知の学習・評価エッジケースバグを修正。
- 自動テストと CI を追加。
- 言語バーで参照されている `i18n/` 配下の翻訳 README を揃える。

## 🤝 貢献

コントリビューションは歓迎します。推奨フロー:

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

- 再現可能なコマンド
- 変更前/変更後のサンプル出力
- チェックポイントまたはデータセット前提の注記

## ❤️ Support

| Donate | PayPal | Stripe |
|---|---|---|
| [![Donate](https://img.shields.io/badge/Donate-LazyingArt-0EA5E9?style=for-the-badge&logo=ko-fi&logoColor=white)](https://chat.lazying.art/donate) | [![PayPal](https://img.shields.io/badge/PayPal-RongzhouChen-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/RongzhouChen) | [![Stripe](https://img.shields.io/badge/Stripe-Donate-635BFF?style=for-the-badge&logo=stripe&logoColor=white)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## 📄 ライセンス

現在のリポジトリスナップショットにはライセンスファイルがありません。

注記: `LICENSE` ファイルが追加されるまでは、再利用・再配布条件は未定義です。
