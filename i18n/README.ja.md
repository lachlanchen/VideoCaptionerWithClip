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

OpenAI CLIP の視覚埋め込みと GPT スタイルの言語モデルを組み合わせ、画像と動画に対して自然言語キャプションを生成する Python ツールキットです。

## ✨ 概要

このリポジトリには次が含まれます。

- 画像キャプション生成と動画字幕生成のための推論スクリプト。
- CLIP の視覚埋め込みから GPT-2 のトークン埋め込みへの写像を学習するトレーニングパイプライン。
- Flickr30k 形式データ向けのデータセット生成ユーティリティ。
- 重みが見つからない場合に、対応モデルサイズのチェックポイントを自動ダウンロード。
- `i18n/` 配下の多言語 README バリアント（上の言語バーを参照）。

現在の実装には新しいスクリプトとレガシースクリプトの両方が含まれています。一部のレガシーファイルは参照用に保持されており、以下で説明しています。

## 🚀 特徴

- `image2caption.py` による単一画像キャプション生成。
- `v2c.py` または `video2caption.py` による動画キャプション生成（フレーム一様サンプリング）。
- 実行時オプションをカスタマイズ可能:
  - フレーム数。
  - モデルサイズ。
  - サンプリング温度。
  - チェックポイント名。
- マルチプロセス/スレッド対応のキャプション生成による高速な動画推論。
- 出力成果物:
  - SRT 字幕ファイル（`.srt`）。
  - `v2c.py` での JSON トランスクリプト（`.json`）。
- CLIP+GPT2 写像実験向けの学習/評価エントリポイント。

### ひと目でわかる構成

| Area | Primary script(s) | Notes |
|---|---|---|
| Image captioning | `image2caption.py`, `i2c.py`, `predict.py` | CLI + 再利用可能クラス |
| Video captioning | `v2c.py` | 推奨される保守パス |
| Legacy video flow | `video2caption.py`, `video2caption_v1.1.py` | マシン依存の前提を含む |
| Dataset build | `dataset_generation.py` | `data/processed/dataset.pkl` を生成 |
| Train / eval | `training.py`, `evaluate.py` | CLIP+GPT2 写像を使用 |

## 🧱 アーキテクチャ（高レベル）

`model/model.py` の中核モデルは 3 つの部分で構成されています。

1. `ImageEncoder`: CLIP 画像埋め込みを抽出。
2. `Mapping`: CLIP 埋め込みを GPT のプレフィックス埋め込み系列へ射影。
3. `TextDecoder`: GPT-2 言語モデルヘッドが自己回帰でキャプショントークンを生成。

学習（`Net.train_forward`）では、事前計算された CLIP 画像埋め込みとトークン化済みキャプションを使用します。
推論（`Net.forward`）では PIL 画像を使い、EOS または `max_len` までトークンをデコードします。

### データフロー

1. データセット準備: `dataset_generation.py` が `data/raw/results.csv` と `data/raw/flickr30k_images/` 内の画像を読み取り、`data/processed/dataset.pkl` を書き出します。
2. 学習: `training.py` が pickle 化されたタプル `(image_name, image_embedding, caption)` を読み込み、mapper/decoder 層を学習します。
3. 評価: `evaluate.py` が保持されたテスト画像に生成キャプションを描画します。
4. 推論実行:
   - 画像: `image2caption.py` / `predict.py` / `i2c.py`。
   - 動画: `v2c.py`（推奨）、`video2caption.py`（レガシー）。

## 🗂️ プロジェクト構成

```text
VideoCaptionerWithClip/
├── README.md
├── image2caption.py               # 単一画像キャプション CLI
├── predict.py                     # 代替の単一画像キャプション CLI
├── i2c.py                         # 再利用可能な ImageCaptioner クラス + CLI
├── v2c.py                         # Video -> SRT + JSON（スレッド化フレームキャプション生成）
├── video2caption.py               # 代替 Video -> SRT 実装（レガシー制約あり）
├── video2caption_v1.1.py          # 旧バリアント
├── video2caption_v1.0_not_work.py # 非動作のレガシーファイルとして明示
├── training.py                    # モデル学習エントリポイント
├── evaluate.py                    # テスト分割評価と描画出力
├── dataset_generation.py          # data/processed/dataset.pkl を構築
├── data/
│   ├── __init__.py
│   └── dataset.py                 # Dataset + DataLoader ヘルパー
├── model/
│   ├── __init__.py
│   ├── model.py                   # CLIP encoder + mapping + GPT2 decoder
│   └── trainer.py                 # 学習/検証/テスト用ユーティリティクラス
├── utils/
│   ├── __init__.py
│   ├── config.py                  # ConfigS / ConfigL デフォルト
│   ├── downloads.py               # Google Drive チェックポイントダウンローダー
│   └── lr_warmup.py               # LR warmup スケジュール
├── i18n/                          # 多言語 README バリアント
└── .auto-readme-work/             # Auto-README パイプライン成果物
```

## 📋 前提条件

- Python `3.10+` 推奨。
- CUDA 対応 GPU は必須ではありませんが、学習や大規模モデル推論では強く推奨。
- 現在のスクリプトでは `ffmpeg` は直接不要です（フレーム抽出は OpenCV を使用）。
- Hugging Face / Google Drive から初回にモデル/チェックポイントをダウンロードするため、インターネット接続が必要です。

現時点では lockfile（`requirements.txt` / `pyproject.toml`）が存在しないため、依存関係は import から推定されます。

## 🛠️ インストール

### 現在のリポジトリ構成に基づく標準セットアップ

```bash
git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers pillow matplotlib numpy tqdm opencv-python pandas wandb gdown
```

### 元 README のインストールスニペット（保持）

以前の README はコードブロック途中で終わっていました。以下の元コマンドは、履歴上の一次情報としてソースどおりに保持しています。

```bash
git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip/src
```

注意: 現在のリポジトリスナップショットでは、スクリプトは `src/` 配下ではなくリポジトリ直下に配置されています。

## ▶️ クイックスタート

### 画像キャプション生成（クイック実行）

```bash
python image2caption.py -I /path/to/image.jpg -S L -C model.pt
```

### 動画キャプション生成（推奨パス）

```bash
python v2c.py -V /path/to/video.mp4 -N 10
```

## 🎯 使い方

### 1. 画像キャプション生成（`image2caption.py`）

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
- `-C, --checkpoint-name`: `weights/{small|large}` 内のチェックポイントファイル名。
- `-R, --res-path`: キャプション描画済み画像の出力ディレクトリ。
- `-T, --temperature`: サンプリング温度。

### 2. 代替画像 CLI（`predict.py`）

```bash
python predict.py \
  -I /path/to/image.jpg \
  -S L \
  -C model.pt \
  -R ./data/result/prediction \
  -T 1.0
```

`predict.py` は `image2caption.py` と機能的にはほぼ同じですが、出力テキストのフォーマットがわずかに異なります。

### 3. 画像キャプション生成クラス API（`i2c.py`）

```bash
python i2c.py -I /path/to/image.jpg -S L -C model.pt -R ./data/result/prediction -T 1.0
```

または自分のスクリプトで import:

```python
from i2c import ImageCaptioner

captioner = ImageCaptioner(model_size="L", checkpoint_name="model.pt")
captioner.set_image_path("/path/to/image.jpg")
caption = captioner.generate_caption(save_image=True)
print(caption)
```

### 4. 動画から字幕 + JSON（`v2c.py`）

```bash
python v2c.py -V /path/to/video.mp4 -N 10
```

入力動画の隣に次を出力します。

- `<video_basename>_caption.srt`
- `<video_basename>_caption.json`
- `<video_basename>_captioning_frames/`

### 5. 代替動画パイプライン（`video2caption.py`）

```bash
python video2caption.py -V /path/to/video.mp4 -N 10
```

重要: このスクリプトには現在、マシン依存のハードコードされたパスが含まれています。

- Python path default: `/home/lachlan/miniconda3/envs/caption/bin/python`
- Caption script path: `/home/lachlan/Projects/image_captioning/clip-gpt-captioning/src/image2caption.py`

これらのパスを意図的に保守する場合を除き、`v2c.py` の使用を推奨します。

### 6. レガシーバリアント（`video2caption_v1.1.py`）

このスクリプトは履歴参照のために保持されています。実運用では `v2c.py` を優先してください。

### 7. データセット生成

```bash
python dataset_generation.py
```

想定される生入力:

- `data/raw/results.csv`（パイプ区切りのキャプションテーブル）。
- `data/raw/flickr30k_images/`（CSV で参照される画像ファイル）。

出力:

- `data/processed/dataset.pkl`

### 8. 学習

```bash
python training.py -S L -C model.pt
```

学習ではデフォルトで Weights & Biases（`wandb`）ロギングを使用します。

### 9. 評価

```bash
python evaluate.py \
  -I ./data/raw/flickr30k_images \
  -R ./data/result/eval \
  -S L \
  -C model.pt \
  -T 1.0
```

評価では、予測キャプションをテスト画像に描画し、次の場所に保存します。

- `<res-path>/<checkpoint_name_without_ext>_<SIZE>/`

## ⚙️ 設定

モデル設定は `utils/config.py` で定義されています。

| Config | CLIP backbone | GPT model | Weights dir |
|---|---|---|---|
| `ConfigS` | `openai/clip-vit-base-patch32` | `gpt2` | `weights/small` |
| `ConfigL` | `openai/clip-vit-large-patch14` | `gpt2-medium` | `weights/large` |

設定クラスの主要デフォルト値:

| Field | `ConfigS` | `ConfigL` |
|---|---:|---:|
| `epochs` | 150 | 120 |
| `lr` | 3e-3 | 5e-3 |
| `batch_size_exp` | 6 | 5 |
| `ep_len` | 4 | 4 |
| `max_len` | 40 | 40 |

チェックポイント自動ダウンロード ID は `utils/downloads.py` にあります。

| Size | Google Drive ID |
|---|---|
| `L` | `1Gh32arzhW06C1ZJyzcJSSfdJDi3RgWoG` |
| `S` | `1pSQruQyg8KJq6VmzhMLFbT_VaHJMdlWF` |

## 📦 出力ファイル

### 画像推論

- `--res-path` に、タイトルを重ねた/生成した画像を保存。
- ファイル名パターン: `<input_stem>-R<SIZE>.jpg`。

### 動画推論（`v2c.py`）

- SRT: `<video_stem>_caption.srt`
- JSON: `<video_stem>_caption.json`
- フレーム画像: `<video_stem>_captioning_frames/`

JSON 要素の例:

```json
{
  "start": "00:00:03,200",
  "end": "00:00:03,700",
  "lang": "en",
  "text": "A dog running through a field."
}
```

## 🧪 例

### 画像キャプションのクイック例

```bash
python image2caption.py -I ./examples/dog.jpg -S S -C model.pt
```

期待される動作:

- `weights/small/model.pt` がない場合はダウンロードされます。
- デフォルトでは `./data/result/prediction` にキャプション付き画像が出力されます。
- キャプションテキストが標準出力に表示されます。

### 動画キャプションのクイック例

```bash
python v2c.py -V ./examples/demo.mp4 -N 8
```

期待される動作:

- 一様サンプリングした 8 フレームにキャプションが付与されます。
- 入力動画と同じ場所に `.srt` と `.json` が生成されます。

### 学習/評価のエンドツーエンド手順

```bash
python dataset_generation.py
python training.py -S L -C model.pt
python evaluate.py -I ./data/raw/flickr30k_images -R ./data/result/eval -S L -C model.pt -T 1.0
```

## 🧭 開発メモ

- `v2c.py`、`video2caption.py`、`video2caption_v1.*` の間にレガシー重複があります。
- `video2caption_v1.0_not_work.py` は、意図的に非動作のレガシーコードとして保持されています。
- `training.py` は現在 `config = ConfigL() if args.size.upper() else ConfigS()` を使っており、`--size` が空でない限り常に `ConfigL` に解決されます。
- `model/trainer.py` の `test_step` は `self.dataset` を使いますが、初期化時には `self.test_dataset` を代入しているため、調整しないと学習実行時のサンプリングが壊れる可能性があります。
- `video2caption_v1.1.py` は `self.config.transform` を参照しますが、`ConfigS`/`ConfigL` は `transform` を定義していません。
- このリポジトリスナップショットには CI/テストスイートが未定義です。
- i18n メモ: この README の先頭には言語リンクがあります。翻訳ファイルは `i18n/` 配下に追加できます。
- 現在の状態メモ: 言語バーは `i18n/README.ru.md` を指していますが、このスナップショットにはそのファイルがありません。

## 🩺 トラブルシューティング

- `AssertionError: Image does not exist`
  - `-I/--img-path` が有効なファイルを指しているか確認してください。
- `Dataset file not found. Downloading...`
  - `data/processed/dataset.pkl` が存在しない場合に `MiniFlickrDataset` がこのエラーを出します。先に `python dataset_generation.py` を実行してください。
- `Path to the test image folder does not exist`
  - `evaluate.py -I` が存在するフォルダを指しているか確認してください。
- 初回実行が遅い/失敗する
  - 初回実行では Hugging Face モデルをダウンロードし、Google Drive からチェックポイントもダウンロードする場合があります。
- `video2caption.py` が空のキャプションを返す
  - ハードコードされたスクリプトパスと Python 実行パスを確認するか、`v2c.py` に切り替えてください。
- 学習中に `wandb` ログインを求められる
  - `wandb login` を実行するか、必要に応じて `training.py` 内でロギングを手動で無効化してください。

## 🛣️ ロードマップ

- 再現可能なインストールのため、依存関係 lockfile（`requirements.txt` または `pyproject.toml`）を追加。
- 重複している動画パイプラインを 1 つの保守実装に統合。
- レガシースクリプトからマシン依存のハードコードパスを削除。
- `training.py` と `model/trainer.py` の既知の学習/評価エッジケースバグを修正。
- 自動テストと CI を追加。
- 言語バーで参照している README 翻訳ファイルを `i18n/` に揃える。

## 🤝 コントリビューション

コントリビューションを歓迎します。推奨ワークフロー:

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

モデル挙動を変更する場合は、次を含めてください。

- 再現可能なコマンド。
- 変更前/変更後のサンプル出力。
- チェックポイントまたはデータセット前提に関するメモ。

## 🙌 サポート

現在のリポジトリスナップショットには、寄付/スポンサー設定は明示されていません。

今後スポンサーリンクが追加された場合は、このセクションに保持してください。

## 📄 ライセンス

現在のリポジトリスナップショットにはライセンスファイルがありません。

前提メモ: `LICENSE` ファイルが追加されるまで、再利用/再配布の条件は未定義です。
