[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lazyingchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lazyingchen/blob/main/figs/banner.png)

# Clip-GPT-Captioning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Status](https://img.shields.io/badge/README-Expanded-success)
![Repo Layout](https://img.shields.io/badge/Layout-Root%20Scripts-informational)
![Legacy Scripts](https://img.shields.io/badge/Legacy%20Scripts-Present-orange)
![i18n](https://img.shields.io/badge/i18n-Enabled-brightgreen)
![Maintained Path](https://img.shields.io/badge/Video-v2c.py-2ea44f)

Ein Python-Toolkit zur Generierung natürlicher Bild- und Videobeschriftungen, indem OpenAI CLIP-Vision-Embeddings mit einem GPT-ähnlichen Sprachmodell kombiniert werden.

## 🧭 Snapshot

| Dimension | Details |
|---|---|
| Aufgabenbereich | Bild- und Videobeschriftung |
| Zentrale Ausgaben | SRT-Untertitel, JSON-Transkripte, beschriftete Bilder |
| Primäre Skripte | `i2c.py`, `v2c.py`, `image2caption.py` |
| Legacy-Pfade | `video2caption.py` und versionsspezifische Brüder (aus historischen Gründen erhalten) |
| Datensatzfluss | `data/raw/results.csv` + `data/raw/flickr30k_images/` |

## ✨ Überblick

Dieses Repository bietet:

- Inferenz-Skripte für Bildbeschriftung und Untertitelung von Videos.
- Eine Trainings-Pipeline, die eine Abbildung von CLIP-Image-Embeddings auf GPT-2-Token-Embeddings lernt.
- Werkzeuge zur Datensatzgenerierung im Stil von Flickr30k.
- Automatischen Checkpoint-Download für unterstützte Modellgrößen, wenn Gewichte fehlen.
- Mehrsprachige README-Varianten in `i18n/` (siehe Sprachleiste oben).

Die aktuelle Implementierung enthält sowohl neuere als auch ältere Skripte. Einige Legacy-Dateien werden als Referenz aufbewahrt und sind unten dokumentiert.

## 🚀 Features

- Einzelbild-Captioning über `image2caption.py`.
- Video-Captioning (gleichmäßiges Frame-Sampling) über `v2c.py` oder `video2caption.py`.
- Anpassbare Laufzeitoptionen:
  - Anzahl der Frames.
  - Modellgröße.
  - Sampling-Temperatur.
  - Checkpoint-Name.
- Multiprocessing-/Threaded-Captioning für schnellere Video-Inferenz.
- Ausgabeartefakte:
  - SRT-Untertiteldateien (`.srt`).
  - JSON-Transkripte (`.json`) in `v2c.py`.
- Trainings- und Evaluations-Einstiegspunkte für CLIP+GPT2-Mapping-Experimente.

### Auf einen Blick

| Bereich | Hauptskript(e) | Hinweise |
|---|---|---|
| Bildbeschriftung | `image2caption.py`, `i2c.py`, `predict.py` | CLI + wiederverwendbare Klasse |
| Videobeschriftung | `v2c.py` | Empfohlener stabiler Pfad |
| Legacy-Videofluss | `video2caption.py`, `video2caption_v1.1.py` | Enthält gerätespezifische Annahmen |
| Datensatzaufbau | `dataset_generation.py` | Erzeugt `data/processed/dataset.pkl` |
| Training / Evaluation | `training.py`, `evaluate.py` | Nutzt CLIP+GPT2-Mapping |

## 🧱 Architektur (High Level)

Das Kernmodell in `model/model.py` hat drei Teile:

1. `ImageEncoder`: extrahiert CLIP-Image-Embeddings.
2. `Mapping`: projiziert CLIP-Embeddings in eine GPT-Prefix-Embedding-Sequenz.
3. `TextDecoder`: GPT-2-Sprachmodellkopf, der Captions autoregressiv tokenweise generiert.

Training (`Net.train_forward`) nutzt vorab berechnete CLIP-Image-Embeddings + tokenisierte Captions.
Inferenz (`Net.forward`) verwendet ein PIL-Bild und dekodiert Tokens bis EOS oder `max_len`.

### Datenfluss

1. Datensatz vorbereiten: `dataset_generation.py` liest `data/raw/results.csv` und Bilder in `data/raw/flickr30k_images/`, schreibt `data/processed/dataset.pkl`.
2. Trainieren: `training.py` lädt gepickelte Tupel `(image_name, image_embedding, caption)` und trainiert Mapping-/Decoder-Schichten.
3. Evaluieren: `evaluate.py` rendert generierte Captions auf zurückgehaltene Testbilder.
4. Inferenz ausführen:
   - Bild: `image2caption.py` / `predict.py` / `i2c.py`.
   - Video: `v2c.py` (empfohlen), `video2caption.py` (Legacy).

## 🗂️ Projektstruktur

```text
VideoCaptionerWithClip/
├── README.md
├── image2caption.py               # Einzelbild-Caption-CLI
├── predict.py                     # Alternative Einzelbild-Caption-CLI
├── i2c.py                         # Wiederverwendbare ImageCaptioner-Klasse + CLI
├── v2c.py                         # Video -> SRT + JSON (threaded Frame-Captioning)
├── video2caption.py               # Alternative Video -> SRT-Implementierung (Legacy-Einschränkungen)
├── video2caption_v1.1.py          # Ältere Variante
├── video2caption_v1.0_not_work.py # Explizit als nicht funktionierende Legacy-Datei markiert
├── training.py                    # Einstiegspunkt für Modelltraining
├── evaluate.py                    # Evaluation auf Test-Split und gerenderte Ausgaben
├── dataset_generation.py          # Baut data/processed/dataset.pkl
├── data/
│   ├── __init__.py
│   └── dataset.py                 # Datensatz- + DataLoader-Helfer
├── model/
│   ├── __init__.py
│   ├── model.py                   # CLIP-Encoder + Mapping + GPT2-Decoder
│   └── trainer.py                 # Hilfsklasse für Training/Validierung/Test
├── utils/
│   ├── __init__.py
│   ├── config.py                  # ConfigS / ConfigL Defaults
│   ├── downloads.py               # Google-Drive-Checkpoint-Downloader
│   └── lr_warmup.py               # LR-Warmup-Zeitplan
├── i18n/                          # Mehrsprachige README-Varianten
└── .auto-readme-work/             # Auto-README-Pipeline-Artefakte
```

## 📋 Voraussetzungen

- Python `3.10+` wird empfohlen.
- Eine CUDA-fähige GPU ist optional, aber für Training und Inferenz großer Modelle stark empfohlen.
- `ffmpeg` wird von den aktuellen Skripten nicht direkt benötigt (OpenCV wird zur Frame-Extraktion verwendet).
- Für den ersten Download von Modellen/Checkpoints aus Hugging Face / Google Drive ist Internetzugang erforderlich.

Aktuell ist keine Lockfile-Datei vorhanden (`requirements.txt` / `pyproject.toml` fehlen), daher werden Abhängigkeiten aus den Imports abgeleitet.

## 🛠️ Installation

### Standard-Setup aus dem aktuellen Repository-Layout

```bash
git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers pillow matplotlib numpy tqdm opencv-python pandas wandb gdown
```

### Ursprüngliches README-Installations-Snippet (beibehalten)

Die frühere README endete in der Mitte eines Blocks. Die ursprünglichen Befehle sind unten exakt als historische Referenz unverändert übernommen:

```bash
git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip/src
```

Hinweis: In der aktuellen Repository-Struktur liegen die Skripte im Root und nicht unter `src/`.

## ▶️ Schnellstart

| Ziel | Befehl |
|---|---|
| Ein Bild beschriften | `python image2caption.py -I /path/to/image.jpg -S L -C model.pt` |
| Ein Video beschriften | `python v2c.py -V /path/to/video.mp4 -N 10` |
| Datensatz aufbauen | `python dataset_generation.py` |

### Bildbeschriftung (schneller Durchlauf)

```bash
python image2caption.py -I /path/to/image.jpg -S L -C model.pt
```

### Video-Beschriftung (empfohlener Weg)

```bash
python v2c.py -V /path/to/video.mp4 -N 10
```

## 🎯 Nutzung

### 1. Bildbeschriftung (`image2caption.py`)

```bash
python image2caption.py \
  -I /path/to/image.jpg \
  -S L \
  -C model.pt \
  -R ./data/result/prediction \
  -T 1.0
```

Argumente:

- `-I, --img-path`: Pfad des Eingabebildes.
- `-S, --size`: Modellgröße (`S` oder `L`).
- `-C, --checkpoint-name`: Checkpoint-Dateiname in `weights/{small|large}`.
- `-R, --res-path`: Ausgabeverzeichnis für gerendertes Bild mit Caption.
- `-T, --temperature`: Sampling-Temperatur.

### 2. Alternative Bild-CLI (`predict.py`)

```bash
python predict.py \
  -I /path/to/image.jpg \
  -S L \
  -C model.pt \
  -R ./data/result/prediction \
  -T 1.0
```

`predict.py` ist funktional ähnlich zu `image2caption.py`; die Textformatierung der Ausgabe unterscheidet sich leicht.

### 3. Bildbeschriftungs-Klassen-API (`i2c.py`)

```bash
python i2c.py -I /path/to/image.jpg -S L -C model.pt -R ./data/result/prediction -T 1.0
```

Oder importieren in einem eigenen Skript:

```python
from i2c import ImageCaptioner

captioner = ImageCaptioner(model_size="L", checkpoint_name="model.pt")
captioner.set_image_path("/path/to/image.jpg")
caption = captioner.generate_caption(save_image=True)
print(caption)
```

### 4. Video zu Untertiteln + JSON (`v2c.py`)

```bash
python v2c.py -V /path/to/video.mp4 -N 10
```

Ausgaben neben dem Eingabevideo:

- `<video_basename>_caption.srt`
- `<video_basename>_caption.json`
- `<video_basename>_captioning_frames/`

### 5. Alternative Videopipeline (`video2caption.py`)

```bash
python video2caption.py -V /path/to/video.mp4 -N 10
```

Wichtig: Dieses Skript enthält derzeit maschinenspezifische hartkodierte Pfade:

- Standard-Python-Pfad: `/home/lachlan/miniconda3/envs/caption/bin/python`
- Caption-Skriptpfad: `/home/lachlan/Projects/image_captioning/clip-gpt-captioning/src/image2caption.py`

Nutze `v2c.py`, es sei denn, du pflegst diese Pfade absichtlich weiter.

### 6. Legacy-Variante (`video2caption_v1.1.py`)

Dieses Skript wird als historische Referenz aufbewahrt. Für aktive Nutzung bitte `v2c.py` bevorzugen.

### 7. Datensatzgenerierung

```bash
python dataset_generation.py
```

Erwartete Rohdaten:

- `data/raw/results.csv` (Pipes-getrennte Caption-Tabelle).
- `data/raw/flickr30k_images/` (Bilddateien, auf die sich die CSV bezieht).

Ausgabe:

- `data/processed/dataset.pkl`

### 8. Training

```bash
python training.py -S L -C model.pt
```

Training nutzt standardmäßig Weights & Biases-Logging (`wandb`).

### 9. Evaluation

```bash
python evaluate.py \
  -I ./data/raw/flickr30k_images \
  -R ./data/result/eval \
  -S L \
  -C model.pt \
  -T 1.0
```

Evaluation rendert vorhergesagte Captions auf Testbildern und speichert sie unter:

- `<res-path>/<checkpoint_name_without_ext>_<SIZE>/`

## ⚙️ Konfiguration

Modellkonfigurationen sind in `utils/config.py` definiert:

| Config | CLIP-Backbone | GPT-Modell | Gewichtsordner |
|---|---|---|---|
| `ConfigS` | `openai/clip-vit-base-patch32` | `gpt2` | `weights/small` |
| `ConfigL` | `openai/clip-vit-large-patch14` | `gpt2-medium` | `weights/large` |

Wichtige Standardwerte aus den Config-Klassen:

| Feld | `ConfigS` | `ConfigL` |
|---|---:|---:|
| `epochs` | 150 | 120 |
| `lr` | 3e-3 | 5e-3 |
| `batch_size_exp` | 6 | 5 |
| `ep_len` | 4 | 4 |
| `max_len` | 40 | 40 |

Automatische Checkpoint-Download-IDs befinden sich in `utils/downloads.py`:

| Größe | Google Drive-ID |
|---|---|
| `L` | `1Gh32arzhW06C1ZJyzcJSSfdJDi3RgWoG` |
| `S` | `1pSQruQyg8KJq6VmzhMLFbT_VaHJMdlWF` |

## 📦 Ausgabedateien

### Bild-Inferenz

- Gespeichertes Bild mit überlagerter / generierter Überschrift unter `--res-path`.
- Dateinamenmuster: `<input_stem>-R<SIZE>.jpg`.

### Video-Inferenz (`v2c.py`)

- SRT: `<video_stem>_caption.srt`
- JSON: `<video_stem>_caption.json`
- Frame-Bilder: `<video_stem>_captioning_frames/`

Beispiel für ein JSON-Element:

```json
{
  "start": "00:00:03,200",
  "end": "00:00:03,700",
  "lang": "en",
  "text": "A dog running through a field."
}
```

## 🧪 Beispiele

### Schnelles Bildbeschriftungsbeispiel

```bash
python image2caption.py -I ./examples/dog.jpg -S S -C model.pt
```

Erwartetes Verhalten:

- Falls `weights/small/model.pt` fehlt, wird sie heruntergeladen.
- Standardmäßig wird ein Bild mit Caption in `./data/result/prediction` geschrieben.
- Der Beschriftungstext wird auf stdout ausgegeben.

### Schnelles Video-Beschriftungsbeispiel

```bash
python v2c.py -V ./examples/demo.mp4 -N 8
```

Erwartetes Verhalten:

- 8 gleichmäßig gesampelte Frames werden beschriftet.
- `.srt`- und `.json`-Dateien werden neben dem Eingabevideo erzeugt.

### End-to-End-Abfolge für Training/Evaluation

```bash
python dataset_generation.py
python training.py -S L -C model.pt
python evaluate.py -I ./data/raw/flickr30k_images -R ./data/result/eval -S L -C model.pt -T 1.0
```

## 🧭 Entwicklungshinweise

- Legacy-Überschneidungen bestehen zwischen `v2c.py`, `video2caption.py` und `video2caption_v1.*`.
- `video2caption_v1.0_not_work.py` wird absichtlich als nicht funktionsfähiger Legacy-Code beibehalten.
- `training.py` wählt derzeit `ConfigL()` über `config = ConfigL() if args.size.upper() else ConfigS()`; für nicht-leere `--size`-Werte wird dadurch immer `ConfigL` verwendet.
- `model/trainer.py` nutzt in `test_step` `self.dataset`, obwohl der Initializer `self.test_dataset` setzt; das kann das Sampling in Trainingsläufen brechen, falls nicht angepasst.
- `video2caption_v1.1.py` referenziert `self.config.transform`, aber `ConfigS`/`ConfigL` definieren `transform` nicht.
- In diesem Repository-Snapshot ist derzeit keine CI/Test-Suite definiert.
- i18n-Hinweis: Sprachlinks sind am Anfang dieser README vorhanden; unter `i18n/` können übersetzte Dateien ergänzt werden.
- Aktueller Stand: Die Sprachleiste verlinkt auf `i18n/README.ru.md`, doch diese Datei ist in diesem Snapshot nicht vorhanden.

## 🩺 Fehlerbehebung

- `AssertionError: Image does not exist`
  - Prüfe, ob `-I/--img-path` auf eine gültige Datei verweist.
- `Dataset file not found. Downloading...`
  - `MiniFlickrDataset` wirft diese Meldung, wenn `data/processed/dataset.pkl` fehlt; zuerst `python dataset_generation.py` ausführen.
- `Path to the test image folder does not exist`
  - Prüfe, ob `evaluate.py -I` auf einen existierenden Ordner zeigt.
- Langsame oder fehlerhafte erste Ausführung
  - Der erste Lauf lädt Modelle von Hugging Face und ggf. Checkpoints von Google Drive herunter.
- `video2caption.py` gibt leere Beschriftungen aus
  - Überprüfe den hartkodierten Skriptpfad und den Python-Executable-Pfad oder wechsle auf `v2c.py`.
- `wandb` fragt beim Training nach Anmeldung
  - `wandb login` ausführen oder Logging in `training.py` bei Bedarf manuell deaktivieren.

## 🛣️ Roadmap

- Abhängigkeits-Lockfiles (`requirements.txt` oder `pyproject.toml`) für reproduzierbare Installationen ergänzen.
- Doppelte Video-Pipelines in eine gepflegte Implementierung konsolidieren.
- Harte, gerätespezifische Pfade aus Legacy-Skripten entfernen.
- Bekannte Trainings-/Evaluations-Edge-Cases in `training.py` und `model/trainer.py` beheben.
- Automatisierte Tests und CI hinzufügen.
- `i18n/` mit den in der Sprachleiste referenzierten Übersetzungen füllen.

## 🤝 Mitwirken

Beiträge sind willkommen. Vorgeschlagener Ablauf:

```bash
# 1) Fork und Klonen
git clone git@github.com:<your-user>/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip

# 2) Feature-Branch erstellen
git checkout -b feat/your-change

# 3) Änderungen durchführen und committen
git add .
git commit -m "feat: describe your change"

# 4) Pushen und PR öffnen
git push origin feat/your-change
```

Wenn du das Modellverhalten änderst, füge bitte hinzu:

- Reproduzierbare(n) Befehl(e).
- Vorher-/Nachher-Beispielausgaben.
- Hinweise zu Checkpoint- oder Datensatzannahmen.

## ❤️ Support

| Donate | PayPal | Stripe |
|---|---|---|
| [![Donate](https://img.shields.io/badge/Donate-LazyingArt-0EA5E9?style=for-the-badge&logo=ko-fi&logoColor=white)](https://chat.lazying.art/donate) | [![PayPal](https://img.shields.io/badge/PayPal-RongzhouChen-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/RongzhouChen) | [![Stripe](https://img.shields.io/badge/Stripe-Donate-635BFF?style=for-the-badge&logo=stripe&logoColor=white)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## 📄 Lizenz

Keine Lizenzdatei ist in der aktuellen Repository-Version vorhanden.

Annahmepostulat: Bis eine `LICENSE`-Datei hinzugefügt wird, sind Nutzungs-/Verteilungsbedingungen nicht festgelegt.
