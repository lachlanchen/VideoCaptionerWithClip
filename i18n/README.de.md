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

Ein Python-Toolkit zur Erzeugung natürlichsprachlicher Bild- und Videobeschreibungen, das OpenAI-CLIP-Visual-Embeddings mit einem GPT-ähnlichen Sprachmodell kombiniert.

## ✨ Überblick

Dieses Repository bietet:

- Inferenz-Skripte für Bildbeschriftung und Video-Untertitel-Erzeugung.
- Eine Trainings-Pipeline, die eine Abbildung von CLIP-Visual-Embeddings auf GPT-2-Token-Embeddings lernt.
- Utilities zur Datensatzgenerierung für Daten im Flickr30k-Stil.
- Automatischen Checkpoint-Download für unterstützte Modellgrößen, wenn Gewichte fehlen.
- Mehrsprachige README-Varianten unter `i18n/` (siehe Sprachleiste oben).

Die aktuelle Implementierung enthält sowohl neuere als auch Legacy-Skripte. Einige Legacy-Dateien bleiben als Referenz erhalten und sind unten dokumentiert.

## 🚀 Features

- Einzelbild-Beschriftung über `image2caption.py`.
- Video-Beschriftung (gleichmäßiges Frame-Sampling) über `v2c.py` oder `video2caption.py`.
- Anpassbare Laufzeitoptionen:
  - Anzahl der Frames.
  - Modellgröße.
  - Sampling-Temperatur.
  - Checkpoint-Name.
- Multiprocessing-/threaded Captioning für schnellere Video-Inferenz.
- Ausgabe-Artefakte:
  - SRT-Untertiteldateien (`.srt`).
  - JSON-Transkripte (`.json`) in `v2c.py`.
- Einstiege für Training und Evaluation von CLIP+GPT2-Mapping-Experimenten.

### Auf einen Blick

| Bereich | Primäre Skript(e) | Hinweise |
|---|---|---|
| Bildbeschriftung | `image2caption.py`, `i2c.py`, `predict.py` | CLI + wiederverwendbare Klasse |
| Video-Beschriftung | `v2c.py` | Empfohlener, aktiv gepflegter Pfad |
| Legacy-Video-Flow | `video2caption.py`, `video2caption_v1.1.py` | Enthält maschinenspezifische Annahmen |
| Datensatzaufbau | `dataset_generation.py` | Erzeugt `data/processed/dataset.pkl` |
| Training / Eval | `training.py`, `evaluate.py` | Nutzt CLIP+GPT2-Mapping |

## 🧱 Architektur (High Level)

Das Kernmodell in `model/model.py` hat drei Teile:

1. `ImageEncoder`: extrahiert CLIP-Bild-Embeddings.
2. `Mapping`: projiziert CLIP-Embeddings in eine GPT-Prefix-Embedding-Sequenz.
3. `TextDecoder`: GPT-2-Sprachmodellkopf, der Caption-Token autoregressiv generiert.

Beim Training (`Net.train_forward`) werden vorab berechnete CLIP-Bild-Embeddings + tokenisierte Beschreibungen verwendet.
Die Inferenz (`Net.forward`) nutzt ein PIL-Bild und dekodiert Token bis EOS oder `max_len`.

### Datenfluss

1. Datensatz vorbereiten: `dataset_generation.py` liest `data/raw/results.csv` und Bilder in `data/raw/flickr30k_images/` und schreibt `data/processed/dataset.pkl`.
2. Training: `training.py` lädt serialisierte Tupel `(image_name, image_embedding, caption)` und trainiert Mapper-/Decoder-Layer.
3. Evaluation: `evaluate.py` rendert generierte Beschreibungen auf Holdout-Testbildern.
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
├── video2caption_v1.0_not_work.py # Explizit als nicht funktionsfähige Legacy-Datei markiert
├── training.py                    # Einstiegspunkt für Modelltraining
├── evaluate.py                    # Evaluation auf Test-Split und gerenderte Ausgaben
├── dataset_generation.py          # Baut data/processed/dataset.pkl
├── data/
│   ├── __init__.py
│   └── dataset.py                 # Datensatz- + DataLoader-Helfer
├── model/
│   ├── __init__.py
│   ├── model.py                   # CLIP-Encoder + Mapping + GPT2-Decoder
│   └── trainer.py                 # Utility-Klasse für Training/Validierung/Test
├── utils/
│   ├── __init__.py
│   ├── config.py                  # ConfigS / ConfigL-Defaults
│   ├── downloads.py               # Google-Drive-Checkpoint-Downloader
│   └── lr_warmup.py               # LR-Warmup-Schedule
├── i18n/                          # Mehrsprachige README-Varianten
└── .auto-readme-work/             # Artefakte der Auto-README-Pipeline
```

## 📋 Voraussetzungen

- Python `3.10+` empfohlen.
- CUDA-fähige GPU ist optional, aber für Training und Inferenz großer Modelle stark empfohlen.
- `ffmpeg` ist für die aktuellen Skripte nicht direkt erforderlich (OpenCV wird für Frame-Extraktion genutzt).
- Internetzugang wird beim ersten Lauf benötigt, um Modelle/Checkpoints von Hugging Face / Google Drive herunterzuladen.

Aktuell existiert kein Lockfile (`requirements.txt` / `pyproject.toml` fehlen), daher werden Abhängigkeiten aus den Imports abgeleitet.

## 🛠️ Installation

### Kanonisches Setup aus dem aktuellen Repository-Layout

```bash
git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers pillow matplotlib numpy tqdm opencv-python pandas wandb gdown
```

### Originaler Installations-Snippet aus der README (beibehalten)

Die vorherige README endete mitten in einem Block. Die ursprünglichen Befehle sind unten exakt als historischer Source-of-Truth-Inhalt beibehalten:

```bash
git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip/src
```

Hinweis: Im aktuellen Repository-Snapshot liegen die Skripte im Repo-Root, nicht unter `src/`.

## ▶️ Quick Start

### Bildbeschriftung (schneller Lauf)

```bash
python image2caption.py -I /path/to/image.jpg -S L -C model.pt
```

### Video-Beschriftung (empfohlener Pfad)

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

- `-I, --img-path`: Eingabebildpfad.
- `-S, --size`: Modellgröße (`S` oder `L`).
- `-C, --checkpoint-name`: Checkpoint-Dateiname in `weights/{small|large}`.
- `-R, --res-path`: Ausgabeverzeichnis für gerendertes Bild mit Beschreibung.
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

`predict.py` ist funktional ähnlich zu `image2caption.py`; die Ausgabeformatierung des Texts unterscheidet sich leicht.

### 3. Bildbeschriftungs-Klassen-API (`i2c.py`)

```bash
python i2c.py -I /path/to/image.jpg -S L -C model.pt -R ./data/result/prediction -T 1.0
```

Oder in ein eigenes Skript importieren:

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

### 5. Alternative Video-Pipeline (`video2caption.py`)

```bash
python video2caption.py -V /path/to/video.mp4 -N 10
```

Wichtig: Dieses Skript enthält derzeit maschinenspezifische, hartkodierte Pfade:

- Standard-Python-Pfad: `/home/lachlan/miniconda3/envs/caption/bin/python`
- Caption-Skriptpfad: `/home/lachlan/Projects/image_captioning/clip-gpt-captioning/src/image2caption.py`

Nutze `v2c.py`, außer du pflegst diese Pfade absichtlich weiter.

### 6. Legacy-Variante (`video2caption_v1.1.py`)

Dieses Skript bleibt als historische Referenz erhalten. Für aktive Nutzung `v2c.py` bevorzugen.

### 7. Datensatzgenerierung

```bash
python dataset_generation.py
```

Erwartete Rohdaten:

- `data/raw/results.csv` (durch Pipe getrennte Caption-Tabelle).
- `data/raw/flickr30k_images/` (Bilddateien, auf die sich die CSV bezieht).

Ausgabe:

- `data/processed/dataset.pkl`

### 8. Training

```bash
python training.py -S L -C model.pt
```

Training nutzt standardmäßig Weights-&-Biases-Logging (`wandb`).

### 9. Evaluation

```bash
python evaluate.py \
  -I ./data/raw/flickr30k_images \
  -R ./data/result/eval \
  -S L \
  -C model.pt \
  -T 1.0
```

Die Evaluation rendert vorhergesagte Beschreibungen auf Testbildern und speichert sie unter:

- `<res-path>/<checkpoint_name_without_ext>_<SIZE>/`

## ⚙️ Konfiguration

Modellkonfigurationen sind in `utils/config.py` definiert:

| Config | CLIP-Backbone | GPT-Modell | Weights-Verzeichnis |
|---|---|---|---|
| `ConfigS` | `openai/clip-vit-base-patch32` | `gpt2` | `weights/small` |
| `ConfigL` | `openai/clip-vit-large-patch14` | `gpt2-medium` | `weights/large` |

Wichtige Defaults aus den Config-Klassen:

| Feld | `ConfigS` | `ConfigL` |
|---|---:|---:|
| `epochs` | 150 | 120 |
| `lr` | 3e-3 | 5e-3 |
| `batch_size_exp` | 6 | 5 |
| `ep_len` | 4 | 4 |
| `max_len` | 40 | 40 |

Checkpoint-Auto-Download-IDs stehen in `utils/downloads.py`:

| Größe | Google-Drive-ID |
|---|---|
| `L` | `1Gh32arzhW06C1ZJyzcJSSfdJDi3RgWoG` |
| `S` | `1pSQruQyg8KJq6VmzhMLFbT_VaHJMdlWF` |

## 📦 Ausgabedateien

### Bild-Inferenz

- Gespeichertes Bild mit überlagertem/generiertem Titel unter `--res-path`.
- Dateinamensmuster: `<input_stem>-R<SIZE>.jpg`.

### Video-Inferenz (`v2c.py`)

- SRT: `<video_stem>_caption.srt`
- JSON: `<video_stem>_caption.json`
- Frame-Bilder: `<video_stem>_captioning_frames/`

Beispiel-JSON-Element:

```json
{
  "start": "00:00:03,200",
  "end": "00:00:03,700",
  "lang": "en",
  "text": "A dog running through a field."
}
```

## 🧪 Beispiele

### Schnelles Beispiel für Bildbeschriftung

```bash
python image2caption.py -I ./examples/dog.jpg -S S -C model.pt
```

Erwartetes Verhalten:

- Falls `weights/small/model.pt` fehlt, wird es heruntergeladen.
- Ein beschriftetes Bild wird standardmäßig nach `./data/result/prediction` geschrieben.
- Der Beschreibungstext wird auf stdout ausgegeben.

### Schnelles Beispiel für Video-Beschriftung

```bash
python v2c.py -V ./examples/demo.mp4 -N 8
```

Erwartetes Verhalten:

- 8 gleichmäßig gesampelte Frames werden beschriftet.
- `.srt`- und `.json`-Dateien werden neben dem Eingabevideo erzeugt.

### End-to-End-Sequenz für Training/Evaluation

```bash
python dataset_generation.py
python training.py -S L -C model.pt
python evaluate.py -I ./data/raw/flickr30k_images -R ./data/result/eval -S L -C model.pt -T 1.0
```

## 🧭 Entwicklungshinweise

- Legacy-Überschneidungen existieren zwischen `v2c.py`, `video2caption.py` und `video2caption_v1.*`.
- `video2caption_v1.0_not_work.py` bleibt absichtlich als nicht funktionsfähiger Legacy-Code erhalten.
- `training.py` wählt derzeit `ConfigL()` über `config = ConfigL() if args.size.upper() else ConfigS()`, was bei nicht-leeren `--size`-Werten immer zu `ConfigL` auflöst.
- `model/trainer.py` nutzt in `test_step` `self.dataset`, während der Initializer `self.test_dataset` setzt; das kann Sampling in Trainingsläufen brechen, wenn es nicht angepasst wird.
- `video2caption_v1.1.py` referenziert `self.config.transform`, aber `ConfigS`/`ConfigL` definieren `transform` nicht.
- In diesem Repository-Snapshot ist derzeit keine CI/Test-Suite definiert.
- i18n-Hinweis: Sprachlinks stehen oben in dieser README; übersetzte Dateien können unter `i18n/` ergänzt werden.
- Aktueller Stand: Die Sprachleiste verlinkt `i18n/README.ru.md`, aber diese Datei ist in diesem Snapshot nicht vorhanden.

## 🩺 Fehlerbehebung

- `AssertionError: Image does not exist`
  - Prüfen, ob `-I/--img-path` auf eine gültige Datei zeigt.
- `Dataset file not found. Downloading...`
  - `MiniFlickrDataset` löst dies aus, wenn `data/processed/dataset.pkl` fehlt; zuerst `python dataset_generation.py` ausführen.
- `Path to the test image folder does not exist`
  - Prüfen, ob `evaluate.py -I` auf einen existierenden Ordner zeigt.
- Langsamer oder fehlschlagender erster Lauf
  - Beim ersten Lauf werden Hugging-Face-Modelle und ggf. Checkpoints von Google Drive geladen.
- `video2caption.py` liefert leere Beschreibungen
  - Hartkodierten Skriptpfad und Python-Executable prüfen oder auf `v2c.py` wechseln.
- `wandb` fordert beim Training einen Login
  - `wandb login` ausführen oder Logging bei Bedarf manuell in `training.py` deaktivieren.

## 🛣️ Roadmap

- Dependency-Lockfiles (`requirements.txt` oder `pyproject.toml`) für reproduzierbare Installationen ergänzen.
- Doppelte Video-Pipelines in eine gepflegte Implementierung zusammenführen.
- Hartkodierte Maschinenpfade aus Legacy-Skripten entfernen.
- Bekannte Edge-Case-Bugs in `training.py` und `model/trainer.py` beheben.
- Automatisierte Tests und CI ergänzen.
- `i18n/` mit übersetzten README-Dateien füllen, die in der Sprachleiste referenziert werden.

## 🤝 Mitwirken

Beiträge sind willkommen. Vorgeschlagener Workflow:

```bash
# 1) Fork und klonen
git clone git@github.com:<your-user>/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip

# 2) Feature-Branch erstellen
git checkout -b feat/your-change

# 3) Änderungen machen und committen
git add .
git commit -m "feat: describe your change"

# 4) Push und PR öffnen
git push origin feat/your-change
```

Wenn du das Modellverhalten änderst, füge Folgendes hinzu:

- Reproduzierbare Befehle.
- Beispielausgaben vor/nach der Änderung.
- Hinweise zu Checkpoint- oder Datensatzannahmen.

## 🙌 Support

Im aktuellen Repository-Snapshot ist keine explizite Konfiguration für Spenden/Sponsoring vorhanden.

Falls später Sponsoring-Links ergänzt werden, sollten sie in diesem Abschnitt erhalten bleiben.

## 📄 Lizenz

Im aktuellen Repository-Snapshot ist keine Lizenzdatei vorhanden.

Annahme-Hinweis: Bis eine `LICENSE`-Datei hinzugefügt wird, sind Bedingungen für Wiederverwendung/Verteilung nicht definiert.
