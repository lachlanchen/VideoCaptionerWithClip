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

## 🧭 Schnellnavigation

| Abschnitt | Wofür ist er gedacht |
|---|---|
| Snapshot | Verzeichnisumfang und aktuelles Skript-Inventar einsehen |
| Überblick | Ziele und Fähigkeiten lesen |
| Nutzung | Exakte CLI/API-Workflows befolgen |
| Fehlerbehebung | Häufige Laufzeitprobleme schnell lösen |
| Roadmap | Bekannte Cleanup- und Verbesserungsziele verfolgen |

---

Ein Python-Toolkit zur Generierung von natürlichsprachigen Bild- und Videobeschriftungen durch Kombination von OpenAI-CLIP-Visuelleinbettungen mit einem GPT-ähnlichen Sprachmodell.

## 🧭 Snapshot

| Dimension | Details |
|---|---|
| Aufgabenbereich | Bild- und Videobeschriftung |
| Zentrale Ausgaben | SRT-Untertitel, JSON-Transkripte, beschriftete Bilder |
| Primäre Skripte | `i2c.py`, `v2c.py`, `image2caption.py` |
| Legacy-Pfade | `video2caption.py` und versionierte Varianten (aus historischen Gründen erhalten) |
| Datensatzfluss | `data/raw/results.csv` + `data/raw/flickr30k_images/` |

## ✨ Übersicht

Dieses Repository bietet:

- Inferenz-Skripte für Bildbeschriftung und Video-Untertitelung.
- Eine Trainings-Pipeline, die eine Abbildung von CLIP-Bild-Embeddings auf GPT-2-Token-Embeddings lernt.
- Werkzeuge zur Datensatzgenerierung im Stil von Flickr30k.
- Automatisches Herunterladen von Checkpoints für unterstützte Modellgrößen, wenn Gewichte fehlen.
- Mehrsprachige README-Varianten unter `i18n/` (siehe Sprachleiste oben).

Die aktuelle Implementierung umfasst sowohl neuere als auch Legacy-Skripte. Einige alte Dateien werden als Referenz aufbewahrt und unten dokumentiert.

## 🚀 Funktionen

- Einzelbild-Captioning über `image2caption.py`.
- Video-Captioning (gleichmäßiges Frame-Sampling) über `v2c.py` oder `video2caption.py`.
- Anpassbare Laufzeitoptionen:
  - Anzahl der Frames.
  - Modellgröße.
  - Sampling-Temperatur.
  - Checkpoint-Name.
- Multiprocessing/Threaded Captioning für schnellere Video-Inferenz.
- Ausgabeartefakte:
  - SRT-Untertiteldateien (`.srt`).
  - JSON-Transkripte (`.json`) in `v2c.py`.
- Trainings- und Evaluations-Einstiegspunkte für CLIP+GPT2-Mapping-Experimente.

### Auf einen Blick

| Bereich | Hauptskript(e) | Hinweise |
|---|---|---|
| Bildbeschriftung | `image2caption.py`, `i2c.py`, `predict.py` | CLI + wiederverwendbare Klasse |
| Video-Beschriftung | `v2c.py` | Empfohlener gepflegter Pfad |
| Legacy-Videoablauf | `video2caption.py`, `video2caption_v1.1.py` | Enthält gerätespezifische Annahmen |
| Dataset-Erstellung | `dataset_generation.py` | Erzeugt `data/processed/dataset.pkl` |
| Training / Eval | `training.py`, `evaluate.py` | Nutzt CLIP+GPT2-Mapping |

## 🧱 Architektur (High Level)

Das Kernmodell in `model/model.py` hat drei Teile:

1. `ImageEncoder`: extrahiert CLIP-Bild-Embeddings.
2. `Mapping`: projiziert CLIP-Embeddings in eine GPT-Prefix-Embedding-Sequenz.
3. `TextDecoder`: GPT-2-Sprachmodell-Kopf, der Caption-Tokens autoregressiv erzeugt.

Training (`Net.train_forward`) nutzt vorab berechnete CLIP-Bild-Embeddings + tokenisierte Captions.
Inferenz (`Net.forward`) verwendet ein PIL-Bild und dekodiert Tokens bis EOS oder `max_len`.

### Datenfluss

1. Datensatz vorbereiten: `dataset_generation.py` liest `data/raw/results.csv` und Bilder in `data/raw/flickr30k_images/` ein und schreibt `data/processed/dataset.pkl`.
2. Trainieren: `training.py` lädt gepickelte Tupel `(image_name, image_embedding, caption)` und trainiert Mapper/Decoder-Schichten.
3. Evaluieren: `evaluate.py` rendert generierte Captions auf ausgeblendete Testbilder.
4. Inferenz bereitstellen:
   - Bild: `image2caption.py` / `predict.py` / `i2c.py`.
   - Video: `v2c.py` (empfohlen), `video2caption.py` (Legacy).

## 🗂️ Projektstruktur

```text
VideoCaptionerWithClip/
├── README.md
├── image2caption.py               # Single-image caption CLI
├── predict.py                     # Alternativer Single-image caption CLI
├── i2c.py                         # Wiederverwendbare ImageCaptioner-Klasse + CLI
├── v2c.py                         # Video -> SRT + JSON (threaded Frame-Captioning)
├── video2caption.py               # Alternative Video -> SRT-Implementierung (Legacy-Einschränkungen)
├── video2caption_v1.1.py          # Ältere Variante
├── video2caption_v1.0_not_work.py # Explizit als nicht funktionierende Legacy-Datei markiert
├── training.py                    # Einstiegspunkt für Training
├── evaluate.py                    # Test-Split-Evaluation und gerenderte Outputs
├── dataset_generation.py          # Baut data/processed/dataset.pkl
├── data/
│   ├── __init__.py
│   └── dataset.py                 # Dataset + DataLoader-Helfer
├── model/
│   ├── __init__.py
│   ├── model.py                   # CLIP-Encoder + Mapping + GPT2-Decoder
│   └── trainer.py                 # Trainings-/Validierungs-/Test-Hilfsklasse
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
- Eine CUDA-fähige GPU ist optional, aber für Training und Inferenz großer Modelle dringend empfohlen.
- `ffmpeg` wird von den aktuellen Skripten nicht direkt benötigt (OpenCV wird für Frame-Extraktion verwendet).
- Internetzugang ist erforderlich, wenn Modelle/Checkpoints von Hugging Face / Google Drive das erste Mal heruntergeladen werden.

Aktuell ist kein Lockfile vorhanden (`requirements.txt` / `pyproject.toml` fehlen), daher werden Abhängigkeiten aus den Imports abgeleitet.

## 🛠️ Installation

### Standard-Setup aus aktueller Repository-Struktur

```bash
git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers pillow matplotlib numpy tqdm opencv-python pandas wandb gdown
```

### Ursprünglicher README-Installationsausschnitt (erhalten)

Das frühere README endete mitten in einem Block. Die Originalbefehle sind unten unverändert als historische Quelle übernommen:

```bash
git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip/src
```

Hinweis: Der aktuelle Repository-Snapshot legt Skripte im Repo-Root ab, nicht unter `src/`.

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

- `-I, --img-path`: Pfad des Eingabebilds.
- `-S, --size`: Modellgröße (`S` oder `L`).
- `-C, --checkpoint-name`: Checkpoint-Dateiname in `weights/{small|large}`.
- `-R, --res-path`: Ausgabeverzeichnis für gerendertes beschriftetes Bild.
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

Oder in einem eigenen Skript importieren:

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

Ausgabe neben dem Eingabevideo:

- `<video_basename>_caption.srt`
- `<video_basename>_caption.json`
- `<video_basename>_captioning_frames/`

### 5. Alternative Video-Pipeline (`video2caption.py`)

```bash
python video2caption.py -V /path/to/video.mp4 -N 10
```

Wichtig: Dieses Skript enthält derzeit maschinenspezifische hartkodierte Pfade:

- Standard-Python-Pfad: `/home/lachlan/miniconda3/envs/caption/bin/python`
- Caption-Skriptpfad: `/home/lachlan/Projects/image_captioning/clip-gpt-captioning/src/image2caption.py`

Nutze `v2c.py`, außer du wartest diese Pfade absichtlich weiter.

### 6. Legacy-Variante (`video2caption_v1.1.py`)

Dieses Skript wird als historische Referenz aufbewahrt. Für die aktive Nutzung bitte `v2c.py` bevorzugen.

### 7. Datensatzgenerierung

```bash
python dataset_generation.py
```

Erwartete Rohdaten:

- `data/raw/results.csv` (Pipe-getrennte Caption-Tabelle).
- `data/raw/flickr30k_images/` (Bilddateien, auf die in der CSV verwiesen wird).

Ausgabe:

- `data/processed/dataset.pkl`

### 8. Training

```bash
python training.py -S L -C model.pt
```

Training verwendet standardmäßig Weights & Biases (`wandb`) Logging.

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

| Config | CLIP-Backbone | GPT-Modell | Gewichtsverzeichnis |
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

Checkpoint-Auto-Download-IDs befinden sich in `utils/downloads.py`:

| Größe | Google-Drive-ID |
|---|---|
| `L` | `1Gh32arzhW06C1ZJyzcJSSfdJDi3RgWoG` |
| `S` | `1pSQruQyg8KJq6VmzhMLFbT_VaHJMdlWF` |

## 📦 Ausgabedateien

### Bild-Inferenz

- Gespeichertes Bild mit überlagertem/erzeugtem Titel unter `--res-path`.
- Dateinamenmuster: `<input_stem>-R<SIZE>.jpg`.

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

### Schnelles Bildbeschriftungsbeispiel

```bash
python image2caption.py -I ./examples/dog.jpg -S S -C model.pt
```

Erwartetes Verhalten:

- Wenn `weights/small/model.pt` fehlt, wird es heruntergeladen.
- Standardmäßig wird ein beschriftetes Bild nach `./data/result/prediction` geschrieben.
- Der Beschriftungstext wird auf stdout ausgegeben.

### Schnelles Video-Beschriftungsbeispiel

```bash
python v2c.py -V ./examples/demo.mp4 -N 8
```

Erwartetes Verhalten:

- Es werden 8 gleichmäßig gesampelte Frames beschriftet.
- `.srt`- und `.json`-Dateien werden neben dem Eingabevideo erzeugt.

### End-to-End-Trainings-/Evaluierungsfolge

```bash
python dataset_generation.py
python training.py -S L -C model.pt
python evaluate.py -I ./data/raw/flickr30k_images -R ./data/result/eval -S L -C model.pt -T 1.0
```

## 🧭 Entwicklungshinweise

- Es gibt Legacy-Überschneidungen zwischen `v2c.py`, `video2caption.py` und `video2caption_v1.*`.
- `video2caption_v1.0_not_work.py` wird absichtlich als nicht funktionierender Legacy-Code beibehalten.
- `training.py` wählt aktuell `ConfigL()` via `config = ConfigL() if args.size.upper() else ConfigS()`; für nicht leere `--size`-Werte wird dadurch immer `ConfigL` verwendet.
- `model/trainer.py` nutzt in `test_step` `self.dataset`, obwohl der Initializer `self.test_dataset` setzt; das kann das Sampling in Trainingsläufen brechen, falls nicht angepasst.
- `video2caption_v1.1.py` referenziert `self.config.transform`, aber `ConfigS`/`ConfigL` definieren `transform` nicht.
- In diesem Repository-Snapshot ist aktuell keine CI/Test-Suite definiert.
- i18n-Hinweis: Sprachlinks stehen am Anfang dieser README; übersetzte Dateien können unter `i18n/` ergänzt werden.
- Aktueller Stand: Die Sprachleiste verweist auf `i18n/README.ru.md`, aber diese Datei ist in diesem Snapshot nicht vorhanden.

## 🩺 Fehlerbehebung

- `AssertionError: Image does not exist`
  - Bestätige, dass `-I/--img-path` auf eine gültige Datei zeigt.
- `Dataset file not found. Downloading...`
  - `MiniFlickrDataset` gibt dies aus, wenn `data/processed/dataset.pkl` fehlt; führe zuerst `python dataset_generation.py` aus.
- `Path to the test image folder does not exist`
  - Bestätige, dass `evaluate.py -I` auf einen existierenden Ordner zeigt.
- Langsame oder fehlerhafte ersten Ausführungsdurchlauf
  - Beim ersten Lauf werden Hugging-Face-Modelle geladen und ggf. Checkpoints von Google Drive heruntergeladen.
- `video2caption.py` liefert leere Captions
  - Prüfe den hartkodierten Skriptpfad und Python-Executable-Pfad oder wechsle zu `v2c.py`.
- `wandb` verlangt beim Training Login
  - `wandb login` ausführen oder Logging in `training.py` bei Bedarf manuell deaktivieren.

## 🛣️ Roadmap

- Abhängigkeits-Lockfiles (`requirements.txt` oder `pyproject.toml`) für reproduzierbare Installationen ergänzen.
- Doppelte Video-Pipelines in eine gepflegte Implementierung vereinheitlichen.
- Harte maschinenspezifische Pfade aus Legacy-Skripten entfernen.
- Bekannte Trainings-/Evaluations-Edge-Case-Bugs in `training.py` und `model/trainer.py` beheben.
- Automatisierte Tests und CI hinzufügen.
- `i18n/` mit den in der Sprachleiste referenzierten README-Übersetzungen befüllen.

## 🤝 Mitwirken

Beiträge sind willkommen. Vorgeschlagener Workflow:

```bash
# 1) Fork und Klonen
git clone git@github.com:<your-user>/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip

# 2) Feature-Branch erstellen
git checkout -b feat/your-change

# 3) Änderungen vornehmen und committen
git add .
git commit -m "feat: describe your change"

# 4) Pushen und PR öffnen
git push origin feat/your-change
```

Wenn du das Modellverhalten änderst, füge bitte Folgendes hinzu:

- Reproduzierbare(r) Befehl(e).
- Vorher-/Nachher-Beispielausgaben.
- Hinweise zu Annahmen bei Checkpoints oder Datensätzen.

## 📄 Lizenz

Im aktuellen Repository-Snapshot ist keine Lizenzdatei vorhanden.

Annahme: Solange keine `LICENSE`-Datei hinzugefügt wurde, sind Nutzungs-/Weitergabe-Bedingungen nicht definiert.


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
