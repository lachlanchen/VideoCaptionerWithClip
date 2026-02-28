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

Une boîte à outils Python pour générer des légendes en langage naturel sur des images et des vidéos en combinant les embeddings visuels OpenAI CLIP avec un modèle de langage de type GPT.

## ✨ Vue d'ensemble

Ce dépôt fournit :

- Des scripts d'inférence pour le sous-titrage d'images et la génération de sous-titres vidéo.
- Un pipeline d'entraînement qui apprend une projection entre les embeddings visuels CLIP et les embeddings de tokens GPT-2.
- Des utilitaires de génération de jeu de données pour des données de type Flickr30k.
- Le téléchargement automatique de checkpoints pour les tailles de modèle prises en charge lorsque les poids sont absents.
- Des variantes multilingues du README sous `i18n/` (voir la barre des langues ci-dessus).

L'implémentation actuelle inclut des scripts récents et hérités. Certains fichiers hérités sont conservés à titre de référence et documentés ci-dessous.

## 🚀 Fonctionnalités

- Légendage d'image unique via `image2caption.py`.
- Légendage vidéo (échantillonnage uniforme des frames) via `v2c.py` ou `video2caption.py`.
- Options d'exécution personnalisables :
  - Nombre de frames.
  - Taille du modèle.
  - Température d'échantillonnage.
  - Nom du checkpoint.
- Légendage multiprocessus/threadé pour accélérer l'inférence vidéo.
- Artéfacts de sortie :
  - Fichiers de sous-titres SRT (`.srt`).
  - Transcriptions JSON (`.json`) dans `v2c.py`.
- Points d'entrée d'entraînement et d'évaluation pour les expériences de projection CLIP+GPT2.

### En un coup d'oeil

| Domaine | Script(s) principal(aux) | Remarques |
|---|---|---|
| Légendage d'image | `image2caption.py`, `i2c.py`, `predict.py` | CLI + classe réutilisable |
| Légendage vidéo | `v2c.py` | Voie recommandée et maintenue |
| Flux vidéo hérité | `video2caption.py`, `video2caption_v1.1.py` | Contient des hypothèses spécifiques à la machine |
| Construction du dataset | `dataset_generation.py` | Produit `data/processed/dataset.pkl` |
| Entraînement / éval | `training.py`, `evaluate.py` | Utilise la projection CLIP+GPT2 |

## 🧱 Architecture (haut niveau)

Le modèle central dans `model/model.py` comporte trois parties :

1. `ImageEncoder` : extrait l'embedding d'image CLIP.
2. `Mapping` : projette l'embedding CLIP vers une séquence d'embeddings de préfixe GPT.
3. `TextDecoder` : tête de modèle de langage GPT-2 qui génère de façon autorégressive les tokens de légende.

L'entraînement (`Net.train_forward`) utilise des embeddings d'image CLIP pré-calculés + des légendes tokenisées.
L'inférence (`Net.forward`) utilise une image PIL et décode les tokens jusqu'à EOS ou `max_len`.

### Flux de données

1. Préparer le dataset : `dataset_generation.py` lit `data/raw/results.csv` et les images dans `data/raw/flickr30k_images/`, puis écrit `data/processed/dataset.pkl`.
2. Entraîner : `training.py` charge les tuples sérialisés `(image_name, image_embedding, caption)` et entraîne les couches mapper/decoder.
3. Évaluer : `evaluate.py` rend les légendes générées sur des images de test.
4. Servir l'inférence :
   - image : `image2caption.py` / `predict.py` / `i2c.py`.
   - vidéo : `v2c.py` (recommandé), `video2caption.py` (hérité).

## 🗂️ Structure du projet

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

## 📋 Prérequis

- Python `3.10+` recommandé.
- Un GPU compatible CUDA est optionnel mais fortement recommandé pour l'entraînement et l'inférence avec les grands modèles.
- `ffmpeg` n'est pas requis directement par les scripts actuels (OpenCV est utilisé pour l'extraction de frames).
- Un accès internet est nécessaire au premier lancement pour télécharger les modèles/checkpoints depuis Hugging Face / Google Drive.

Aucun lockfile n'est actuellement présent (`requirements.txt` / `pyproject.toml` absents), les dépendances sont donc déduites des imports.

## 🛠️ Installation

### Configuration canonique depuis la structure actuelle du dépôt

```bash
git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers pillow matplotlib numpy tqdm opencv-python pandas wandb gdown
```

### Extrait d'installation du README d'origine (préservé)

Le README précédent se terminait au milieu d'un bloc. Les commandes d'origine sont conservées ci-dessous exactement comme contenu historique de référence :

```bash
git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip/src
```

Remarque : dans l'instantané actuel du dépôt, les scripts se trouvent à la racine du dépôt, pas sous `src/`.

## ▶️ Démarrage rapide

### Légendage d'image (exécution rapide)

```bash
python image2caption.py -I /path/to/image.jpg -S L -C model.pt
```

### Légendage vidéo (voie recommandée)

```bash
python v2c.py -V /path/to/video.mp4 -N 10
```

## 🎯 Utilisation

### 1. Légendage d'image (`image2caption.py`)

```bash
python image2caption.py \
  -I /path/to/image.jpg \
  -S L \
  -C model.pt \
  -R ./data/result/prediction \
  -T 1.0
```

Arguments :

- `-I, --img-path` : chemin de l'image d'entrée.
- `-S, --size` : taille du modèle (`S` ou `L`).
- `-C, --checkpoint-name` : nom de fichier du checkpoint dans `weights/{small|large}`.
- `-R, --res-path` : répertoire de sortie pour l'image légendée rendue.
- `-T, --temperature` : température d'échantillonnage.

### 2. CLI image alternative (`predict.py`)

```bash
python predict.py \
  -I /path/to/image.jpg \
  -S L \
  -C model.pt \
  -R ./data/result/prediction \
  -T 1.0
```

`predict.py` est fonctionnellement similaire à `image2caption.py` ; le format du texte de sortie diffère légèrement.

### 3. API de classe pour légendage d'image (`i2c.py`)

```bash
python i2c.py -I /path/to/image.jpg -S L -C model.pt -R ./data/result/prediction -T 1.0
```

Ou importez-la dans votre propre script :

```python
from i2c import ImageCaptioner

captioner = ImageCaptioner(model_size="L", checkpoint_name="model.pt")
captioner.set_image_path("/path/to/image.jpg")
caption = captioner.generate_caption(save_image=True)
print(caption)
```

### 4. Vidéo vers sous-titres + JSON (`v2c.py`)

```bash
python v2c.py -V /path/to/video.mp4 -N 10
```

Sorties à côté de la vidéo d'entrée :

- `<video_basename>_caption.srt`
- `<video_basename>_caption.json`
- `<video_basename>_captioning_frames/`

### 5. Pipeline vidéo alternatif (`video2caption.py`)

```bash
python video2caption.py -V /path/to/video.mp4 -N 10
```

Important : ce script contient actuellement des chemins codés en dur, spécifiques à une machine :

- Python path default: `/home/lachlan/miniconda3/envs/caption/bin/python`
- Caption script path: `/home/lachlan/Projects/image_captioning/clip-gpt-captioning/src/image2caption.py`

Utilisez `v2c.py` sauf si vous maintenez volontairement ces chemins.

### 6. Variante héritée (`video2caption_v1.1.py`)

Ce script est conservé à des fins de référence historique. Préférez `v2c.py` pour un usage actif.

### 7. Génération du dataset

```bash
python dataset_generation.py
```

Entrées brutes attendues :

- `data/raw/results.csv` (table de légendes séparée par des pipes).
- `data/raw/flickr30k_images/` (fichiers image référencés par le CSV).

Sortie :

- `data/processed/dataset.pkl`

### 8. Entraînement

```bash
python training.py -S L -C model.pt
```

L'entraînement utilise par défaut la journalisation Weights & Biases (`wandb`).

### 9. Évaluation

```bash
python evaluate.py \
  -I ./data/raw/flickr30k_images \
  -R ./data/result/eval \
  -S L \
  -C model.pt \
  -T 1.0
```

L'évaluation rend les légendes prédites sur les images de test et les enregistre sous :

- `<res-path>/<checkpoint_name_without_ext>_<SIZE>/`

## ⚙️ Configuration

Les configurations de modèle sont définies dans `utils/config.py` :

| Config | CLIP backbone | GPT model | Weights dir |
|---|---|---|---|
| `ConfigS` | `openai/clip-vit-base-patch32` | `gpt2` | `weights/small` |
| `ConfigL` | `openai/clip-vit-large-patch14` | `gpt2-medium` | `weights/large` |

Principales valeurs par défaut des classes de configuration :

| Field | `ConfigS` | `ConfigL` |
|---|---:|---:|
| `epochs` | 150 | 120 |
| `lr` | 3e-3 | 5e-3 |
| `batch_size_exp` | 6 | 5 |
| `ep_len` | 4 | 4 |
| `max_len` | 40 | 40 |

Les IDs de téléchargement automatique des checkpoints sont dans `utils/downloads.py` :

| Size | Google Drive ID |
|---|---|
| `L` | `1Gh32arzhW06C1ZJyzcJSSfdJDi3RgWoG` |
| `S` | `1pSQruQyg8KJq6VmzhMLFbT_VaHJMdlWF` |

## 📦 Fichiers de sortie

### Inférence image

- Image enregistrée avec titre superposé/généré dans `--res-path`.
- Schéma de nom de fichier : `<input_stem>-R<SIZE>.jpg`.

### Inférence vidéo (`v2c.py`)

- SRT: `<video_stem>_caption.srt`
- JSON: `<video_stem>_caption.json`
- Images de frames : `<video_stem>_captioning_frames/`

Exemple d'élément JSON :

```json
{
  "start": "00:00:03,200",
  "end": "00:00:03,700",
  "lang": "en",
  "text": "A dog running through a field."
}
```

## 🧪 Exemples

### Exemple rapide de légendage d'image

```bash
python image2caption.py -I ./examples/dog.jpg -S S -C model.pt
```

Comportement attendu :

- Si `weights/small/model.pt` est absent, il est téléchargé.
- Une image légendée est écrite par défaut dans `./data/result/prediction`.
- Le texte de la légende est affiché sur stdout.

### Exemple rapide de légendage vidéo

```bash
python v2c.py -V ./examples/demo.mp4 -N 8
```

Comportement attendu :

- 8 frames échantillonnées uniformément sont légendées.
- Des fichiers `.srt` et `.json` sont générés à côté de la vidéo d'entrée.

### Séquence entraînement/évaluation de bout en bout

```bash
python dataset_generation.py
python training.py -S L -C model.pt
python evaluate.py -I ./data/raw/flickr30k_images -R ./data/result/eval -S L -C model.pt -T 1.0
```

## 🧭 Notes de développement

- Un chevauchement hérité existe entre `v2c.py`, `video2caption.py` et `video2caption_v1.*`.
- `video2caption_v1.0_not_work.py` est intentionnellement conservé comme code hérité non fonctionnel.
- `training.py` sélectionne actuellement `ConfigL()` via `config = ConfigL() if args.size.upper() else ConfigS()`, ce qui résout toujours vers `ConfigL` pour les valeurs non vides de `--size`.
- `model/trainer.py` utilise `self.dataset` dans `test_step`, tandis que l'initialiseur assigne `self.test_dataset` ; cela peut casser l'échantillonnage pendant l'entraînement sans ajustement.
- `video2caption_v1.1.py` référence `self.config.transform`, mais `ConfigS`/`ConfigL` ne définissent pas `transform`.
- Aucune suite de tests/CI n'est actuellement définie dans cet instantané du dépôt.
- Note i18n : des liens de langue sont présents en haut de ce README ; des fichiers traduits peuvent être ajoutés sous `i18n/`.
- Note sur l'état actuel : la barre de langue référence `i18n/README.ru.md`, mais ce fichier n'est pas présent dans cet instantané.

## 🩺 Dépannage

- `AssertionError: Image does not exist`
  - Vérifiez que `-I/--img-path` pointe vers un fichier valide.
- `Dataset file not found. Downloading...`
  - `MiniFlickrDataset` déclenche ceci quand `data/processed/dataset.pkl` est manquant ; exécutez d'abord `python dataset_generation.py`.
- `Path to the test image folder does not exist`
  - Vérifiez que `evaluate.py -I` pointe vers un dossier existant.
- Premier lancement lent ou en échec
  - Le premier lancement télécharge les modèles Hugging Face et peut télécharger des checkpoints depuis Google Drive.
- `video2caption.py` renvoie des légendes vides
  - Vérifiez le chemin du script codé en dur et le chemin de l'exécutable Python, ou passez à `v2c.py`.
- `wandb` demande une connexion pendant l'entraînement
  - Exécutez `wandb login` ou désactivez manuellement la journalisation dans `training.py` si nécessaire.

## 🛣️ Feuille de route

- Ajouter des lockfiles de dépendances (`requirements.txt` ou `pyproject.toml`) pour des installations reproductibles.
- Unifier les pipelines vidéo dupliqués en une implémentation maintenue.
- Supprimer les chemins machine codés en dur des scripts hérités.
- Corriger les bugs connus de cas limites d'entraînement/évaluation dans `training.py` et `model/trainer.py`.
- Ajouter des tests automatisés et de la CI.
- Compléter `i18n/` avec les README traduits référencés dans la barre des langues.

## 🤝 Contribution

Les contributions sont les bienvenues. Workflow suggéré :

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

Si vous modifiez le comportement du modèle, incluez :

- Des commandes reproductibles.
- Des exemples de sorties avant/après.
- Des notes sur les hypothèses liées aux checkpoints ou au dataset.

## 🙌 Support

Aucune configuration explicite de donation/sponsoring n'est présente dans l'instantané actuel du dépôt.

Si des liens de sponsoring sont ajoutés plus tard, ils doivent être préservés dans cette section.

## 📄 Licence

Aucun fichier de licence n'est présent dans l'instantané actuel du dépôt.

Note d'hypothèse : tant qu'un fichier `LICENSE` n'est pas ajouté, les conditions de réutilisation/distribution sont indéfinies.
