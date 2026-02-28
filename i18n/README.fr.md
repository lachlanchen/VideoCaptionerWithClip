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

Une boîte à outils Python pour générer des légendes en langage naturel sur des images et des vidéos en combinant les embeddings visuels OpenAI CLIP avec un modèle de langage de type GPT.

## 🧭 Snapshot

| Dimension | Détails |
|---|---|
| Couverture de tâche | Légende d'images et de vidéos |
| Sorties principales | Sous-titres SRT, transcriptions JSON, images légendées |
| Scripts principaux | `i2c.py`, `v2c.py`, `image2caption.py` |
| Chemins hérités | `video2caption.py` et variantes versionnées (conservées pour l'historique) |
| Flux de données | `data/raw/results.csv` + `data/raw/flickr30k_images/` |

## ✨ Vue d'ensemble

Ce dépôt fournit :

- Des scripts d'inférence pour le sous-titrage d'images et la génération de sous-titres vidéo.
- Un pipeline d'entraînement qui apprend une projection des embeddings visuels CLIP vers les embeddings de tokens GPT-2.
- Des utilitaires de génération de jeu de données pour des données de style Flickr30k.
- Le téléchargement automatique de checkpoints pour les tailles de modèles prises en charge quand les poids sont manquants.
- Des variantes du README multilingues sous `i18n/` (voir la barre de langues ci-dessus).

L'implémentation actuelle inclut à la fois des scripts récents et des scripts hérités. Certains fichiers hérités sont conservés à des fins de référence et sont documentés ci-dessous.

## 🚀 Fonctionnalités

- Génération de légende d'une image via `image2caption.py`.
- Génération de légende vidéo (échantillonnage uniforme des frames) via `v2c.py` ou `video2caption.py`.
- Options d'exécution personnalisables :
  - Nombre de frames.
  - Taille du modèle.
  - Température d'échantillonnage.
  - Nom du checkpoint.
- Légendage multiprocessus/threadé pour une inférence vidéo plus rapide.
- Artefacts de sortie :
  - Fichiers de sous-titres SRT (`.srt`).
  - Transcriptions JSON (`.json`) dans `v2c.py`.
- Points d'entrée entraînement et évaluation pour les expériences de mapping CLIP+GPT2.

### À vue d'ensemble

| Domaine | Script(s) principal(aux) | Remarques |
|---|---|---|
| Légende d'images | `image2caption.py`, `i2c.py`, `predict.py` | CLI + classe réutilisable |
| Légende de vidéos | `v2c.py` | Chemin maintenu recommandé |
| Flux vidéo hérité | `video2caption.py`, `video2caption_v1.1.py` | Contient des hypothèses spécifiques à la machine |
| Construction du dataset | `dataset_generation.py` | Produit `data/processed/dataset.pkl` |
| Entraînement / évaluation | `training.py`, `evaluate.py` | Utilise le mapping CLIP+GPT2 |

## 🧱 Architecture (vue d'ensemble)

Le modèle central dans `model/model.py` comporte trois parties :

1. `ImageEncoder` : extrait l'embedding d'image CLIP.
2. `Mapping` : projette l'embedding CLIP vers une séquence d'embeddings de préfixe GPT.
3. `TextDecoder` : tête GPT-2 qui génère de manière autorégressive les tokens de légende.

L'entraînement (`Net.train_forward`) utilise des embeddings d'image CLIP pré-calculés + des légendes tokenisées.
L'inférence (`Net.forward`) utilise une image PIL et décode les tokens jusqu'à EOS ou `max_len`.

### Flux de données

1. Préparer le dataset : `dataset_generation.py` lit `data/raw/results.csv` et les images dans `data/raw/flickr30k_images/`, puis écrit `data/processed/dataset.pkl`.
2. Entraîner : `training.py` charge des tuples picklés `(image_name, image_embedding, caption)` et entraîne les couches mapper/décodeur.
3. Évaluer : `evaluate.py` applique les légendes générées aux images du jeu de test.
4. Servir l'inférence :
   - image : `image2caption.py` / `predict.py` / `i2c.py`.
   - vidéo : `v2c.py` (recommandé), `video2caption.py` (hérité).

## 🗂️ Structure du projet

```text
VideoCaptionerWithClip/
├── README.md
├── image2caption.py               # CLI de légende d'image unique
├── predict.py                     # CLI alternatif de légende d'image
├── i2c.py                         # Classe ImageCaptioner réutilisable + CLI
├── v2c.py                         # Vidéo -> SRT + JSON (légende de frames en threads)
├── video2caption.py               # Implémentation alternative vidéo -> SRT (contraintes héritées)
├── video2caption_v1.1.py          # Variante plus ancienne
├── video2caption_v1.0_not_work.py # Fichier explicitement marqué non fonctionnel
├── training.py                    # Point d'entrée de l'entraînement
├── evaluate.py                    # Évaluation sur split test et rendu des sorties
├── dataset_generation.py          # Génère data/processed/dataset.pkl
├── data/
│   ├── __init__.py
│   └── dataset.py                 # Dataset + utilitaires DataLoader
├── model/
│   ├── __init__.py
│   ├── model.py                   # encodeur CLIP + mapping + décodeur GPT2
│   └── trainer.py                 # classe utilitaire entraînement/validation/test
├── utils/
│   ├── __init__.py
│   ├── config.py                  # valeurs par défaut ConfigS / ConfigL
│   ├── downloads.py               # téléchargement checkpoint Google Drive
│   └── lr_warmup.py               # planification de warmup de LR
├── i18n/                          # variantes du README multilingues
└── .auto-readme-work/             # artefacts pipeline auto-README
```

## 📋 Prérequis

- Python `3.10+` recommandé.
- Un GPU compatible CUDA est optionnel, mais fortement recommandé pour l'entraînement et l'inférence de grands modèles.
- `ffmpeg` n'est pas requis directement par les scripts actuels (OpenCV est utilisé pour l'extraction des frames).
- Un accès Internet est nécessaire au premier téléchargement des modèles/checkpoints depuis Hugging Face / Google Drive.

Aucun lockfile n'est présent actuellement (`requirements.txt` / `pyproject.toml` absent), donc les dépendances sont déduites depuis les imports.

## 🛠️ Installation

### Configuration canonique à partir de la structure du dépôt actuelle

```bash

git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers pillow matplotlib numpy tqdm opencv-python pandas wandb gdown
```

### Extrait d'installation du README original (préservé)

Le README précédent se terminait au milieu d'un bloc. Les commandes d'origine sont conservées ci-dessous exactement comme contenu historique source :

```bash
git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip/src
```

Note : l'instantané actuel du dépôt place les scripts à la racine du dépôt, pas sous `src/`.

## ▶️ Démarrage rapide

| Objectif | Commande |
|---|---|
| Légender une image | `python image2caption.py -I /path/to/image.jpg -S L -C model.pt` |
| Légender une vidéo | `python v2c.py -V /path/to/video.mp4 -N 10` |
| Générer le dataset | `python dataset_generation.py` |

### Légende d'une image (exécution rapide)

```bash
python image2caption.py -I /path/to/image.jpg -S L -C model.pt
```

### Légende vidéo (chemin recommandé)

```bash
python v2c.py -V /path/to/video.mp4 -N 10
```

## 🎯 Utilisation

### 1. Légende d'image (`image2caption.py`)

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
- `-C, --checkpoint-name` : nom du checkpoint dans `weights/{small|large}`.
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

`predict.py` est fonctionnellement proche de `image2caption.py` ; le format texte de sortie diffère légèrement.

### 3. API de classe pour la légende d'image (`i2c.py`)

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

Important : ce script contient actuellement des chemins codés en dur spécifiques à la machine :

- Python path par défaut : `/home/lachlan/miniconda3/envs/caption/bin/python`
- Chemin du script de légende : `/home/lachlan/Projects/image_captioning/clip-gpt-captioning/src/image2caption.py`

Utilisez `v2c.py` sauf si vous maintenez volontairement ces chemins.

### 6. Variante héritée (`video2caption_v1.1.py`)

Ce script est conservé à titre de référence historique. Préférez `v2c.py` pour une utilisation active.

### 7. Génération du dataset

```bash
python dataset_generation.py
```

Entrées attendues :

- `data/raw/results.csv` (table de légendes séparées par `|`).
- `data/raw/flickr30k_images/` (fichiers image référencés par le CSV).

Sortie :

- `data/processed/dataset.pkl`

### 8. Entraînement

```bash
python training.py -S L -C model.pt
```

L'entraînement utilise la journalisation Weights & Biases (`wandb`) par défaut.

### 9. Évaluation

```bash
python evaluate.py \
  -I ./data/raw/flickr30k_images \
  -R ./data/result/eval \
  -S L \
  -C model.pt \
  -T 1.0
```

L'évaluation rend les légendes prédites sur les images de test et les enregistre dans :

- `<res-path>/<checkpoint_name_without_ext>_<SIZE>/`

## ⚙️ Configuration

Les configurations de modèle sont définies dans `utils/config.py` :

| Config | Backbone CLIP | Modèle GPT | Dossier des poids |
|---|---|---|---|
| `ConfigS` | `openai/clip-vit-base-patch32` | `gpt2` | `weights/small` |
| `ConfigL` | `openai/clip-vit-large-patch14` | `gpt2-medium` | `weights/large` |

Principaux paramètres par défaut des classes de configuration :

| Champ | `ConfigS` | `ConfigL` |
|---|---:|---:|
| `epochs` | 150 | 120 |
| `lr` | 3e-3 | 5e-3 |
| `batch_size_exp` | 6 | 5 |
| `ep_len` | 4 | 4 |
| `max_len` | 40 | 40 |

Les IDs de téléchargement automatique des checkpoints sont dans `utils/downloads.py` :

| Taille | ID Google Drive |
|---|---|
| `L` | `1Gh32arzhW06C1ZJyzcJSSfdJDi3RgWoG` |
| `S` | `1pSQruQyg8KJq6VmzhMLFbT_VaHJMdlWF` |

## 📦 Fichiers de sortie

### Inférence image

- Image enregistrée avec titre superposé/généré dans `--res-path`.
- Motif de nom de fichier : `<input_stem>-R<SIZE>.jpg`.

### Inférence vidéo (`v2c.py`)

- SRT : `<video_stem>_caption.srt`
- JSON : `<video_stem>_caption.json`
- Images des frames : `<video_stem>_captioning_frames/`

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

### Exemple rapide de légende d'image

```bash
python image2caption.py -I ./examples/dog.jpg -S S -C model.pt
```

Comportement attendu :

- Si `weights/small/model.pt` est manquant, il est téléchargé.
- Une image légendée est écrite par défaut dans `./data/result/prediction`.
- Le texte de la légende est affiché sur stdout.

### Exemple rapide de légende vidéo

```bash
python v2c.py -V ./examples/demo.mp4 -N 8
```

Comportement attendu :

- 8 frames sont légendées par échantillonnage uniforme.
- Des fichiers `.srt` et `.json` sont générés à côté de la vidéo d'entrée.

### Chaîne entraînement/évaluation de bout en bout

```bash
python dataset_generation.py
python training.py -S L -C model.pt
python evaluate.py -I ./data/raw/flickr30k_images -R ./data/result/eval -S L -C model.pt -T 1.0
```

## 🧭 Notes de développement

- Un chevauchement hérité existe entre `v2c.py`, `video2caption.py` et `video2caption_v1.*`.
- `video2caption_v1.0_not_work.py` est conservé intentionnellement comme code hérité non fonctionnel.
- `training.py` sélectionne actuellement `ConfigL()` via `config = ConfigL() if args.size.upper() else ConfigS()`, ce qui se résout toujours vers `ConfigL` pour des valeurs `--size` non vides.
- `model/trainer.py` utilise `self.dataset` dans `test_step`, tandis que l'initialiseur assigne `self.test_dataset` ; cela peut casser l'échantillonnage pendant les runs d'entraînement si ce n'est pas ajusté.
- `video2caption_v1.1.py` référence `self.config.transform`, alors que `ConfigS`/`ConfigL` ne définissent pas `transform`.
- Aucun suite de CI/tests n'est actuellement définie dans cet instantané de dépôt.
- Note i18n : des liens de langue sont présents en haut de ce README ; des fichiers traduits peuvent être ajoutés sous `i18n/`.
- Note d'état actuelle : la barre de langue référence `i18n/README.ru.md`, mais ce fichier n'est pas présent dans cet instantané.

## 🩺 Dépannage

- `AssertionError: Image does not exist`
  - Vérifiez que `-I/--img-path` pointe vers un fichier valide.
- `Dataset file not found. Downloading...`
  - `MiniFlickrDataset` lève ce message quand `data/processed/dataset.pkl` est absent ; exécutez d'abord `python dataset_generation.py`.
- `Path to the test image folder does not exist`
  - Vérifiez que `evaluate.py -I` pointe vers un dossier existant.
- Exécution initiale lente ou en échec
  - La première exécution télécharge les modèles Hugging Face et peut télécharger des checkpoints Google Drive.
- `video2caption.py` renvoie des légendes vides
  - Vérifiez le chemin du script codé en dur et le chemin de l'exécutable Python, ou passez à `v2c.py`.
- `wandb` demande une connexion pendant l'entraînement
  - Exécutez `wandb login` ou désactivez la journalisation dans `training.py` si nécessaire.

## 🛣️ Feuille de route

- Ajouter des fichiers lock de dépendances (`requirements.txt` ou `pyproject.toml`) pour des installations reproductibles.
- Unifier les pipelines vidéo dupliqués en une implémentation maintenue.
- Supprimer les chemins machines codés en dur des scripts hérités.
- Corriger les bugs connus de cas limites d'entraînement/évaluation dans `training.py` et `model/trainer.py`.
- Ajouter des tests automatisés et une CI.
- Compléter `i18n/` avec les README traduits référencés dans la barre de langues.

## 🤝 Contribution

Les contributions sont les bienvenues. Workflow suggéré :

```bash
# 1) Fork et clone
 git clone git@github.com:<your-user>/VideoCaptionerWithClip.git
 cd VideoCaptionerWithClip

# 2) Créer une branche de fonctionnalité
 git checkout -b feat/your-change

# 3) Faire les changements et valider
 git add .
 git commit -m "feat: describe your change"

# 4) Pousser et ouvrir une PR
 git push origin feat/your-change
```

Si vous modifiez le comportement du modèle, incluez :

- Une(ou plusieurs) commande(s) reproductibles.
- Des exemples de sortie avant/après.
- Des notes sur les hypothèses liées aux checkpoints ou au dataset.

## ❤️ Support

| Donate | PayPal | Stripe |
|---|---|---|
| [![Donate](https://img.shields.io/badge/Donate-LazyingArt-0EA5E9?style=for-the-badge&logo=ko-fi&logoColor=white)](https://chat.lazying.art/donate) | [![PayPal](https://img.shields.io/badge/PayPal-RongzhouChen-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/RongzhouChen) | [![Stripe](https://img.shields.io/badge/Stripe-Donate-635BFF?style=for-the-badge&logo=stripe&logoColor=white)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## 📄 License

Aucun fichier de licence n'est présent dans l'instantané actuel du dépôt.

Note d'hypothèse : tant qu'un fichier `LICENSE` n'est pas ajouté, les conditions de réutilisation/distribution restent indéfinies.
