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

Una herramienta de Python para generar subtítulos y descripciones en lenguaje natural sobre imágenes y videos combinando embeddings visuales de OpenAI CLIP con un modelo de lenguaje estilo GPT.

## 🧭 Instantánea

| Dimensión | Detalles |
|---|---|
| Cobertura de tarea | Captionado de imágenes y videos |
| Salidas principales | Subtítulos SRT, transcripciones JSON, imágenes con caption |
| Scripts principales | `i2c.py`, `v2c.py`, `image2caption.py` |
| Rutas heredadas | `video2caption.py` y versiones históricas (se conservan por referencia) |
| Flujo de datos | `data/raw/results.csv` + `data/raw/flickr30k_images/` |

## ✨ Descripción general

Este repositorio proporciona:

- Scripts de inferencia para captionado de imágenes y generación de subtítulos de video.
- Un pipeline de entrenamiento que aprende el mapeo entre embeddings visuales de CLIP y embeddings de tokens de GPT-2.
- Utilidades para generar datasets al estilo Flickr30k.
- Descarga automática de checkpoints para tamaños de modelo soportados cuando faltan pesos.
- Variantes del README en múltiples idiomas bajo `i18n/` (ver la barra de idiomas arriba).

La implementación actual incluye scripts nuevos y heredados. Algunos archivos legacy se mantienen por referencia y se documentan más abajo.

## 🚀 Características

- Captionado de una sola imagen vía `image2caption.py`.
- Captionado de video (muestreo uniforme de frames) vía `v2c.py` o `video2caption.py`.
- Opciones de ejecución personalizables:
  - Número de frames.
  - Tamaño del modelo.
  - Temperatura de muestreo.
  - Nombre del checkpoint.
- Captionado multiproceso para acelerar inferencia de video.
- Artefactos de salida:
  - Archivos de subtítulos SRT (`.srt`).
  - Transcripciones JSON (`.json`) en `v2c.py`.
- Puntos de entrada para entrenamiento y evaluación de experimentos de mapeo CLIP+GPT2.

### A simple vista

| Área | Script(s) principal(es) | Notas |
|---|---|---|
| Captionado de imagen | `image2caption.py`, `i2c.py`, `predict.py` | CLI + clase reutilizable |
| Captionado de video | `v2c.py` | Ruta mantenida recomendada |
| Flujo legacy de video | `video2caption.py`, `video2caption_v1.1.py` | Incluye suposiciones específicas de máquina |
| Construcción de dataset | `dataset_generation.py` | Produce `data/processed/dataset.pkl` |
| Entrenamiento / evaluación | `training.py`, `evaluate.py` | Usa mapeo CLIP+GPT2 |

## 🧱 Arquitectura (visión general)

El modelo central en `model/model.py` tiene tres componentes:

1. `ImageEncoder`: extrae embedding de imagen de CLIP.
2. `Mapping`: proyecta el embedding de CLIP a una secuencia de embeddings de prefijo para GPT.
3. `TextDecoder`: cabeza de lenguaje basada en GPT-2 que genera tokens del caption de forma autorregresiva.

El entrenamiento (`Net.train_forward`) usa embeddings de imagen CLIP precomputados + captions tokenizados.
La inferencia (`Net.forward`) usa una imagen PIL y decodifica tokens hasta EOS o `max_len`.

### Flujo de datos

1. Preparar dataset: `dataset_generation.py` lee `data/raw/results.csv` y las imágenes en `data/raw/flickr30k_images/`, y escribe `data/processed/dataset.pkl`.
2. Entrenar: `training.py` carga tuplas pickled `(image_name, image_embedding, caption)` y entrena capas de mapper/decoder.
3. Evaluar: `evaluate.py` genera captions sobre imágenes de prueba en el split de validación.
4. Ejecutar inferencia:
   - imagen: `image2caption.py` / `predict.py` / `i2c.py`.
   - video: `v2c.py` (recomendado), `video2caption.py` (legacy).

## 🗂️ Estructura del proyecto

```text
VideoCaptionerWithClip/
├── README.md
├── image2caption.py               # CLI de captionado para imagen única
├── predict.py                     # CLI alternativa para imagen única
├── i2c.py                         # Clase ImageCaptioner reutilizable + CLI
├── v2c.py                         # Video -> SRT + JSON (captionado de frames en paralelo)
├── video2caption.py               # Implementación alternativa video -> SRT (legacy, con limitaciones)
├── video2caption_v1.1.py          # Variante anterior
├── video2caption_v1.0_not_work.py # Archivo legacy explícitamente marcado como no funcional
├── training.py                    # Entrada de entrenamiento de modelo
├── evaluate.py                    # Evaluación en split de prueba y outputs renderizados
├── dataset_generation.py          # Construye data/processed/dataset.pkl
├── data/
│   ├── __init__.py
│   └── dataset.py                 # Ayudas de Dataset + DataLoader
├── model/
│   ├── __init__.py
│   ├── model.py                   # Encoder CLIP + mapping + decodificador GPT2
│   └── trainer.py                 # Clase utilitaria train/val/test
├── utils/
│   ├── __init__.py
│   ├── config.py                  # ConfigS / ConfigL por defecto
│   ├── downloads.py               # Descarga de checkpoints desde Google Drive
│   └── lr_warmup.py               # Programador de warmup de LR
├── i18n/                          # Variantes del README en varios idiomas
└── .auto-readme-work/             # Artefactos del pipeline auto-README
```

## 📋 Requisitos previos

- Python `3.10+` recomendado.
- Se recomienda una GPU con CUDA para entrenamiento e inferencia en modelos grandes, aunque es opcional.
- `ffmpeg` no es requerido directamente por los scripts actuales (OpenCV se usa para extracción de frames).
- Se necesita acceso a internet la primera vez que se descargan modelos/checkpoints desde Hugging Face / Google Drive.

Actualmente no hay lockfile (`requirements.txt` / `pyproject.toml` faltantes), por lo que las dependencias se infieren desde los imports.

## 🛠️ Instalación

### Configuración canónica desde la estructura actual del repositorio

```bash

git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers pillow matplotlib numpy tqdm opencv-python pandas wandb gdown
```

### Fragmento de instalación del README original (conservar)

El README anterior terminaba a mitad de bloque. Los comandos originales se mantienen abajo exactamente como referencia histórica:

```bash
git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip/src
```

Nota: la instantánea actual del repositorio coloca los scripts en la raíz, no bajo `src/`.

## ▶️ Inicio rápido

| Objetivo | Comando |
|---|---|
| Hacer caption de una imagen | `python image2caption.py -I /path/to/image.jpg -S L -C model.pt` |
| Hacer caption de un video | `python v2c.py -V /path/to/video.mp4 -N 10` |
| Construir dataset | `python dataset_generation.py` |

### Captionado de imagen (ejecución rápida)

```bash
python image2caption.py -I /path/to/image.jpg -S L -C model.pt
```

### Captionado de video (ruta recomendada)

```bash
python v2c.py -V /path/to/video.mp4 -N 10
```

## 🎯 Uso

### 1. Captionado de imágenes (`image2caption.py`)

```bash
python image2caption.py \
  -I /path/to/image.jpg \
  -S L \
  -C model.pt \
  -R ./data/result/prediction \
  -T 1.0
```

Argumentos:

- `-I, --img-path`: ruta de la imagen de entrada.
- `-S, --size`: tamaño del modelo (`S` o `L`).
- `-C, --checkpoint-name`: nombre del checkpoint en `weights/{small|large}`.
- `-R, --res-path`: directorio de salida para la imagen con caption renderizado.
- `-T, --temperature`: temperatura de muestreo.

### 2. CLI alternativa de imagen (`predict.py`)

```bash
python predict.py \
  -I /path/to/image.jpg \
  -S L \
  -C model.pt \
  -R ./data/result/prediction \
  -T 1.0
```

`predict.py` es funcionalmente similar a `image2caption.py`; el formato de texto de salida difiere ligeramente.

### 3. API de clase para captionado de imágenes (`i2c.py`)

```bash
python i2c.py -I /path/to/image.jpg -S L -C model.pt -R ./data/result/prediction -T 1.0
```

O impórtalo en tu propio script:

```python
from i2c import ImageCaptioner

captioner = ImageCaptioner(model_size="L", checkpoint_name="model.pt")
captioner.set_image_path("/path/to/image.jpg")
caption = captioner.generate_caption(save_image=True)
print(caption)
```

### 4. De video a subtítulos + JSON (`v2c.py`)

```bash
python v2c.py -V /path/to/video.mp4 -N 10
```

Output junto al video de entrada:

- `<video_basename>_caption.srt`
- `<video_basename>_caption.json`
- `<video_basename>_captioning_frames/`

### 5. Pipeline alternativo de video (`video2caption.py`)

```bash
python video2caption.py -V /path/to/video.mp4 -N 10
```

Importante: este script actualmente contiene rutas fijas específicas de máquina:

- Python path por defecto: `/home/lachlan/miniconda3/envs/caption/bin/python`
- Ruta del script de caption: `/home/lachlan/Projects/image_captioning/clip-gpt-captioning/src/image2caption.py`

Usa `v2c.py` a menos que mantengas esas rutas de forma intencional.

### 6. Variante legacy (`video2caption_v1.1.py`)

Este script se conserva como referencia histórica. Para uso activo, preferir `v2c.py`.

### 7. Generación de dataset

```bash
python dataset_generation.py
```

Entradas esperadas:

- `data/raw/results.csv` (tabla de captions separada por `|`).
- `data/raw/flickr30k_images/` (archivos de imagen referenciados por el CSV).

Salida:

- `data/processed/dataset.pkl`

### 8. Entrenamiento

```bash
python training.py -S L -C model.pt
```

El entrenamiento usa logging de Weights & Biases (`wandb`) por defecto.

### 9. Evaluación

```bash
python evaluate.py \
  -I ./data/raw/flickr30k_images \
  -R ./data/result/eval \
  -S L \
  -C model.pt \
  -T 1.0
```

La evaluación renderiza captions predichos sobre imágenes de prueba y los guarda en:

- `<res-path>/<checkpoint_name_without_ext>_<SIZE>/`

## ⚙️ Configuración

Las configuraciones del modelo se definen en `utils/config.py`:

| Config | Backbone CLIP | Modelo GPT | Directorio de pesos |
|---|---|---|---|
| `ConfigS` | `openai/clip-vit-base-patch32` | `gpt2` | `weights/small` |
| `ConfigL` | `openai/clip-vit-large-patch14` | `gpt2-medium` | `weights/large` |

Valores por defecto de las clases de configuración:

| Campo | `ConfigS` | `ConfigL` |
|---|---:|---:|
| `epochs` | 150 | 120 |
| `lr` | 3e-3 | 5e-3 |
| `batch_size_exp` | 6 | 5 |
| `ep_len` | 4 | 4 |
| `max_len` | 40 | 40 |

Las IDs de auto-descarga de checkpoints están en `utils/downloads.py`:

| Tamaño | ID de Google Drive |
|---|---|
| `L` | `1Gh32arzhW06C1ZJyzcJSSfdJDi3RgWoG` |
| `S` | `1pSQruQyg8KJq6VmzhMLFbT_VaHJMdlWF` |

## 📦 Archivos de salida

### Inferencia de imagen

- Imagen guardada con título/render sobrepuesto en `--res-path`.
- Patrón de nombre: `<input_stem>-R<SIZE>.jpg`.

### Inferencia de video (`v2c.py`)

- SRT: `<video_stem>_caption.srt`
- JSON: `<video_stem>_caption.json`
- Imágenes de frames: `<video_stem>_captioning_frames/`

Ejemplo de elemento JSON:

```json
{
  "start": "00:00:03,200",
  "end": "00:00:03,700",
  "lang": "en",
  "text": "A dog running through a field."
}
```

## 🧪 Ejemplos

### Ejemplo rápido de imagen

```bash
python image2caption.py -I ./examples/dog.jpg -S S -C model.pt
```

Comportamiento esperado:

- Si falta `weights/small/model.pt`, se descarga.
- Por defecto se guarda una imagen con caption en `./data/result/prediction`.
- El texto del caption se imprime en stdout.

### Ejemplo rápido de caption de video

```bash
python v2c.py -V ./examples/demo.mp4 -N 8
```

Comportamiento esperado:

- Se generan captions para 8 frames muestreados uniformemente.
- Los archivos `.srt` y `.json` se generan junto al video de entrada.

### Secuencia completa de entrenamiento/evaluación

```bash
python dataset_generation.py
python training.py -S L -C model.pt
python evaluate.py -I ./data/raw/flickr30k_images -R ./data/result/eval -S L -C model.pt -T 1.0
```

## 🧭 Notas de desarrollo

- Existe solapamiento legacy entre `v2c.py`, `video2caption.py` y `video2caption_v1.*`.
- `video2caption_v1.0_not_work.py` se conserva intencionalmente como código legacy no funcional.
- `training.py` actualmente selecciona `ConfigL()` mediante `config = ConfigL() if args.size.upper() else ConfigS()`, lo que resuelve siempre a `ConfigL` para valores no vacíos de `--size`.
- `model/trainer.py` usa `self.dataset` en `test_step`, mientras el inicializador asigna `self.test_dataset`; esto puede romper el muestreo en ejecuciones de entrenamiento si no se ajusta.
- `video2caption_v1.1.py` referencia `self.config.transform`, pero `ConfigS`/`ConfigL` no definen `transform`.
- Actualmente no existe suite de CI/pruebas en este snapshot del repositorio.
- Nota de i18n: en la parte superior de este README hay enlaces de idioma; se pueden añadir traducciones en `i18n/`.
- Nota de estado actual: los enlaces del language bar apuntan a `i18n/README.ru.md`, pero ese archivo no está presente en este snapshot.

## 🩺 Solución de problemas

- `AssertionError: Image does not exist`
  - Confirma que `-I/--img-path` apunta a un archivo válido.
- `Dataset file not found. Downloading...`
  - `MiniFlickrDataset` lanza esto cuando falta `data/processed/dataset.pkl`; ejecuta primero `python dataset_generation.py`.
- `Path to the test image folder does not exist`
  - Confirma que `evaluate.py -I` apunte a una carpeta existente.
- La primera ejecución es lenta o falla
  - La corrida inicial descarga modelos de Hugging Face y puede descargar checkpoints desde Google Drive.
- `video2caption.py` devuelve captions vacíos
  - Verifica la ruta del script hardcodeada y la ruta de Python, o cambia a `v2c.py`.
- `wandb` pide inicio de sesión durante entrenamiento
  - Ejecuta `wandb login` o desactiva el logging manualmente en `training.py` si hace falta.

## 🛣️ Hoja de ruta

- Añadir lockfiles de dependencias (`requirements.txt` o `pyproject.toml`) para instalaciones reproducibles.
- Unificar pipelines de video duplicados en una única implementación mantenida.
- Eliminar rutas hardcodeadas de máquina de scripts legacy.
- Corregir errores conocidos en casos límite de entrenamiento/evaluación en `training.py` y `model/trainer.py`.
- Añadir pruebas automatizadas y CI.
- Poblar `i18n/` con README traducidos referenciados en la barra de idiomas.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Flujo sugerido:

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

Si cambias el comportamiento del modelo, incluye:

- Comando(s) reproducible(s).
- Salidas de ejemplo antes/después.
- Notas sobre supuestos de checkpoint o dataset.

## ❤️ Support

| Donate | PayPal | Stripe |
|---|---|---|
| [![Donate](https://img.shields.io/badge/Donate-LazyingArt-0EA5E9?style=for-the-badge&logo=ko-fi&logoColor=white)](https://chat.lazying.art/donate) | [![PayPal](https://img.shields.io/badge/PayPal-RongzhouChen-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/RongzhouChen) | [![Stripe](https://img.shields.io/badge/Stripe-Donate-635BFF?style=for-the-badge&logo=stripe&logoColor=white)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## 📄 Licencia

No existe un archivo de licencia en el snapshot actual del repositorio.

Nota de supuesto: hasta que se añada un archivo `LICENSE`, los términos de reutilización/distribución permanecen indefinidos.
