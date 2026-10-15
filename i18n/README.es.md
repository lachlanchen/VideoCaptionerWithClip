[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner — apps and four buddies](https://lazying.art/media/brand/lazyingart-intro.gif)](https://lazying.art)

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

## 🧭 Navegación rápida

| Sección | Para qué sirve |
|---|---|
| Snapshot | Ver el alcance del repositorio y el inventario actual de scripts |
| Overview | Ver objetivos y capacidades |
| Uso | Seguir los flujos CLI/API exactos |
| Solución de problemas | Resolver incidencias comunes rápidamente |
| Hoja de ruta | Seguir objetivos de limpieza y mejora conocidos |

---

Un toolkit de Python para generar subtítulos y textos en lenguaje natural sobre imágenes y vídeos combinando embeddings visuales de OpenAI CLIP con un modelo de lenguaje tipo GPT.

## 🧭 Resumen

| Dimensión | Detalles |
|---|---|
| Cobertura de tareas | Captionado de imagen y vídeo |
| Salidas principales | Subtítulos SRT, transcripciones JSON, imágenes con pie de imagen |
| Scripts principales | `i2c.py`, `v2c.py`, `image2caption.py` |
| Rutas heredadas | `video2caption.py` y sus variantes versionadas (conservadas por historial) |
| Flujo de datos | `data/raw/results.csv` + `data/raw/flickr30k_images/` |

## ✨ Visión general

Este repositorio ofrece:

- Scripts de inferencia para captionado de imágenes y generación de subtítulos de vídeo.
- Un pipeline de entrenamiento que aprende un mapeo entre embeddings visuales de CLIP y embeddings de tokens de GPT-2.
- Utilidades para generar datasets con estilo Flickr30k.
- Descarga automática de checkpoints para tamaños de modelo compatibles cuando faltan los pesos.
- Variantes de README multilingües en `i18n/` (ver la barra de idiomas arriba).

La implementación actual incluye scripts nuevos y heredados. Algunos ficheros legacy se conservan para referencia y están documentados abajo.

## 🚀 Funcionalidades

- Captionado de una sola imagen mediante `image2caption.py`.
- Captionado de vídeo (muestreo uniforme de fotogramas) con `v2c.py` o `video2caption.py`.
- Opciones de ejecución personalizables:
  - Número de fotogramas.
  - Tamaño del modelo.
  - Temperatura de muestreo.
  - Nombre del checkpoint.
- Captionado en multiproceso para acelerar inferencia de vídeo.
- Artefactos de salida:
  - Archivos de subtítulos SRT (`.srt`).
  - Transcripciones JSON (`.json`) en `v2c.py`.
- Entradas de entrenamiento y evaluación para experimentos de mapeo CLIP+GPT2.

### A simple vista

| Área | Script principal | Notas |
|---|---|---|
| Captionado de imagen | `image2caption.py`, `i2c.py`, `predict.py` | CLI + clase reutilizable |
| Captionado de vídeo | `v2c.py` | Ruta mantenida recomendada |
| Flujo legacy de vídeo | `video2caption.py`, `video2caption_v1.1.py` | Incluye suposiciones específicas de máquina |
| Construcción de dataset | `dataset_generation.py` | Genera `data/processed/dataset.pkl` |
| Entrenamiento / evaluación | `training.py`, `evaluate.py` | Usa mapeo CLIP+GPT2 |

## 🧱 Arquitectura (vista general)

El modelo principal en `model/model.py` tiene tres partes:

1. `ImageEncoder`: extrae embeddings de imagen de CLIP.
2. `Mapping`: proyecta el embedding de CLIP en una secuencia de embeddings de prefijo GPT.
3. `TextDecoder`: cabecera de lenguaje basada en GPT-2 que genera tokens de forma autoregresiva.

El entrenamiento (`Net.train_forward`) usa embeddings de imagen CLIP precalculados + captions tokenizados.
La inferencia (`Net.forward`) usa una imagen PIL y decodifica tokens hasta EOS o `max_len`.

### Flujo de datos

1. Preparar dataset: `dataset_generation.py` lee `data/raw/results.csv` y las imágenes en `data/raw/flickr30k_images/`, y escribe `data/processed/dataset.pkl`.
2. Entrenar: `training.py` carga tuplas serializadas `(image_name, image_embedding, caption)` y entrena capas mapper/decoder.
3. Evaluar: `evaluate.py` genera captions para imágenes de prueba del split retenido.
4. Ejecutar inferencia:
   - imagen: `image2caption.py` / `predict.py` / `i2c.py`.
   - vídeo: `v2c.py` (recomendado), `video2caption.py` (legacy).

## 🗂️ Estructura del proyecto

```text
VideoCaptionerWithClip/
├── README.md
├── image2caption.py               # CLI de captionado para imagen única
├── predict.py                     # CLI alternativa para imagen única
├── i2c.py                         # Clase reutilizable ImageCaptioner + CLI
├── v2c.py                         # Vídeo -> SRT + JSON (captionado de fotogramas con hilos)
├── video2caption.py               # Implementación alternativa vídeo -> SRT (legacy, con limitaciones)
├── video2caption_v1.1.py          # Variante anterior
├── video2caption_v1.0_not_work.py # Archivo legacy explícitamente marcado como no funcional
├── training.py                    # Punto de entrada de entrenamiento del modelo
├── evaluate.py                    # Evaluación en split de prueba y salidas renderizadas
├── dataset_generation.py          # Construye data/processed/dataset.pkl
├── data/
│   ├── __init__.py
│   └── dataset.py                 # Helpers de Dataset + DataLoader
├── model/
│   ├── __init__.py
│   ├── model.py                   # Encodificador CLIP + mapping + decodificador GPT2
│   └── trainer.py                 # Clase utilitaria train/val/test
├── utils/
│   ├── __init__.py
│   ├── config.py                  # Defaults ConfigS / ConfigL
│   ├── downloads.py               # Descargador de checkpoints desde Google Drive
│   └── lr_warmup.py               # Planificador de warmup de LR
├── i18n/                          # Variantes del README en varios idiomas
└── .auto-readme-work/             # Artefactos del pipeline auto-README
```

## 📋 Requisitos previos

- Python `3.10+` recomendado.
- Se recomienda GPU con CUDA, especialmente para entrenamiento y inferencia con modelos grandes, aunque es opcional.
- `ffmpeg` no es requisito directo de los scripts actuales (OpenCV se usa para la extracción de fotogramas).
- Se requiere acceso a internet para la primera descarga de modelos/checkpoints desde Hugging Face / Google Drive.

Actualmente no hay lockfile presente (`requirements.txt` / `pyproject.toml` faltantes), por lo que las dependencias se infieren desde los imports.

## 🛠️ Instalación

### Configuración canónica para el layout actual del repositorio

```bash
git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers pillow matplotlib numpy tqdm opencv-python pandas wandb gdown
```

### Fragmento de instalación del README original (conservado)

El README anterior terminaba a mitad de bloque. Los comandos originales se conservan abajo exactamente como fuente histórica:

```bash
git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip/src
```

Nota: la instantánea actual del repositorio coloca los scripts en la raíz, no bajo `src/`.

## ▶️ Inicio rápido

| Objetivo | Comando |
|---|---|
| Captionar una imagen | `python image2caption.py -I /path/to/image.jpg -S L -C model.pt` |
| Captionar un vídeo | `python v2c.py -V /path/to/video.mp4 -N 10` |
| Construir dataset | `python dataset_generation.py` |

### Captionado de imagen (ejecución rápida)

```bash
python image2caption.py -I /path/to/image.jpg -S L -C model.pt
```

### Captionado de vídeo (ruta recomendada)

```bash
python v2c.py -V /path/to/video.mp4 -N 10
```

## 🎯 Uso

### 1. Captionado de imagen (`image2caption.py`)

```bash
python image2caption.py \
  -I /path/to/image.jpg \
  -S L \
  -C model.pt \
  -R ./data/result/prediction \
  -T 1.0
```

Argumentos:

- `-I, --img-path`: ruta de imagen de entrada.
- `-S, --size`: tamaño del modelo (`S` o `L`).
- `-C, --checkpoint-name`: nombre del checkpoint dentro de `weights/{small|large}`.
- `-R, --res-path`: directorio de salida para la imagen con el caption renderizado.
- `-T, --temperature`: temperatura de muestreo.

### 2. CLI alternativa para imagen (`predict.py`)

```bash
python predict.py \
  -I /path/to/image.jpg \
  -S L \
  -C model.pt \
  -R ./data/result/prediction \
  -T 1.0
```

`predict.py` es funcionalmente similar a `image2caption.py`; el formato del texto de salida difiere levemente.

### 3. API de clase para captionado de imagen (`i2c.py`)

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

### 4. De vídeo a subtítulos + JSON (`v2c.py`)

```bash
python v2c.py -V /path/to/video.mp4 -N 10
```

Salida junto al vídeo de entrada:

- `<video_basename>_caption.srt`
- `<video_basename>_caption.json`
- `<video_basename>_captioning_frames/`

### 5. Pipeline alternativo de vídeo (`video2caption.py`)

```bash
python video2caption.py -V /path/to/video.mp4 -N 10
```

Importante: este script contiene rutas duras específicas de máquina:

- Python path predeterminado: `/home/lachlan/miniconda3/envs/caption/bin/python`
- Ruta del script de caption: `/home/lachlan/Projects/image_captioning/clip-gpt-captioning/src/image2caption.py`

Usa `v2c.py` a menos que mantengas estas rutas de forma intencionada.

### 6. Variante heredada (`video2caption_v1.1.py`)

Este script se conserva como referencia histórica. Para uso activo, prefiere `v2c.py`.

### 7. Generación de dataset

```bash
python dataset_generation.py
```

Entradas esperadas:

- `data/raw/results.csv` (tabla de captions separada por `|`).
- `data/raw/flickr30k_images/` (ficheros de imagen referenciados por el CSV).

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

La evaluación renderiza los captions predichos sobre imágenes de prueba y los guarda en:

- `<res-path>/<checkpoint_name_without_ext>_<SIZE>/`

## ⚙️ Configuración

Las configuraciones del modelo se definen en `utils/config.py`:

| Config | Backbone de CLIP | Modelo GPT | Carpeta de pesos |
|---|---|---|---|
| `ConfigS` | `openai/clip-vit-base-patch32` | `gpt2` | `weights/small` |
| `ConfigL` | `openai/clip-vit-large-patch14` | `gpt2-medium` | `weights/large` |

Valores predeterminados de las clases de configuración:

| Campo | `ConfigS` | `ConfigL` |
|---|---:|---:|
| `epochs` | 150 | 120 |
| `lr` | 3e-3 | 5e-3 |
| `batch_size_exp` | 6 | 5 |
| `ep_len` | 4 | 4 |
| `max_len` | 40 | 40 |

Los IDs de auto-descarga de checkpoints están en `utils/downloads.py`:

| Tamaño | ID de Google Drive |
|---|---|
| `L` | `1Gh32arzhW06C1ZJyzcJSSfdJDi3RgWoG` |
| `S` | `1pSQruQyg8KJq6VmzhMLFbT_VaHJMdlWF` |

## 📦 Archivos de salida

### Inferencia de imagen

- Imagen guardada con título o texto overlay en `--res-path`.
- Patrón de nombre de archivo: `<input_stem>-R<SIZE>.jpg`.

### Inferencia de vídeo (`v2c.py`)

- SRT: `<video_stem>_caption.srt`
- JSON: `<video_stem>_caption.json`
- Imágenes de fotogramas: `<video_stem>_captioning_frames/`

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

- Si falta `weights/small/model.pt`, se descarga automáticamente.
- Por defecto, se guarda una imagen con caption en `./data/result/prediction`.
- El texto del caption se imprime en stdout.

### Ejemplo rápido de caption de vídeo

```bash
python v2c.py -V ./examples/demo.mp4 -N 8
```

Comportamiento esperado:

- Se generan captions para 8 fotogramas muestreados uniformemente.
- Los archivos `.srt` y `.json` se generan junto al vídeo de entrada.

### Secuencia completa entrenamiento/evaluación

```bash
python dataset_generation.py
python training.py -S L -C model.pt
python evaluate.py -I ./data/raw/flickr30k_images -R ./data/result/eval -S L -C model.pt -T 1.0
```

## 🧭 Notas de desarrollo

- Existe solapamiento legacy entre `v2c.py`, `video2caption.py` y `video2caption_v1.*`.
- `video2caption_v1.0_not_work.py` se conserva intencionadamente como código legado no funcional.
- `training.py` actualmente selecciona `ConfigL()` en `config = ConfigL() if args.size.upper() else ConfigS()`, lo que siempre resuelve a `ConfigL` para valores no vacíos de `--size`.
- `model/trainer.py` usa `self.dataset` en `test_step`, mientras el inicializador asigna `self.test_dataset`; esto puede romper el muestreo en ejecuciones de entrenamiento si no se ajusta.
- `video2caption_v1.1.py` referencia `self.config.transform`, pero `ConfigS`/`ConfigL` no definen `transform`.
- Actualmente no existe suite de CI/pruebas en este snapshot del repositorio.
- Nota de i18n: en este README hay enlaces de idiomas arriba; se pueden agregar traducciones en `i18n/`.
- Nota de estado actual: los enlaces de idioma apuntan a `i18n/README.ru.md`, pero ese archivo no está presente en este snapshot.

## 🩺 Solución de problemas

- `AssertionError: Image does not exist`
  - Verifica que `-I/--img-path` apunte a un archivo válido.
- `Dataset file not found. Downloading...`
  - `MiniFlickrDataset` lanza esto cuando falta `data/processed/dataset.pkl`; ejecuta primero `python dataset_generation.py`.
- `Path to the test image folder does not exist`
  - Confirma que `evaluate.py -I` apunte a una carpeta existente.
- Primera corrida lenta o fallida
  - La primera ejecución descarga modelos de Hugging Face y puede descargar checkpoints desde Google Drive.
- `video2caption.py` devuelve captions vacíos
  - Verifica la ruta del script hardcodeada y la ruta del ejecutable Python, o cambia a `v2c.py`.
- `wandb` solicita login durante entrenamiento
  - Ejecuta `wandb login` o desactiva el logging manualmente en `training.py` si hace falta.

## 🛣️ Hoja de ruta

- Añadir lockfiles de dependencias (`requirements.txt` o `pyproject.toml`) para instalaciones reproducibles.
- Unificar pipelines de vídeo duplicados en una implementación mantenida.
- Eliminar rutas hardcodeadas de máquina de los scripts legacy.
- Corregir bugs conocidos en casos límite de entrenamiento/evaluación en `training.py` y `model/trainer.py`.
- Añadir pruebas y CI automatizados.
- Poblar `i18n/` con los README traducidos referenciados en la barra de idiomas.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Flujo sugerido:

```bash
# 1) Fork and clone
git clone git@github.com:<your-user>/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip

# 2) Crear rama de funcionalidades
git checkout -b feat/your-change

# 3) Hacer cambios y confirmar

git add .
git commit -m "feat: describe your change"

# 4) Enviar y abrir PR
git push origin feat/your-change
```

Si cambias el comportamiento del modelo, incluye:

- Comando(s) reproducibles.
- Ejemplos de salida antes/después.
- Notas sobre supuestos de checkpoint o dataset.

---

## 📄 Licencia

No existe un archivo de licencia en el snapshot actual del repositorio.

Nota de supuesto: hasta que se añada un archivo `LICENSE`, los términos de reutilización/distribución permanecen indefinidos.


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
