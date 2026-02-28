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

Un toolkit en Python para generar subtítulos en lenguaje natural para imágenes y videos, combinando embeddings visuales de OpenAI CLIP con un modelo de lenguaje estilo GPT.

## ✨ Resumen

Este repositorio ofrece:

- Scripts de inferencia para subtitulado de imágenes y generación de subtítulos de video.
- Un pipeline de entrenamiento que aprende un mapeo desde embeddings visuales de CLIP hacia embeddings de tokens de GPT-2.
- Utilidades de generación de datasets para datos estilo Flickr30k.
- Descarga automática de checkpoints para tamaños de modelo compatibles cuando faltan pesos.
- Variantes multilingües del README en `i18n/` (consulta la barra de idiomas arriba).

La implementación actual incluye scripts nuevos y heredados. Algunos archivos heredados se conservan como referencia y se documentan más abajo.

## 🚀 Funcionalidades

- Subtitulado de imagen individual mediante `image2caption.py`.
- Subtitulado de video (muestreo uniforme de frames) mediante `v2c.py` o `video2caption.py`.
- Opciones de ejecución personalizables:
  - Número de frames.
  - Tamaño del modelo.
  - Temperatura de muestreo.
  - Nombre del checkpoint.
- Subtitulado con multiproceso/hilos para acelerar la inferencia en video.
- Artefactos de salida:
  - Archivos de subtítulos SRT (`.srt`).
  - Transcripciones JSON (`.json`) en `v2c.py`.
- Puntos de entrada de entrenamiento y evaluación para experimentos de mapeo CLIP+GPT2.

### De un vistazo

| Área | Script(s) principal(es) | Notas |
|---|---|---|
| Subtitulado de imágenes | `image2caption.py`, `i2c.py`, `predict.py` | CLI + clase reutilizable |
| Subtitulado de video | `v2c.py` | Ruta recomendada y mantenida |
| Flujo de video heredado | `video2caption.py`, `video2caption_v1.1.py` | Incluye supuestos específicos de máquina |
| Construcción de dataset | `dataset_generation.py` | Produce `data/processed/dataset.pkl` |
| Entrenamiento / evaluación | `training.py`, `evaluate.py` | Usa mapeo CLIP+GPT2 |

## 🧱 Arquitectura (alto nivel)

El modelo central en `model/model.py` tiene tres partes:

1. `ImageEncoder`: extrae el embedding de imagen de CLIP.
2. `Mapping`: proyecta el embedding de CLIP en una secuencia de embeddings de prefijo de GPT.
3. `TextDecoder`: cabecera de modelo de lenguaje GPT-2 que genera tokens de subtítulo de forma autorregresiva.

El entrenamiento (`Net.train_forward`) usa embeddings de imagen CLIP precomputados + subtítulos tokenizados.
La inferencia (`Net.forward`) usa una imagen PIL y decodifica tokens hasta EOS o `max_len`.

### Flujo de datos

1. Preparar dataset: `dataset_generation.py` lee `data/raw/results.csv` e imágenes en `data/raw/flickr30k_images/`, escribe `data/processed/dataset.pkl`.
2. Entrenar: `training.py` carga tuplas serializadas `(image_name, image_embedding, caption)` y entrena capas de mapper/decoder.
3. Evaluar: `evaluate.py` renderiza subtítulos generados sobre imágenes de prueba retenidas.
4. Servir inferencia:
   - imagen: `image2caption.py` / `predict.py` / `i2c.py`.
   - video: `v2c.py` (recomendado), `video2caption.py` (heredado).

## 🗂️ Estructura del proyecto

```text
VideoCaptionerWithClip/
├── README.md
├── image2caption.py               # CLI de subtitulado de imagen individual
├── predict.py                     # CLI alternativo de subtitulado de imagen individual
├── i2c.py                         # Clase ImageCaptioner reutilizable + CLI
├── v2c.py                         # Video -> SRT + JSON (subtitulado de frames con hilos)
├── video2caption.py               # Implementación alterna de video -> SRT (restricciones heredadas)
├── video2caption_v1.1.py          # Variante más antigua
├── video2caption_v1.0_not_work.py # Archivo heredado marcado explícitamente como no funcional
├── training.py                    # Punto de entrada de entrenamiento del modelo
├── evaluate.py                    # Evaluación en split de prueba y salidas renderizadas
├── dataset_generation.py          # Construye data/processed/dataset.pkl
├── data/
│   ├── __init__.py
│   └── dataset.py                 # Helpers de Dataset + DataLoader
├── model/
│   ├── __init__.py
│   ├── model.py                   # Codificador CLIP + mapeo + decodificador GPT2
│   └── trainer.py                 # Clase utilitaria de entrenamiento/validación/prueba
├── utils/
│   ├── __init__.py
│   ├── config.py                  # Valores por defecto de ConfigS / ConfigL
│   ├── downloads.py               # Descargador de checkpoints desde Google Drive
│   └── lr_warmup.py               # Programación de calentamiento de LR
├── i18n/                          # Variantes multilingües del README
└── .auto-readme-work/             # Artefactos del pipeline Auto-README
```

## 📋 Requisitos previos

- Se recomienda Python `3.10+`.
- Una GPU compatible con CUDA es opcional, pero muy recomendable para entrenamiento e inferencia con modelos grandes.
- `ffmpeg` no es requerido directamente por los scripts actuales (se usa OpenCV para extraer frames).
- Se necesita acceso a Internet la primera vez que se descargan modelos/checkpoints desde Hugging Face / Google Drive.

Actualmente no hay lockfile (`requirements.txt` / `pyproject.toml` faltan), por lo que las dependencias se infieren de los imports.

## 🛠️ Instalación

### Configuración canónica según el layout actual del repositorio

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

El README previo terminaba a mitad de bloque. Los comandos originales se conservan abajo exactamente como contenido histórico fuente de verdad:

```bash
git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip/src
```

Nota: en el snapshot actual del repositorio, los scripts están en la raíz del repositorio, no bajo `src/`.

## ▶️ Inicio rápido

### Subtitulado de imágenes (ejecución rápida)

```bash
python image2caption.py -I /path/to/image.jpg -S L -C model.pt
```

### Subtitulado de video (ruta recomendada)

```bash
python v2c.py -V /path/to/video.mp4 -N 10
```

## 🎯 Uso

### 1. Subtitulado de imágenes (`image2caption.py`)

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
- `-C, --checkpoint-name`: nombre del archivo checkpoint en `weights/{small|large}`.
- `-R, --res-path`: directorio de salida para la imagen subtitulada renderizada.
- `-T, --temperature`: temperatura de muestreo.

### 2. CLI alternativo para imagen (`predict.py`)

```bash
python predict.py \
  -I /path/to/image.jpg \
  -S L \
  -C model.pt \
  -R ./data/result/prediction \
  -T 1.0
```

`predict.py` es funcionalmente similar a `image2caption.py`; el formato del texto de salida difiere ligeramente.

### 3. API de clase para subtitulado de imágenes (`i2c.py`)

```bash
python i2c.py -I /path/to/image.jpg -S L -C model.pt -R ./data/result/prediction -T 1.0
```

O importa en tu propio script:

```python
from i2c import ImageCaptioner

captioner = ImageCaptioner(model_size="L", checkpoint_name="model.pt")
captioner.set_image_path("/path/to/image.jpg")
caption = captioner.generate_caption(save_image=True)
print(caption)
```

### 4. Video a subtítulos + JSON (`v2c.py`)

```bash
python v2c.py -V /path/to/video.mp4 -N 10
```

Salidas junto al video de entrada:

- `<video_basename>_caption.srt`
- `<video_basename>_caption.json`
- `<video_basename>_captioning_frames/`

### 5. Pipeline alterno de video (`video2caption.py`)

```bash
python video2caption.py -V /path/to/video.mp4 -N 10
```

Importante: este script actualmente contiene rutas hardcodeadas específicas de máquina:

- Ruta Python por defecto: `/home/lachlan/miniconda3/envs/caption/bin/python`
- Ruta del script de subtitulado: `/home/lachlan/Projects/image_captioning/clip-gpt-captioning/src/image2caption.py`

Usa `v2c.py` a menos que quieras mantener intencionalmente estas rutas.

### 6. Variante heredada (`video2caption_v1.1.py`)

Este script se conserva como referencia histórica. Para uso activo, prefiere `v2c.py`.

### 7. Generación de dataset

```bash
python dataset_generation.py
```

Entradas raw esperadas:

- `data/raw/results.csv` (tabla de subtítulos separada por tuberías).
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

La evaluación renderiza subtítulos predichos sobre imágenes de prueba y los guarda en:

- `<res-path>/<checkpoint_name_without_ext>_<SIZE>/`

## ⚙️ Configuración

Las configuraciones del modelo se definen en `utils/config.py`:

| Config | Backbone CLIP | Modelo GPT | Directorio de pesos |
|---|---|---|---|
| `ConfigS` | `openai/clip-vit-base-patch32` | `gpt2` | `weights/small` |
| `ConfigL` | `openai/clip-vit-large-patch14` | `gpt2-medium` | `weights/large` |

Valores por defecto clave de las clases de configuración:

| Campo | `ConfigS` | `ConfigL` |
|---|---:|---:|
| `epochs` | 150 | 120 |
| `lr` | 3e-3 | 5e-3 |
| `batch_size_exp` | 6 | 5 |
| `ep_len` | 4 | 4 |
| `max_len` | 40 | 40 |

Los IDs de descarga automática de checkpoints están en `utils/downloads.py`:

| Tamaño | ID de Google Drive |
|---|---|
| `L` | `1Gh32arzhW06C1ZJyzcJSSfdJDi3RgWoG` |
| `S` | `1pSQruQyg8KJq6VmzhMLFbT_VaHJMdlWF` |

## 📦 Archivos de salida

### Inferencia de imagen

- Imagen guardada con título superpuesto/generado en `--res-path`.
- Patrón de nombre de archivo: `<input_stem>-R<SIZE>.jpg`.

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

### Ejemplo rápido de subtitulado de imagen

```bash
python image2caption.py -I ./examples/dog.jpg -S S -C model.pt
```

Comportamiento esperado:

- Si falta `weights/small/model.pt`, se descarga.
- Se escribe una imagen subtitulada en `./data/result/prediction` por defecto.
- El texto del subtítulo se imprime en stdout.

### Ejemplo rápido de subtitulado de video

```bash
python v2c.py -V ./examples/demo.mp4 -N 8
```

Comportamiento esperado:

- Se subtitulan 8 frames muestreados uniformemente.
- Se generan archivos `.srt` y `.json` junto al video de entrada.

### Secuencia completa de entrenamiento/evaluación

```bash
python dataset_generation.py
python training.py -S L -C model.pt
python evaluate.py -I ./data/raw/flickr30k_images -R ./data/result/eval -S L -C model.pt -T 1.0
```

## 🧭 Notas de desarrollo

- Hay superposición heredada entre `v2c.py`, `video2caption.py` y `video2caption_v1.*`.
- `video2caption_v1.0_not_work.py` se conserva intencionalmente como código heredado no funcional.
- `training.py` actualmente selecciona `ConfigL()` vía `config = ConfigL() if args.size.upper() else ConfigS()`, lo que siempre resuelve en `ConfigL` para valores no vacíos de `--size`.
- `model/trainer.py` usa `self.dataset` en `test_step`, mientras que el inicializador asigna `self.test_dataset`; esto puede romper el muestreo en ejecuciones de entrenamiento si no se ajusta.
- `video2caption_v1.1.py` referencia `self.config.transform`, pero `ConfigS`/`ConfigL` no definen `transform`.
- No hay un suite de CI/tests definido actualmente en este snapshot del repositorio.
- Nota i18n: hay enlaces de idiomas en la parte superior de este README; los archivos traducidos pueden añadirse en `i18n/`.
- Nota de estado actual: la barra de idiomas enlaza `i18n/README.ru.md`, pero ese archivo no está presente en este snapshot.

## 🩺 Solución de problemas

- `AssertionError: Image does not exist`
  - Confirma que `-I/--img-path` apunta a un archivo válido.
- `Dataset file not found. Downloading...`
  - `MiniFlickrDataset` lanza esto cuando falta `data/processed/dataset.pkl`; ejecuta `python dataset_generation.py` primero.
- `Path to the test image folder does not exist`
  - Confirma que `evaluate.py -I` apunta a una carpeta existente.
- Primera ejecución lenta o fallida
  - La ejecución inicial descarga modelos de Hugging Face y puede descargar checkpoints desde Google Drive.
- `video2caption.py` devuelve subtítulos vacíos
  - Verifica la ruta hardcodeada del script y la ruta del ejecutable de Python, o cambia a `v2c.py`.
- `wandb` pide login durante el entrenamiento
  - Ejecuta `wandb login` o desactiva manualmente el logging en `training.py` si hace falta.

## 🛣️ Hoja de ruta

- Añadir lockfiles de dependencias (`requirements.txt` o `pyproject.toml`) para instalaciones reproducibles.
- Unificar pipelines de video duplicados en una implementación mantenida.
- Eliminar rutas hardcodeadas de máquina en scripts heredados.
- Corregir bugs conocidos de casos límite de entrenamiento/evaluación en `training.py` y `model/trainer.py`.
- Añadir tests automatizados y CI.
- Completar `i18n/` con los README traducidos referenciados en la barra de idiomas.

## 🤝 Contribuir

Las contribuciones son bienvenidas. Flujo sugerido:

```bash
# 1) Fork y clon
git clone git@github.com:<your-user>/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip

# 2) Crear una rama de feature
git checkout -b feat/your-change

# 3) Hacer cambios y commit
git add .
git commit -m "feat: describe your change"

# 4) Push y abrir un PR
git push origin feat/your-change
```

Si cambias el comportamiento del modelo, incluye:

- Comando(s) reproducible(s).
- Salidas de muestra antes/después.
- Notas sobre supuestos de checkpoint o dataset.

## 🙌 Soporte

No hay una configuración explícita de donaciones/patrocinios en el snapshot actual del repositorio.

Si se agregan enlaces de patrocinio más adelante, deberían conservarse en esta sección.

## 📄 Licencia

No hay archivo de licencia presente en el snapshot actual del repositorio.

Nota de suposición: hasta que se agregue un archivo `LICENSE`, los términos de reutilización/distribución no están definidos.
