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

## 🧭 Быстрая навигация

| Раздел | Для чего использовать |
|---|---|
| Snapshot | Смотреть границы репозитория и текущий список скриптов |
| Overview | Читать цели и возможности |
| Usage | Следовать точным сценариям CLI/API |
| Troubleshooting | Быстро устранять типичные ошибки во время выполнения |
| Roadmap | Отслеживать известные цели по очистке и улучшениям |

---

Набор инструментов на Python для генерации описаний изображений и видео на естественном языке, объединяющий визуальные эмбеддинги OpenAI CLIP и язык-модель в стиле GPT.

## 🧭 Snapshot

| Измерение | Детали |
|---|---|
| Покрытие задач | Описание изображений и видео |
| Основные выходные данные | SRT-субтитры, JSON-транскрипты, аннотированные изображения |
| Основные скрипты | `i2c.py`, `v2c.py`, `image2caption.py` |
| Устаревшие пути | `video2caption.py` и версии для истории (оставлены для справки) |
| Поток датасета | `data/raw/results.csv` + `data/raw/flickr30k_images/` |

## ✨ Обзор

Этот репозиторий предоставляет:

- Скрипты инференса для генерации подписей к изображениям и видео.
- Обучающий конвейер, который учит отображение от эмбеддингов CLIP к токен-эмбеддингам GPT-2.
- Инструменты генерации датасета для данных в стиле Flickr30k.
- Автоматическую загрузку контрольных точек для поддерживаемых размеров модели, если веса отсутствуют.
- Мультиязычные варианты README в `i18n/` (см. языковую полосу выше).

Текущая реализация включает как новые, так и устаревшие скрипты. Часть из них сохранены как справочные и описаны ниже.

## 🚀 Возможности

- Одиночная генерация подписи через `image2caption.py`.
- Генерация подписей для видео (равномерная выборка кадров) через `v2c.py` или `video2caption.py`.
- Настраиваемые параметры выполнения:
  - Количество кадров.
  - Размер модели.
  - Температура сэмплирования.
  - Имя контрольной точки.
- Многопоточная/многопроцессная captioning-процедура для более быстрого инференса видео.
- Артефакты вывода:
  - SRT-файлы субтитров (`.srt`).
  - JSON-транскрипты (`.json`) в `v2c.py`.
- Точки входа для обучения и валидации экспериментов по отображению CLIP+GPT2.

### Вкратце

| Область | Основной(ые) скрипт(ы) | Примечания |
|---|---|---|
| Подпись изображений | `image2caption.py`, `i2c.py`, `predict.py` | CLI + переиспользуемый класс |
| Подпись видео | `v2c.py` | Рекомендуемый поддерживаемый путь |
| Устаревший видеопоток | `video2caption.py`, `video2caption_v1.1.py` | Содержит машинно-специфические предположения |
| Построение датасета | `dataset_generation.py` | Формирует `data/processed/dataset.pkl` |
| Обучение / оценка | `training.py`, `evaluate.py` | Использует отображение CLIP+GPT2 |

## 🧱 Архитектура (высокоуровнево)

Ядро модели в `model/model.py` состоит из трёх частей:

1. `ImageEncoder`: извлекает векторное представление CLIP изображения.
2. `Mapping`: проецирует эмбеддинг CLIP в последовательность prefix-эмбеддингов GPT.
3. `TextDecoder`: глава на базе GPT-2, которая автрегрессионно генерирует токены подписи.

Обучение (`Net.train_forward`) использует предвычисленные эмбеддинги изображений CLIP и токенизированные подписи.
Инференс (`Net.forward`) принимает PIL-изображение и декодирует токены до `EOS` или `max_len`.

### Поток данных

1. Подготовка датасета: `dataset_generation.py` считывает `data/raw/results.csv` и изображения из `data/raw/flickr30k_images/`, пишет `data/processed/dataset.pkl`.
2. Обучение: `training.py` загружает сериализованные кортежи `(image_name, image_embedding, caption)` и обучает слои маппера/декодера.
3. Оценка: `evaluate.py` визуализирует сгенерированные подписи на отложенных тестовых изображениях.
4. Инференс:
   - изображение: `image2caption.py` / `predict.py` / `i2c.py`.
   - видео: `v2c.py` (рекомендуется), `video2caption.py` (устаревший).

## 🗂️ Структура проекта

```text
VideoCaptionerWithClip/
├── README.md
├── image2caption.py               # CLI для одного изображения
├── predict.py                     # Альтернативный CLI для одного изображения
├── i2c.py                         # Переиспользуемый класс ImageCaptioner + CLI
├── v2c.py                         # Видео -> SRT + JSON (многопоточный captioning кадров)
├── video2caption.py               # Альтернативная реализация видео -> SRT (ограничения legacy)
├── video2caption_v1.1.py          # Более старая вариация
├── video2caption_v1.0_not_work.py # Явно помеченный нерабочий legacy-файл
├── training.py                    # Точка входа для обучения
├── evaluate.py                    # Оценка на тесте и сохранение рендеров
├── dataset_generation.py          # Формирует data/processed/dataset.pkl
├── data/
│   ├── __init__.py
│   └── dataset.py                 # Набор данных и вспомогательные DataLoader
├── model/
│   ├── __init__.py
│   ├── model.py                   # CLIP encoder + mapping + GPT2 decoder
│   └── trainer.py                 # Класс утилит обучения/валидации/тестов
├── utils/
│   ├── __init__.py
│   ├── config.py                  # Значения по умолчанию ConfigS / ConfigL
│   ├── downloads.py               # Скачивание чекпоинтов с Google Drive
│   └── lr_warmup.py               # Расписание разогрева LR
├── i18n/                          # Многоязычные README
└── .auto-readme-work/             # Артефакты пайплайна автогенерации README
```

## 📋 Предпосылки

- Рекомендуется `Python` `3.10+`.
- Для обучения и инференса больших моделей желательно использовать GPU с CUDA (не обязательно, но сильно желательно).
- `ffmpeg` непосредственно текущими скриптами не требуется (`opencv` используется для извлечения кадров).
- Для первого скачивания моделей/весов с Hugging Face / Google Drive нужен доступ в Интернет.

На текущий момент lockfile отсутствует (`requirements.txt` / `pyproject.toml` отсутствуют), поэтому зависимости выводятся по импортам.

## 🛠️ Установка

### Каноничная установка из текущей структуры репозитория

```bash

git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers pillow matplotlib numpy tqdm opencv-python pandas wandb gdown
```

### Оригинальный фрагмент установки из README (сохранён)

Предыдущая версия README обрывалась в середине блока. Команды ниже сохранены как исторически-истинное содержимое:

```bash
git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip/src
```

Примечание: текущий снимок репозитория помещает скрипты в корень репозитория, а не в `src/`.

## ▶️ Быстрый старт

| Цель | Команда |
|---|---|
| Сгенерировать подпись для изображения | `python image2caption.py -I /path/to/image.jpg -S L -C model.pt` |
| Сгенерировать подпись для видео | `python v2c.py -V /path/to/video.mp4 -N 10` |
| Собрать датасет | `python dataset_generation.py` |

### Запуск для изображений (быстрый пример)

```bash
python image2caption.py -I /path/to/image.jpg -S L -C model.pt
```

### Запуск для видео (рекомендуемый путь)

```bash
python v2c.py -V /path/to/video.mp4 -N 10
```

## 🎯 Использование

### 1. Изображения (`image2caption.py`)

```bash
python image2caption.py \
  -I /path/to/image.jpg \
  -S L \
  -C model.pt \
  -R ./data/result/prediction \
  -T 1.0
```

Аргументы:

- `-I, --img-path`: путь к входному изображению.
- `-S, --size`: размер модели (`S` или `L`).
- `-C, --checkpoint-name`: имя контрольной точки в `weights/{small|large}`.
- `-R, --res-path`: папка для сохранения отрендеренного изображения с подписью.
- `-T, --temperature`: температура сэмплирования.

### 2. Альтернативный image CLI (`predict.py`)

```bash
python predict.py \
  -I /path/to/image.jpg \
  -S L \
  -C model.pt \
  -R ./data/result/prediction \
  -T 1.0
```

`predict.py` функционально похож на `image2caption.py`; формат вывода текста немного отличается.

### 3. Класс API captioning изображений (`i2c.py`)

```bash
python i2c.py -I /path/to/image.jpg -S L -C model.pt -R ./data/result/prediction -T 1.0
```

Или импортируйте в собственном скрипте:

```python
from i2c import ImageCaptioner

captioner = ImageCaptioner(model_size="L", checkpoint_name="model.pt")
captioner.set_image_path("/path/to/image.jpg")
caption = captioner.generate_caption(save_image=True)
print(caption)
```

### 4. Видео в субтитры + JSON (`v2c.py`)

```bash
python v2c.py -V /path/to/video.mp4 -N 10
```

Создаёт файлы рядом с исходным видео:

- `<video_basename>_caption.srt`
- `<video_basename>_caption.json`
- `<video_basename>_captioning_frames/`

### 5. Альтернативный видеопоток (`video2caption.py`)

```bash
python video2caption.py -V /path/to/video.mp4 -N 10
```

Важно: этот скрипт пока содержит жестко заданные пути, специфичные для машины:

- Python path по умолчанию: `/home/lachlan/miniconda3/envs/caption/bin/python`
- Путь скрипта подписи: `/home/lachlan/Projects/image_captioning/clip-gpt-captioning/src/image2caption.py`

Используйте `v2c.py`, если только вы целенаправленно не поддерживаете эти пути.

### 6. Устаревший вариант (`video2caption_v1.1.py`)

Этот скрипт сохраняется как историческая ссылка. Для активного использования предпочтительнее `v2c.py`.

### 7. Генерация датасета

```bash
python dataset_generation.py
```

Ожидаемые исходные данные:

- `data/raw/results.csv` (таблица подписей, разделённая вертикальной чертой).
- `data/raw/flickr30k_images/` (изображения, на которые ссылается CSV).

Выход:

- `data/processed/dataset.pkl`

### 8. Обучение

```bash
python training.py -S L -C model.pt
```

По умолчанию обучение пишет логи в Weights & Biases (`wandb`).

### 9. Оценка

```bash
python evaluate.py \
  -I ./data/raw/flickr30k_images \
  -R ./data/result/eval \
  -S L \
  -C model.pt \
  -T 1.0
```

Оценка рендерит предсказания на тестовых изображениях и сохраняет их в:

- `<res-path>/<checkpoint_name_without_ext>_<SIZE>/`

## ⚙️ Конфигурация

Модельные конфигурации определены в `utils/config.py`:

| Конфиг | CLIP backbone | GPT модель | Каталог весов |
|---|---|---|---|
| `ConfigS` | `openai/clip-vit-base-patch32` | `gpt2` | `weights/small` |
| `ConfigL` | `openai/clip-vit-large-patch14` | `gpt2-medium` | `weights/large` |

Ключевые значения по умолчанию из классов конфигурации:

| Поле | `ConfigS` | `ConfigL` |
|---|---:|---:|
| `epochs` | 150 | 120 |
| `lr` | 3e-3 | 5e-3 |
| `batch_size_exp` | 6 | 5 |
| `ep_len` | 4 | 4 |
| `max_len` | 40 | 40 |

ID автоматических скачиваний чекпоинтов находятся в `utils/downloads.py`:

| Размер | Google Drive ID |
|---|---|
| `L` | `1Gh32arzhW06C1ZJyzcJSSfdJDi3RgWoG` |
| `S` | `1pSQruQyg8KJq6VmzhMLFbT_VaHJMdlWF` |

## 📦 Выходные файлы

### Инференс изображений

- Сохранённое изображение с наложенной/сгенерированной подписью в каталоге `--res-path`.
- Шаблон имени файла: `<input_stem>-R<SIZE>.jpg`.

### Инференс видео (`v2c.py`)

- SRT: `<video_stem>_caption.srt`
- JSON: `<video_stem>_caption.json`
- Кадры: `<video_stem>_captioning_frames/`

Пример элемента JSON:

```json
{
  "start": "00:00:03,200",
  "end": "00:00:03,700",
  "lang": "en",
  "text": "A dog running through a field."
}
```

## 🧪 Примеры

### Быстрый пример подписи изображения

```bash
python image2caption.py -I ./examples/dog.jpg -S S -C model.pt
```

Ожидаемое поведение:

- Если `weights/small/model.pt` отсутствует, он будет загружен.
- По умолчанию в `./data/result/prediction` сохраняется изображение с подписью.
- Текст подписи печатается в stdout.

### Быстрый пример подписи видео

```bash
python v2c.py -V ./examples/demo.mp4 -N 8
```

Ожидаемое поведение:

- Обрабатываются и подписываются 8 кадров, выбранных равномерно.
- `.srt` и `.json` генерируются рядом с исходным видео.

### Полный цикл обучения и оценки

```bash
python dataset_generation.py
python training.py -S L -C model.pt
python evaluate.py -I ./data/raw/flickr30k_images -R ./data/result/eval -S L -C model.pt -T 1.0
```

## 🧭 Заметки по разработке

- Есть перекрытия legacy-кода между `v2c.py`, `video2caption.py` и `video2caption_v1.*`.
- `video2caption_v1.0_not_work.py` намеренно оставлен как неработающий legacy-код.
- Сейчас `training.py` выбирает `ConfigL()` через `config = ConfigL() if args.size.upper() else ConfigS()`, что всегда приводит к `ConfigL` при непустом значении `--size`.
- `model/trainer.py` использует `self.dataset` в `test_step`, тогда как в инициализаторе присваивается `self.test_dataset`; это может ломать семплинг во время запуска обучения, если не откорректировать код.
- `video2caption_v1.1.py` обращается к `self.config.transform`, но `ConfigS`/`ConfigL` не определяют `transform`.
- В текущем срезе репозитория нет CI/тестов.
- Примечание по i18n: ссылки на языки находятся в начале этого README; переводы можно добавлять в `i18n/`.
- Текущая пометка: языковая полоса ссылается на `i18n/README.ru.md`, но этот файл отсутствовал в исходном снимке.

## 🩺 Устранение неполадок

- `AssertionError: Image does not exist`
  - Проверьте, что `-I/--img-path` указывает на существующий файл.
- `Dataset file not found. Downloading...`
  - `MiniFlickrDataset` вызывает это, если `data/processed/dataset.pkl` отсутствует; сначала выполните `python dataset_generation.py`.
- `Path to the test image folder does not exist`
  - Убедитесь, что путь в `evaluate.py -I` существует.
- Медленный или падающий первый запуск
  - На первом запуске идёт загрузка моделей с Hugging Face и, возможно, контрольных точек с Google Drive.
- `video2caption.py` возвращает пустые подписи
  - Проверьте жёстко заданный путь к скрипту и путь Python, или переключитесь на `v2c.py`.
- `wandb` запрашивает вход во время обучения
  - Выполните `wandb login` или отключите логирование вручную в `training.py`, если нужно.

## 🛣️ Дорожная карта

- Добавить lockfile зависимостей (`requirements.txt` или `pyproject.toml`) для воспроизводимых установок.
- Объединить дублирующиеся видеопроцессы в одну поддерживаемую реализацию.
- Убрать жестко заданные пути машин из legacy-скриптов.
- Исправить известные граничные ошибки обучения/оценки в `training.py` и `model/trainer.py`.
- Добавить автоматические тесты и CI.
- Заполнить `i18n/` файлами README для языков из языковой полосы.

## 🤝 Вклад

Приветствуется участие. Предлагаемый рабочий процесс:

```bash
# 1) Fork и clone
git clone git@github.com:<your-user>/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip

# 2) Создать feature branch
git checkout -b feat/your-change

# 3) Внести изменения и сделать commit
git add .
git commit -m "feat: describe your change"

# 4) Push и открыть PR
git push origin feat/your-change
```

Если вы меняете поведение модели, приложите:

- воспроизводимые команды;
- до/после примеры вывода;
- заметки о предположениях, касающихся чекпоинтов или датасета.

---

## 📄 Лицензия

В текущем снимке репозитория отсутствует файл лицензии.

Примечание: до добавления `LICENSE` условия повторного использования и распространения не определены.


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
