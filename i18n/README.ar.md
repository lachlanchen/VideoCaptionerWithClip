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

أداة Python لتوليد تسميات توضيحية لغوية طبيعية للصور ومقاطع الفيديو عبر دمج التضمينات البصرية من OpenAI CLIP مع نموذج لغوي بأسلوب GPT.

## ✨ نظرة عامة

يوفّر هذا المستودع:

- سكربتات استدلال لوصف الصور وتوليد ترجمات فرعية للفيديو.
- مسار تدريب يتعلّم إسقاطًا من التضمينات البصرية لـ CLIP إلى تضمينات رموز GPT-2.
- أدوات لتوليد مجموعة بيانات على نمط Flickr30k.
- تنزيلًا تلقائيًا لنقاط الحفظ (checkpoints) لأحجام النماذج المدعومة عند غياب الأوزان.
- نسخ README متعددة اللغات ضمن `i18n/` (راجع شريط اللغات أعلاه).

يتضمن التنفيذ الحالي سكربتات أحدث وأخرى قديمة. بعض الملفات القديمة محفوظة للمرجعية ومُوثقة أدناه.

## 🚀 الميزات

- وصف صورة واحدة عبر `image2caption.py`.
- وصف الفيديو (باختيار إطارات بعينة موحّدة) عبر `v2c.py` أو `video2caption.py`.
- خيارات تشغيل قابلة للتخصيص:
  - عدد الإطارات.
  - حجم النموذج.
  - درجة حرارة أخذ العينات.
  - اسم نقطة الحفظ.
- وصف فيديو متعدد العمليات/متعدد الخيوط لتسريع الاستدلال.
- نواتج الإخراج:
  - ملفات ترجمة فرعية SRT (`.srt`).
  - سجلات JSON (`.json`) في `v2c.py`.
- نقاط دخول للتدريب والتقييم لتجارب الربط بين CLIP وGPT2.

### لمحة سريعة

| المجال | السكربت(ات) الأساسية | ملاحظات |
|---|---|---|
| وصف الصور | `image2caption.py`, `i2c.py`, `predict.py` | CLI + صنف قابل لإعادة الاستخدام |
| وصف الفيديو | `v2c.py` | المسار المُوصى به والمحفوظ حاليًا |
| تدفق فيديو قديم | `video2caption.py`, `video2caption_v1.1.py` | يتضمن افتراضات خاصة بجهاز معيّن |
| بناء مجموعة البيانات | `dataset_generation.py` | يُنتج `data/processed/dataset.pkl` |
| التدريب / التقييم | `training.py`, `evaluate.py` | يستخدم ربط CLIP+GPT2 |

## 🧱 البنية (مستوى عالٍ)

يتكوّن النموذج الأساسي في `model/model.py` من ثلاثة أجزاء:

1. `ImageEncoder`: يستخرج تضمين صورة CLIP.
2. `Mapping`: يُسقط تضمين CLIP إلى تسلسل تضمينات بادئة GPT.
3. `TextDecoder`: رأس نموذج GPT-2 اللغوي الذي يولّد رموز الوصف بشكل توليدي ذاتي.

التدريب (`Net.train_forward`) يستخدم تضمينات صور CLIP المحسوبة مسبقًا + التسميات النصية بعد ترميزها.
الاستدلال (`Net.forward`) يستخدم صورة PIL ويفك الرموز حتى EOS أو `max_len`.

### تدفّق البيانات

1. تجهيز مجموعة البيانات: `dataset_generation.py` يقرأ `data/raw/results.csv` والصور في `data/raw/flickr30k_images/`، ثم يكتب `data/processed/dataset.pkl`.
2. التدريب: `training.py` يحمّل العناصر المحفوظة بصيغة `(image_name, image_embedding, caption)` ويدرّب طبقات mapper/decoder.
3. التقييم: `evaluate.py` يعرض التسميات المولدة فوق صور اختبار محجوزة.
4. تقديم الاستدلال:
   - الصور: `image2caption.py` / `predict.py` / `i2c.py`.
   - الفيديو: `v2c.py` (موصى به)، `video2caption.py` (قديم).

## 🗂️ هيكل المشروع

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

## 📋 المتطلبات المسبقة

- يُوصى بـ Python `3.10+`.
- وجود GPU يدعم CUDA اختياري لكنه موصى به بشدة للتدريب والاستدلال بالنماذج الكبيرة.
- `ffmpeg` غير مطلوب مباشرة في السكربتات الحالية (يُستخدم OpenCV لاستخراج الإطارات).
- يلزم اتصال بالإنترنت في أول مرة لتنزيل النماذج/نقاط الحفظ من Hugging Face / Google Drive.

لا يوجد lockfile حاليًا (`requirements.txt` / `pyproject.toml` غير موجودين)، لذا يتم استنتاج الاعتماديات من الاستيرادات.

## 🛠️ التثبيت

### الإعداد القياسي وفق البنية الحالية للمستودع

```bash
git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers pillow matplotlib numpy tqdm opencv-python pandas wandb gdown
```

### مقطع التثبيت من README الأصلي (محفوظ)

انتهت النسخة السابقة من README في منتصف كتلة. الأوامر الأصلية محفوظة أدناه كما هي تمامًا باعتبارها محتوى تاريخيًا مرجعيًا:

```bash
git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip/src
```

ملاحظة: اللقطة الحالية من المستودع تضع السكربتات في الجذر، وليس ضمن `src/`.

## ▶️ بدء سريع

### وصف الصور (تشغيل سريع)

```bash
python image2caption.py -I /path/to/image.jpg -S L -C model.pt
```

### وصف الفيديو (المسار الموصى به)

```bash
python v2c.py -V /path/to/video.mp4 -N 10
```

## 🎯 الاستخدام

### 1. وصف الصور (`image2caption.py`)

```bash
python image2caption.py \
  -I /path/to/image.jpg \
  -S L \
  -C model.pt \
  -R ./data/result/prediction \
  -T 1.0
```

المعاملات:

- `-I, --img-path`: مسار صورة الإدخال.
- `-S, --size`: حجم النموذج (`S` أو `L`).
- `-C, --checkpoint-name`: اسم ملف نقطة الحفظ داخل `weights/{small|large}`.
- `-R, --res-path`: مجلد الإخراج للصورة المرسومة مع الوصف.
- `-T, --temperature`: درجة حرارة أخذ العينات.

### 2. واجهة CLI بديلة للصور (`predict.py`)

```bash
python predict.py \
  -I /path/to/image.jpg \
  -S L \
  -C model.pt \
  -R ./data/result/prediction \
  -T 1.0
```

`predict.py` مشابه وظيفيًا لـ `image2caption.py`؛ تنسيق نص الإخراج يختلف قليلًا.

### 3. واجهة الصنف البرمجية لوصف الصور (`i2c.py`)

```bash
python i2c.py -I /path/to/image.jpg -S L -C model.pt -R ./data/result/prediction -T 1.0
```

أو استورده في سكربتك الخاص:

```python
from i2c import ImageCaptioner

captioner = ImageCaptioner(model_size="L", checkpoint_name="model.pt")
captioner.set_image_path("/path/to/image.jpg")
caption = captioner.generate_caption(save_image=True)
print(caption)
```

### 4. فيديو إلى ترجمة فرعية + JSON (`v2c.py`)

```bash
python v2c.py -V /path/to/video.mp4 -N 10
```

المخرجات بجوار ملف الفيديو المُدخل:

- `<video_basename>_caption.srt`
- `<video_basename>_caption.json`
- `<video_basename>_captioning_frames/`

### 5. مسار فيديو بديل (`video2caption.py`)

```bash
python video2caption.py -V /path/to/video.mp4 -N 10
```

مهم: هذا السكربت يتضمن حاليًا مسارات hardcoded خاصة بجهاز معيّن:

- مسار Python الافتراضي: `/home/lachlan/miniconda3/envs/caption/bin/python`
- مسار سكربت الوصف: `/home/lachlan/Projects/image_captioning/clip-gpt-captioning/src/image2caption.py`

استخدم `v2c.py` ما لم تكن تقصد صيانة هذه المسارات.

### 6. النسخة القديمة (`video2caption_v1.1.py`)

هذا السكربت محفوظ كمرجع تاريخي. للاستخدام الفعلي، يُفضّل `v2c.py`.

### 7. توليد مجموعة البيانات

```bash
python dataset_generation.py
```

مدخلات الخام المتوقعة:

- `data/raw/results.csv` (جدول تسميات مفصول بعلامة pipe).
- `data/raw/flickr30k_images/` (ملفات الصور المشار إليها في CSV).

الإخراج:

- `data/processed/dataset.pkl`

### 8. التدريب

```bash
python training.py -S L -C model.pt
```

يستخدم التدريب تسجيل Weights & Biases (`wandb`) افتراضيًا.

### 9. التقييم

```bash
python evaluate.py \
  -I ./data/raw/flickr30k_images \
  -R ./data/result/eval \
  -S L \
  -C model.pt \
  -T 1.0
```

يقوم التقييم برسم التسميات المتوقعة على صور الاختبار ويحفظها ضمن:

- `<res-path>/<checkpoint_name_without_ext>_<SIZE>/`

## ⚙️ الإعداد

تعريفات إعدادات النموذج موجودة في `utils/config.py`:

| Config | CLIP backbone | GPT model | Weights dir |
|---|---|---|---|
| `ConfigS` | `openai/clip-vit-base-patch32` | `gpt2` | `weights/small` |
| `ConfigL` | `openai/clip-vit-large-patch14` | `gpt2-medium` | `weights/large` |

أهم القيم الافتراضية من أصناف الإعداد:

| Field | `ConfigS` | `ConfigL` |
|---|---:|---:|
| `epochs` | 150 | 120 |
| `lr` | 3e-3 | 5e-3 |
| `batch_size_exp` | 6 | 5 |
| `ep_len` | 4 | 4 |
| `max_len` | 40 | 40 |

مُعرّفات التنزيل التلقائي لنقاط الحفظ موجودة في `utils/downloads.py`:

| Size | Google Drive ID |
|---|---|
| `L` | `1Gh32arzhW06C1ZJyzcJSSfdJDi3RgWoG` |
| `S` | `1pSQruQyg8KJq6VmzhMLFbT_VaHJMdlWF` |

## 📦 ملفات الإخراج

### استدلال الصور

- صورة محفوظة مع عنوان/وصف مولَّد ومُركّب عليها في `--res-path`.
- نمط اسم الملف: `<input_stem>-R<SIZE>.jpg`.

### استدلال الفيديو (`v2c.py`)

- SRT: `<video_stem>_caption.srt`
- JSON: `<video_stem>_caption.json`
- صور الإطارات: `<video_stem>_captioning_frames/`

مثال عنصر JSON:

```json
{
  "start": "00:00:03,200",
  "end": "00:00:03,700",
  "lang": "en",
  "text": "A dog running through a field."
}
```

## 🧪 أمثلة

### مثال سريع لوصف صورة

```bash
python image2caption.py -I ./examples/dog.jpg -S S -C model.pt
```

السلوك المتوقع:

- إذا كان `weights/small/model.pt` مفقودًا، فسيتم تنزيله.
- تُكتب صورة موصوفة في `./data/result/prediction` افتراضيًا.
- يُطبع نص الوصف إلى stdout.

### مثال سريع لوصف فيديو

```bash
python v2c.py -V ./examples/demo.mp4 -N 8
```

السلوك المتوقع:

- يتم وصف 8 إطارات مأخوذة بعينة موحّدة.
- يتم إنشاء ملفي `.srt` و`.json` بجوار فيديو الإدخال.

### تسلسل تدريب/تقييم من البداية للنهاية

```bash
python dataset_generation.py
python training.py -S L -C model.pt
python evaluate.py -I ./data/raw/flickr30k_images -R ./data/result/eval -S L -C model.pt -T 1.0
```

## 🧭 ملاحظات التطوير

- يوجد تداخل قديم بين `v2c.py` و`video2caption.py` و`video2caption_v1.*`.
- تم الاحتفاظ بـ `video2caption_v1.0_not_work.py` عمدًا ككود قديم غير عامل.
- يختار `training.py` حاليًا `ConfigL()` عبر `config = ConfigL() if args.size.upper() else ConfigS()`، وهو ما ينتهي دائمًا إلى `ConfigL` عند تمرير قيمة غير فارغة لـ `--size`.
- يستخدم `model/trainer.py` المتغير `self.dataset` داخل `test_step`، بينما يعيّن المُهيّئ `self.test_dataset`؛ وقد يؤدي هذا إلى كسر أخذ العينات أثناء التدريب ما لم يُعدّل.
- يشير `video2caption_v1.1.py` إلى `self.config.transform`، لكن `ConfigS`/`ConfigL` لا يعرّفان `transform`.
- لا توجد حاليًا مجموعة اختبارات/تكامل CI معرفة في لقطة المستودع هذه.
- ملاحظة i18n: روابط اللغات موجودة أعلى هذا README؛ ويمكن إضافة الملفات المترجمة ضمن `i18n/`.
- ملاحظة الحالة الحالية: شريط اللغات يربط إلى `i18n/README.ru.md`، لكن هذا الملف غير موجود في هذه اللقطة.

## 🩺 استكشاف الأخطاء وإصلاحها

- `AssertionError: Image does not exist`
  - تأكد أن `-I/--img-path` يشير إلى ملف صالح.
- `Dataset file not found. Downloading...`
  - يطلق `MiniFlickrDataset` هذه الرسالة عند غياب `data/processed/dataset.pkl`؛ شغّل `python dataset_generation.py` أولًا.
- `Path to the test image folder does not exist`
  - تأكد أن `evaluate.py -I` يشير إلى مجلد موجود.
- بطء التشغيل الأول أو فشله
  - التشغيل الأول ينزّل نماذج Hugging Face وقد ينزّل نقاط حفظ من Google Drive.
- `video2caption.py` يعيد أوصافًا فارغة
  - تحقّق من مسار السكربت ومسار Python المضمّنين، أو انتقل إلى `v2c.py`.
- `wandb` يطلب تسجيل الدخول أثناء التدريب
  - شغّل `wandb login` أو عطّل التسجيل يدويًا في `training.py` إذا لزم.

## 🛣️ خارطة الطريق

- إضافة lockfiles للاعتماديات (`requirements.txt` أو `pyproject.toml`) لتثبيتات قابلة لإعادة الإنتاج.
- توحيد مسارات الفيديو المكررة في تنفيذ واحد مُصان.
- إزالة مسارات الأجهزة hardcoded من السكربتات القديمة.
- إصلاح أخطاء الحواف المعروفة في التدريب/التقييم داخل `training.py` و`model/trainer.py`.
- إضافة اختبارات آلية وتكامل CI.
- ملء `i18n/` بملفات README مترجمة مشار إليها في شريط اللغات.

## 🤝 المساهمة

المساهمات مرحّب بها. سير عمل مقترح:

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

إذا غيّرت سلوك النموذج، أرفق:

- أوامر قابلة لإعادة الإنتاج.
- عينات مخرجات قبل/بعد.
- ملاحظات حول افتراضات نقطة الحفظ أو مجموعة البيانات.

## 🙌 الدعم

لا توجد إعدادات صريحة للتبرعات/الرعاية في لقطة المستودع الحالية.

إذا أضيفت روابط رعاية لاحقًا، فيجب الحفاظ عليها في هذا القسم.

## 📄 الترخيص

لا يوجد ملف ترخيص في لقطة المستودع الحالية.

ملاحظة افتراضية: إلى أن يُضاف ملف `LICENSE`، تبقى شروط إعادة الاستخدام/التوزيع غير محددة.
