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

أداة Python لتوليد تعليقات توضيحية بلغة طبيعية للصور ومقاطع الفيديو عبر دمج تضمينات التصوير من OpenAI CLIP مع نموذج لغة على نمط GPT.

## 🧭 Snapshot

| البُعد | التفاصيل |
|---|---|
| تغطية المهمة | وصف الصور والفيديو |
| المخرجات الأساسية | ترجمات SRT، نسخ نُصوص JSON، صور مع التعليقات |
| السكربتات الأساسية | `i2c.py`، `v2c.py`، `image2caption.py` |
| المسارات القديمة | `video2caption.py` والنُسخ ذات الأرقام (محتفظ بها للتاريخ) |
| تدفق مجموعة البيانات | `data/raw/results.csv` + `data/raw/flickr30k_images/` |

## ✨ نظرة عامة

يوفّر هذا المستودع:

- سكربتات استنتاج لوصف الصور وإنشاء ترجمات الفيديو.
- خط أنابيب تدريب يتعلم تحويل تضمينات CLIP البصرية إلى تضمينات رموز GPT-2.
- أدوات إنشاء مجموعة بيانات بنمط Flickr30k.
- تنزيل تلقائي لنقاط الحفظ المدعومة عندما تكون الملفات غير موجودة.
- نسخ README متعددة اللغات ضمن `i18n/` (انظر شريط اللغات أعلى الصفحة).

التنفيذ الحالي يجمع بين السكربتات الأحدث والقديمة. تُحتفظ بعض الملفات القديمة كمرجع ومُوثقة أدناه.

## 🚀 الميزات

- وصف صورة واحدة عبر `image2caption.py`.
- وصف الفيديو (بمعاينة إطارات متجانسة) عبر `v2c.py` أو `video2caption.py`.
- خيارات تشغيل قابلة للتخصيص:
  - عدد الإطارات.
  - حجم النموذج.
  - درجة حرارة العينة.
  - اسم نقطة الحفظ.
- استنتاج فيديو متعدد العمليات/الخيوط لسرعة أعلى.
- نتائج المخرجات:
  - ملفات الترجمة الفرعية `SRT` (`.srt`).
  - نسخ JSON (`.json`) في `v2c.py`.
- نقاط دخول للتدريب والتقييم لتجارب ربط CLIP+GPT2.

### لمحة سريعة

| المجال | السكربت(ات) الأساسية | ملاحظات |
|---|---|---|
| وصف الصور | `image2caption.py`، `i2c.py`، `predict.py` | CLI + class قابلة لإعادة الاستخدام |
| وصف الفيديو | `v2c.py` | المسار المُصان الموصى به |
| تدفق الفيديو القديم | `video2caption.py`، `video2caption_v1.1.py` | يحتوي على افتراضات خاصة ببيئة معينة |
| بناء مجموعة البيانات | `dataset_generation.py` | ينتج `data/processed/dataset.pkl` |
| التدريب/التقييم | `training.py`، `evaluate.py` | يستخدم ربط CLIP+GPT2 |

## 🧱 المعمارية (مستوى عالٍ)

النموذج الأساسي في `model/model.py` يتكوّن من ثلاثة أجزاء:

1. `ImageEncoder`: يستخرج embedding صورة CLIP.
2. `Mapping`: يُحوّل embedding CLIP إلى تسلسل embedding تمهيدي لنموذج GPT.
3. `TextDecoder`: رأس نموذج GPT-2 الذي يولد رموز التسمية توضيحياً بشكل تلقائي.

التدريب (`Net.train_forward`) يستخدم تضمينات صور CLIP المحسوبة مسبقًا + التسميات النصية المرمزة.
الاستنتاج (`Net.forward`) يستخدم صورة PIL ويفك الرموز حتى EOS أو `max_len`.

### تدفق البيانات

1. إعداد المجموعة: `dataset_generation.py` يقرأ `data/raw/results.csv` وصور `data/raw/flickr30k_images/`، ثم يكتب `data/processed/dataset.pkl`.
2. التدريب: `training.py` يحمل tuples محجوزة بصيغة `(image_name, image_embedding, caption)` ويدرب طبقات الـ mapper والdecoder.
3. التقييم: `evaluate.py` يعرض التسميات المولدة على صور الاختبار المحتفظ بها.
4. تنفيذ الاستنتاج:
   - الصور: `image2caption.py` / `predict.py` / `i2c.py`.
   - الفيديو: `v2c.py` (موصى به)، `video2caption.py` (قديم).

## 🗂️ هيكل المشروع

```text
VideoCaptionerWithClip/
├── README.md
├── image2caption.py               # واجهة CLI لوصف صورة واحدة
├── predict.py                     # واجهة بديلة لوصف صورة واحدة
├── i2c.py                         # class ImageCaptioner قابلة لإعادة الاستخدام + CLI
├── v2c.py                         # فيديو -> SRT + JSON (استنتاج متعدد الخيوط لكل إطار)
├── video2caption.py               # تنفيذ بديل لفيديو -> SRT (قيود قديمة)
├── video2caption_v1.1.py          # نسخة أقدم
├── video2caption_v1.0_not_work.py # ملف قديم معلن أنه غير يعمل
├── training.py                    # نقطة دخول التدريب
├── evaluate.py                    # تقييم على مجموعة الاختبار وحفظ النتائج المرئية
├── dataset_generation.py          # ينشئ data/processed/dataset.pkl
├── data/
│   ├── __init__.py
│   └── dataset.py                 # Dataset + مساعدات DataLoader
├── model/
│   ├── __init__.py
│   ├── model.py                   # CLIP encoder + mapping + GPT2 decoder
│   └── trainer.py                 # فئة أدوات التدريب/التحقق/الاختبار
├── utils/
│   ├── __init__.py
│   ├── config.py                  # القيم الافتراضية ConfigS / ConfigL
│   ├── downloads.py               # تنزيل نقاط الحفظ من Google Drive
│   └── lr_warmup.py               # جدول زيادة معدل التعلم
├── i18n/                          # نسخ README متعددة اللغات
└── .auto-readme-work/             # مخرجات خط أنابيب auto-README
```

## 📋 المتطلبات المسبقة

- Python `3.10+` موصى به.
- وجود GPU مع دعم CUDA اختياري لكنه موصى به بشدة للتدريب واستنتاج النماذج الكبيرة.
- `ffmpeg` ليس مطلوبًا مباشرةً بواسطة السكربتات الحالية (يُستخدم OpenCV لاستخراج الإطارات).
- يلزم اتصال بالإنترنت في أول مرة فقط لتحميل النماذج/نقاط الحفظ من Hugging Face أو Google Drive.

لا يوجد ملف lockfile حاليًا (`requirements.txt` / `pyproject.toml` مفقود)، لذلك تُستنتج الاعتماديات من الاستيرادات.

## 🛠️ التثبيت

### إعداد قياسي من بنية المستودع الحالية

```bash

git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers pillow matplotlib numpy tqdm opencv-python pandas wandb gdown
```

### مقطع التثبيت المحفوظ من README الأصلي

انتهت نسخة README السابقة في منتصف كتلة. الأوامر الأصلية محفوظة أدناه كما هي، وتُعد مرجعًا تاريخيًا:

```bash
git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip/src
```

ملاحظة: نسخة المستودع الحالية تضع السكربتات في جذر المشروع، وليست داخل `src/`.

## ▶️ بداية سريعة

| الهدف | الأمر |
|---|---|
| وصف صورة | `python image2caption.py -I /path/to/image.jpg -S L -C model.pt` |
| وصف فيديو | `python v2c.py -V /path/to/video.mp4 -N 10` |
| بناء مجموعة البيانات | `python dataset_generation.py` |

### وصف صورة (تشغيل سريع)

```bash
python image2caption.py -I /path/to/image.jpg -S L -C model.pt
```

### وصف فيديو (المسار الموصى به)

```bash
python v2c.py -V /path/to/video.mp4 -N 10
```

## 🎯 الاستخدام

### 1) وصف صورة (`image2caption.py`)

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
- `-R, --res-path`: مجلد الإخراج للصورة المولدة.
- `-T, --temperature`: درجة حرارة العينة.

### 2) CLI بديلة للصور (`predict.py`)

```bash
python predict.py \
  -I /path/to/image.jpg \
  -S L \
  -C model.pt \
  -R ./data/result/prediction \
  -T 1.0
```

`predict.py` وظيفيًا مشابه لـ `image2caption.py`؛ تنسيق النص الناتج يختلف قليلًا.

### 3) class API لوصف الصور (`i2c.py`)

```bash
python i2c.py -I /path/to/image.jpg -S L -C model.pt -R ./data/result/prediction -T 1.0
```

أو استيرادها داخل سكربتك:

```python
from i2c import ImageCaptioner

captioner = ImageCaptioner(model_size="L", checkpoint_name="model.pt")
captioner.set_image_path("/path/to/image.jpg")
caption = captioner.generate_caption(save_image=True)
print(caption)
```

### 4) فيديو إلى ترجمات فرعية + JSON (`v2c.py`)

```bash
python v2c.py -V /path/to/video.mp4 -N 10
```

المخرجات بجانب ملف الفيديو الأصلي:

- `<video_basename>_caption.srt`
- `<video_basename>_caption.json`
- `<video_basename>_captioning_frames/`

### 5) مسار فيديو بديل (`video2caption.py`)

```bash
python video2caption.py -V /path/to/video.mp4 -N 10
```

مهم: هذا السكربت يحتوي حاليًا على مسارات ثابتة مخصصة لجهاز معيّن:

- مسار Python الافتراضي: `/home/lachlan/miniconda3/envs/caption/bin/python`
- مسار سكربت التسمية: `/home/lachlan/Projects/image_captioning/clip-gpt-captioning/src/image2caption.py`

استخدم `v2c.py` ما لم تكن تنوي صيانة هذه المسارات يدويًا.

### 6) النسخة القديمة (`video2caption_v1.1.py`)

هذا السكربت مُحافَظ عليه للمرجعية التاريخية. يُفضّل استخدام `v2c.py` للاستخدام اليومي.

### 7) توليد مجموعة البيانات

```bash
python dataset_generation.py
```

المدخلات المتوقعة:

- `data/raw/results.csv` (جدول تسميات مفصول بعلامة pipe).
- `data/raw/flickr30k_images/` (ملفات الصور المشار إليها في CSV).

المخرج:

- `data/processed/dataset.pkl`

### 8) التدريب

```bash
python training.py -S L -C model.pt
```

يستخدم التدريب سجل `Weights & Biases` (`wandb`) افتراضيًا.

### 9) التقييم

```bash
python evaluate.py \
  -I ./data/raw/flickr30k_images \
  -R ./data/result/eval \
  -S L \
  -C model.pt \
  -T 1.0
```

يقوم التقييم برسم التسميات المولدة على صور الاختبار ويخزنها ضمن:

- `<res-path>/<checkpoint_name_without_ext>_<SIZE>/`

## ⚙️ الإعداد

تعريفات النموذج موجودة في `utils/config.py`:

| Config | CLIP backbone | نموذج GPT | مجلد الوزن |
|---|---|---|---|
| `ConfigS` | `openai/clip-vit-base-patch32` | `gpt2` | `weights/small` |
| `ConfigL` | `openai/clip-vit-large-patch14` | `gpt2-medium` | `weights/large` |

الاعدادات الأساسية من أصناف الإعداد:

| الحقل | `ConfigS` | `ConfigL` |
|---|---:|---:|
| `epochs` | 150 | 120 |
| `lr` | 3e-3 | 5e-3 |
| `batch_size_exp` | 6 | 5 |
| `ep_len` | 4 | 4 |
| `max_len` | 40 | 40 |

معرّفات التنزيل التلقائي لنقاط الحفظ موجودة في `utils/downloads.py`:

| الحجم | معرف Google Drive |
|---|---|
| `L` | `1Gh32arzhW06C1ZJyzcJSSfdJDi3RgWoG` |
| `S` | `1pSQruQyg8KJq6VmzhMLFbT_VaHJMdlWF` |

## 📦 ملفات الإخراج

### استنتاج الصور

- يتم حفظ الصورة مع العنوان/النص المُولد في `--res-path`.
- نمط اسم الملف: `<input_stem>-R<SIZE>.jpg`.

### استنتاج الفيديو (`v2c.py`)

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

## 🧪 الأمثلة

### مثال سريع لوصف صورة

```bash
python image2caption.py -I ./examples/dog.jpg -S S -C model.pt
```

السلوك المتوقع:

- إذا لم يوجد `weights/small/model.pt`، سيتم تنزيله.
- تُكتب صورة موصوفة إلى `./data/result/prediction` افتراضيًا.
- يتم طباعة نص التسمية إلى stdout.

### مثال سريع لوصف فيديو

```bash
python v2c.py -V ./examples/demo.mp4 -N 8
```

السلوك المتوقع:

- يتم وصف 8 إطارات مختارة بالتساوي.
- يتم إنشاء ملفات `.srt` و`.json` بجوار ملف الفيديو الأصلي.

### تسلسل تدريجي كامل للتدريب والتقييم

```bash
python dataset_generation.py
python training.py -S L -C model.pt
python evaluate.py -I ./data/raw/flickr30k_images -R ./data/result/eval -S L -C model.pt -T 1.0
```

## 🧭 ملاحظات التطوير

- يوجد تداخل قديم بين `v2c.py` و`video2caption.py` و`video2caption_v1.*`.
- ملف `video2caption_v1.0_not_work.py` محتفظ به عن قصد كرمز قديم غير عامل.
- `training.py` يختار حاليًا `ConfigL()` عبر `config = ConfigL() if args.size.upper() else ConfigS()`، وهو يُرجع دائمًا `ConfigL` لقيمة غير فارغة في `--size`.
- `model/trainer.py` يستخدم `self.dataset` داخل `test_step`، بينما البادئ يعيّن `self.test_dataset`؛ قد يسبب هذا فشلًا في العينة أثناء التدريب إن لم يُعدَّل.
- `video2caption_v1.1.py` يشير إلى `self.config.transform`، لكن `ConfigS`/`ConfigL` لا يعرّفان `transform`.
- لا توجد حالياً أي مجموعة اختبارات أو CI في لقطة هذا المستودع.
- ملاحظة i18n: توجد روابط اللغات أعلى README؛ قد تُضاف ملفات مترجمة إضافية تحت `i18n/`.
- ملاحظة الحالة الحالية: شريط اللغات يشير إلى `i18n/README.ru.md`، لكن هذا الملف غير موجود في هذه اللقطة.

## 🩺 استكشاف الأخطاء وإصلاحها

- `AssertionError: Image does not exist`
  - تأكد أن `-I/--img-path` يشير إلى ملف صحيح.
- `Dataset file not found. Downloading...`
  - يطلق `MiniFlickrDataset` هذا الخطأ إذا كان `data/processed/dataset.pkl` مفقودًا؛ شغّل `python dataset_generation.py` أولًا.
- `Path to the test image folder does not exist`
  - تأكد أن `evaluate.py -I` يشير إلى مجلد موجود.
- بطء أو فشل في أول تشغيل
  - أول تشغيل قد يقوم بتنزيل نماذج Hugging Face وقد يجلب نقاط حفظ من Google Drive.
- `video2caption.py` يرجّع تسميات فارغة
  - تحقق من مسارات السكربت ومسار تنفيذ Python المحددين صلبًا، أو استخدم `v2c.py`.
- ظهور طلب تسجيل الدخول في `wandb` أثناء التدريب
  - شغّل `wandb login` أو عطل التسجيل يدويًا في `training.py` إذا لزم الأمر.

## 🛣️ خارطة الطريق

- إضافة lockfiles للاعتماديات (`requirements.txt` أو `pyproject.toml`) لتثبيتات قابلة لإعادة الإنتاج.
- توحيد مسارات الفيديو المكررة في تنفيذ واحد مُصان.
- إزالة المسارات الصلبة الخاصة بجهاز محدد من السكربتات القديمة.
- إصلاح أخطاء الحواف المعروفة في `training.py` و`model/trainer.py`.
- إضافة اختبارات وتكامل CI تلقائي.
- تعبئة `i18n/` بملفات README مترجمة كما هو موضح في شريط اللغات.

## 🤝 المساهمة

المساهمات مرحب بها. مقترح سير العمل:

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

إذا قمت بتغيير سلوك النموذج، أضف:

- أمرًا قابلًا لإعادة الإنتاج.
- عينات مخرجات قبل/بعد.
- ملاحظات حول افتراضات نقاط الحفظ أو مجموعة البيانات.

## ❤️ Support

| Donate | PayPal | Stripe |
|---|---|---|
| [![Donate](https://img.shields.io/badge/Donate-LazyingArt-0EA5E9?style=for-the-badge&logo=ko-fi&logoColor=white)](https://chat.lazying.art/donate) | [![PayPal](https://img.shields.io/badge/PayPal-RongzhouChen-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/RongzhouChen) | [![Stripe](https://img.shields.io/badge/Stripe-Donate-635BFF?style=for-the-badge&logo=stripe&logoColor=white)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## 📄 الترخيص

لا يوجد ملف ترخيص في لقطة المستودع الحالية.

ملاحظة افتراضية: حتى يتم إضافة ملف `LICENSE`، تظل شروط إعادة الاستخدام أو التوزيع غير محددة.
