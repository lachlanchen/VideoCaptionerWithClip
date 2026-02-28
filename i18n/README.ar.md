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

## 🧭 تنقل سريع

| القسم | ما هو استخدامه |
|---|---|
| لقطة سريعة | عرض نطاق المستودع وسجل السكربتات الحالية |
| النظرة العامة | قراءة الأهداف والقدرات |
| الاستخدام | اتباع سير عمل CLI / API بدقة |
| استكشاف الأخطاء | حل المشكلات الشائعة أثناء التشغيل بسرعة |
| خارطة الطريق | تتبّع أهداف التنظيف والتحسين المعروفة |

---

مجموعة أدوات بايثون لتوليد التسميات التوضيحية النصية للصور والفيديو عبر دمج تمثيلات الرؤية من OpenAI CLIP مع نموذج لغة على نمط GPT.

## 🧭 لقطة سريعة

| البعد | التفاصيل |
|---|---|
| نطاق المهمة | تسمية الصور والفيديو |
| المخرجات الأساسية | ملفات ترجمة SRT، نسخ مكتوبة JSON، صور مرفقة بالتسمية |
| السكربتات الأساسية | `i2c.py`، `v2c.py`، `image2caption.py` |
| المسارات القديمة | `video2caption.py` والإصدارات المماثلة (محتفظ بها للتاريخ) |
| تدفق البيانات | `data/raw/results.csv` + `data/raw/flickr30k_images/` |

## ✨ نظرة عامة

يوفّر هذا المستودع:

- سكربتات استدلال لتسمية الصور وإنشاء ترجمة فيديو.
- خط أنابيب تدريبي يتعلم تحويل تمثيلات CLIP البصرية إلى تمثيلات توكنز GPT-2.
- أدوات إنشاء مجموعات بيانات بنمط Flickr30k.
- تنزيل تلقائي للنقاط المرجعية للنماذج المدعومة عندما تكون الأوزان مفقودة.
- نسخ README متعددة اللغات ضمن `i18n/` (راجع شريط اللغات أعلاه).

التنفيذ الحالي يضم سكربتات حديثة ووراثية. بعض الملفات القديمة محتفظ بها للمرجعية ومذكورة أدناه.

## 🚀 الميزات

- تسمية صورة مفردة عبر `image2caption.py`.
- تسمية فيديو (أخذ إطارات منتظمة) عبر `v2c.py` أو `video2caption.py`.
- خيارات تشغيل قابلة للتخصيص:
  - عدد الإطارات.
  - حجم النموذج.
  - حرارة العينة (temperature).
  - اسم النقطة المرجعية.
- تسمية متعددة العمليات/المُشددة لتسريع استدلال الفيديو.
- مخرجات:
  - ملفات ترجمة SRT (`.srt`).
  - نسخ مكتوبة JSON (`.json`) في `v2c.py`.
- نقاط دخول للتدريب والتقييم لتجارب خريطة CLIP+GPT2.

### نظرة سريعة

| المجال | السكربت الأساسي | ملاحظات |
|---|---|---|
| تسمية الصور | `image2caption.py`، `i2c.py`، `predict.py` | CLI + class قابلة للإعادة |
| تسمية الفيديو | `v2c.py` | المسار المُعتنى به الموصى به |
| مسار الفيديو القديم | `video2caption.py`، `video2caption_v1.1.py` | يحتوي افتراضات خاصة بالجهاز |
| بناء مجموعة البيانات | `dataset_generation.py` | ينتج `data/processed/dataset.pkl` |
| التدريب / التقييم | `training.py`، `evaluate.py` | يستخدم خريطة CLIP+GPT2 |

## 🧱 البنية المعمارية (عام)

النموذج الأساسي في `model/model.py` يتكوّن من ثلاثة أجزاء:

1. `ImageEncoder`: لاستخراج embedding للصورة من CLIP.
2. `Mapping`: إسقاط embedding الصورة في GPT prefix embedding sequence.
3. `TextDecoder`: رأس نموذج لغة GPT-2 الذي يولّد توكنات التسمية بطريقة autoregressive.

التدريب (`Net.train_forward`) يستخدم تمثيلات الصور المعدة مسبقًا من CLIP + التسميات المرمزة.
الاستنتاج (`Net.forward`) يستخدم صورة PIL ويفك الشيفرة حتى EOS أو `max_len`.

### تدفق البيانات

1. إعداد مجموعة البيانات: `dataset_generation.py` تقرأ `data/raw/results.csv` والصور في `data/raw/flickr30k_images/` وتكتب `data/processed/dataset.pkl`.
2. التدريب: `training.py` يحمل tuples الملتفزة `(image_name, image_embedding, caption)` ويدرب طبقات mapper/decoder.
3. التقييم: `evaluate.py` تعرض التسميات المولدة على صور اختبار احتياطية.
4. تقديم الاستنتاج:
  - صورة: `image2caption.py` / `predict.py` / `i2c.py`.
  - فيديو: `v2c.py` (مُوصى به)، `video2caption.py` (قديم).

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

- Python `3.10+` موصى به.
- وجود GPU يدعم CUDA اختياري لكنه قويًا جدًا للتدريب والاستدلال بالنماذج الكبيرة.
- `ffmpeg` غير مطلوب مباشرة من السكربتات الحالية (يُستخدم OpenCV لاستخراج الإطارات).
- يحتاج الوصول إلى الإنترنت عند أول تنزيل للنماذج/النقاط المرجعية من Hugging Face / Google Drive.

لا يوجد lockfile حاليًا (`requirements.txt` / `pyproject.toml` مفقودان)، لذلك تُستنتج التبعيات من الواردات.

## 🛠️ التثبيت

### الإعداد القياسي من ترتيب المستودع الحالي

```bash
git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers pillow matplotlib numpy tqdm opencv-python pandas wandb gdown
```

### مقتطف التثبيت من README الأصلي (مُحافظ عليه)

اكتمل README السابق في منتصف كتلة الأوامر. الأوامر الأصلية محفوظة أدناه كما هي كنصّ مرجعي تاريخي:

```bash
git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip/src
```

ملاحظة: لقطة المستودع الحالية تضع السكربتات في جذر المستودع، وليس داخل `src/`.

## ▶️ بدء سريع

| الهدف | الأمر |
|---|---|
| تسمية صورة | `python image2caption.py -I /path/to/image.jpg -S L -C model.pt` |
| تسمية فيديو | `python v2c.py -V /path/to/video.mp4 -N 10` |
| بناء مجموعة بيانات | `python dataset_generation.py` |

### تشغيل سريع لتسمية صورة

```bash
python image2caption.py -I /path/to/image.jpg -S L -C model.pt
```

### توصية تسمية فيديو

```bash
python v2c.py -V /path/to/video.mp4 -N 10
```

## 🎯 الاستخدام

### 1. تسمية الصورة (`image2caption.py`)

```bash
python image2caption.py \
  -I /path/to/image.jpg \
  -S L \
  -C model.pt \
  -R ./data/result/prediction \
  -T 1.0
```

المعاملات:

- `-I, --img-path`: مسار الصورة المدخل.
- `-S, --size`: حجم النموذج (`S` أو `L`).
- `-C, --checkpoint-name`: اسم النقطة المرجعية داخل `weights/{small|large}`.
- `-R, --res-path`: مجلد الإخراج للصورة المولّفة بالتسمية.
- `-T, --temperature`: حرارة العينة.

### 2. واجهة CLI البديلة (`predict.py`)

```bash
python predict.py \
  -I /path/to/image.jpg \
  -S L \
  -C model.pt \
  -R ./data/result/prediction \
  -T 1.0
```

`predict.py` يعمل وظيفيًا بشكل مشابه لـ `image2caption.py`، مع اختلاف طفيف في تنسيق نص الناتج.

### 3. API فئة التسمية للصورة (`i2c.py`)

```bash
python i2c.py -I /path/to/image.jpg -S L -C model.pt -R ./data/result/prediction -T 1.0
```

أو استيراد داخل سكربتك:

```python
from i2c import ImageCaptioner

captioner = ImageCaptioner(model_size="L", checkpoint_name="model.pt")
captioner.set_image_path("/path/to/image.jpg")
caption = captioner.generate_caption(save_image=True)
print(caption)
```

### 4. فيديو إلى ترجمة + JSON (`v2c.py`)

```bash
python v2c.py -V /path/to/video.mp4 -N 10
```

المخرجات تظهر بجانب الفيديو المدخل:

- `<video_basename>_caption.srt`
- `<video_basename>_caption.json`
- `<video_basename>_captioning_frames/`

### 5. مسار الفيديو البديل (`video2caption.py`)

```bash
python video2caption.py -V /path/to/video.mp4 -N 10
```

مهم: يحتوي هذا السكربت حاليًا على مسارات صلبة محددة لجهاز معيّن:

- مسار Python الافتراضي: `/home/lachlan/miniconda3/envs/caption/bin/python`
- مسار سكربت التسمية: `/home/lachlan/Projects/image_captioning/clip-gpt-captioning/src/image2caption.py`

استخدم `v2c.py` ما لم تكن تقصد الحفاظ على هذه المسارات يدويًا.

### 6. نسخة قديمة (`video2caption_v1.1.py`)

هذا السكربت محتفظ به كمراجع تاريخية. يفضّل استخدام `v2c.py` للاستخدام النشط.

### 7. إنشاء مجموعة البيانات

```bash
python dataset_generation.py
```

المُدخلات المتوقعة:

- `data/raw/results.csv` (جدول التعليقات مفصول بعلامة `|`).
- `data/raw/flickr30k_images/` (ملفات الصور المشار إليها في CSV).

المخرجات:

- `data/processed/dataset.pkl`

### 8. التدريب

```bash
python training.py -S L -C model.pt
```

يتضمن التدريب تسجيلًا افتراضيًا إلى Weights & Biases (`wandb`).

### 9. التقييم

```bash
python evaluate.py \
  -I ./data/raw/flickr30k_images \
  -R ./data/result/eval \
  -S L \
  -C model.pt \
  -T 1.0
```

يعرض التقييم التسميات المتوقعة على صور الاختبار ويحفظها في:

- `<res-path>/<checkpoint_name_without_ext>_<SIZE>/`

## ⚙️ الإعدادات

تُعرّف إعدادات النموذج في `utils/config.py`:

| الإعداد | واجهة CLIP الأساسية | نموذج GPT | مجلد الأوزان |
|---|---|---|
| `ConfigS` | `openai/clip-vit-base-patch32` | `gpt2` | `weights/small` |
| `ConfigL` | `openai/clip-vit-large-patch14` | `gpt2-medium` | `weights/large` |

القيم الافتراضية الرئيسية من فئات الإعداد:

| الحقل | `ConfigS` | `ConfigL` |
|---|---:|---:|
| `epochs` | 150 | 120 |
| `lr` | 3e-3 | 5e-3 |
| `batch_size_exp` | 6 | 5 |
| `ep_len` | 4 | 4 |
| `max_len` | 40 | 40 |

معرفات التنزيل التلقائي للنقاط المرجعية موجودة في `utils/downloads.py`:

| الحجم | معرف Google Drive |
|---|---|
| `L` | `1Gh32arzhW06C1ZJyzcJSSfdJDi3RgWoG` |
| `S` | `1pSQruQyg8KJq6VmzhMLFbT_VaHJMdlWF` |

## 📦 ملفات المخرجات

### استدلال الصورة

- صورة محفوظة بعنوان أو عنوان مُدرج فوق الصورة في `--res-path`.
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

### مثال سريع لتسمية صورة

```bash
python image2caption.py -I ./examples/dog.jpg -S S -C model.pt
```

السلوك المتوقع:

- إذا كان `weights/small/model.pt` مفقودًا، يتم تنزيله تلقائيًا.
- تُكتب صورة مرفقة بالتسمية تلقائيًا في `./data/result/prediction` بشكل افتراضي.
- يُطبع نص التسمية على stdout.

### مثال سريع لتسمية فيديو

```bash
python v2c.py -V ./examples/demo.mp4 -N 8
```

السلوك المتوقع:

- سيتم تسمية 8 إطارات مُؤخذة بشكل منتظم.
- تُنتج ملفات `.srt` و`.json` بجانب ملف الفيديو المدخل.

### تسلسل تدريبي/تقييمي من النهاية إلى النهاية

```bash
python dataset_generation.py
python training.py -S L -C model.pt
python evaluate.py -I ./data/raw/flickr30k_images -R ./data/result/eval -S L -C model.pt -T 1.0
```

## 🧭 ملاحظات التطوير

- يوجد تداخل قديم بين `v2c.py` و `video2caption.py` و `video2caption_v1.*`.
- ملف `video2caption_v1.0_not_work.py` محتفظ به عمدًا ككود قديم غير عامل.
- `training.py` يختار حاليًا `ConfigL()` عبر `config = ConfigL() if args.size.upper() else ConfigS()`، وهذا يعود دائمًا إلى `ConfigL` لقيم `--size` غير الفارغة.
- `model/trainer.py` يستخدم `self.dataset` داخل `test_step`، بينما المُهيئ يعيّن `self.test_dataset`؛ هذا قد يسبب انقطاعًا في العينة أثناء تشغيلات التدريب ما لم تتم المعالجة.
- `video2caption_v1.1.py` يشير إلى `self.config.transform` لكن `ConfigS` / `ConfigL` لا يعرّفان `transform`.
- لا توجد سير عمل CI / اختبارات مفعّلة في لقطة هذا المستودع.
- ملاحظة i18n: روابط اللغات موجودة أعلى هذا الـREADME، ويمكن إضافة ملفات مترجمة تحت `i18n/`.
- ملاحظة الحالة الحالية: رابط شريط اللغة يشير إلى `i18n/README.ru.md`، لكن هذا الملف غير موجود في هذه اللقطة.

## 🩺 استكشاف المشكلات

- `AssertionError: Image does not exist`
  - تأكد من أن `-I/--img-path` يشير إلى ملف صحيح.
- `Dataset file not found. Downloading...`
  - يرفع `MiniFlickrDataset` هذا التحذير عند فقدان `data/processed/dataset.pkl`; شغّل `python dataset_generation.py` أولًا.
- `Path to the test image folder does not exist`
  - تأكد من أن `evaluate.py -I` يشير إلى مجلد موجود.
- بطء أو فشل في التشغيل الأول
  - التشغيل الأول ينزّل نماذج Hugging Face وقد يجلب نقاطًا مرجعية من Google Drive.
- `video2caption.py` تعيد تسميات فارغة
  - تحقق من مسارات السكربت ومسار Python الصلبة، أو انتقل إلى `v2c.py`.
- `wandb` يطلب تسجيل الدخول أثناء التدريب
  - نفّذ `wandb login` أو عطّل التسجيل يدويًا في `training.py` إذا لزم الأمر.

## 🛣️ خارطة الطريق

- إضافة ملفات قفل تبعيات (`requirements.txt` أو `pyproject.toml`) لتثبيت قابل لإعادة الإنتاج.
- توحيد مسارات الفيديو المكررة ضمن تنفيذ واحد مُعتنى به.
- إزالة المسارات الصلبة للأجهزة من السكربتات القديمة.
- إصلاح أخطاء الحالات الحدّية المعروفة في `training.py` و `model/trainer.py`.
- إضافة اختبارات آلية و CI.
- تعبئة `i18n/` بملفات README مترجمة كما هو مذكور في شريط اللغات.

## 🤝 المساهمة

المساهمات مرحّبة. سير عمل مقترح:

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

إذا غيّرت سلوك النموذج، أدرج:

- أوامر قابلة للإعادة.
- مخرجات نموذجية قبل/بعد.
- ملاحظات بشأن افتراضات نقاط المرجعية أو مجموعة البيانات.

---

## 📄 الترخيص

لا يوجد ملف ترخيص في لقطة المستودع الحالية.

ملاحظة افتراضية: حتى تتم إضافة ملف `LICENSE`، تبقى شروط إعادة الاستخدام والتوزيع غير محددة.


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
