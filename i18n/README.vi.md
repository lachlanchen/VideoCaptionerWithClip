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

## 🧭 Quick Navigation

| Mục | Dùng để |
|---|---|
| Snapshot | Xem phạm vi repo và danh mục script hiện tại |
| Overview | Đọc mục tiêu và phạm vi năng lực |
| Usage | Thực hiện đúng các quy trình CLI/API |
| Troubleshooting | Khắc phục nhanh các lỗi chạy thường gặp |
| Roadmap | Theo dõi các mục tối ưu/sửa lỗi đã biết |

---

Bộ công cụ Python sinh mô tả ngôn ngữ tự nhiên cho ảnh và video bằng cách kết hợp embedding hình ảnh từ OpenAI CLIP với mô hình ngôn ngữ kiểu GPT.

## 🧭 Snapshot

| Phạm vi | Chi tiết |
|---|---|
| Phạm vi tác vụ | Sinh caption cho ảnh và video |
| Kết quả chính | Subtitle SRT, transcript JSON, ảnh đã gắn caption |
| Script chính | `i2c.py`, `v2c.py`, `image2caption.py` |
| Đường đi legacy | `video2caption.py` và các phiên bản liên quan (giữ để tham chiếu lịch sử) |
| Luồng dữ liệu | `data/raw/results.csv` + `data/raw/flickr30k_images/` |

## ✨ Tổng quan

Repo này cung cấp:

- Script suy luận cho caption ảnh và sinh phụ đề video.
- Pipeline huấn luyện ánh xạ embedding ảnh CLIP sang embedding token của GPT-2.
- Tiện ích tạo bộ dữ liệu theo phong cách Flickr30k.
- Tự động tải checkpoint theo kích thước mô hình khi thiếu file trọng số.
- Các bản README đa ngôn ngữ trong `i18n/` (xem thanh ngôn ngữ phía trên).

Triển khai hiện tại có cả script mới và script kế thừa. Một số file cũ được giữ lại để tham khảo và được mô tả bên dưới.

## 🚀 Tính năng

- Sinh caption cho ảnh đơn qua `image2caption.py`.
- Sinh caption video (lấy mẫu frame đều nhau) qua `v2c.py` hoặc `video2caption.py`.
- Tùy chỉnh thời gian chạy:
  - Số frame.
  - Kích thước mô hình.
  - Nhiệt độ lấy mẫu.
  - Tên checkpoint.
- Song song/multi-process để suy luận video nhanh hơn.
- Đầu ra:
  - File subtitle SRT (`.srt`).
  - Transcript JSON (`.json`) trong `v2c.py`.
- Điểm khởi đầu huấn luyện và đánh giá cho thử nghiệm ánh xạ CLIP+GPT2.

### Tóm tắt nhanh

| Khu vực | Script chính | Ghi chú |
|---|---|---|
| Caption ảnh | `image2caption.py`, `i2c.py`, `predict.py` | Có cả CLI và class có thể tái sử dụng |
| Caption video | `v2c.py` | Đường dẫn đang được duy trì khuyến nghị |
| Luồng kế thừa video | `video2caption.py`, `video2caption_v1.1.py` | Chứa giả định riêng cho máy cụ thể |
| Tạo dataset | `dataset_generation.py` | Tạo `data/processed/dataset.pkl` |
| Train / eval | `training.py`, `evaluate.py` | Dùng ánh xạ CLIP+GPT2 |

## 🧱 Kiến trúc (Tổng quan)

Mô hình lõi trong `model/model.py` gồm ba phần:

1. `ImageEncoder`: trích xuất embedding ảnh từ CLIP.
2. `Mapping`: chiếu embedding CLIP thành dãy embedding tiền tố cho GPT.
3. `TextDecoder`: head mô hình GPT-2 sinh token caption theo autoregressive.

Huấn luyện (`Net.train_forward`) dùng trước embedding ảnh CLIP đã tiền xử lý + caption đã tokenize.
Suy luận (`Net.forward`) nhận ảnh PIL và giải mã token đến khi gặp EOS hoặc `max_len`.

### Luồng dữ liệu

1. Chuẩn bị dataset: `dataset_generation.py` đọc `data/raw/results.csv` và ảnh trong `data/raw/flickr30k_images/`, ghi ra `data/processed/dataset.pkl`.
2. Huấn luyện: `training.py` đọc tuple pickle `(image_name, image_embedding, caption)` và huấn luyện các lớp mapper/decoder.
3. Đánh giá: `evaluate.py` render caption sinh ra lên tập ảnh test.
4. Thực thi suy luận:
   - ảnh: `image2caption.py` / `predict.py` / `i2c.py`.
   - video: `v2c.py` (khuyến nghị), `video2caption.py` (legacy).

## 🗂️ Cấu trúc dự án

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

## 📋 Yêu cầu

- Khuyến nghị dùng Python `3.10+`.
- GPU có hỗ trợ CUDA là không bắt buộc nhưng rất khuyến nghị cho huấn luyện và suy luận mô hình lớn.
- `ffmpeg` không bắt buộc trực tiếp cho các script hiện tại (OpenCV được dùng để trích frame).
- Cần có truy cập Internet cho lần tải đầu tiên mô hình/checkpoint từ Hugging Face hoặc Google Drive.

Hiện chưa có lockfile (`requirements.txt` / `pyproject.toml` chưa có), nên phụ thuộc được suy ra từ import trong mã nguồn.

## 🛠️ Cài đặt

### Cài đặt chuẩn theo cấu trúc repo hiện tại

```bash
git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers pillow matplotlib numpy tqdm opencv-python pandas wandb gdown
```

### Snippet cài đặt từ README cũ (được giữ nguyên)

README trước kết thúc giữa khối lệnh. Các lệnh gốc được giữ đúng như nguồn lịch sử bên dưới:

```bash
git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip/src
```

Lưu ý: snapshot hiện tại đặt tất cả script ở root repo, không nằm trong `src/`.

## ▶️ Bắt đầu nhanh

| Mục tiêu | Lệnh |
|---|---|
| Caption một ảnh | `python image2caption.py -I /path/to/image.jpg -S L -C model.pt` |
| Caption một video | `python v2c.py -V /path/to/video.mp4 -N 10` |
| Tạo dataset | `python dataset_generation.py` |

### Caption ảnh (chạy nhanh)

```bash
python image2caption.py -I /path/to/image.jpg -S L -C model.pt
```

### Caption video (đường dẫn khuyến nghị)

```bash
python v2c.py -V /path/to/video.mp4 -N 10
```

## 🎯 Hướng dẫn sử dụng

### 1. Caption ảnh (`image2caption.py`)

```bash
python image2caption.py \
  -I /path/to/image.jpg \
  -S L \
  -C model.pt \
  -R ./data/result/prediction \
  -T 1.0
```

Đối số:

- `-I, --img-path`: đường dẫn ảnh đầu vào.
- `-S, --size`: kích thước mô hình (`S` hoặc `L`).
- `-C, --checkpoint-name`: tên file checkpoint trong `weights/{small|large}`.
- `-R, --res-path`: thư mục output cho ảnh đã render caption.
- `-T, --temperature`: tham số nhiệt độ sampling.

### 2. CLI ảnh thay thế (`predict.py`)

```bash
python predict.py \
  -I /path/to/image.jpg \
  -S L \
  -C model.pt \
  -R ./data/result/prediction \
  -T 1.0
```

`predict.py` hoạt động tương tự `image2caption.py`; chỉ khác một chút phần format đầu ra.

### 3. API class caption ảnh (`i2c.py`)

```bash
python i2c.py -I /path/to/image.jpg -S L -C model.pt -R ./data/result/prediction -T 1.0
```

Hoặc import trong script riêng:

```python
from i2c import ImageCaptioner

captioner = ImageCaptioner(model_size="L", checkpoint_name="model.pt")
captioner.set_image_path("/path/to/image.jpg")
caption = captioner.generate_caption(save_image=True)
print(caption)
```

### 4. Caption video thành subtitle + JSON (`v2c.py`)

```bash
python v2c.py -V /path/to/video.mp4 -N 10
```

Kết quả xuất ra cạnh file video gốc:

- `<video_basename>_caption.srt`
- `<video_basename>_caption.json`
- `<video_basename>_captioning_frames/`

### 5. Pipeline thay thế (`video2caption.py`)

```bash
python video2caption.py -V /path/to/video.mp4 -N 10
```

Lưu ý quan trọng: script này hiện còn chứa một số đường dẫn cứng theo máy cụ thể:

- Python mặc định: `/home/lachlan/miniconda3/envs/caption/bin/python`
- Đường dẫn caption script: `/home/lachlan/Projects/image_captioning/clip-gpt-captioning/src/image2caption.py`

Nên dùng `v2c.py` nếu bạn không có nhu cầu duy trì cố định các đường dẫn trên.

### 6. Phiên bản legacy (`video2caption_v1.1.py`)

Script này được giữ lại để tham chiếu lịch sử. Với sử dụng hằng ngày nên chọn `v2c.py`.

### 7. Tạo dataset

```bash
python dataset_generation.py
```

Đầu vào thô mong đợi:

- `data/raw/results.csv` (bảng caption phân tách bằng pipe).
- `data/raw/flickr30k_images/` (các file ảnh được tham chiếu trong CSV).

Đầu ra:

- `data/processed/dataset.pkl`

### 8. Huấn luyện

```bash
python training.py -S L -C model.pt
```

Huấn luyện mặc định log bằng Weights & Biases (`wandb`).

### 9. Đánh giá

```bash
python evaluate.py \
  -I ./data/raw/flickr30k_images \
  -R ./data/result/eval \
  -S L \
  -C model.pt \
  -T 1.0
```

Kết quả đánh giá render caption lên ảnh test và lưu tại:

- `<res-path>/<checkpoint_name_without_ext>_<SIZE>/`

## ⚙️ Cấu hình

Các cấu hình mô hình nằm trong `utils/config.py`:

| Cấu hình | CLIP backbone | Mô hình GPT | Thư mục weights |
|---|---|---|---|
| `ConfigS` | `openai/clip-vit-base-patch32` | `gpt2` | `weights/small` |
| `ConfigL` | `openai/clip-vit-large-patch14` | `gpt2-medium` | `weights/large` |

Các tham số mặc định:

| Trường | `ConfigS` | `ConfigL` |
|---|---:|---:|
| `epochs` | 150 | 120 |
| `lr` | 3e-3 | 5e-3 |
| `batch_size_exp` | 6 | 5 |
| `ep_len` | 4 | 4 |
| `max_len` | 40 | 40 |

ID checkpoint tự động tải được lưu trong `utils/downloads.py`:

| Kích thước | Google Drive ID |
|---|---|
| `L` | `1Gh32arzhW06C1ZJyzcJSSfdJDi3RgWoG` |
| `S` | `1pSQruQyg8KJq6VmzhMLFbT_VaHJMdlWF` |

## 📦 File đầu ra

### Suy luận ảnh

- Ảnh kết quả với chữ caption overlay được lưu tại `--res-path`.
- Mẫu tên file: `<input_stem>-R<SIZE>.jpg`.

### Suy luận video (`v2c.py`)

- SRT: `<video_stem>_caption.srt`
- JSON: `<video_stem>_caption.json`
- Ảnh frame: `<video_stem>_captioning_frames/`

Ví dụ phần tử JSON:

```json
{
  "start": "00:00:03,200",
  "end": "00:00:03,700",
  "lang": "en",
  "text": "A dog running through a field."
}
```

## 🧪 Ví dụ

### Ví dụ caption ảnh nhanh

```bash
python image2caption.py -I ./examples/dog.jpg -S S -C model.pt
```

Hành vi mong đợi:

- Nếu `weights/small/model.pt` chưa có, nó sẽ tự tải.
- Mặc định sẽ tạo ảnh có caption trong `./data/result/prediction`.
- Text caption được in ra stdout.

### Ví dụ caption video nhanh

```bash
python v2c.py -V ./examples/demo.mp4 -N 8
```

Hành vi mong đợi:

- 8 frame được lấy mẫu đều và được caption.
- File `.srt` và `.json` sẽ sinh cạnh video đầu vào.

### Chuỗi huấn luyện + đánh giá end-to-end

```bash
python dataset_generation.py
python training.py -S L -C model.pt
python evaluate.py -I ./data/raw/flickr30k_images -R ./data/result/eval -S L -C model.pt -T 1.0
```

## 🧭 Ghi chú phát triển

- `v2c.py`, `video2caption.py`, và `video2caption_v1.*` có phần chức năng lặp lại.
- `video2caption_v1.0_not_work.py` được giữ có chủ đích như legacy không còn dùng.
- `training.py` hiện đang chọn `ConfigL()` qua `config = ConfigL() if args.size.upper() else ConfigS()`, nên gần như luôn trả về `ConfigL` khi `--size` không rỗng.
- `model/trainer.py` dùng `self.dataset` trong `test_step`, trong khi initializer lại gán `self.test_dataset`; điểm này có thể làm lỗi lấy mẫu trong một số lần train nếu không chỉnh.
- `video2caption_v1.1.py` tham chiếu `self.config.transform`, nhưng `ConfigS`/`ConfigL` không có trường `transform`.
- Repo hiện chưa có CI/test suite.
- Ghi chú i18n: thanh ngôn ngữ nằm đầu README; các file dịch có thể được thêm vào `i18n/`.
- Hiện trạng: thanh ngôn ngữ có liên kết `i18n/README.ru.md`, nhưng file này chưa có trong snapshot này.

## 🩺 Khắc phục sự cố

- `AssertionError: Image does not exist`
  - Kiểm tra `-I/--img-path` trỏ tới một file hợp lệ.
- `Dataset file not found. Downloading...`
  - `MiniFlickrDataset` phát ra lỗi khi thiếu `data/processed/dataset.pkl`; hãy chạy `python dataset_generation.py` trước.
- `Path to the test image folder does not exist`
  - Kiểm tra `evaluate.py -I` trỏ đúng thư mục đã tồn tại.
- Chạy đầu tiên chậm/không ổn
  - Lần đầu có thể phải tải mô hình từ Hugging Face và/hoặc checkpoint từ Google Drive.
- `video2caption.py` trả về caption rỗng
  - Kiểm tra lại đường dẫn hardcode và đường dẫn python executable, hoặc chuyển sang `v2c.py`.
- `wandb` yêu cầu đăng nhập trong quá trình train
  - Chạy `wandb login` hoặc tắt logging thủ công trong `training.py` nếu cần.

## 🛣️ Lộ trình

- Thêm lockfile phụ thuộc (`requirements.txt` hoặc `pyproject.toml`) để cài đặt tái lập.
- Gộp các pipeline video trùng lặp về một bản duy nhất đang duy trì.
- Loại bỏ đường dẫn máy cứng trong các script legacy.
- Sửa lỗi biên đã biết trong huấn luyện/đánh giá tại `training.py` và `model/trainer.py`.
- Thêm test tự động và CI.
- Bổ sung đầy đủ README dịch trong `i18n/` theo đúng ngôn ngữ trên thanh điều hướng.

## 🤝 Đóng góp

Mọi đóng góp đều được chào đón. Quy trình gợi ý:

```bash
# 1) Fork và clone
git clone git@github.com:<your-user>/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip

# 2) Tạo nhánh feature
git checkout -b feat/your-change

# 3) Chỉnh sửa và commit
git add .
git commit -m "feat: describe your change"

# 4) Push và mở PR
git push origin feat/your-change
```

Nếu bạn sửa đổi hành vi model, cần kèm:

- Lệnh reproduce được.
- Ví dụ đầu ra trước/sau.
- Ghi chú về giả định checkpoint hoặc dữ liệu.

---

## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## 📄 License

No license file is present in the current repository snapshot.

Assumption note: until a `LICENSE` file is added, reuse/distribution terms are undefined.
